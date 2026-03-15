"""Stage 1 Training: Video Tokenizer (VAE + FSQ)

Trains the video encoder/decoder for reconstruction with FSQ quantization.

Objectives:
- High-quality video reconstruction
- Efficient latent representation
- Stable FSQ quantization

Usage:
    python scripts/train_stage1.py --config configs/default.yaml
    python scripts/train_stage1.py --config configs/default.yaml --resume outputs/stage1/latest.pt
"""

import argparse
import sys
from pathlib import Path
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Video Tokenizer Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data/videos")
    parser.add_argument("--output", type=str, default="outputs/stage1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="genesis-stage1")
    parser.add_argument("--dry-run", action="store_true", help="Run one batch and exit")
    parser.add_argument("--local-rank", type=int, default=-1)

    return parser.parse_args()


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    rank, world_size, local_rank, distributed = setup_distributed()

    is_main = rank == 0

    if is_main:
        print("=" * 60)
        print("Genesis Stage 1: Video Tokenizer Training")
        print("=" * 60)
        print(f"Distributed: {distributed} (world_size={world_size})")

    # Setup device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Genesis components
    from genesis import Genesis, GenesisConfig
    from genesis.tokenizer.decoder import VideoTokenizer
    from training.losses import GenesisCriterion
    from training.checkpoints import CheckpointManager
    from training.data import VideoDataset, VideoTransforms

    # Create model (tokenizer only for stage 1)
    model_config = GenesisConfig()
    tokenizer = VideoTokenizer(model_config.tokenizer)
    tokenizer = tokenizer.to(device)

    if distributed:
        tokenizer = DDP(tokenizer, device_ids=[local_rank])

    # Create criterion
    criterion = GenesisCriterion(stage=1)

    # Create optimizer with cosine schedule
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Cosine LR scheduler with warmup
    total_steps = args.epochs * 1000  # Estimate
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=str(output_dir),
        keep_last_n=5,
        keep_best_n=3,
        metric_name="loss",
        metric_mode="min",
    )

    # Resume if specified
    start_epoch = 0
    global_step = 0

    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        ckpt = ckpt_manager.load(
            path=args.resume,
            model=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)

    # Create dataset
    transform = VideoTransforms.compose(
        VideoTransforms.random_horizontal_flip(0.5),
        VideoTransforms.random_crop((0.9, 1.0)),
        VideoTransforms.color_jitter(0.05, 0.05, 0.05),
    )

    dataset = VideoDataset(
        root=args.data_dir,
        split="train",
        num_frames=16,
        frame_rate=8,
        resolution=(256, 256),
        transform=transform,
    )

    if is_main:
        print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None

    # Wandb logging
    if args.wandb and is_main and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=f"stage1_lr{args.lr}_bs{args.batch_size}",
            config=vars(args),
        )

    # Count parameters
    if is_main:
        n_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

    # Training loop
    tokenizer.train()

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            sampler.set_epoch(epoch)

        if is_main:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(pbar):
            video = batch["video"].to(device)  # [B, T, C, H, W]

            # Forward pass with mixed precision
            with autocast(enabled=args.mixed_precision):
                outputs = tokenizer(video)
                targets = {"video": video}
                losses = criterion(outputs, targets)
                loss = losses["total"] / args.grad_accum

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += losses["total"].item()
            n_batches += 1

            # Logging
            if is_main and batch_idx % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.4f}",
                    "recon": f"{losses.get('recon', torch.tensor(0)).item():.4f}",
                    "lr": f"{lr:.2e}",
                })

                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "loss": losses["total"].item(),
                        "recon_loss": losses.get("recon", torch.tensor(0)).item(),
                        "perceptual_loss": losses.get("perceptual", torch.tensor(0)).item(),
                        "lr": lr,
                        "step": global_step,
                    })

            # Dry run exit
            if args.dry_run:
                if is_main:
                    print("\nDry run complete!")
                    print(f"Loss: {losses['total'].item():.4f}")
                    print(f"Reconstruction shape: {outputs['recon'].shape}")
                cleanup_distributed()
                return

        # End of epoch
        avg_loss = epoch_loss / max(n_batches, 1)

        if is_main:
            print(f"\nEpoch {epoch} complete. Avg loss: {avg_loss:.4f}")

            # Save checkpoint
            ckpt_manager.save(
                model=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                metrics={"loss": avg_loss},
            )

    # Final save
    if is_main:
        final_path = output_dir / "stage1_final.pt"
        torch.save({
            "model_state": tokenizer.module.state_dict() if distributed else tokenizer.state_dict(),
            "config": model_config.tokenizer,
        }, final_path)
        print(f"\nTraining complete! Final model: {final_path}")

        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
