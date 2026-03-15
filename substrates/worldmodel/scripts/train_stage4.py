"""Stage 4 Training: End-to-End Fine-tuning

Fine-tunes the entire Genesis model end-to-end with:
- LPIPS perceptual loss
- Temporal stability losses
- Optional user interaction simulation

Requires: All previous stage checkpoints

Usage:
    python scripts/train_stage4.py \
        --tokenizer-ckpt outputs/stage1/final.pt \
        --stage2-ckpt outputs/stage2/final.pt \
        --stage3-ckpt outputs/stage3/final.pt
"""

import argparse
import sys
from pathlib import Path
import os

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
    parser = argparse.ArgumentParser(description="Stage 4: End-to-End Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--stage2-ckpt", type=str, required=True)
    parser.add_argument("--stage3-ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/videos")
    parser.add_argument("--output", type=str, default="outputs/stage4")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-rank", type=int, default=-1)

    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
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
        print("Genesis Stage 4: End-to-End Fine-tuning")
        print("=" * 60)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports
    from genesis import Genesis, GenesisConfig
    from training.losses import LPIPSLoss, GenesisCriterion
    from training.checkpoints import CheckpointManager
    from training.data import VideoDataset, VideoTransforms

    # Create full model
    model_config = GenesisConfig()
    model = Genesis(model_config, use_stubs=True)

    # Load all stage checkpoints
    if is_main:
        print("Loading stage checkpoints...")

    # Stage 1: Tokenizer
    tok_ckpt = torch.load(args.tokenizer_ckpt, map_location=device)
    model.tokenizer.load_state_dict(tok_ckpt["model_state"])

    # Stage 2: Dynamics + LAM
    s2_ckpt = torch.load(args.stage2_ckpt, map_location=device)
    model.dynamics.load_state_dict(s2_ckpt["dynamics_state"])
    model.lam.load_state_dict(s2_ckpt["lam_state"])
    model.action_predictor.load_state_dict(s2_ckpt["action_predictor_state"])

    # Stage 3: DeltaV
    s3_ckpt = torch.load(args.stage3_ckpt, map_location=device)
    model.deltav.load_state_dict(s3_ckpt["deltav_state"])

    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Losses
    lpips_loss = LPIPSLoss(net="vgg")
    criterion = GenesisCriterion(stage=4)

    # Optimizer with different LR for different components
    param_groups = [
        {"params": model.module.tokenizer.parameters() if distributed else model.tokenizer.parameters(),
         "lr": args.lr * 0.1},  # Lower LR for tokenizer
        {"params": model.module.dynamics.parameters() if distributed else model.dynamics.parameters(),
         "lr": args.lr},
        {"params": model.module.lam.parameters() if distributed else model.lam.parameters(),
         "lr": args.lr},
        {"params": model.module.deltav.parameters() if distributed else model.deltav.parameters(),
         "lr": args.lr},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    total_steps = args.epochs * 200
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if distributed:
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)

    # Dataset
    transform = VideoTransforms.random_horizontal_flip(0.5)

    dataset = VideoDataset(
        root=args.data_dir,
        split="train",
        num_frames=16,
        frame_rate=8,
        resolution=(256, 256),
        transform=transform,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
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

    scaler = GradScaler() if args.mixed_precision else None

    if args.wandb and is_main and WANDB_AVAILABLE:
        wandb.init(project="genesis-stage4", config=vars(args))

    if is_main:
        param_count = model.module.get_param_count() if distributed else model.get_param_count()
        print(f"Total parameters: {param_count['total']:,}")

    # Training
    model.train()

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if is_main else dataloader
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(pbar):
            video = batch["video"].to(device)  # [B, T, C, H, W]

            with autocast(enabled=args.mixed_precision):
                # Full forward pass
                outputs = model(video)

                # Reconstruction loss
                recon = outputs.get("reconstruction")
                if recon is not None:
                    # Match temporal dimensions
                    T_recon = recon.shape[1]
                    T_target = video.shape[1]
                    min_T = min(T_recon, T_target)

                    recon_loss = nn.functional.l1_loss(
                        recon[:, :min_T],
                        video[:, :min_T],
                    )

                    # LPIPS loss on frames
                    lpips_frames_loss = 0.0
                    for t in range(min_T):
                        lpips_frames_loss += lpips_loss(recon[:, t], video[:, t])
                    lpips_frames_loss = lpips_frames_loss / min_T
                else:
                    recon_loss = torch.tensor(0.0, device=device)
                    lpips_frames_loss = torch.tensor(0.0, device=device)

                # Temporal stability loss
                pred_features = outputs.get("predicted_features")
                if pred_features is not None and pred_features.shape[1] > 1:
                    temporal_diff = pred_features[:, 1:] - pred_features[:, :-1]
                    stability_loss = temporal_diff.pow(2).mean()
                else:
                    stability_loss = torch.tensor(0.0, device=device)

                # Action diversity
                actions = outputs.get("actions")
                if actions is not None and actions.numel() > 0:
                    action_var = actions.var(dim=0).mean()
                    diversity_loss = (action_var - 0.1).pow(2)
                else:
                    diversity_loss = torch.tensor(0.0, device=device)

                # Total loss
                loss = (
                    recon_loss +
                    0.1 * lpips_frames_loss +
                    0.05 * stability_loss +
                    0.01 * diversity_loss
                ) / args.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            total_loss = loss.item() * args.grad_accum
            epoch_loss += total_loss
            n_batches += 1

            if is_main and batch_idx % 5 == 0:
                pbar.set_postfix({
                    "recon": f"{recon_loss.item():.4f}",
                    "lpips": f"{lpips_frames_loss.item():.4f}" if isinstance(lpips_frames_loss, torch.Tensor) else "0.0",
                    "stable": f"{stability_loss.item():.4f}",
                })

                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "recon_loss": recon_loss.item(),
                        "lpips_loss": lpips_frames_loss.item() if isinstance(lpips_frames_loss, torch.Tensor) else 0.0,
                        "stability_loss": stability_loss.item(),
                        "diversity_loss": diversity_loss.item(),
                        "step": global_step,
                    })

            if args.dry_run:
                if is_main:
                    print(f"\nDry run complete! Loss: {total_loss:.4f}")
                cleanup_distributed()
                return

        avg_loss = epoch_loss / max(n_batches, 1)

        if is_main:
            print(f"\nEpoch {epoch} complete. Avg loss: {avg_loss:.4f}")

            save_dict = {
                "model_state": model.module.state_dict() if distributed else model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": model_config,
                "epoch": epoch,
                "step": global_step,
                "loss": avg_loss,
            }
            torch.save(save_dict, output_dir / f"checkpoint_epoch{epoch}.pt")

    # Final save
    if is_main:
        final_path = output_dir / "genesis_final.pt"
        torch.save({
            "model_state": model.module.state_dict() if distributed else model.state_dict(),
            "config": model_config,
        }, final_path)
        print(f"\nTraining complete! Final model: {final_path}")

        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
