"""Stage 2 Training: Dynamics Backbone + Latent Action Model

Trains the autoregressive transformer for next-frame prediction
with unsupervised latent action discovery.

Objectives:
- Accurate next-frame feature prediction
- Diverse latent action space
- Temporal consistency

Requires: Trained Stage 1 tokenizer checkpoint

Usage:
    python scripts/train_stage2.py --tokenizer-ckpt outputs/stage1/stage1_final.pt
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
    parser = argparse.ArgumentParser(description="Stage 2: Dynamics + LAM Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True,
                       help="Path to trained Stage 1 tokenizer")
    parser.add_argument("--data-dir", type=str, default="data/videos")
    parser.add_argument("--output", type=str, default="outputs/stage2")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--freeze-tokenizer", action="store_true", default=True)
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
        print("Genesis Stage 2: Dynamics + LAM Training")
        print("=" * 60)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports
    from genesis import GenesisConfig
    from genesis.tokenizer.decoder import VideoTokenizer
    from genesis.dynamics.backbone import DynamicsBackbone
    from genesis.action.lam import LatentActionModel, ActionConditionedPredictor
    from training.losses import GenesisCriterion
    from training.checkpoints import CheckpointManager
    from training.data import VideoDataset, VideoTransforms

    # Create models
    model_config = GenesisConfig()

    # Load frozen tokenizer
    tokenizer = VideoTokenizer(model_config.tokenizer)
    ckpt = torch.load(args.tokenizer_ckpt, map_location=device)
    tokenizer.load_state_dict(ckpt["model_state"])
    tokenizer = tokenizer.to(device)
    tokenizer.eval()

    if args.freeze_tokenizer:
        for p in tokenizer.parameters():
            p.requires_grad = False

    # Create trainable components
    dynamics = DynamicsBackbone(
        model_config.dynamics,
        latent_channels=model_config.tokenizer.latent_channels,
    ).to(device)

    lam = LatentActionModel(
        latent_channels=model_config.tokenizer.latent_channels,
        config=model_config.action.lam,
    ).to(device)

    action_predictor = ActionConditionedPredictor(
        latent_channels=model_config.tokenizer.latent_channels,
        action_dim=len(model_config.action.lam.fsq_levels),
    ).to(device)

    if distributed:
        dynamics = DDP(dynamics, device_ids=[local_rank])
        lam = DDP(lam, device_ids=[local_rank])
        action_predictor = DDP(action_predictor, device_ids=[local_rank])

    # Optimizer for trainable components
    trainable_params = list(dynamics.parameters()) + \
                       list(lam.parameters()) + \
                       list(action_predictor.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Scheduler
    total_steps = args.epochs * 1000
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoint manager
    ckpt_manager = CheckpointManager(str(output_dir), keep_last_n=5)

    # Resume
    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        dynamics_state = ckpt.get("dynamics_state", ckpt.get("model_state"))
        if distributed:
            dynamics.module.load_state_dict(dynamics_state)
            lam.module.load_state_dict(ckpt["lam_state"])
            action_predictor.module.load_state_dict(ckpt["action_predictor_state"])
        else:
            dynamics.load_state_dict(dynamics_state)
            lam.load_state_dict(ckpt["lam_state"])
            action_predictor.load_state_dict(ckpt["action_predictor_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)

    # Dataset
    transform = VideoTransforms.compose(
        VideoTransforms.random_horizontal_flip(0.5),
    )

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
        wandb.init(project="genesis-stage2", config=vars(args))

    if is_main:
        n_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

    # Training
    dynamics.train()
    lam.train()
    action_predictor.train()

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if is_main else dataloader
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(pbar):
            video = batch["video"].to(device)  # [B, T, C, H, W]
            B, T = video.shape[:2]

            with autocast(enabled=args.mixed_precision):
                # Encode video with frozen tokenizer
                with torch.no_grad():
                    latent = tokenizer.encode(video)  # [B, T, H, W, C]

                # Infer latent actions between frames
                actions = []
                lam_losses = []

                for t in range(T - 1):
                    z_t = latent[:, t].permute(0, 3, 1, 2)
                    z_t1 = latent[:, t + 1].permute(0, 3, 1, 2)

                    # LAM forward
                    action_embed, action_q, action_idx = lam(z_t, z_t1)
                    actions.append(action_q)

                    # LAM reconstruction loss
                    z_t1_pred = action_predictor(z_t, action_q)
                    lam_loss = nn.functional.mse_loss(z_t1_pred, z_t1)
                    lam_losses.append(lam_loss)

                if len(actions) == 0:
                    continue

                actions = torch.stack(actions, dim=1)  # [B, T-1, A]

                # Dynamics prediction
                pred_features = []
                for t in range(min(T - 1, actions.shape[1])):
                    history = latent[:, :t + 1]
                    action_history = actions[:, :t + 1]
                    features = dynamics(history, action_history)
                    pred_features.append(features)

                if len(pred_features) == 0:
                    continue

                pred_features = torch.stack(pred_features, dim=1)

                # Dynamics loss: predict next latent
                target_features = latent[:, 1:pred_features.shape[1] + 1]
                dynamics_loss = nn.functional.mse_loss(
                    pred_features,
                    target_features.mean(dim=(-2, -1)),  # Pool spatial dims
                )

                # Action variance loss (encourage diversity)
                action_var = actions.var(dim=0).mean()
                action_var_loss = (action_var - 0.1).pow(2)

                # Temporal consistency
                if pred_features.shape[1] > 1:
                    temporal_diff = pred_features[:, 1:] - pred_features[:, :-1]
                    temporal_loss = temporal_diff.pow(2).mean()
                else:
                    temporal_loss = torch.tensor(0.0, device=device)

                # Total loss
                lam_loss_total = sum(lam_losses) / len(lam_losses)
                loss = (
                    dynamics_loss +
                    0.5 * lam_loss_total +
                    0.1 * action_var_loss +
                    0.05 * temporal_loss
                ) / args.grad_accum

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            total_loss = loss.item() * args.grad_accum
            epoch_loss += total_loss
            n_batches += 1

            if is_main and batch_idx % 10 == 0:
                pbar.set_postfix({
                    "dyn": f"{dynamics_loss.item():.4f}",
                    "lam": f"{lam_loss_total.item():.4f}",
                    "var": f"{action_var.item():.4f}",
                })

                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "dynamics_loss": dynamics_loss.item(),
                        "lam_loss": lam_loss_total.item(),
                        "action_variance": action_var.item(),
                        "temporal_loss": temporal_loss.item(),
                        "step": global_step,
                    })

            if args.dry_run:
                if is_main:
                    print(f"\nDry run complete! Total loss: {total_loss:.4f}")
                cleanup_distributed()
                return

        avg_loss = epoch_loss / max(n_batches, 1)

        if is_main:
            print(f"\nEpoch {epoch} complete. Avg loss: {avg_loss:.4f}")

            # Save checkpoint
            save_dict = {
                "dynamics_state": dynamics.module.state_dict() if distributed else dynamics.state_dict(),
                "lam_state": lam.module.state_dict() if distributed else lam.state_dict(),
                "action_predictor_state": action_predictor.module.state_dict() if distributed else action_predictor.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "loss": avg_loss,
            }
            torch.save(save_dict, output_dir / f"checkpoint_epoch{epoch}.pt")

    if is_main:
        print("\nStage 2 training complete!")
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
