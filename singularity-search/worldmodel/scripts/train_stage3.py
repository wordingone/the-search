"""Stage 3 Training: DeltaV Predictor with 3D Supervision

Trains the sparse voxel delta predictor using synthetic 3D data
with ground truth voxel labels.

Objectives:
- Accurate voxel coordinate prediction
- Correct operation classification (add/modify/remove)
- PBR feature prediction

Requires: Trained Stage 2 checkpoint

Usage:
    python scripts/train_stage3.py --stage2-ckpt outputs/stage2/latest.pt
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
    parser = argparse.ArgumentParser(description="Stage 3: DeltaV Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--stage2-ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--output", type=str, default="outputs/stage3")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
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
        print("Genesis Stage 3: DeltaV Predictor Training")
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
    from genesis.deltav.predictor import DeltaVPredictor
    from training.losses import ChamferLoss, HungarianMatcher
    from training.checkpoints import CheckpointManager
    from training.data import SyntheticDataset, MultiViewTransforms

    # Create models
    model_config = GenesisConfig()

    # Load frozen tokenizer
    tokenizer = VideoTokenizer(model_config.tokenizer)
    ckpt = torch.load(args.tokenizer_ckpt, map_location=device)
    tokenizer.load_state_dict(ckpt["model_state"])
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # Load frozen dynamics
    dynamics = DynamicsBackbone(
        model_config.dynamics,
        latent_channels=model_config.tokenizer.latent_channels,
    )
    stage2_ckpt = torch.load(args.stage2_ckpt, map_location=device)
    dynamics.load_state_dict(stage2_ckpt["dynamics_state"])
    dynamics = dynamics.to(device)
    dynamics.eval()
    for p in dynamics.parameters():
        p.requires_grad = False

    # Trainable DeltaV predictor
    deltav = DeltaVPredictor(model_config.deltav).to(device)

    if distributed:
        deltav = DDP(deltav, device_ids=[local_rank])

    # Losses
    chamfer_loss = ChamferLoss(symmetric=True)
    matcher = HungarianMatcher(cost_coord=1.0, cost_feature=0.5, cost_class=0.5)

    # Optimizer
    optimizer = torch.optim.AdamW(
        deltav.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    total_steps = args.epochs * 500
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
            deltav.module.load_state_dict(ckpt["deltav_state"])
        else:
            deltav.load_state_dict(ckpt["deltav_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)

    # Dataset
    transform = MultiViewTransforms.normalize_voxels(256)

    dataset = SyntheticDataset(
        root=args.data_dir,
        split="train",
        num_views=8,
        resolution=(256, 256),
        voxel_resolution=64,
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
        wandb.init(project="genesis-stage3", config=vars(args))

    if is_main:
        n_params = sum(p.numel() for p in deltav.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")

    # Training
    deltav.train()

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}") if is_main else dataloader
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)  # [B, V, 3, H, W]
            target_coords = batch["voxel_coords"].to(device)
            target_features = batch["voxel_features"].to(device)

            B, V = images.shape[:2]

            with autocast(enabled=args.mixed_precision):
                # Use first view as "video frame" for tokenizer
                first_view = images[:, 0].unsqueeze(1)  # [B, 1, 3, H, W]

                with torch.no_grad():
                    latent = tokenizer.encode(first_view)  # [B, 1, H, W, C]

                    # Create dummy actions for dynamics
                    action_dim = len(model_config.action.lam.fsq_levels)
                    actions = torch.zeros(B, 1, action_dim, device=device)

                    features = dynamics(latent, actions)  # [B, H, W, D]

                # Predict deltas
                pred_delta = deltav(features)

                # Compute losses
                losses = {}

                # Chamfer loss on coordinates
                for b in range(B):
                    pred_coords_b = pred_delta.coords[pred_delta.coords[:, 0] == b][:, 1:].float()
                    target_coords_b = target_coords[b].float()

                    if pred_coords_b.shape[0] > 0 and target_coords_b.shape[0] > 0:
                        chamfer = chamfer_loss(pred_coords_b, target_coords_b)
                        losses[f"chamfer_{b}"] = chamfer

                if losses:
                    coord_loss = sum(losses.values()) / len(losses)
                else:
                    coord_loss = torch.tensor(0.0, device=device, requires_grad=True)

                # Feature loss via Hungarian matching
                feature_losses = []
                for b in range(B):
                    pred_mask = pred_delta.coords[:, 0] == b
                    if pred_mask.sum() == 0:
                        continue

                    pred_idx, target_idx = matcher(
                        pred_delta.coords[pred_mask][:, 1:].float(),
                        pred_delta.features[pred_mask],
                        pred_delta.op_type[pred_mask],
                        target_coords[b].float(),
                        target_features[b],
                        torch.ones(target_coords[b].shape[0], dtype=torch.long, device=device),  # Default: add
                    )

                    if len(pred_idx) > 0:
                        feat_loss = nn.functional.l1_loss(
                            pred_delta.features[pred_mask][pred_idx],
                            target_features[b][target_idx],
                        )
                        feature_losses.append(feat_loss)

                if feature_losses:
                    feature_loss = sum(feature_losses) / len(feature_losses)
                else:
                    feature_loss = torch.tensor(0.0, device=device, requires_grad=True)

                # Sparsity regularization
                sparsity = pred_delta.num_deltas / model_config.deltav.max_deltas_per_frame
                sparsity_loss = (sparsity - 0.1) ** 2

                # Total loss
                loss = (
                    coord_loss +
                    0.5 * feature_loss +
                    0.01 * sparsity_loss
                ) / args.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(deltav.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(deltav.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            total_loss = loss.item() * args.grad_accum
            epoch_loss += total_loss
            n_batches += 1

            if is_main and batch_idx % 10 == 0:
                pbar.set_postfix({
                    "coord": f"{coord_loss.item():.4f}",
                    "feat": f"{feature_loss.item():.4f}",
                    "sparse": f"{sparsity:.3f}",
                })

                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "coord_loss": coord_loss.item(),
                        "feature_loss": feature_loss.item(),
                        "sparsity": sparsity,
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
                "deltav_state": deltav.module.state_dict() if distributed else deltav.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "loss": avg_loss,
            }
            torch.save(save_dict, output_dir / f"checkpoint_epoch{epoch}.pt")

    if is_main:
        print("\nStage 3 training complete!")
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
