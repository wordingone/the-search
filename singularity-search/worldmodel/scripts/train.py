"""Training script for Genesis."""

import argparse
import sys
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Genesis world model")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                       help="Training stage (1=tokenizer, 2=dynamics, 3=deltav, 4=e2e)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    print(f"Genesis Training - Stage {args.stage}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")

    # Load config
    config = load_config(args.config)

    # Setup device
    device = torch.device(args.device)

    # Import here to avoid circular imports
    from genesis import Genesis, GenesisConfig
    from training.losses import GenesisCriterion

    # Create model
    model_config = GenesisConfig()
    model = Genesis(model_config, use_stubs=True)
    model = model.to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = Genesis.load_checkpoint(args.checkpoint, device)

    # Create criterion
    criterion = GenesisCriterion(stage=args.stage)

    # Get training config for this stage
    stage_config = config['training'][f'stage{args.stage}']

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_config['lr'],
        betas=tuple(config['training']['optimizer']['betas']),
        weight_decay=config['training']['optimizer']['weight_decay'],
    )

    print(f"\nModel parameters: {model.get_param_count()['total']:,}")
    print(f"Learning rate: {stage_config['lr']}")
    print(f"Batch size: {stage_config['batch_size']}")
    print(f"Total steps: {stage_config['total_steps']}")

    # Training loop placeholder
    print("\n[Training loop would go here]")
    print("Data loaders not implemented yet - see training/data/")

    # Example forward pass
    print("\nRunning example forward pass...")
    with torch.no_grad():
        # Dummy video
        video = torch.randn(1, 16, 3, 256, 256, device=device)

        # Forward
        outputs = model(video)

        print(f"  Latent shape: {outputs['latent'].shape}")
        if 'reconstruction' in outputs:
            print(f"  Reconstruction shape: {outputs['reconstruction'].shape}")

        # Compute losses
        targets = {'video': video}
        losses = criterion(outputs, targets)

        print(f"  Total loss: {losses['total'].item():.4f}")

    print("\nTraining setup complete!")


if __name__ == "__main__":
    main()
