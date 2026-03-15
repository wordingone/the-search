"""
Train and Evaluate ANIMA on Synthetic Multi-Modal Data
=======================================================

Trains ANIMA variants on synthetic data across all 7 modalities,
then evaluates to find optimal architecture at low parameter counts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from .synthetic_multimodal import (
    SyntheticTextData,
    SyntheticVisionData,
    SyntheticVideoData,
    SyntheticAudioData,
    SyntheticPointCloudData,
    SyntheticSensorData,
    SyntheticMolecularData,
    run_synthetic_benchmark,
    print_benchmark_results,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.001
    seq_len: int = 64
    samples_per_epoch: int = 1000
    device: str = "cuda"


def train_on_modality(
    model: nn.Module,
    generator,
    task: str,
    config: TrainingConfig,
    sensory_dim: int,
) -> Tuple[float, float]:
    """
    Train model on a single modality.

    Returns:
        (final_loss, final_accuracy)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    final_loss = 0.0

    num_batches = config.samples_per_epoch // config.batch_size

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for _ in range(num_batches):
            x, y = generator.generate_batch(config.batch_size, task)

            # Ensure correct dimensions
            if x.shape[-1] != sensory_dim:
                if x.shape[-1] < sensory_dim:
                    pad = torch.zeros(*x.shape[:-1], sensory_dim - x.shape[-1])
                    x = torch.cat([x, pad], dim=-1)
                else:
                    x = x[..., :sensory_dim]

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward
            out = model(x)
            out_pooled = out.mean(dim=1)

            # Match output to num classes
            num_classes = y.max().item() + 1
            out_pooled = out_pooled[:, :num_classes]

            loss = criterion(out_pooled, y)
            loss.backward()
            optimizer.step()

            # Track metrics
            preds = out_pooled.argmax(dim=-1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += config.batch_size
            epoch_loss += loss.item()

        epoch_acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / num_batches

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        final_loss = avg_loss

    return final_loss, best_acc


def train_multimodal(
    model: nn.Module,
    config: TrainingConfig,
    sensory_dim: int = 8,
    output_dim: int = 10,
    verbose: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """
    Train model on all modalities.

    Returns:
        Dict mapping modality name to (loss, accuracy)
    """
    # Define modalities
    modalities = [
        ("Text", "next_token", SyntheticTextData(vocab_size=100, seq_len=config.seq_len, num_classes=output_dim)),
        ("Vision", "classify", SyntheticVisionData(num_patches=config.seq_len, patch_dim=sensory_dim, num_classes=output_dim)),
        ("Video", "action", SyntheticVideoData(num_frames=config.seq_len, frame_dim=sensory_dim, num_classes=min(5, output_dim))),
        ("Audio", "phoneme", SyntheticAudioData(num_frames=config.seq_len, freq_bins=sensory_dim, num_classes=min(5, output_dim))),
        ("3D", "shape", SyntheticPointCloudData(num_points=config.seq_len, point_dim=sensory_dim, num_classes=min(5, output_dim))),
        ("Sensor", "anomaly", SyntheticSensorData(seq_len=config.seq_len, num_channels=sensory_dim, num_classes=min(4, output_dim))),
        ("Science", "property", SyntheticMolecularData(max_atoms=config.seq_len, atom_dim=sensory_dim, num_classes=min(5, output_dim))),
    ]

    results = {}

    for name, task, generator in modalities:
        if verbose:
            print(f"  Training on {name}...", end=" ", flush=True)

        # Reset model weights for fair comparison per modality
        # (or we could train jointly - let's do per-modality for analysis)

        loss, acc = train_on_modality(model, generator, task, config, sensory_dim)
        results[name] = (loss, acc)

        if verbose:
            print(f"Loss: {loss:.4f}, Acc: {acc:.1%}")

    return results


def run_training_comparison(
    model_configs: List[Tuple[str, int, int]],
    sensory_dim: int = 8,
    output_dim: int = 10,
    epochs: int = 30,
    samples_per_epoch: int = 500,
):
    """
    Compare training performance across model sizes.

    Args:
        model_configs: List of (name, d_model, bottleneck_dim)
    """
    from anima.core import Anima

    config = TrainingConfig(
        epochs=epochs,
        batch_size=64,
        lr=0.001,
        seq_len=64,
        samples_per_epoch=samples_per_epoch,
    )

    all_results = {}

    print("=" * 80)
    print("ANIMA TRAINING COMPARISON")
    print(f"Epochs: {epochs}, Samples/epoch: {samples_per_epoch}")
    print("=" * 80)

    for name, d_model, bottleneck in model_configs:
        model = Anima(
            sensory_dim=sensory_dim,
            d_model=d_model,
            bottleneck_dim=bottleneck,
            output_dim=output_dim,
        )
        params = sum(p.numel() for p in model.parameters())

        print(f"\n{name} ({params:,} params)")
        print("-" * 40)

        results = train_multimodal(model, config, sensory_dim, output_dim)
        all_results[name] = {
            "params": params,
            "results": results,
            "avg_acc": np.mean([acc for _, acc in results.values()]),
        }

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Params':>10} {'Text':>8} {'Vision':>8} {'Video':>8} {'Audio':>8} {'3D':>8} {'Sensor':>8} {'Science':>8} {'AVG':>8}")
    print("-" * 80)

    for name, data in all_results.items():
        print(f"{name:<15} {data['params']:>10,}", end="")
        for modality in ["Text", "Vision", "Video", "Audio", "3D", "Sensor", "Science"]:
            _, acc = data["results"][modality]
            print(f" {acc:>7.1%}", end="")
        print(f" {data['avg_acc']:>7.1%}")

    print("=" * 80)

    return all_results


def quick_train_test():
    """Quick training test for development."""
    from anima.core import Anima

    print("Quick Training Test")
    print("=" * 50)

    model = Anima(sensory_dim=8, d_model=32, bottleneck_dim=16, output_dim=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params")

    config = TrainingConfig(
        epochs=20,
        batch_size=64,
        lr=0.001,
        seq_len=64,
        samples_per_epoch=500,
    )

    results = train_multimodal(model, config, sensory_dim=8, output_dim=10)

    print("\nResults:")
    for name, (loss, acc) in results.items():
        print(f"  {name}: {acc:.1%}")

    avg_acc = np.mean([acc for _, acc in results.values()])
    print(f"\nAverage: {avg_acc:.1%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_train_test()
    else:
        # Full comparison
        configs = [
            ("Tiny", 16, 8),
            ("Small", 32, 16),
            ("Medium", 64, 32),
            ("Large", 128, 64),
        ]

        run_training_comparison(
            configs,
            sensory_dim=8,
            output_dim=10,
            epochs=30,
            samples_per_epoch=500,
        )
