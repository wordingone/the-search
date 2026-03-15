"""
Optimized Training for ANIMA
============================

Tests various training optimizations:
1. Learning rate scheduling
2. Weight decay
3. Gradient clipping
4. Layer normalization
5. Residual connections
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
import time

from .synthetic_multimodal import (
    SyntheticTextData,
    SyntheticVisionData,
    SyntheticVideoData,
    SyntheticAudioData,
    SyntheticPointCloudData,
    SyntheticSensorData,
    SyntheticMolecularData,
)


def train_with_config(
    model: nn.Module,
    generators: list,
    config: dict,
    device: torch.device,
) -> Dict[str, float]:
    """Train model with specific configuration."""
    model = model.to(device)
    model.train()

    # Optimizer
    if config.get('optimizer') == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.01)
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0)
        )

    # Scheduler
    if config.get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['lr'] * 0.01
        )
    elif config.get('scheduler') == 'warmup':
        def lr_lambda(epoch):
            warmup = config.get('warmup_epochs', 5)
            if epoch < warmup:
                return epoch / warmup
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    grad_clip = config.get('grad_clip', None)

    results = {}

    for name, task, generator in generators:
        best_acc = 0.0

        for epoch in range(config['epochs']):
            epoch_correct = 0
            epoch_total = 0

            for _ in range(config['batches_per_epoch']):
                x, y = generator.generate_batch(config['batch_size'], task)

                # Ensure dimensions
                if x.shape[-1] < config['sensory_dim']:
                    pad = torch.zeros(*x.shape[:-1], config['sensory_dim'] - x.shape[-1])
                    x = torch.cat([x, pad], dim=-1)
                elif x.shape[-1] > config['sensory_dim']:
                    x = x[..., :config['sensory_dim']]

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                out = model(x)
                out_pooled = out.mean(dim=1)

                num_classes = y.max().item() + 1
                out_pooled = out_pooled[:, :num_classes]

                loss = criterion(out_pooled, y)
                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                preds = out_pooled.argmax(dim=-1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += config['batch_size']

            if scheduler:
                scheduler.step()

            acc = epoch_correct / epoch_total
            if acc > best_acc:
                best_acc = acc

        results[name] = best_acc

    return results


def run_optimization_study():
    """Compare different training configurations."""
    from anima.core import Anima

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SENSORY_DIM = 8
    OUTPUT_DIM = 10
    SEQ_LEN = 64

    # Generators
    generators = [
        ("Text", "next_token", SyntheticTextData(100, SEQ_LEN, OUTPUT_DIM)),
        ("Vision", "classify", SyntheticVisionData(SEQ_LEN, SENSORY_DIM, OUTPUT_DIM)),
        ("Video", "action", SyntheticVideoData(SEQ_LEN, SENSORY_DIM, 5)),
        ("Audio", "phoneme", SyntheticAudioData(SEQ_LEN, SENSORY_DIM, 5)),
        ("3D", "shape", SyntheticPointCloudData(SEQ_LEN, SENSORY_DIM, 5)),
        ("Sensor", "anomaly", SyntheticSensorData(SEQ_LEN, SENSORY_DIM, 4)),
        ("Science", "property", SyntheticMolecularData(SEQ_LEN, SENSORY_DIM, 5)),
    ]

    # Training configurations to test
    configs = {
        "baseline": {
            'lr': 0.001,
            'optimizer': 'adam',
            'epochs': 30,
            'batch_size': 64,
            'batches_per_epoch': 8,
            'sensory_dim': SENSORY_DIM,
        },
        "adamw": {
            'lr': 0.001,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'epochs': 30,
            'batch_size': 64,
            'batches_per_epoch': 8,
            'sensory_dim': SENSORY_DIM,
        },
        "cosine_lr": {
            'lr': 0.003,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'epochs': 30,
            'batch_size': 64,
            'batches_per_epoch': 8,
            'sensory_dim': SENSORY_DIM,
        },
        "warmup": {
            'lr': 0.003,
            'optimizer': 'adam',
            'scheduler': 'warmup',
            'warmup_epochs': 5,
            'epochs': 30,
            'batch_size': 64,
            'batches_per_epoch': 8,
            'sensory_dim': SENSORY_DIM,
        },
        "grad_clip": {
            'lr': 0.001,
            'optimizer': 'adam',
            'grad_clip': 1.0,
            'epochs': 30,
            'batch_size': 64,
            'batches_per_epoch': 8,
            'sensory_dim': SENSORY_DIM,
        },
        "all_tricks": {
            'lr': 0.003,
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'scheduler': 'cosine',
            'grad_clip': 1.0,
            'epochs': 30,
            'batch_size': 64,
            'batches_per_epoch': 8,
            'sensory_dim': SENSORY_DIM,
        },
    }

    # Model sizes to test
    model_sizes = [
        ("Small", 32, 16),     # ~21K
        ("Large", 128, 64),    # ~339K
    ]

    print("=" * 90)
    print("TRAINING OPTIMIZATION STUDY")
    print("=" * 90)

    all_results = {}

    for model_name, d_model, bottleneck in model_sizes:
        print(f"\n{model_name} Model (d_model={d_model})")
        print("-" * 90)

        for config_name, config in configs.items():
            # Fresh model for each config
            model = Anima(
                sensory_dim=SENSORY_DIM,
                d_model=d_model,
                bottleneck_dim=bottleneck,
                output_dim=OUTPUT_DIM,
            )
            params = sum(p.numel() for p in model.parameters())

            print(f"  {config_name:<15}", end=" ", flush=True)

            results = train_with_config(model, generators, config, device)
            avg = np.mean(list(results.values()))

            print(f"Avg: {avg:.1%}", end="")
            for name in ["Text", "Vision", "Video", "Audio", "3D", "Sensor", "Science"]:
                print(f"  {name[:3]}:{results[name]:.0%}", end="")
            print()

            all_results[f"{model_name}_{config_name}"] = {
                'params': params,
                'config': config_name,
                'avg': avg,
                'results': results,
            }

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY - Best configs per model size")
    print("=" * 90)

    for model_name, _, _ in model_sizes:
        relevant = {k: v for k, v in all_results.items() if k.startswith(model_name)}
        best = max(relevant.items(), key=lambda x: x[1]['avg'])
        print(f"{model_name}: Best config = {best[1]['config']} ({best[1]['avg']:.1%})")


def quick_lr_sweep():
    """Quick learning rate sweep."""
    from anima.core import Anima

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generators = [
        ("Video", "action", SyntheticVideoData(64, 8, 5)),
        ("Sensor", "anomaly", SyntheticSensorData(64, 8, 4)),
    ]

    print("Learning Rate Sweep")
    print("-" * 50)

    for lr in [0.0001, 0.0003, 0.001, 0.003, 0.01]:
        model = Anima(sensory_dim=8, d_model=64, bottleneck_dim=32, output_dim=10)

        config = {
            'lr': lr,
            'optimizer': 'adam',
            'epochs': 20,
            'batch_size': 64,
            'batches_per_epoch': 8,
            'sensory_dim': 8,
        }

        results = train_with_config(model, generators, config, device)
        avg = np.mean(list(results.values()))
        print(f"  lr={lr:.4f}: {avg:.1%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "lr":
        quick_lr_sweep()
    else:
        run_optimization_study()
