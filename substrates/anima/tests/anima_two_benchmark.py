"""
ANIMA-Two Benchmark: Fair Comparison with ANIMA-Zero and ANIMA-One
===================================================================

Tests the Hierarchical Temporal Correction architecture against:
- ANIMA-Zero (baseline, 96.75% accuracy)
- ANIMA-One (parallel, 92% accuracy)
- Transformer (control, 92% accuracy)

All models matched to ~25,000 parameters for fair comparison.

Expected Results:
- ANIMA-Two should achieve ~96% accuracy (matching Zero)
- ANIMA-Two should have better parallelism than Zero
- ANIMA-Two should maintain κ=1, λ≈0⁺, Φ>0 properties
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from anima.core.anima_zero import ANIMA, ANIMAConfig
from anima.core.anima_one import ANIMA1, ANIMA1Config
from anima.core.anima_two import ANIMATwo, ANIMATwoConfig, create_anima_two

# =====================
# Constants
# =====================
TARGET_PARAMS = 25000
TOLERANCE = 0.05  # 5% tolerance for parameter matching
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES = 100
SEQ_LEN = 10
TRAINING_STEPS = 200
LEARNING_RATE = 0.003


# =====================
# Benchmark Tasks
# =====================

class BenchmarkTask:
    """Base class for benchmark tasks."""

    def __init__(self, num_samples: int, seq_len: int, seed: int = 42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.rng = torch.Generator().manual_seed(seed)

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def evaluate(self, model: nn.Module, device: torch.device) -> float:
        inputs, targets = self.generate_data()
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.eval()
        with torch.no_grad():
            if hasattr(model, 'reset'):
                model.reset(inputs.shape[0], device)
                outputs = []
                for t in range(inputs.shape[1]):
                    result = model.step(inputs[:, t])
                    outputs.append(result['action'])
                predictions = torch.stack(outputs, dim=1)
            elif hasattr(model, 'forward'):
                result = model(inputs)
                if isinstance(result, dict):
                    predictions = result['action']
                else:
                    predictions = result
            else:
                predictions = model(inputs)

        # Get final timestep
        pred = predictions[:, -1] if predictions.dim() > 2 else predictions
        tgt = targets[:, -1] if targets.dim() > 2 else targets

        # Shape alignment
        if pred.shape != tgt.shape:
            min_dim = min(pred.shape[-1], tgt.shape[-1])
            pred = pred[..., :min_dim]
            tgt = tgt[..., :min_dim]

        return self._compute_accuracy(pred, tgt)

    def _compute_accuracy(self, pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Default regression accuracy with 20% tolerance."""
        diff = torch.abs(pred - tgt)
        tol = 0.2 * torch.abs(tgt).clamp(min=0.1).mean().item()
        tol = max(tol, 0.2)
        return (diff < tol).float().mean().item()


class SequenceTask(BenchmarkTask):
    """Arithmetic sequence prediction."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            start = torch.rand(1, generator=self.rng).item() * 10
            step = torch.rand(1, generator=self.rng).item() * 5 + 0.5
            seq = torch.tensor([start + i * step for i in range(self.seq_len + 1)]) / 100.0
            inp = torch.zeros(self.seq_len, 8)
            inp[:, 0] = seq[:-1]
            tgt = torch.zeros(self.seq_len, 4)
            tgt[:, 0] = seq[1:]
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)


class PatternTask(BenchmarkTask):
    """Repeating pattern recognition."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            plen = int(torch.randint(2, 5, (1,), generator=self.rng).item())
            pattern = torch.rand(plen, generator=self.rng)
            repeats = (self.seq_len // plen) + 2
            full = pattern.repeat(repeats)[:self.seq_len + 1]
            inp = torch.zeros(self.seq_len, 8)
            inp[:, 0] = full[:-1]
            tgt = torch.zeros(self.seq_len, 4)
            tgt[:, 0] = full[1:]
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)


class ConditionalTask(BenchmarkTask):
    """Conditional logic (if-then-else)."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            condition = torch.rand(1, generator=self.rng).item() > 0.5
            value_a = torch.rand(1, generator=self.rng).item()
            value_b = torch.rand(1, generator=self.rng).item()
            result = value_a if condition else value_b
            inp = torch.zeros(self.seq_len, 8)
            inp[0, 0] = 1.0 if condition else 0.0
            inp[0, 1] = value_a
            inp[0, 2] = value_b
            tgt = torch.zeros(self.seq_len, 4)
            tgt[-1, 0] = result
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)


class AnalogyTask(BenchmarkTask):
    """Analogy completion (A:B::C:?)."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            a = torch.rand(1, generator=self.rng).item()
            transform = torch.rand(1, generator=self.rng).item() * 2
            b = a * transform
            c = torch.rand(1, generator=self.rng).item()
            d = c * transform
            inp = torch.zeros(self.seq_len, 8)
            inp[0, 0] = a
            inp[0, 1] = b
            inp[0, 2] = c
            tgt = torch.zeros(self.seq_len, 4)
            tgt[-1, 0] = d / 10.0
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)


class ProjectileTask(BenchmarkTask):
    """Projectile trajectory prediction - tests continuous temporal dynamics."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            v0 = torch.rand(1, generator=self.rng).item() * 10 + 5
            t = torch.linspace(0, 1, self.seq_len + 1)
            positions = v0 * t / 20.0
            inp = torch.zeros(self.seq_len, 8)
            inp[:, 0] = positions[:-1]
            tgt = torch.zeros(self.seq_len, 4)
            tgt[:, 0] = positions[1:]
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)


class CollisionTask(BenchmarkTask):
    """Collision prediction (binary classification)."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            x1 = torch.rand(1, generator=self.rng).item() * 0.5
            v1 = torch.rand(1, generator=self.rng).item() * 0.3 + 0.1
            x2 = torch.rand(1, generator=self.rng).item() * 0.5 + 0.5
            v2 = -torch.rand(1, generator=self.rng).item() * 0.3 - 0.1
            will_collide = 1.0 if v1 > -v2 else 0.0
            inp = torch.zeros(self.seq_len, 8)
            inp[0, :4] = torch.tensor([x1, v1, x2, v2])
            tgt = torch.tensor([will_collide])
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)

    def _compute_accuracy(self, pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Binary classification accuracy."""
        pred_binary = (pred.squeeze() > 0.5).float()
        tgt_binary = tgt.squeeze().float()
        return (pred_binary == tgt_binary).float().mean().item()


class GoalTask(BenchmarkTask):
    """Goal-directed navigation."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            pos = torch.rand(2, generator=self.rng) - 0.5
            goal = torch.rand(2, generator=self.rng) - 0.5
            direction = goal - pos
            direction = direction / (direction.norm() + 1e-6)
            inp = torch.zeros(self.seq_len, 8)
            inp[0, :2] = pos
            inp[0, 2:4] = goal
            tgt = torch.zeros(self.seq_len, 4)
            tgt[-1, :2] = direction
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)

    def _compute_accuracy(self, pred: torch.Tensor, tgt: torch.Tensor) -> float:
        """Directional accuracy (cosine similarity > 0.7)."""
        pred_norm = torch.nn.functional.normalize(pred, dim=-1)
        tgt_norm = torch.nn.functional.normalize(tgt, dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)
        return (cos_sim > 0.7).float().mean().item()


class MomentumTask(BenchmarkTask):
    """Momentum conservation prediction."""

    def generate_data(self):
        inputs, targets = [], []
        for _ in range(self.num_samples):
            m1 = torch.rand(1, generator=self.rng).item() * 0.8 + 0.2
            v1 = torch.rand(1, generator=self.rng).item() * 0.8 + 0.2
            m2 = torch.rand(1, generator=self.rng).item() * 0.8 + 0.2
            v2 = -torch.rand(1, generator=self.rng).item() * 0.8 - 0.2
            v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
            target_norm = (v_final + 1) / 2
            inp = torch.zeros(self.seq_len, 8)
            inp[0, :4] = torch.tensor([m1, v1, m2, v2])
            tgt = torch.zeros(self.seq_len, 4)
            tgt[-1, 0] = target_norm
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)


# =====================
# Model Creation
# =====================

def create_anima_zero_matched(target_params: int = TARGET_PARAMS) -> ANIMA:
    """Create ANIMA-Zero with approximately target_params parameters."""
    config = ANIMAConfig()

    # Binary search for optimal dimensions
    def count_params(d):
        cfg = ANIMAConfig(world_dim=d, internal_dim=d, action_dim=d)
        model = ANIMA(cfg)
        return sum(p.numel() for p in model.parameters())

    low, high = 8, 64
    while high - low > 1:
        mid = (low + high) // 2
        if count_params(mid) > target_params:
            high = mid
        else:
            low = mid

    config.world_dim = low
    config.internal_dim = low
    config.action_dim = low

    return ANIMA(config)


def create_anima_one_matched(target_params: int = TARGET_PARAMS) -> ANIMA1:
    """Create ANIMA-One with approximately target_params parameters."""
    config = ANIMA1Config(max_params=target_params)

    def count_params(d_model, d_state, d_bottleneck):
        cfg = ANIMA1Config(d_model=d_model, d_state=d_state, d_bottleneck=d_bottleneck)
        model = ANIMA1(cfg)
        return sum(p.numel() for p in model.parameters())

    # Search for good dimensions
    best_config = (32, 32, 16)
    best_diff = float('inf')

    for d_model in [16, 24, 32, 40]:
        for d_state in [16, 24, 32, 40]:
            for d_bottleneck in [8, 12, 16, 20]:
                try:
                    params = count_params(d_model, d_state, d_bottleneck)
                    diff = abs(params - target_params)
                    if diff < best_diff and params <= target_params * (1 + TOLERANCE):
                        best_diff = diff
                        best_config = (d_model, d_state, d_bottleneck)
                except:
                    continue

    config.d_model = best_config[0]
    config.d_state = best_config[1]
    config.d_bottleneck = best_config[2]

    return ANIMA1(config)


def create_anima_two_matched(target_params: int = TARGET_PARAMS) -> ANIMATwo:
    """Create ANIMA-Two with approximately target_params parameters."""
    config = ANIMATwoConfig(target_params=target_params)

    def count_params(d_model, d_state, d_bottleneck):
        cfg = ANIMATwoConfig(d_model=d_model, d_state=d_state, d_bottleneck=d_bottleneck)
        model = ANIMATwo(cfg)
        return sum(p.numel() for p in model.parameters())

    # Search for good dimensions
    best_config = (32, 32, 16)
    best_diff = float('inf')

    for d_model in [16, 20, 24, 28, 32]:
        for d_state in [16, 20, 24, 28, 32]:
            for d_bottleneck in [8, 10, 12, 14, 16]:
                try:
                    params = count_params(d_model, d_state, d_bottleneck)
                    diff = abs(params - target_params)
                    if diff < best_diff and params <= target_params * (1 + TOLERANCE):
                        best_diff = diff
                        best_config = (d_model, d_state, d_bottleneck)
                except:
                    continue

    config.d_model = best_config[0]
    config.d_state = best_config[1]
    config.d_bottleneck = best_config[2]

    return ANIMATwo(config)


class TransformerMatched(nn.Module):
    """Transformer baseline matched to target parameters."""

    def __init__(self, sensory_dim: int = 8, output_dim: int = 4, d_model: int = 32):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(sensory_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)

        # Single transformer layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.output = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x = self.embed(x) + self.pos_enc[:, :seq_len]

        # Self-attention with causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        x = self.norm2(x + self.ffn(x))

        return self.output(x)


def create_transformer_matched(target_params: int = TARGET_PARAMS) -> TransformerMatched:
    """Create Transformer with approximately target_params parameters."""
    def count_params(d):
        # d_model must be divisible by num_heads (2)
        d = d - (d % 2)
        if d < 4:
            d = 4
        model = TransformerMatched(d_model=d)
        return sum(p.numel() for p in model.parameters())

    # Search even numbers only (divisible by 2 for num_heads)
    best_d = 32
    best_diff = float('inf')

    for d in range(16, 65, 2):
        try:
            params = count_params(d)
            diff = abs(params - target_params)
            if diff < best_diff and params <= target_params * (1 + TOLERANCE):
                best_diff = diff
                best_d = d
        except:
            continue

    return TransformerMatched(d_model=best_d)


# =====================
# Training
# =====================

def train_model(model: nn.Module, tasks: Dict[str, BenchmarkTask], steps: int = TRAINING_STEPS):
    """Quick training to initialize model weights."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Generate combined training data
    all_inputs = []
    all_targets = []
    for task in tasks.values():
        inp, tgt = task.generate_data()
        all_inputs.append(inp)
        # Ensure targets have consistent shape
        if tgt.dim() == 2:
            # Expand to match sequence format
            tgt_expanded = torch.zeros(inp.shape[0], inp.shape[1], 4)
            tgt_expanded[:, -1, :tgt.shape[-1]] = tgt
            tgt = tgt_expanded
        all_targets.append(tgt)

    inputs = torch.cat(all_inputs, dim=0).to(DEVICE)
    targets = torch.cat(all_targets, dim=0).to(DEVICE)

    dataset_size = inputs.shape[0]
    batch_size = min(32, dataset_size)

    for step in range(steps):
        # Random batch
        idx = torch.randint(0, dataset_size, (batch_size,))
        batch_x = inputs[idx]
        batch_y = targets[idx]

        optimizer.zero_grad()

        # Forward pass
        if hasattr(model, 'reset'):
            model.reset(batch_size, DEVICE)
            outputs = []
            for t in range(batch_x.shape[1]):
                result = model.step(batch_x[:, t])
                outputs.append(result['action'])
            pred = torch.stack(outputs, dim=1)
        elif hasattr(model, 'forward'):
            result = model(batch_x)
            if isinstance(result, dict):
                pred = result['action']
            else:
                pred = result
        else:
            pred = model(batch_x)

        # Loss on final timestep
        pred_final = pred[:, -1]
        tgt_final = batch_y[:, -1]

        # Align dimensions
        min_dim = min(pred_final.shape[-1], tgt_final.shape[-1])
        pred_final = pred_final[..., :min_dim]
        tgt_final = tgt_final[..., :min_dim]

        loss = nn.functional.mse_loss(pred_final, tgt_final)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{steps}, Loss: {loss.item():.4f}")

    model.eval()


# =====================
# Main Benchmark
# =====================

def run_benchmark():
    """Run the complete ANIMA-Two benchmark."""
    print("=" * 70)
    print("ANIMA-Two Benchmark: Hierarchical Temporal Correction Validation")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Target Parameters: {TARGET_PARAMS:,}")
    print(f"Tolerance: ±{TOLERANCE*100:.0f}%")
    print()

    # Create tasks
    tasks = {
        'sequence': SequenceTask(NUM_SAMPLES, SEQ_LEN),
        'pattern': PatternTask(NUM_SAMPLES, SEQ_LEN),
        'conditional': ConditionalTask(NUM_SAMPLES, SEQ_LEN),
        'analogy': AnalogyTask(NUM_SAMPLES, SEQ_LEN),
        'projectile': ProjectileTask(NUM_SAMPLES, SEQ_LEN),
        'collision': CollisionTask(NUM_SAMPLES, SEQ_LEN),
        'goal': GoalTask(NUM_SAMPLES, SEQ_LEN),
        'momentum': MomentumTask(NUM_SAMPLES, SEQ_LEN),
    }

    reasoning_tasks = ['sequence', 'pattern', 'conditional', 'analogy']
    physics_tasks = ['projectile', 'collision', 'goal', 'momentum']

    # Create models
    print("Creating models...")
    models = {
        'ANIMA-Zero': create_anima_zero_matched(TARGET_PARAMS),
        'ANIMA-One': create_anima_one_matched(TARGET_PARAMS),
        'ANIMA-Two': create_anima_two_matched(TARGET_PARAMS),
        'Transformer': create_transformer_matched(TARGET_PARAMS),
    }

    # Move to device and count params
    for name, model in models.items():
        model.to(DEVICE)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} parameters")

    # Train models
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\n  Training {name}...")
        train_model(model, tasks, TRAINING_STEPS)

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    results = {}

    for name, model in models.items():
        print(f"\n{name}:")
        model_results = {
            'params': sum(p.numel() for p in model.parameters()),
            'reasoning': {},
            'physics': {},
        }

        # Reasoning tasks
        print("  Reasoning:")
        reasoning_scores = []
        for task_name in reasoning_tasks:
            acc = tasks[task_name].evaluate(model, DEVICE)
            model_results['reasoning'][task_name] = acc
            reasoning_scores.append(acc)
            print(f"    {task_name}: {acc:.2%}")

        # Physics tasks
        print("  Physics:")
        physics_scores = []
        for task_name in physics_tasks:
            acc = tasks[task_name].evaluate(model, DEVICE)
            model_results['physics'][task_name] = acc
            physics_scores.append(acc)
            print(f"    {task_name}: {acc:.2%}")

        # Averages
        model_results['reasoning_avg'] = sum(reasoning_scores) / len(reasoning_scores)
        model_results['physics_avg'] = sum(physics_scores) / len(physics_scores)
        model_results['overall'] = (model_results['reasoning_avg'] + model_results['physics_avg']) / 2

        print(f"  Reasoning Avg: {model_results['reasoning_avg']:.2%}")
        print(f"  Physics Avg: {model_results['physics_avg']:.2%}")
        print(f"  Overall: {model_results['overall']:.2%}")

        results[name] = model_results

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Params':<10} {'Reasoning':<12} {'Physics':<12} {'Overall':<12}")
    print("-" * 61)

    for name, res in results.items():
        print(f"{name:<15} {res['params']:<10,} {res['reasoning_avg']:<12.2%} {res['physics_avg']:<12.2%} {res['overall']:<12.2%}")

    # Highlight key metrics
    print("\n" + "=" * 70)
    print("KEY METRICS")
    print("=" * 70)

    zero_overall = results['ANIMA-Zero']['overall']
    one_overall = results['ANIMA-One']['overall']
    two_overall = results['ANIMA-Two']['overall']

    print(f"\nANIMA-Zero (baseline): {zero_overall:.2%}")
    print(f"ANIMA-One (parallel): {one_overall:.2%}")
    print(f"ANIMA-Two (HTC): {two_overall:.2%}")

    gap_one_zero = (one_overall - zero_overall) * 100
    gap_two_zero = (two_overall - zero_overall) * 100
    gap_two_one = (two_overall - one_overall) * 100

    print(f"\nGap Analysis:")
    print(f"  ANIMA-One vs Zero: {gap_one_zero:+.2f}pp")
    print(f"  ANIMA-Two vs Zero: {gap_two_zero:+.2f}pp")
    print(f"  ANIMA-Two vs One: {gap_two_one:+.2f}pp")

    # Physics deep dive (where the gap was largest)
    print("\nPhysics Task Deep Dive (where ANIMA-One failed):")
    for task in physics_tasks:
        zero_acc = results['ANIMA-Zero']['physics'][task]
        one_acc = results['ANIMA-One']['physics'][task]
        two_acc = results['ANIMA-Two']['physics'][task]
        print(f"  {task:<12} Zero: {zero_acc:.2%}  One: {one_acc:.2%}  Two: {two_acc:.2%}")

    # Save results
    results['target_params'] = TARGET_PARAMS
    results['tolerance'] = TOLERANCE

    output_path = Path(__file__).parent / 'anima_two_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Theoretical validation
    print("\n" + "=" * 70)
    print("THEORETICAL VALIDATION")
    print("=" * 70)

    if two_overall >= zero_overall * 0.95:
        print("\n[OK] ANIMA-Two achieves near ANIMA-Zero accuracy")
        print("  -> Hierarchical Temporal Correction WORKS")
    else:
        print("\n[FAIL] ANIMA-Two below target accuracy")
        print("  -> May need more correction steps or deeper boundary network")

    if two_overall > one_overall:
        print("\n[OK] ANIMA-Two outperforms ANIMA-One")
        print("  -> Boundary correction restores temporal coherence")
    else:
        print("\n[FAIL] ANIMA-Two not better than ANIMA-One")
        print("  -> Need to investigate boundary correction implementation")

    two_projectile = results['ANIMA-Two']['physics']['projectile']
    one_projectile = results['ANIMA-One']['physics']['projectile']
    if two_projectile > one_projectile:
        print(f"\n[OK] Projectile task improved: {one_projectile:.2%} -> {two_projectile:.2%}")
        print("  -> Continuous trajectory dynamics restored")
    else:
        print(f"\n[FAIL] Projectile task not improved: {one_projectile:.2%} -> {two_projectile:.2%}")

    return results


if __name__ == '__main__':
    run_benchmark()
