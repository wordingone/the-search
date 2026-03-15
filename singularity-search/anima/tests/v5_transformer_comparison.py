"""
V5 vs Vanilla Transformer Comparison
=====================================

Exact benchmark values for all V5 variants against vanilla transformer.

Tests:
1. Reasoning tasks: sequence, pattern, conditional, analogy
2. Physics tasks: projectile, collision, goal, momentum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import V5 variants
from anima.core import (
    Anima, AnimaV4, AnimaTelos,
    AnimaV5, AnimaV5Fixed, AnimaV5Adaptive, AnimaV5Full
)


# ============================================================================
# VANILLA TRANSFORMER BASELINE
# ============================================================================

@dataclass
class VanillaTransformerConfig:
    """Config for vanilla transformer matched to V5 params (~25k)."""
    input_dim: int = 8
    output_dim: int = 4
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 64
    max_seq_len: int = 32
    dropout: float = 0.0


class VanillaTransformer(nn.Module):
    """
    Standard transformer baseline for comparison.
    Matched to ~25k parameters (similar to V5).
    """

    def __init__(self, config: Optional[VanillaTransformerConfig] = None):
        super().__init__()
        self.config = config or VanillaTransformerConfig()

        # Input projection
        self.input_proj = nn.Linear(self.config.input_dim, self.config.d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.config.max_seq_len, self.config.d_model) * 0.1
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_ff,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.n_layers)

        # Output projection
        self.output_proj = nn.Linear(self.config.d_model, self.config.output_dim)

        # State for sequential processing
        self.history = []
        self.max_history = self.config.max_seq_len

    def reset(self):
        """Reset history for new sequence."""
        self.history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on sequence."""
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.output_proj(x[:, -1, :])  # Last token output

    def step(self, observation: torch.Tensor) -> Dict[str, Any]:
        """Process single observation (for compatibility with Anima interface)."""
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Add to history
        self.history.append(observation)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Stack history
        seq = torch.cat(self.history, dim=0).unsqueeze(0)  # (1, seq_len, input_dim)

        # Forward
        with torch.no_grad():
            output = self.forward(seq)

        return {
            'action': output,
            'alive': True
        }


# ============================================================================
# TASK GENERATION
# ============================================================================

def create_task_examples(task_type: str, n_examples: int = 100) -> List[Dict]:
    """Create task examples."""
    np.random.seed(42)
    examples = []

    if task_type == "sequence":
        for _ in range(n_examples):
            start = np.random.randint(1, 10)
            step = np.random.randint(1, 5)
            seq = [start + i * step for i in range(5)]
            target = start + 5 * step
            inp = seq + [0, 0, 0]
            examples.append({
                'input': torch.tensor(inp[:8], dtype=torch.float32) / 50,
                'target': torch.tensor([target / 50], dtype=torch.float32),
            })

    elif task_type == "pattern":
        patterns = [
            ([1, 2, 1, 2, 1, 2, 1, 2], 1),
            ([1, 1, 2, 2, 1, 1, 2, 2], 1),
            ([1, 2, 3, 1, 2, 3, 1, 2], 3),
            ([1, 2, 4, 8, 1, 2, 4, 8], 16),
        ]
        for _ in range(n_examples):
            pattern, target = patterns[np.random.randint(0, len(patterns))]
            examples.append({
                'input': torch.tensor(pattern[:8], dtype=torch.float32) / 20,
                'target': torch.tensor([target / 20], dtype=torch.float32),
            })

    elif task_type == "conditional":
        for _ in range(n_examples):
            a = np.random.randint(0, 10)
            b = np.random.randint(0, 10)
            cond = np.random.choice([0, 1, 2])
            if cond == 0:
                target = max(a, b)
            elif cond == 1:
                target = min(a, b)
            else:
                target = a if a == b else 0
            inp = [a/10, b/10, cond/3, 0, 0, 0, 0, 0]
            examples.append({
                'input': torch.tensor(inp, dtype=torch.float32),
                'target': torch.tensor([target / 10], dtype=torch.float32),
            })

    elif task_type == "analogy":
        for _ in range(n_examples):
            a = np.random.randint(1, 10)
            diff = np.random.randint(1, 5)
            b = a + diff
            c = np.random.randint(1, 10)
            target = c + diff
            inp = [a/20, b/20, c/20, diff/10, 0, 0, 0, 0]
            examples.append({
                'input': torch.tensor(inp, dtype=torch.float32),
                'target': torch.tensor([target / 20], dtype=torch.float32),
            })

    elif task_type == "projectile":
        for _ in range(n_examples):
            v0 = np.random.uniform(5, 15)
            angle = np.random.uniform(30, 60) * np.pi / 180
            g = 9.8
            t_flight = 2 * v0 * np.sin(angle) / g
            x_land = v0 * np.cos(angle) * t_flight
            inp = [v0/20, angle, g/10, 0, 0, 0, 0, 0]
            examples.append({
                'input': torch.tensor(inp, dtype=torch.float32),
                'target': torch.tensor([x_land / 50], dtype=torch.float32),
            })

    elif task_type == "collision":
        for _ in range(n_examples):
            x1, y1 = np.random.uniform(0, 10, 2)
            vx1, vy1 = np.random.uniform(-2, 2, 2)
            x2, y2 = np.random.uniform(0, 10, 2)
            vx2, vy2 = np.random.uniform(-2, 2, 2)

            will_collide = 0
            for t in range(10):
                nx1, ny1 = x1 + vx1 * t, y1 + vy1 * t
                nx2, ny2 = x2 + vx2 * t, y2 + vy2 * t
                if np.sqrt((nx1-nx2)**2 + (ny1-ny2)**2) < 1.0:
                    will_collide = 1
                    break

            inp = [x1/10, y1/10, vx1/5, vy1/5, x2/10, y2/10, vx2/5, vy2/5]
            examples.append({
                'input': torch.tensor(inp, dtype=torch.float32),
                'target': torch.tensor([float(will_collide)], dtype=torch.float32),
                'is_binary': True,
            })

    elif task_type == "goal":
        for _ in range(n_examples):
            pos = np.random.uniform(0, 10, 2)
            goal = np.random.uniform(0, 10, 2)
            direction = goal - pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            inp = [pos[0]/10, pos[1]/10, goal[0]/10, goal[1]/10, 0, 0, 0, 0]
            examples.append({
                'input': torch.tensor(inp, dtype=torch.float32),
                'target': torch.tensor([direction[0], direction[1]], dtype=torch.float32),
            })

    elif task_type == "momentum":
        for _ in range(n_examples):
            m1, m2 = np.random.uniform(1, 5, 2)
            v1 = np.random.uniform(-5, 5)
            v2 = np.random.uniform(-5, 5)
            v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            inp = [m1/5, v1/10, m2/5, v2/10, 0, 0, 0, 0]
            examples.append({
                'input': torch.tensor(inp, dtype=torch.float32),
                'target': torch.tensor([v1_final / 10], dtype=torch.float32),
            })

    return examples


def evaluate_model(model, examples: List[Dict], warmup: int = 30) -> Dict[str, float]:
    """Evaluate model and return exact score."""
    # Reset if transformer
    if hasattr(model, 'reset'):
        model.reset()

    # Warmup
    for _ in range(warmup):
        obs = torch.randn(1, 8)
        try:
            if hasattr(model, 'step'):
                model.step(obs)
            else:
                model(obs)
        except:
            pass

    correct = 0
    total = 0

    for ex in examples:
        try:
            inp = ex['input'].unsqueeze(0)

            if hasattr(model, 'step'):
                result = model.step(inp)
                if result is None or not result.get('alive', True):
                    if hasattr(model, 'reset_state'):
                        model.reset_state()
                    elif hasattr(model, 'reset'):
                        model.reset()
                    continue
                pred = result.get('action', torch.zeros(1, 4))
            else:
                pred = model(inp)

            target = ex['target']
            is_binary = ex.get('is_binary', False)

            if is_binary:
                pred_val = pred[0, 0].item() if pred.dim() > 1 else pred[0].item()
                target_val = target[0].item()
                if (pred_val > 0.5) == (target_val > 0.5):
                    correct += 1
            else:
                pred_val = pred[0, 0].item() if pred.dim() > 1 else pred[0].item()
                target_val = target[0].item()
                if abs(pred_val - target_val) < 0.3:
                    correct += 1

            total += 1

        except Exception as e:
            total += 1

    return {'score': correct / total if total > 0 else 0.0, 'correct': correct, 'total': total}


def count_params(model) -> int:
    """Count parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 80)
    print("V5 vs VANILLA TRANSFORMER - EXACT VALUES")
    print("=" * 80)

    # All models to test
    models_config = [
        ('VanillaTransformer', lambda: VanillaTransformer()),
        ('V1-Base', lambda: Anima()),
        ('V4-MinEff', lambda: AnimaV4()),
        ('V4-Telos', lambda: AnimaTelos()),
        ('V5-A-Fixed', lambda: AnimaV5Fixed()),
        ('V5-B-Adaptive', lambda: AnimaV5Adaptive()),
        ('V5-C-Full', lambda: AnimaV5Full()),
    ]

    # Tasks
    reasoning_tasks = ['sequence', 'pattern', 'conditional', 'analogy']
    physics_tasks = ['projectile', 'collision', 'goal', 'momentum']

    results = {}

    for name, model_fn in models_config:
        print(f"\n{'='*60}")
        print(f"MODEL: {name}")
        print(f"{'='*60}")

        model = model_fn()
        params = count_params(model)
        print(f"Parameters: {params:,}")

        results[name] = {
            'params': params,
            'reasoning': {},
            'physics': {}
        }

        # Reasoning tests
        print("\n--- REASONING TASKS ---")
        for task in reasoning_tasks:
            examples = create_task_examples(task, n_examples=100)
            eval_result = evaluate_model(model, examples)
            score = eval_result['score']
            results[name]['reasoning'][task] = {
                'score': round(score, 4),
                'correct': eval_result['correct'],
                'total': eval_result['total'],
                'percent': f"{score*100:.2f}%"
            }
            print(f"  {task}: {eval_result['correct']}/{eval_result['total']} = {score*100:.2f}%")

            # Reset model
            if hasattr(model, 'reset_state'):
                model.reset_state()
            elif hasattr(model, 'reset'):
                model.reset()

        # Physics tests
        print("\n--- PHYSICS TASKS ---")
        for task in physics_tasks:
            examples = create_task_examples(task, n_examples=100)
            eval_result = evaluate_model(model, examples)
            score = eval_result['score']
            results[name]['physics'][task] = {
                'score': round(score, 4),
                'correct': eval_result['correct'],
                'total': eval_result['total'],
                'percent': f"{score*100:.2f}%"
            }
            print(f"  {task}: {eval_result['correct']}/{eval_result['total']} = {score*100:.2f}%")

            if hasattr(model, 'reset_state'):
                model.reset_state()
            elif hasattr(model, 'reset'):
                model.reset()

        # Calculate averages
        reasoning_scores = [results[name]['reasoning'][t]['score'] for t in reasoning_tasks]
        physics_scores = [results[name]['physics'][t]['score'] for t in physics_tasks]

        results[name]['reasoning_avg'] = round(np.mean(reasoning_scores), 4)
        results[name]['physics_avg'] = round(np.mean(physics_scores), 4)
        results[name]['overall'] = round((results[name]['reasoning_avg'] + results[name]['physics_avg']) / 2, 4)

        print(f"\n  Reasoning Avg: {results[name]['reasoning_avg']*100:.2f}%")
        print(f"  Physics Avg:   {results[name]['physics_avg']*100:.2f}%")
        print(f"  OVERALL:       {results[name]['overall']*100:.2f}%")

    # ========================================================================
    # EXACT VALUES TABLE
    # ========================================================================
    print("\n" + "=" * 100)
    print("EXACT VALUES - REASONING TASKS")
    print("=" * 100)

    header = f"{'Model':<20}"
    for task in reasoning_tasks:
        header += f" | {task:>12}"
    header += f" | {'AVG':>10}"
    print(header)
    print("-" * 100)

    for name in [m[0] for m in models_config]:
        row = f"{name:<20}"
        for task in reasoning_tasks:
            score = results[name]['reasoning'][task]['score']
            row += f" | {score*100:>11.2f}%"
        row += f" | {results[name]['reasoning_avg']*100:>9.2f}%"
        print(row)

    print("\n" + "=" * 100)
    print("EXACT VALUES - PHYSICS TASKS")
    print("=" * 100)

    header = f"{'Model':<20}"
    for task in physics_tasks:
        header += f" | {task:>12}"
    header += f" | {'AVG':>10}"
    print(header)
    print("-" * 100)

    for name in [m[0] for m in models_config]:
        row = f"{name:<20}"
        for task in physics_tasks:
            score = results[name]['physics'][task]['score']
            row += f" | {score*100:>11.2f}%"
        row += f" | {results[name]['physics_avg']*100:>9.2f}%"
        print(row)

    # ========================================================================
    # OVERALL SUMMARY
    # ========================================================================
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY (Sorted by Overall Score)")
    print("=" * 100)

    sorted_models = sorted(results.keys(), key=lambda m: results[m]['overall'], reverse=True)

    print(f"\n{'Rank':<6} {'Model':<20} {'Params':>10} {'Reasoning':>12} {'Physics':>12} {'OVERALL':>12}")
    print("-" * 80)

    for i, name in enumerate(sorted_models, 1):
        r = results[name]
        marker = " <-- TRANSFORMER" if name == "VanillaTransformer" else ""
        print(f"{i:<6} {name:<20} {r['params']:>10,} {r['reasoning_avg']*100:>11.2f}% {r['physics_avg']*100:>11.2f}% {r['overall']*100:>11.2f}%{marker}")

    # ========================================================================
    # V5 vs TRANSFORMER COMPARISON
    # ========================================================================
    print("\n" + "=" * 100)
    print("V5 vs VANILLA TRANSFORMER - EXACT DELTAS")
    print("=" * 100)

    transformer_results = results['VanillaTransformer']

    print(f"\n{'Metric':<15} {'Transformer':>12} {'V5-A':>12} {'V5-B':>12} {'V5-C':>12}")
    print("-" * 70)

    # All tasks
    all_tasks = reasoning_tasks + physics_tasks
    for task in all_tasks:
        if task in reasoning_tasks:
            t_score = transformer_results['reasoning'][task]['score']
            v5a = results['V5-A-Fixed']['reasoning'][task]['score']
            v5b = results['V5-B-Adaptive']['reasoning'][task]['score']
            v5c = results['V5-C-Full']['reasoning'][task]['score']
        else:
            t_score = transformer_results['physics'][task]['score']
            v5a = results['V5-A-Fixed']['physics'][task]['score']
            v5b = results['V5-B-Adaptive']['physics'][task]['score']
            v5c = results['V5-C-Full']['physics'][task]['score']

        print(f"{task:<15} {t_score*100:>11.2f}% {v5a*100:>11.2f}% {v5b*100:>11.2f}% {v5c*100:>11.2f}%")

    print("-" * 70)
    print(f"{'Reasoning Avg':<15} {transformer_results['reasoning_avg']*100:>11.2f}% "
          f"{results['V5-A-Fixed']['reasoning_avg']*100:>11.2f}% "
          f"{results['V5-B-Adaptive']['reasoning_avg']*100:>11.2f}% "
          f"{results['V5-C-Full']['reasoning_avg']*100:>11.2f}%")
    print(f"{'Physics Avg':<15} {transformer_results['physics_avg']*100:>11.2f}% "
          f"{results['V5-A-Fixed']['physics_avg']*100:>11.2f}% "
          f"{results['V5-B-Adaptive']['physics_avg']*100:>11.2f}% "
          f"{results['V5-C-Full']['physics_avg']*100:>11.2f}%")
    print(f"{'OVERALL':<15} {transformer_results['overall']*100:>11.2f}% "
          f"{results['V5-A-Fixed']['overall']*100:>11.2f}% "
          f"{results['V5-B-Adaptive']['overall']*100:>11.2f}% "
          f"{results['V5-C-Full']['overall']*100:>11.2f}%")

    # Save results
    output_path = Path(__file__).parent / 'v5_transformer_exact_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
