"""
V5 Temporal Horizon Benchmark
=============================

Tests V5 (Temporal Horizon) against V4-MinimalEfficient baseline.

Key Hypothesis: V5's H = (H_short, H_long, alpha) enhances ALL tasks:
- Reasoning tasks: H_long captures pattern structure (alpha < 0.5)
- Reactive tasks: H_short provides immediacy (alpha > 0.5)
- Navigation tasks: Both horizons balanced (alpha ~ 0.5)

Targets:
- V5 should beat V4-MinEff overall (>65.5%)
- V5 should beat V4-Telos on goal (>20%)
- V5 should NOT hurt reasoning tasks (unlike V4-Telos)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from typing import Dict, Any, List
from collections import defaultdict

from anima.core import (
    Anima, AnimaV4, AnimaTelos,
    AnimaV5, AnimaV5Fixed, AnimaV5Adaptive, AnimaV5Full
)


def create_task_examples(task_type: str, n_examples: int = 100) -> List[Dict]:
    """Create task examples for benchmarking."""
    np.random.seed(42)
    examples = []

    if task_type == "sequence":
        # Pattern: arithmetic sequence prediction
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
        # Pattern: repeating pattern recognition
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
        # Logic: conditional operations
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
        # Reasoning: a:b::c:?
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
        # Physics: projectile landing prediction
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
        # Physics: collision prediction (binary)
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
        # Navigation: direction to goal
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
        # Physics: elastic collision final velocity
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


def evaluate_model(model, examples: List[Dict], warmup: int = 30, track_alpha: bool = False) -> Dict[str, Any]:
    """Evaluate model on task with optional alpha tracking."""
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
    alpha_values = []

    for ex in examples:
        try:
            inp = ex['input'].unsqueeze(0)

            if hasattr(model, 'step'):
                result = model.step(inp)
                if result is None or not result.get('alive', True):
                    # Reset model
                    try:
                        if hasattr(model, 'reset_state'):
                            model.reset_state()
                        else:
                            model.__init__(model.config if hasattr(model, 'config') else None)
                    except:
                        pass
                    continue
                pred = result.get('action', torch.zeros(1, 4))

                # Track alpha for V5 models
                if track_alpha and 'alpha' in result:
                    alpha_val = result['alpha']
                    if isinstance(alpha_val, torch.Tensor):
                        alpha_values.append(alpha_val.item())
                    else:
                        alpha_values.append(alpha_val)
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

    score = correct / total if total > 0 else 0.0
    result = {'score': score}

    if alpha_values:
        result['alpha_mean'] = np.mean(alpha_values)
        result['alpha_std'] = np.std(alpha_values)

    return result


def count_params(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_variant(name: str, model, track_alpha: bool = False) -> Dict[str, Any]:
    """Test a variant on all tasks."""
    print(f"\n  {name}...")

    results = {'reasoning': {}, 'physics': {}, 'alpha': {}}

    # Reasoning tasks (expect alpha < 0.5 for V5)
    for task in ['sequence', 'pattern', 'conditional', 'analogy']:
        examples = create_task_examples(task)
        eval_result = evaluate_model(model, examples, track_alpha=track_alpha)
        results['reasoning'][task] = round(eval_result['score'], 3)
        if 'alpha_mean' in eval_result:
            results['alpha'][task] = round(eval_result['alpha_mean'], 3)
        print(f"    {task}: {eval_result['score']:.1%}", end='')
        if 'alpha_mean' in eval_result:
            print(f" (alpha={eval_result['alpha_mean']:.2f})", end='')
        print()

        # Reset model state
        try:
            if hasattr(model, 'reset_state'):
                model.reset_state()
            else:
                model.__init__(model.config if hasattr(model, 'config') else None)
        except:
            pass

    # Physics tasks (expect alpha > 0.5 for V5 on reactive tasks)
    for task in ['projectile', 'collision', 'goal', 'momentum']:
        examples = create_task_examples(task)
        eval_result = evaluate_model(model, examples, track_alpha=track_alpha)
        results['physics'][task] = round(eval_result['score'], 3)
        if 'alpha_mean' in eval_result:
            results['alpha'][task] = round(eval_result['alpha_mean'], 3)
        print(f"    {task}: {eval_result['score']:.1%}", end='')
        if 'alpha_mean' in eval_result:
            print(f" (alpha={eval_result['alpha_mean']:.2f})", end='')
        print()

        try:
            if hasattr(model, 'reset_state'):
                model.reset_state()
            else:
                model.__init__(model.config if hasattr(model, 'config') else None)
        except:
            pass

    results['reasoning_avg'] = round(np.mean(list(results['reasoning'].values())), 4)
    results['physics_avg'] = round(np.mean(list(results['physics'].values())), 4)
    results['overall'] = round((results['reasoning_avg'] + results['physics_avg']) / 2, 4)
    results['params'] = count_params(model)

    print(f"    Reasoning: {results['reasoning_avg']:.1%}")
    print(f"    Physics: {results['physics_avg']:.1%}")
    print(f"    Overall: {results['overall']:.1%}")
    print(f"    Params: {results['params']:,}")

    return results


def main():
    print("=" * 70)
    print("V5 TEMPORAL HORIZON BENCHMARK")
    print("Testing if H = (H_short, H_long, alpha) enhances ALL tasks")
    print("=" * 70)

    results = {}

    # Baselines
    print("\n--- BASELINES ---")

    print("\n--- V1-Base (reference) ---")
    model = Anima()
    results['V1-Base'] = test_variant('V1-Base', model)

    print("\n--- V4-MinimalEfficient (main baseline) ---")
    model = AnimaV4()
    results['V4-MinEff'] = test_variant('V4-MinEff', model)

    print("\n--- V4-Telos (goal-only improvement) ---")
    model = AnimaTelos()
    results['V4-Telos'] = test_variant('V4-Telos', model)

    # V5 Variants
    print("\n--- V5 TEMPORAL HORIZON VARIANTS ---")

    print("\n--- V5-A: Fixed Horizons (alpha=0.5) ---")
    model = AnimaV5Fixed()
    results['V5-A-Fixed'] = test_variant('V5-A-Fixed', model, track_alpha=True)

    print("\n--- V5-B: Adaptive Horizons (main variant) ---")
    model = AnimaV5Adaptive()
    results['V5-B-Adaptive'] = test_variant('V5-B-Adaptive', model, track_alpha=True)

    print("\n--- V5-C: Full Adaptive (alpha + decay) ---")
    model = AnimaV5Full()
    results['V5-C-Full'] = test_variant('V5-C-Full', model, track_alpha=True)

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Efficiency ranking
    def efficiency(r):
        return r['overall'] / (r['params'] / 10000) if r['params'] > 0 else 0

    sorted_results = sorted(results.items(), key=lambda x: x[1]['overall'], reverse=True)

    print(f"\n{'Rank':<5} {'Variant':<20} {'Overall':<10} {'Reasoning':<10} {'Physics':<10} {'Goal':<8} {'Params':<10}")
    print("-" * 75)
    for i, (name, data) in enumerate(sorted_results, 1):
        goal = data['physics'].get('goal', 0)
        print(f"{i:<5} {name:<20} {data['overall']:.1%}      "
              f"{data['reasoning_avg']:.1%}       {data['physics_avg']:.1%}      "
              f"{goal:.1%}    {data['params']:>8,}")

    # V5 Analysis
    print("\n" + "=" * 70)
    print("V5 vs V4 ANALYSIS")
    print("=" * 70)

    v5b = results['V5-B-Adaptive']
    v4m = results['V4-MinEff']
    v4t = results['V4-Telos']

    print(f"\nV5-B vs V4-MinEff:")
    print(f"  Overall: {v5b['overall']:.1%} vs {v4m['overall']:.1%} ({'+' if v5b['overall'] > v4m['overall'] else ''}{(v5b['overall']-v4m['overall'])*100:.1f}pp)")
    print(f"  Reasoning: {v5b['reasoning_avg']:.1%} vs {v4m['reasoning_avg']:.1%}")
    print(f"  Physics: {v5b['physics_avg']:.1%} vs {v4m['physics_avg']:.1%}")
    print(f"  Goal: {v5b['physics']['goal']:.1%} vs {v4m['physics']['goal']:.1%}")

    print(f"\nV5-B vs V4-Telos (goal comparison):")
    print(f"  Goal: {v5b['physics']['goal']:.1%} vs {v4t['physics']['goal']:.1%}")
    print(f"  Overall: {v5b['overall']:.1%} vs {v4t['overall']:.1%}")
    print(f"  (V4-Telos improved goal but hurt everything else)")

    # Alpha Analysis
    if 'alpha' in v5b and v5b['alpha']:
        print("\n" + "=" * 70)
        print("ALPHA (HORIZON BALANCE) ANALYSIS")
        print("=" * 70)

        print("\nExpected:")
        print("  Reasoning tasks: alpha < 0.5 (long-horizon biased)")
        print("  Reactive tasks:  alpha > 0.5 (short-horizon biased)")

        print("\nObserved (V5-B):")
        for task, alpha in v5b['alpha'].items():
            task_type = "reasoning" if task in ['sequence', 'pattern', 'conditional', 'analogy'] else "reactive"
            expected = "< 0.5" if task_type == "reasoning" else "> 0.5"
            actual = "OK" if (alpha < 0.5 and task_type == "reasoning") or (alpha >= 0.5 and task_type != "reasoning") else "UNEXPECTED"
            print(f"  {task}: alpha = {alpha:.3f} (expected {expected}) [{actual}]")

    # Task-by-task comparison
    print("\n" + "=" * 70)
    print("TASK-BY-TASK COMPARISON: V5-B vs V4-MinEff vs V4-Telos")
    print("=" * 70)

    print(f"\n{'Task':<15} {'V5-B':<10} {'V4-MinEff':<10} {'V4-Telos':<10} {'V5 vs V4M':<12}")
    print("-" * 60)

    all_tasks = list(v5b['reasoning'].keys()) + list(v5b['physics'].keys())
    for task in all_tasks:
        v5_score = v5b['reasoning'].get(task, v5b['physics'].get(task, 0))
        v4m_score = v4m['reasoning'].get(task, v4m['physics'].get(task, 0))
        v4t_score = v4t['reasoning'].get(task, v4t['physics'].get(task, 0))
        delta = v5_score - v4m_score
        delta_str = f"{'+' if delta > 0 else ''}{delta*100:.1f}pp"
        print(f"{task:<15} {v5_score:.1%}     {v4m_score:.1%}       {v4t_score:.1%}      {delta_str}")

    # Save results
    output_path = Path(__file__).parent / 'v5_horizon_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Key Finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    if v5b['overall'] > v4m['overall'] and v5b['physics']['goal'] > v4m['physics']['goal']:
        print("\nSUCCESS: V5-B improves BOTH overall AND goal-seeking!")
        print("  The temporal horizon structure enhances multiple tasks.")
    elif v5b['overall'] > v4t['overall'] and v5b['physics']['goal'] >= v4t['physics']['goal']:
        print("\nPARTIAL SUCCESS: V5-B beats V4-Telos overall while matching goal.")
        print("  Temporal integration is better than separate goal pathway.")
    else:
        print("\nNEEDS ITERATION: V5-B did not achieve target improvement.")
        print("  Consider adjusting horizon parameters or architecture.")

    return results


if __name__ == '__main__':
    main()
