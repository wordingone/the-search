"""
V4 Efficiency Benchmark
=======================

Tests V4 (Minimal Efficient) against V1-Base and other sub-100k variants.

Key question: Does V4's targeted additions (goal pathway, urgency gate)
improve over V1-Base while staying parameter-efficient?

Targets:
- V4 should beat V1-Base overall (>61.6%)
- V4 should stay under 60k params
- V4 should show improvement specifically in goal-seeking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from typing import Dict, Any, List

from anima.core import Anima, AnimaV2, AnimaV4, AnimaTelos
from anima.variants import AnimaMortal, AnimaMetamorphic


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
            # Pad to 8 dims
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
            cond = np.random.choice([0, 1, 2])  # gt, lt, eq
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
        # This is V4's target improvement area
        for _ in range(n_examples):
            pos = np.random.uniform(0, 10, 2)
            goal = np.random.uniform(0, 10, 2)
            direction = goal - pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            # Input: [pos_x, pos_y, goal_x, goal_y, ...]
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


def evaluate_model(model, examples: List[Dict], warmup: int = 30) -> float:
    """Evaluate model on task."""
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
                    # Reset model
                    try:
                        model.__init__(model.config if hasattr(model, 'config') else None)
                    except:
                        pass
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

    return correct / total if total > 0 else 0.0


def count_params(model) -> int:
    """Count parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_variant(name: str, model) -> Dict[str, Any]:
    """Test a variant on all tasks."""
    print(f"\n  {name}...")

    results = {'reasoning': {}, 'physics': {}}

    # Reasoning
    for task in ['sequence', 'pattern', 'conditional', 'analogy']:
        examples = create_task_examples(task)
        score = evaluate_model(model, examples)
        results['reasoning'][task] = round(score, 3)
        print(f"    {task}: {score:.1%}")
        # Reset
        try:
            model.__init__(model.config if hasattr(model, 'config') else None)
        except:
            pass

    # Physics
    for task in ['projectile', 'collision', 'goal', 'momentum']:
        examples = create_task_examples(task)
        score = evaluate_model(model, examples)
        results['physics'][task] = round(score, 3)
        print(f"    {task}: {score:.1%}")
        try:
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
    print("=" * 60)
    print("V4 EFFICIENCY BENCHMARK")
    print("Testing parameter efficiency under 100k limit")
    print("=" * 60)

    results = {}

    # V1-Base (reference)
    print("\n--- V1-Base (38k params, reference) ---")
    model = Anima()
    results['V1-Base'] = test_variant('V1-Base', model)

    # V1 Variants
    print("\n--- V1 Variants ---")
    model = AnimaMortal()
    results['V1-Mortal'] = test_variant('V1-Mortal', model)

    model = AnimaMetamorphic()
    results['V1-Metamorphic'] = test_variant('V1-Metamorphic', model)

    # V2-Core (98k, at limit)
    print("\n--- V2-Core (98k params) ---")
    model = AnimaV2()
    results['V2-Core'] = test_variant('V2-Core', model)

    # V4 (NEW - minimal efficient)
    print("\n--- V4-MinimalEfficient (target: <60k) ---")
    model = AnimaV4()
    results['V4-MinimalEfficient'] = test_variant('V4-MinimalEfficient', model)

    # V4 variants
    print("\n--- V4 without goal pathway ---")
    from anima.core.anima_v4 import AnimaV4Config
    config = AnimaV4Config(use_goal_pathway=False)
    model = AnimaV4(config)
    results['V4-NoGoal'] = test_variant('V4-NoGoal', model)

    print("\n--- V4 without urgency gate ---")
    config = AnimaV4Config(use_urgency_gate=False)
    model = AnimaV4(config)
    results['V4-NoUrgency'] = test_variant('V4-NoUrgency', model)

    print("\n--- V4 base only (no pathways) ---")
    config = AnimaV4Config(use_goal_pathway=False, use_urgency_gate=False)
    model = AnimaV4(config)
    results['V4-BaseOnly'] = test_variant('V4-BaseOnly', model)

    # V4-Telos (NEW - goal as embodied preference)
    print("\n--- V4-Telos (Goal as Embodied Preference) ---")
    model = AnimaTelos()
    results['V4-Telos'] = test_variant('V4-Telos', model)

    # Summary
    print("\n" + "=" * 60)
    print("EFFICIENCY SUMMARY (Under 100k params)")
    print("=" * 60)

    # Sort by efficiency (overall / params * 10000)
    def efficiency(r):
        return r['overall'] / (r['params'] / 10000) if r['params'] > 0 else 0

    sorted_results = sorted(results.items(), key=lambda x: efficiency(x[1]), reverse=True)

    print(f"\n{'Rank':<5} {'Variant':<20} {'Overall':<10} {'Params':<10} {'Efficiency':<12}")
    print("-" * 60)
    for i, (name, data) in enumerate(sorted_results, 1):
        eff = efficiency(data)
        print(f"{i:<5} {name:<20} {data['overall']:.1%}      {data['params']:>7,}   {eff:.2f}")

    # V4 Analysis
    print("\n" + "=" * 60)
    print("V4 ANALYSIS")
    print("=" * 60)

    v4 = results['V4-MinimalEfficient']
    v1 = results['V1-Base']

    print(f"\nV4 vs V1-Base:")
    print(f"  Overall: {v4['overall']:.1%} vs {v1['overall']:.1%} ({'+' if v4['overall'] > v1['overall'] else ''}{(v4['overall']-v1['overall'])*100:.1f}pp)")
    print(f"  Params:  {v4['params']:,} vs {v1['params']:,} ({v4['params']/v1['params']:.2f}x)")
    print(f"  Goal:    {v4['physics']['goal']:.1%} vs {v1['physics']['goal']:.1%}")

    print(f"\nGoal pathway contribution:")
    v4_ng = results['V4-NoGoal']
    print(f"  With goal pathway:    {v4['physics']['goal']:.1%}")
    print(f"  Without goal pathway: {v4_ng['physics']['goal']:.1%}")
    print(f"  Improvement: {'+' if v4['physics']['goal'] > v4_ng['physics']['goal'] else ''}{(v4['physics']['goal']-v4_ng['physics']['goal'])*100:.1f}pp")

    print(f"\nUrgency gate contribution:")
    v4_nu = results['V4-NoUrgency']
    print(f"  With urgency:    {v4['physics']['projectile']:.1%}")
    print(f"  Without urgency: {v4_nu['physics']['projectile']:.1%}")
    print(f"  Improvement: {'+' if v4['physics']['projectile'] > v4_nu['physics']['projectile'] else ''}{(v4['physics']['projectile']-v4_nu['physics']['projectile'])*100:.1f}pp")

    # Save
    output_path = Path(__file__).parent / 'v4_efficiency_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
