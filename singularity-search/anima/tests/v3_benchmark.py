"""
V3 Comprehensive Benchmark
==========================

Tests AnimaV3 (Collective-Democratic) against all previous variants
and the Transformer baseline.

Expected Targets (from v3_analysis.md):
- Reasoning: >9.5% avg (beat V1-Metamorphic)
- Physics: >52% avg (beat V1-Collective)
- Overall: >30% (beat all existing variants)
"""

import sys
import os
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from typing import Dict, Any, List

# Import all variants
from anima.core import Anima, AnimaV2, AnimaV3
from anima.variants import (
    AnimaMortal,
    AnimaMetamorphic,
    AnimaCollective, AnimaSwarm,
    AnimaNeuroplastic,
    AnimaAdaptive,
    AnimaResonant,
    AnimaPhoenix,
    AnimaPressured,
)
from anima.tests.tiny_transformer_baseline import TinyTransformer


def create_reasoning_task(task_type: str) -> List[Dict]:
    """Create reasoning task examples."""
    np.random.seed(42)  # Reproducibility

    examples = []

    if task_type == "sequence":
        # Number sequences: predict next number
        for _ in range(100):
            start = np.random.randint(1, 10)
            step = np.random.randint(1, 5)
            seq = [start + i * step for i in range(5)]
            target = start + 5 * step
            examples.append({
                'input': torch.tensor(seq, dtype=torch.float32) / 50,  # Normalize
                'target': torch.tensor([target / 50], dtype=torch.float32),
                'seq': seq,
                'answer': target
            })

    elif task_type == "pattern":
        # Pattern completion
        patterns = [
            [1, 2, 1, 2, 1],  # Alternating
            [1, 1, 2, 2, 1],  # Repeat pairs
            [1, 2, 3, 1, 2],  # Cyclic
            [1, 2, 4, 8, 16], # Doubling
        ]
        for _ in range(100):
            pattern = patterns[np.random.randint(0, len(patterns))]
            if pattern == [1, 2, 1, 2, 1]:
                target = 2
            elif pattern == [1, 1, 2, 2, 1]:
                target = 1
            elif pattern == [1, 2, 3, 1, 2]:
                target = 3
            else:
                target = 32
            examples.append({
                'input': torch.tensor(pattern, dtype=torch.float32) / 32,
                'target': torch.tensor([target / 32], dtype=torch.float32),
                'pattern': pattern,
                'answer': target
            })

    elif task_type == "conditional":
        # If-then reasoning
        for _ in range(100):
            a = np.random.randint(0, 10)
            b = np.random.randint(0, 10)
            condition = np.random.choice(['gt', 'lt', 'eq'])
            if condition == 'gt':
                target = a if a > b else b
            elif condition == 'lt':
                target = a if a < b else b
            else:
                target = a if a == b else 0
            cond_enc = {'gt': 0.33, 'lt': 0.66, 'eq': 1.0}[condition]
            examples.append({
                'input': torch.tensor([a/10, b/10, cond_enc], dtype=torch.float32),
                'target': torch.tensor([target / 10], dtype=torch.float32),
                'a': a, 'b': b, 'condition': condition,
                'answer': target
            })

    elif task_type == "analogy":
        # A:B :: C:?
        for _ in range(100):
            # Simple relationship: A+diff = B, C+diff = ?
            a = np.random.randint(1, 10)
            diff = np.random.randint(1, 5)
            b = a + diff
            c = np.random.randint(1, 10)
            target = c + diff
            examples.append({
                'input': torch.tensor([a/20, b/20, c/20], dtype=torch.float32),
                'target': torch.tensor([target / 20], dtype=torch.float32),
                'a': a, 'b': b, 'c': c,
                'answer': target
            })

    return examples


def create_physics_task(task_type: str) -> List[Dict]:
    """Create physics simulation task examples."""
    np.random.seed(42)

    examples = []

    if task_type == "projectile":
        # Predict landing position
        for _ in range(100):
            v0 = np.random.uniform(5, 15)
            angle = np.random.uniform(30, 60) * np.pi / 180
            g = 9.8
            t_flight = 2 * v0 * np.sin(angle) / g
            x_land = v0 * np.cos(angle) * t_flight
            examples.append({
                'input': torch.tensor([v0/20, angle, g/10], dtype=torch.float32),
                'target': torch.tensor([x_land / 50], dtype=torch.float32),
                'v0': v0, 'angle': angle,
                'answer': x_land
            })

    elif task_type == "collision":
        # Binary: will objects collide?
        for _ in range(100):
            x1, y1 = np.random.uniform(0, 10, 2)
            vx1, vy1 = np.random.uniform(-2, 2, 2)
            x2, y2 = np.random.uniform(0, 10, 2)
            vx2, vy2 = np.random.uniform(-2, 2, 2)

            # Simple collision check (10 timesteps)
            will_collide = 0
            for t in range(10):
                nx1, ny1 = x1 + vx1 * t, y1 + vy1 * t
                nx2, ny2 = x2 + vx2 * t, y2 + vy2 * t
                if np.sqrt((nx1-nx2)**2 + (ny1-ny2)**2) < 1.0:
                    will_collide = 1
                    break

            examples.append({
                'input': torch.tensor([x1/10, y1/10, vx1/5, vy1/5, x2/10, y2/10, vx2/5, vy2/5], dtype=torch.float32),
                'target': torch.tensor([float(will_collide)], dtype=torch.float32),
                'will_collide': will_collide
            })

    elif task_type == "goal":
        # Navigate to goal
        for _ in range(100):
            pos = np.random.uniform(0, 10, 2)
            goal = np.random.uniform(0, 10, 2)
            direction = goal - pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            examples.append({
                'input': torch.tensor([pos[0]/10, pos[1]/10, goal[0]/10, goal[1]/10], dtype=torch.float32),
                'target': torch.tensor([direction[0], direction[1]], dtype=torch.float32),
                'pos': pos, 'goal': goal,
                'answer': direction
            })

    elif task_type == "momentum":
        # Conservation of momentum
        for _ in range(100):
            m1, m2 = np.random.uniform(1, 5, 2)
            v1 = np.random.uniform(-5, 5)
            v2 = np.random.uniform(-5, 5)
            # Elastic collision
            v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            examples.append({
                'input': torch.tensor([m1/5, v1/10, m2/5, v2/10], dtype=torch.float32),
                'target': torch.tensor([v1_final / 10], dtype=torch.float32),
                'm1': m1, 'v1': v1, 'm2': m2, 'v2': v2,
                'answer': v1_final
            })

    return examples


def get_sensory_dim(model) -> int:
    """Get the expected sensory dimension for a model."""
    if hasattr(model, 'config') and hasattr(model.config, 'sensory_dim'):
        return model.config.sensory_dim
    elif hasattr(model, 'members') and len(model.members) > 0:
        # AnimaSwarm
        return get_sensory_dim(model.members[0])
    elif hasattr(model, 'agents') and len(model.agents) > 0:
        # AnimaV3
        return model.agents[0].config.sensory_dim
    elif hasattr(model, 'd_model'):
        # Transformer
        return 16
    return 8  # Default for V1


def evaluate_model(model, task_examples: List[Dict], is_binary: bool = False,
                   warmup_steps: int = 50) -> float:
    """Evaluate model on task examples."""
    correct = 0
    total = 0

    sensory_dim = get_sensory_dim(model)

    # Warmup
    for _ in range(warmup_steps):
        try:
            obs = torch.randn(1, sensory_dim)
            if hasattr(model, 'step'):
                model.step(obs)
            elif hasattr(model, 'forward'):
                model(obs)
        except:
            pass

    for example in task_examples:
        try:
            inp = example['input']

            # Pad/truncate input to match sensory_dim
            if inp.dim() == 1:
                if inp.shape[0] < sensory_dim:
                    inp = torch.cat([inp, torch.zeros(sensory_dim - inp.shape[0])])
                elif inp.shape[0] > sensory_dim:
                    inp = inp[:sensory_dim]
                inp = inp.unsqueeze(0)

            # Get prediction
            if hasattr(model, 'step'):
                result = model.step(inp)
                if result is None or not result.get('alive', True):
                    # Model died, reset
                    try:
                        if hasattr(model, '__class__'):
                            model.__init__(model.config if hasattr(model, 'config') else None)
                    except:
                        pass
                    continue
                pred = result.get('action', torch.zeros(1, 4))
            else:
                pred = model(inp)

            target = example['target']

            # Compare
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
            # Skip on error
            total += 1

    return correct / total if total > 0 else 0.0


def test_variant(variant_name: str, model) -> Dict[str, Any]:
    """Test a variant on all reasoning and physics tasks."""
    print(f"\n  Testing {variant_name}...")

    results = {
        'reasoning': {},
        'physics': {},
    }

    # Reasoning tasks
    for task in ['sequence', 'pattern', 'conditional', 'analogy']:
        examples = create_reasoning_task(task)
        score = evaluate_model(model, examples, is_binary=False)
        results['reasoning'][task.capitalize()] = round(score, 2)
        print(f"    {task}: {score:.1%}")

        # Reset model
        try:
            if hasattr(model, '__class__'):
                model.__init__(model.config if hasattr(model, 'config') else None)
        except:
            pass

    # Physics tasks
    for task in ['projectile', 'collision', 'goal', 'momentum']:
        examples = create_physics_task(task)
        is_binary = task == 'collision'
        score = evaluate_model(model, examples, is_binary=is_binary)
        results['physics'][task.capitalize()] = round(score, 4) if score < 0.01 else round(score, 2)
        print(f"    {task}: {score:.1%}")

        # Reset model
        try:
            if hasattr(model, '__class__'):
                model.__init__(model.config if hasattr(model, 'config') else None)
        except:
            pass

    # Averages
    results['reasoning_avg'] = round(np.mean(list(results['reasoning'].values())), 4)
    results['physics_avg'] = round(np.mean(list(results['physics'].values())), 4)
    results['overall'] = round((results['reasoning_avg'] + results['physics_avg']) / 2, 4)

    print(f"    Reasoning Avg: {results['reasoning_avg']:.1%}")
    print(f"    Physics Avg: {results['physics_avg']:.1%}")
    print(f"    Overall: {results['overall']:.1%}")

    return results


def count_parameters(model) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 70)
    print("V3 COMPREHENSIVE BENCHMARK")
    print("Testing AnimaV3 (Collective-Democratic) against all variants")
    print("=" * 70)

    results = {}

    # V1 Variants
    print("\n--- V1 VARIANTS ---")

    print("\nV1-Base:")
    model = Anima()
    results['V1-Base'] = test_variant('V1-Base', model)
    results['V1-Base']['params'] = count_parameters(model)

    print("\nV1-Mortal:")
    model = AnimaMortal()
    results['V1-Mortal'] = test_variant('V1-Mortal', model)
    results['V1-Mortal']['params'] = count_parameters(model)

    print("\nV1-Metamorphic:")
    model = AnimaMetamorphic()
    results['V1-Metamorphic'] = test_variant('V1-Metamorphic', model)
    results['V1-Metamorphic']['params'] = count_parameters(model)

    print("\nV1-Collective (Swarm of 3):")
    from anima.variants.collective import AnimaCollectiveConfig
    collective_config = AnimaCollectiveConfig(swarm_size=3)
    model = AnimaSwarm(collective_config)
    results['V1-Collective'] = test_variant('V1-Collective', model)
    results['V1-Collective']['params'] = sum(count_parameters(m) for m in model.members)

    print("\nV1-Neuroplastic:")
    model = AnimaNeuroplastic()
    results['V1-Neuroplastic'] = test_variant('V1-Neuroplastic', model)
    results['V1-Neuroplastic']['params'] = count_parameters(model)

    # V2 Variants
    print("\n--- V2 VARIANTS ---")

    print("\nV2-Core:")
    model = AnimaV2()
    results['V2-Core'] = test_variant('V2-Core', model)
    results['V2-Core']['params'] = count_parameters(model)

    print("\nV2-Adaptive:")
    model = AnimaAdaptive()
    results['V2-Adaptive'] = test_variant('V2-Adaptive', model)
    results['V2-Adaptive']['params'] = count_parameters(model)

    print("\nV2-Resonant:")
    model = AnimaResonant()
    results['V2-Resonant'] = test_variant('V2-Resonant', model)
    results['V2-Resonant']['params'] = count_parameters(model)

    print("\nV2-Phoenix:")
    model = AnimaPhoenix()
    results['V2-Phoenix'] = test_variant('V2-Phoenix', model)
    results['V2-Phoenix']['params'] = count_parameters(model)

    print("\nV2-Pressured:")
    model = AnimaPressured()
    results['V2-Pressured'] = test_variant('V2-Pressured', model)
    results['V2-Pressured']['params'] = count_parameters(model)

    # V3
    print("\n--- V3 (NEW) ---")

    print("\nV3-CollectiveDemocratic:")
    model = AnimaV3()
    results['V3-CollectiveDemocratic'] = test_variant('V3-CollectiveDemocratic', model)
    results['V3-CollectiveDemocratic']['params'] = count_parameters(model)

    # Transformer baseline
    print("\n--- BASELINE ---")

    print("\nTransformer (~103k params):")
    model = TinyTransformer()
    results['Transformer'] = test_variant('Transformer', model)
    results['Transformer']['params'] = count_parameters(model)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Sort by overall
    sorted_results = sorted(results.items(), key=lambda x: x[1]['overall'], reverse=True)

    print(f"\n{'Rank':<5} {'Variant':<25} {'Reasoning':<10} {'Physics':<10} {'Overall':<10} {'Params':<10}")
    print("-" * 70)
    for i, (name, data) in enumerate(sorted_results, 1):
        print(f"{i:<5} {name:<25} {data['reasoning_avg']:.1%}      {data['physics_avg']:.1%}      {data['overall']:.1%}      {data['params']:,}")

    # V3 Analysis
    v3_data = results['V3-CollectiveDemocratic']
    print("\n" + "=" * 70)
    print("V3 PERFORMANCE ANALYSIS")
    print("=" * 70)

    print(f"\nV3 Overall: {v3_data['overall']:.1%}")
    print(f"Target: >30%")
    print(f"Status: {'ACHIEVED' if v3_data['overall'] > 0.30 else 'NOT YET'}")

    print(f"\nV3 Reasoning: {v3_data['reasoning_avg']:.1%}")
    print(f"Target: >9.5%")
    print(f"Status: {'ACHIEVED' if v3_data['reasoning_avg'] > 0.095 else 'NOT YET'}")

    print(f"\nV3 Physics: {v3_data['physics_avg']:.1%}")
    print(f"Target: >52%")
    print(f"Status: {'ACHIEVED' if v3_data['physics_avg'] > 0.52 else 'NOT YET'}")

    # Save results
    output_path = Path(__file__).parent / 'v3_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
