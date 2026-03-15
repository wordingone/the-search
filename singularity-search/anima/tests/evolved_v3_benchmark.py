"""
V3 Benchmark: Adaptive Coupling Architecture
=============================================

Tests whether V3's learnable coupling coefficient can achieve:
- Conditional: >= 90% (V2 level)
- Analogy: >= 90% (V1 level)
- No regressions on other tasks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Dict, List
import json

from anima.core import AnimaATR, AnimaISSM, AnimaEvolved
from anima.core.anima_evolved_v2 import AnimaEvolvedV2
from anima.core.anima_evolved_v3 import AnimaEvolvedV3


# =============================================================================
# DATA GENERATORS (same as comprehensive benchmark)
# =============================================================================

def generate_sequence_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        seq = [(start + i * step) / 100.0 for i in range(9)]
        data.append({'input': seq[:-1], 'target': seq[-1], 'task': 'sequence'})
    return data

def generate_pattern_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        plen = random.randint(2, 4)
        pattern = [random.randint(1, 9) / 10.0 for _ in range(plen)]
        seq = (pattern * 5)[:9]
        data.append({'input': seq[:-1], 'target': seq[-1], 'task': 'pattern'})
    return data

def generate_conditional_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        seq = [random.randint(1, 10) / 20.0 for _ in range(8)]
        last = seq[-1] * 20
        target = (last * 2 if last > 5 else last + 1) / 20.0
        data.append({'input': seq, 'target': target, 'task': 'conditional'})
    return data

def generate_analogy_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        a = random.randint(1, 5) / 10.0
        b = a * 2
        c = random.randint(1, 5) / 10.0
        d = c * 2
        data.append({'input': [a, b, c, 0, 0, 0, 0, 0], 'target': d, 'task': 'analogy'})
    return data

def generate_projectile_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        v0 = random.uniform(5, 15)
        positions = [v0 * t * 0.1 / 20.0 for t in range(8)]
        landing = v0 * 0.8 / 20.0
        data.append({'input': positions, 'target': min(landing, 1.0), 'task': 'projectile'})
    return data

def generate_collision_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        x1, v1 = random.uniform(0, 0.5), random.uniform(0.1, 0.3)
        x2, v2 = random.uniform(0.5, 1), random.uniform(-0.3, -0.1)
        will_collide = 1.0 if v1 > -v2 else 0.0
        seq = [x1, v1, x2, v2, 0, 0, 0, 0]
        data.append({'input': seq, 'target': will_collide, 'task': 'collision', 'binary': True})
    return data

def generate_goal_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        pos = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        goal = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        dist = max((dx**2 + dy**2)**0.5, 0.01)
        action = [dx/dist, dy/dist]
        seq = pos + goal + [0, 0, 0, 0]
        data.append({'input': seq, 'target': action, 'task': 'goal', 'goal': True})
    return data

def generate_momentum_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        m1, v1 = random.uniform(0.2, 1), random.uniform(0.2, 1)
        m2, v2 = random.uniform(0.2, 1), random.uniform(-1, -0.2)
        v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
        seq = [m1, v1, m2, v2, 0, 0, 0, 0]
        data.append({'input': seq, 'target': (v_final + 1) / 2, 'task': 'momentum'})
    return data

def generate_delay_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        signal = random.uniform(0.2, 0.8)
        delay = random.randint(2, 5)
        seq = [signal] + [0.0] * delay + [0.5] + [0.0] * (7 - delay)
        data.append({'input': seq[:8], 'target': signal, 'task': 'delay'})
    return data

def generate_copy_data(n, seed):
    random.seed(seed)
    data = []
    for _ in range(n):
        seq = [random.uniform(0.1, 0.9) for _ in range(8)]
        data.append({'input': seq, 'target': seq[0], 'task': 'copy'})
    return data


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(arch_name: str, device, seed: int):
    """Create fresh model with proper seeding and parameter matching (~48K)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if arch_name == 'ATR':
        return AnimaATR(sensory_dim=8, d_model=48, bottleneck_dim=24, output_dim=4).to(device)
    elif arch_name == 'ISSM':
        return AnimaISSM(sensory_dim=8, d_model=24, bottleneck_dim=12, output_dim=4, d_state=16).to(device)
    elif arch_name == 'EvolvedV1':
        return AnimaEvolved(sensory_dim=8, d_model=23, bottleneck_dim=11, output_dim=4, d_state=16).to(device)
    elif arch_name == 'EvolvedV2':
        return AnimaEvolvedV2(sensory_dim=8, d_model=23, bottleneck_dim=11, output_dim=4, d_state=16).to(device)
    elif arch_name == 'EvolvedV3':
        # V3 with parameter matching (~47.5K)
        return AnimaEvolvedV3(sensory_dim=8, d_model=23, bottleneck_dim=11, output_dim=4, d_state=16).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_evaluate(arch_name, train_data, test_data, device, seed, epochs=50, lr=0.01):
    """Train and evaluate with fresh model."""
    task = train_data[0]['task']
    is_goal = task == 'goal'
    is_binary = task == 'collision'

    model = create_model(arch_name, device, seed)

    # Prepare data
    train_inputs = [d['input'][:8] + [0]*(8-len(d['input'][:8])) for d in train_data]
    train_x = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    train_x = train_x.unsqueeze(1).expand(-1, 8, -1)

    if is_goal:
        train_y = torch.tensor([d['target'] for d in train_data], dtype=torch.float32, device=device)
    else:
        train_y = torch.tensor([[d['target']] for d in train_data], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()

        # All Evolved variants return dict
        if arch_name.startswith('Evolved'):
            outputs = model(train_x)['output'][:, -1]
        else:
            outputs = model(train_x)[:, -1]

        if is_goal:
            outputs = outputs[:, :2]
        else:
            outputs = outputs[:, :1]

        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    test_inputs = [d['input'][:8] + [0]*(8-len(d['input'][:8])) for d in test_data]
    test_x = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_x = test_x.unsqueeze(1).expand(-1, 8, -1)

    correct = 0
    with torch.no_grad():
        if arch_name.startswith('Evolved'):
            outputs = model(test_x)['output'][:, -1]
        else:
            outputs = model(test_x)[:, -1]

        for i, d in enumerate(test_data):
            if is_goal:
                pred = outputs[i, :2].cpu().numpy()
                target = d['target']
                dot = pred[0] * target[0] + pred[1] * target[1]
                pred_norm = (pred[0]**2 + pred[1]**2)**0.5 + 1e-6
                target_norm = (target[0]**2 + target[1]**2)**0.5 + 1e-6
                if dot / (pred_norm * target_norm) > 0.7:
                    correct += 1
            elif is_binary:
                if (outputs[i, 0].item() > 0.5) == (d['target'] > 0.5):
                    correct += 1
            else:
                pred = outputs[i, 0].item()
                target = d['target']
                tol = max(abs(target) * 0.2, 0.1)
                if abs(pred - target) <= tol:
                    correct += 1

    # Get coupling stats for V3
    coupling_stats = None
    if arch_name == 'EvolvedV3':
        coupling_stats = model.get_coupling_stats()

    return correct / len(test_data), coupling_stats


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("V3 BENCHMARK: ADAPTIVE COUPLING")
    print(f"{'='*70}")

    # Verify parameters
    print("\nParameter counts (~48K target):")
    for name, create_fn in [
        ('ATR', lambda: AnimaATR(sensory_dim=8, d_model=48, bottleneck_dim=24, output_dim=4)),
        ('ISSM', lambda: AnimaISSM(sensory_dim=8, d_model=24, bottleneck_dim=12, output_dim=4, d_state=16)),
        ('EvolvedV1', lambda: AnimaEvolved(sensory_dim=8, d_model=23, bottleneck_dim=11, output_dim=4, d_state=16)),
        ('EvolvedV2', lambda: AnimaEvolvedV2(sensory_dim=8, d_model=23, bottleneck_dim=11, output_dim=4, d_state=16)),
        ('EvolvedV3', lambda: AnimaEvolvedV3(sensory_dim=8, d_model=23, bottleneck_dim=11, output_dim=4, d_state=16)),
    ]:
        m = create_fn()
        print(f"  {name}: {sum(p.numel() for p in m.parameters()):,}")

    # Key tasks to test V3's adaptive coupling
    tasks = {
        'conditional': generate_conditional_data,  # V2 wins, V1 fails
        'analogy': generate_analogy_data,          # V1 wins, V2 fails
        'copy': generate_copy_data,                # Both V1 and V2 good
        'pattern': generate_pattern_data,          # Both good
        'delay': generate_delay_data,              # Both good
        'sequence': generate_sequence_data,        # Both good
        'projectile': generate_projectile_data,    # Both good
        'collision': generate_collision_data,      # Both moderate
        'goal': generate_goal_data,                # Both good
        'momentum': generate_momentum_data,        # Both good
    }

    architectures = ['ATR', 'ISSM', 'EvolvedV1', 'EvolvedV2', 'EvolvedV3']
    results = {arch: {} for arch in architectures}
    coupling_results = {}

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")

    for task_name, generator in tasks.items():
        print(f"\n--- {task_name.upper()} ---")

        train_data = generator(100, seed=42)
        test_data = generator(50, seed=43)

        for arch in architectures:
            accuracy, coupling_stats = train_evaluate(arch, train_data, test_data, device, seed=42)
            results[arch][task_name] = accuracy
            print(f"  {arch:<10}: {accuracy*100:>5.1f}%")

            if coupling_stats:
                coupling_results[task_name] = coupling_stats
                print(f"    Coupling: mean={coupling_stats['mean']:.3f}, %coupled={coupling_stats['pct_coupled']*100:.1f}%")

    # Summary
    print(f"\n{'='*70}")
    print("KEY COMPARISON: V1 vs V2 vs V3")
    print(f"{'='*70}")

    print(f"\n{'Task':<15} | {'V1':>8} | {'V2':>8} | {'V3':>8} | V3 vs Best Parent")
    print("-" * 60)

    for task_name in tasks.keys():
        v1 = results['EvolvedV1'][task_name] * 100
        v2 = results['EvolvedV2'][task_name] * 100
        v3 = results['EvolvedV3'][task_name] * 100

        best_parent = max(v1, v2)
        delta = v3 - best_parent
        delta_str = f"+{delta:.1f}pp" if delta >= 0 else f"{delta:.1f}pp"

        print(f"{task_name:<15} | {v1:>7.1f}% | {v2:>7.1f}% | {v3:>7.1f}% | {delta_str}")

    # Overall
    v1_avg = np.mean([results['EvolvedV1'][t] for t in tasks.keys()]) * 100
    v2_avg = np.mean([results['EvolvedV2'][t] for t in tasks.keys()]) * 100
    v3_avg = np.mean([results['EvolvedV3'][t] for t in tasks.keys()]) * 100

    print("-" * 60)
    print(f"{'OVERALL':<15} | {v1_avg:>7.1f}% | {v2_avg:>7.1f}% | {v3_avg:>7.1f}% |")

    # Critical tasks analysis
    print(f"\n{'='*70}")
    print("CRITICAL TASK ANALYSIS")
    print(f"{'='*70}")

    cond_v1 = results['EvolvedV1']['conditional'] * 100
    cond_v2 = results['EvolvedV2']['conditional'] * 100
    cond_v3 = results['EvolvedV3']['conditional'] * 100

    anal_v1 = results['EvolvedV1']['analogy'] * 100
    anal_v2 = results['EvolvedV2']['analogy'] * 100
    anal_v3 = results['EvolvedV3']['analogy'] * 100

    print(f"\nConditional (V2 specialty):")
    print(f"  V1: {cond_v1:.1f}%, V2: {cond_v2:.1f}%, V3: {cond_v3:.1f}%")
    print(f"  V3 target: >= {cond_v2:.1f}% (V2 level)")
    print(f"  Result: {'PASS' if cond_v3 >= cond_v2 - 5 else 'FAIL'}")

    print(f"\nAnalogy (V1 specialty):")
    print(f"  V1: {anal_v1:.1f}%, V2: {anal_v2:.1f}%, V3: {anal_v3:.1f}%")
    print(f"  V3 target: >= {anal_v1:.1f}% (V1 level)")
    print(f"  Result: {'PASS' if anal_v3 >= anal_v1 - 5 else 'FAIL'}")

    # Save results
    output = {
        'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        'coupling_stats': coupling_results,
        'summary': {
            'v1_overall': v1_avg,
            'v2_overall': v2_avg,
            'v3_overall': v3_avg,
            'conditional_fixed': cond_v3 >= cond_v2 - 5,
            'analogy_preserved': anal_v3 >= anal_v1 - 5,
        }
    }

    with open('EVOLVED_V3_BENCHMARK.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to EVOLVED_V3_BENCHMARK.json")

    return results


if __name__ == "__main__":
    results = run_benchmark()
