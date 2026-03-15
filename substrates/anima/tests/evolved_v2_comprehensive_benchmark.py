"""
Comprehensive Benchmark: AnimaEvolved V1 vs V2 vs Parents
=========================================================

Tests across ALL benchmark domains with parameter matching.
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


# =============================================================================
# ALL DATA GENERATORS
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

# Additional multimodal-inspired tasks
def generate_frequency_data(n, seed):
    """Audio-like: Detect dominant frequency class."""
    random.seed(seed)
    data = []
    for _ in range(n):
        freq_class = random.randint(0, 4)
        t = np.linspace(0, 2*np.pi, 8)
        freq = 1 + freq_class * 0.5
        seq = list(np.sin(freq * t) * 0.5 + 0.5 + np.random.randn(8) * 0.05)
        target = freq_class / 4.0
        data.append({'input': seq, 'target': target, 'task': 'frequency'})
    return data

def generate_trend_data(n, seed):
    """Time series: Detect up/down trend."""
    random.seed(seed)
    data = []
    for _ in range(n):
        is_up = random.random() > 0.5
        if is_up:
            seq = list(np.linspace(0.2, 0.8, 8) + np.random.randn(8) * 0.05)
        else:
            seq = list(np.linspace(0.8, 0.2, 8) + np.random.randn(8) * 0.05)
        target = 1.0 if is_up else 0.0
        data.append({'input': seq, 'target': target, 'task': 'trend', 'binary': True})
    return data

def generate_xor_data(n, seed):
    """Logic: XOR of first two inputs."""
    random.seed(seed)
    data = []
    for _ in range(n):
        a = random.random() > 0.5
        b = random.random() > 0.5
        result = (a and not b) or (not a and b)
        seq = [1.0 if a else 0.0, 1.0 if b else 0.0] + [0.0] * 6
        target = 1.0 if result else 0.0
        data.append({'input': seq, 'target': target, 'task': 'xor', 'binary': True})
    return data

def generate_count_data(n, seed):
    """Counting: Count elements above 0.5."""
    random.seed(seed)
    data = []
    for _ in range(n):
        seq = [random.random() for _ in range(8)]
        count = sum(1 for x in seq if x > 0.5)
        target = count / 8.0
        data.append({'input': seq, 'target': target, 'task': 'count'})
    return data


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(arch_name: str, device, seed: int):
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


def train_evaluate(arch_name, train_data, test_data, device, seed, epochs=50, lr=0.01):
    task = train_data[0]['task']
    is_goal = 'goal' in train_data[0]
    is_binary = 'binary' in train_data[0]

    model = create_model(arch_name, device, seed)

    train_inputs = [d['input'][:8] + [0]*(8-len(d['input'][:8])) for d in train_data]
    train_x = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    train_x = train_x.unsqueeze(1).expand(-1, 8, -1)

    if is_goal:
        train_y = torch.tensor([d['target'] for d in train_data], dtype=torch.float32, device=device)
    else:
        train_y = torch.tensor([[d['target']] for d in train_data], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        if 'Evolved' in arch_name:
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

    model.eval()
    test_inputs = [d['input'][:8] + [0]*(8-len(d['input'][:8])) for d in test_data]
    test_x = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_x = test_x.unsqueeze(1).expand(-1, 8, -1)

    correct = 0
    with torch.no_grad():
        if 'Evolved' in arch_name:
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

    return correct / len(test_data)


def run_comprehensive_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK: ATR vs ISSM vs EvolvedV1 vs EvolvedV2")
    print(f"{'='*80}")

    # Verify parameters
    print("\nParameter counts (~48K target):")
    for name in ['ATR', 'ISSM', 'EvolvedV1', 'EvolvedV2']:
        m = create_model(name, 'cpu', 42)
        print(f"  {name}: {sum(p.numel() for p in m.parameters()):,}")

    # All tasks
    tasks = {
        # Reasoning
        'sequence': generate_sequence_data,
        'pattern': generate_pattern_data,
        'conditional': generate_conditional_data,
        'analogy': generate_analogy_data,
        # Physics
        'projectile': generate_projectile_data,
        'collision': generate_collision_data,
        'goal': generate_goal_data,
        'momentum': generate_momentum_data,
        # Memory
        'delay': generate_delay_data,
        'copy': generate_copy_data,
        # Additional
        'frequency': generate_frequency_data,
        'trend': generate_trend_data,
        'xor': generate_xor_data,
        'count': generate_count_data,
    }

    categories = {
        'reasoning': ['sequence', 'pattern', 'conditional', 'analogy'],
        'physics': ['projectile', 'collision', 'goal', 'momentum'],
        'memory': ['delay', 'copy'],
        'additional': ['frequency', 'trend', 'xor', 'count'],
    }

    architectures = ['ATR', 'ISSM', 'EvolvedV1', 'EvolvedV2']
    results = {arch: {} for arch in architectures}

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")

    for task_name, generator in tasks.items():
        print(f"\n--- {task_name.upper()} ---")
        train_data = generator(100, seed=42)
        test_data = generator(50, seed=43)

        for arch in architectures:
            accuracy = train_evaluate(arch, train_data, test_data, device, seed=42)
            results[arch][task_name] = accuracy
            print(f"  {arch:10s}: {accuracy*100:5.1f}%")

    # Compute category averages
    print(f"\n{'='*80}")
    print("CATEGORY SUMMARIES")
    print(f"{'='*80}")

    for arch in architectures:
        for cat, task_list in categories.items():
            avg = np.mean([results[arch][t] for t in task_list])
            results[arch][f'{cat}_avg'] = avg
        results[arch]['overall_avg'] = np.mean([results[arch][t] for t in tasks.keys()])

    for arch in architectures:
        print(f"\n{arch}:")
        for cat in categories.keys():
            print(f"  {cat.capitalize():12s}: {results[arch][f'{cat}_avg']*100:5.1f}%")
        print(f"  {'OVERALL':12s}: {results[arch]['overall_avg']*100:5.1f}%")

    # Comparison table
    print(f"\n{'='*80}")
    print("FULL COMPARISON TABLE")
    print(f"{'='*80}")

    header = f"{'Task':<15} | {'ATR':>7} | {'ISSM':>7} | {'V1':>7} | {'V2':>7} | {'Winner':<10}"
    print(header)
    print("-" * len(header))

    for task_name in tasks.keys():
        atr = results['ATR'][task_name] * 100
        issm = results['ISSM'][task_name] * 100
        v1 = results['EvolvedV1'][task_name] * 100
        v2 = results['EvolvedV2'][task_name] * 100

        scores = {'ATR': atr, 'ISSM': issm, 'V1': v1, 'V2': v2}
        winner = max(scores, key=scores.get)
        if max(scores.values()) - min(scores.values()) < 3:
            winner = "Tie"

        print(f"{task_name:<15} | {atr:>6.1f}% | {issm:>6.1f}% | {v1:>6.1f}% | {v2:>6.1f}% | {winner:<10}")

    print("-" * len(header))

    for cat in categories.keys():
        atr = results['ATR'][f'{cat}_avg'] * 100
        issm = results['ISSM'][f'{cat}_avg'] * 100
        v1 = results['EvolvedV1'][f'{cat}_avg'] * 100
        v2 = results['EvolvedV2'][f'{cat}_avg'] * 100
        scores = {'ATR': atr, 'ISSM': issm, 'V1': v1, 'V2': v2}
        winner = max(scores, key=scores.get)
        print(f"{cat.upper()+' AVG':<15} | {atr:>6.1f}% | {issm:>6.1f}% | {v1:>6.1f}% | {v2:>6.1f}% | {winner:<10}")

    print("-" * len(header))
    atr = results['ATR']['overall_avg'] * 100
    issm = results['ISSM']['overall_avg'] * 100
    v1 = results['EvolvedV1']['overall_avg'] * 100
    v2 = results['EvolvedV2']['overall_avg'] * 100
    scores = {'ATR': atr, 'ISSM': issm, 'V1': v1, 'V2': v2}
    winner = max(scores, key=scores.get)
    print(f"{'OVERALL':<15} | {atr:>6.1f}% | {issm:>6.1f}% | {v1:>6.1f}% | {v2:>6.1f}% | {winner:<10}")

    # V1 vs V2 specific comparison
    print(f"\n{'='*80}")
    print("V1 vs V2 DELTA (V2 - V1)")
    print(f"{'='*80}")

    for task_name in tasks.keys():
        v1 = results['EvolvedV1'][task_name] * 100
        v2 = results['EvolvedV2'][task_name] * 100
        delta = v2 - v1
        marker = "+" if delta > 0 else ""
        print(f"  {task_name:<15}: {marker}{delta:5.1f}pp")

    return results


if __name__ == "__main__":
    results = run_comprehensive_benchmark()

    with open('EVOLVED_V2_COMPREHENSIVE.json', 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print("\nResults saved to EVOLVED_V2_COMPREHENSIVE.json")
