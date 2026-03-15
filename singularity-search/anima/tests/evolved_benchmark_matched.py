"""
Parameter-Matched Benchmark: AnimaEvolved vs Parents
=====================================================

Fair comparison with ~48K parameters for all architectures.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Dict, List, Tuple
import json

# Set seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from anima.core import AnimaATR, AnimaISSM, AnimaEvolved


# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_sequence_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        seq = [(start + i * step) / 100.0 for i in range(seq_len + 1)]
        data.append({'input': seq[:-1], 'target': seq[-1], 'task': 'sequence'})
    return data

def generate_pattern_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        plen = random.randint(2, 4)
        pattern = [random.randint(1, 9) / 10.0 for _ in range(plen)]
        seq = (pattern * 5)[:seq_len + 1]
        data.append({'input': seq[:-1], 'target': seq[-1], 'task': 'pattern'})
    return data

def generate_conditional_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        seq = [random.randint(1, 10) / 20.0 for _ in range(seq_len)]
        last = seq[-1] * 20
        target = (last * 2 if last > 5 else last + 1) / 20.0
        data.append({'input': seq, 'target': target, 'task': 'conditional'})
    return data

def generate_analogy_data(n=100):
    data = []
    for _ in range(n):
        a = random.randint(1, 5) / 10.0
        b = a * 2
        c = random.randint(1, 5) / 10.0
        d = c * 2
        data.append({'input': [a, b, c, 0, 0, 0, 0, 0], 'target': d, 'task': 'analogy'})
    return data

def generate_projectile_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        v0 = random.uniform(5, 15)
        positions = [v0 * t * 0.1 / 20.0 for t in range(seq_len)]
        landing = v0 * 0.8 / 20.0
        data.append({'input': positions, 'target': min(landing, 1.0), 'task': 'projectile'})
    return data

def generate_collision_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        x1, v1 = random.uniform(0, 0.5), random.uniform(0.1, 0.3)
        x2, v2 = random.uniform(0.5, 1), random.uniform(-0.3, -0.1)
        will_collide = 1.0 if v1 > -v2 else 0.0
        seq = [x1, v1, x2, v2] + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': will_collide, 'task': 'collision', 'binary': True})
    return data

def generate_goal_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        pos = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        goal = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        dist = max((dx**2 + dy**2)**0.5, 0.01)
        action = [dx/dist, dy/dist]
        seq = pos + goal + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': action, 'task': 'goal', 'goal': True})
    return data

def generate_momentum_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        m1, v1 = random.uniform(0.2, 1), random.uniform(0.2, 1)
        m2, v2 = random.uniform(0.2, 1), random.uniform(-1, -0.2)
        v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
        seq = [m1, v1, m2, v2] + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': (v_final + 1) / 2, 'task': 'momentum'})
    return data

def generate_delay_data(n=100, seq_len=16):
    data = []
    for _ in range(n):
        signal = random.uniform(0.2, 0.8)
        delay = random.randint(4, 10)
        seq = [signal] + [0.0] * delay + [0.5] + [0.0] * (seq_len - delay - 2)
        data.append({'input': seq[:seq_len], 'target': signal, 'task': 'delay'})
    return data

def generate_copy_data(n=100, seq_len=8):
    data = []
    for _ in range(n):
        seq = [random.uniform(0.1, 0.9) for _ in range(seq_len)]
        data.append({'input': seq, 'target': seq[0], 'task': 'copy'})
    return data


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_evaluate(model, model_name, train_data, test_data, epochs=50, lr=0.01, device='cuda'):
    task = train_data[0]['task']
    is_goal = task == 'goal'
    is_binary = task == 'collision'

    # Prepare data
    train_inputs = []
    for d in train_data:
        inp = d['input']
        if len(inp) < 8:
            inp = inp + [0.0] * (8 - len(inp))
        train_inputs.append(inp[:8])

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

        if model_name == 'Evolved':
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
    correct = 0
    total = len(test_data)

    test_inputs = []
    for d in test_data:
        inp = d['input']
        if len(inp) < 8:
            inp = inp + [0.0] * (8 - len(inp))
        test_inputs.append(inp[:8])
    test_x = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_x = test_x.unsqueeze(1).expand(-1, 8, -1)

    with torch.no_grad():
        if model_name == 'Evolved':
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
                cos_sim = dot / (pred_norm * target_norm)
                if cos_sim > 0.7:
                    correct += 1
            elif is_binary:
                pred = outputs[i, 0].item()
                if (pred > 0.5) == (d['target'] > 0.5):
                    correct += 1
            else:
                pred = outputs[i, 0].item()
                target = d['target']
                tol = max(abs(target) * 0.2, 0.1)
                if abs(pred - target) <= tol:
                    correct += 1

    return correct / total


def run_matched_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("PARAMETER-MATCHED BENCHMARK (~48K params)")
    print(f"{'='*70}")

    # Parameter-matched configurations
    configs = {
        'ATR': {'d_model': 48, 'bottleneck_dim': 24, 'd_state': None},
        'ISSM': {'d_model': 24, 'bottleneck_dim': 12, 'd_state': 16},
        'Evolved': {'d_model': 23, 'bottleneck_dim': 11, 'd_state': 16},
    }

    # Create models
    models = {
        'ATR': AnimaATR(sensory_dim=8, d_model=48, bottleneck_dim=24, output_dim=4).to(device),
        'ISSM': AnimaISSM(sensory_dim=8, d_model=24, bottleneck_dim=12, output_dim=4, d_state=16).to(device),
        'Evolved': AnimaEvolved(sensory_dim=8, d_model=23, bottleneck_dim=11, output_dim=4, d_state=16).to(device),
    }

    print("\nModel Parameters:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        cfg = configs[name]
        print(f"  {name}: {params:,} (d_model={cfg['d_model']}, d_state={cfg['d_state']})")

    # Tasks
    tasks = {
        'sequence': generate_sequence_data,
        'pattern': generate_pattern_data,
        'conditional': generate_conditional_data,
        'analogy': generate_analogy_data,
        'projectile': generate_projectile_data,
        'collision': generate_collision_data,
        'goal': generate_goal_data,
        'momentum': generate_momentum_data,
        'delay': generate_delay_data,
        'copy': generate_copy_data,
    }

    reasoning_tasks = ['sequence', 'pattern', 'conditional', 'analogy']
    physics_tasks = ['projectile', 'collision', 'goal', 'momentum']
    memory_tasks = ['delay', 'copy']

    results = {name: {} for name in models.keys()}

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")

    for task_name, generator in tasks.items():
        print(f"\n--- {task_name.upper()} ---")
        train_data = generator(100)
        test_data = generator(50)

        for model_name, model in models.items():
            # Reset seeds and reinitialize
            torch.manual_seed(42)
            model._init_weights()

            accuracy = train_evaluate(model, model_name, train_data, test_data, device=device)
            results[model_name][task_name] = accuracy
            print(f"  {model_name}: {accuracy*100:.1f}%")

    # Compute summaries
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for model_name in models.keys():
        reasoning_avg = np.mean([results[model_name][t] for t in reasoning_tasks])
        physics_avg = np.mean([results[model_name][t] for t in physics_tasks])
        memory_avg = np.mean([results[model_name][t] for t in memory_tasks])
        overall_avg = np.mean([results[model_name][t] for t in tasks.keys()])

        results[model_name]['reasoning_avg'] = reasoning_avg
        results[model_name]['physics_avg'] = physics_avg
        results[model_name]['memory_avg'] = memory_avg
        results[model_name]['overall_avg'] = overall_avg

        print(f"\n{model_name}:")
        print(f"  Reasoning: {reasoning_avg*100:.1f}%")
        print(f"  Physics:   {physics_avg*100:.1f}%")
        print(f"  Memory:    {memory_avg*100:.1f}%")
        print(f"  Overall:   {overall_avg*100:.1f}%")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")

    header = f"{'Task':<15} | {'ATR':>8} | {'ISSM':>8} | {'Evolved':>8} | {'Winner':<12}"
    print(header)
    print("-" * len(header))

    for task_name in tasks.keys():
        atr = results['ATR'][task_name] * 100
        issm = results['ISSM'][task_name] * 100
        evolved = results['Evolved'][task_name] * 100

        scores = {'ATR': atr, 'ISSM': issm, 'Evolved': evolved}
        winner = max(scores, key=scores.get)
        if max(scores.values()) - min(scores.values()) < 2:
            winner = "Tie"

        print(f"{task_name:<15} | {atr:>7.1f}% | {issm:>7.1f}% | {evolved:>7.1f}% | {winner:<12}")

    print("-" * len(header))

    for cat, cat_tasks in [('Reasoning', reasoning_tasks), ('Physics', physics_tasks), ('Memory', memory_tasks)]:
        atr_avg = np.mean([results['ATR'][t] for t in cat_tasks]) * 100
        issm_avg = np.mean([results['ISSM'][t] for t in cat_tasks]) * 100
        evolved_avg = np.mean([results['Evolved'][t] for t in cat_tasks]) * 100

        scores = {'ATR': atr_avg, 'ISSM': issm_avg, 'Evolved': evolved_avg}
        winner = max(scores, key=scores.get)

        print(f"{cat + ' Avg':<15} | {atr_avg:>7.1f}% | {issm_avg:>7.1f}% | {evolved_avg:>7.1f}% | {winner:<12}")

    print("-" * len(header))

    atr_overall = results['ATR']['overall_avg'] * 100
    issm_overall = results['ISSM']['overall_avg'] * 100
    evolved_overall = results['Evolved']['overall_avg'] * 100

    print(f"{'OVERALL':<15} | {atr_overall:>7.1f}% | {issm_overall:>7.1f}% | {evolved_overall:>7.1f}% | {max({'ATR': atr_overall, 'ISSM': issm_overall, 'Evolved': evolved_overall}, key=lambda k: {'ATR': atr_overall, 'ISSM': issm_overall, 'Evolved': evolved_overall}[k]):<12}")

    return results


if __name__ == "__main__":
    results = run_matched_benchmark()

    # Save results
    serializable = {}
    for model_name, model_results in results.items():
        serializable[model_name] = {k: float(v) for k, v in model_results.items()}

    with open('EVOLVED_BENCHMARK_MATCHED.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print("\n\nResults saved to EVOLVED_BENCHMARK_MATCHED.json")
