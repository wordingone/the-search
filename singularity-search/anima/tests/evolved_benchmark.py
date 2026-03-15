"""
AnimaEvolved vs Parents Benchmark
=================================

Comprehensive comparison of AnimaEvolved against its parent architectures
(AnimaATR and AnimaISSM) across all benchmark domains.

This validates that AnimaEvolved preserves the strengths of both parents
while potentially improving on their weaknesses.
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
from dataclasses import dataclass
import time

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from anima.core import AnimaATR, AnimaISSM, AnimaEvolved


# =============================================================================
# DATA GENERATORS (Standard ANIMA Benchmark Tasks)
# =============================================================================

def generate_sequence_data(n=100, seq_len=8):
    """Arithmetic sequence prediction - tests temporal pattern learning."""
    data = []
    for _ in range(n):
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        seq = [(start + i * step) / 100.0 for i in range(seq_len + 1)]
        data.append({'input': seq[:-1], 'target': seq[-1], 'task': 'sequence'})
    return data


def generate_pattern_data(n=100, seq_len=8):
    """Repeating pattern recognition - tests memory."""
    data = []
    for _ in range(n):
        plen = random.randint(2, 4)
        pattern = [random.randint(1, 9) / 10.0 for _ in range(plen)]
        seq = (pattern * 5)[:seq_len + 1]
        data.append({'input': seq[:-1], 'target': seq[-1], 'task': 'pattern'})
    return data


def generate_conditional_data(n=100, seq_len=8):
    """Conditional logic (if-then-else) - tests reasoning."""
    data = []
    for _ in range(n):
        seq = [random.randint(1, 10) / 20.0 for _ in range(seq_len)]
        last = seq[-1] * 20
        target = (last * 2 if last > 5 else last + 1) / 20.0
        data.append({'input': seq, 'target': target, 'task': 'conditional'})
    return data


def generate_analogy_data(n=100):
    """Analogy completion (A:B::C:?) - tests abstract reasoning."""
    data = []
    for _ in range(n):
        a = random.randint(1, 5) / 10.0
        b = a * 2  # Doubling relationship
        c = random.randint(1, 5) / 10.0
        d = c * 2
        data.append({'input': [a, b, c, 0, 0, 0, 0, 0], 'target': d, 'task': 'analogy'})
    return data


def generate_projectile_data(n=100, seq_len=8):
    """Projectile trajectory - tests physics modeling."""
    data = []
    for _ in range(n):
        v0 = random.uniform(5, 15)
        positions = [v0 * t * 0.1 / 20.0 for t in range(seq_len)]
        landing = v0 * 0.8 / 20.0
        data.append({'input': positions, 'target': min(landing, 1.0), 'task': 'projectile'})
    return data


def generate_collision_data(n=100, seq_len=8):
    """Collision prediction (binary) - tests physics reasoning."""
    data = []
    for _ in range(n):
        x1, v1 = random.uniform(0, 0.5), random.uniform(0.1, 0.3)
        x2, v2 = random.uniform(0.5, 1), random.uniform(-0.3, -0.1)
        will_collide = 1.0 if v1 > -v2 else 0.0
        seq = [x1, v1, x2, v2] + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': will_collide, 'task': 'collision', 'binary': True})
    return data


def generate_goal_data(n=100, seq_len=8):
    """Goal-directed navigation - tests action planning."""
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
    """Momentum conservation - tests physics invariants."""
    data = []
    for _ in range(n):
        m1, v1 = random.uniform(0.2, 1), random.uniform(0.2, 1)
        m2, v2 = random.uniform(0.2, 1), random.uniform(-1, -0.2)
        v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
        seq = [m1, v1, m2, v2] + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': (v_final + 1) / 2, 'task': 'momentum'})
    return data


def generate_delay_data(n=100, seq_len=16):
    """Delayed recall - tests long-term memory."""
    data = []
    for _ in range(n):
        signal = random.uniform(0.2, 0.8)
        delay = random.randint(4, 10)
        seq = [signal] + [0.0] * delay + [0.5] + [0.0] * (seq_len - delay - 2)
        data.append({'input': seq[:seq_len], 'target': signal, 'task': 'delay'})
    return data


def generate_copy_data(n=100, seq_len=8):
    """Copy task - tests memory fidelity."""
    data = []
    for _ in range(n):
        seq = [random.uniform(0.1, 0.9) for _ in range(seq_len)]
        # Target is the first element after seeing whole sequence
        data.append({'input': seq, 'target': seq[0], 'task': 'copy'})
    return data


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class ModelWrapper:
    """Unified interface for all ANIMA variants."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.device = next(model.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence and return last step output."""
        batch = x.shape[0]

        # All models now use forward() which returns [batch, seq_len, output_dim]
        # We take the last timestep
        if self.name == 'AnimaEvolved':
            # AnimaEvolved returns dict with 'output' key
            result = self.model(x)
            return result['output'][:, -1]  # Last timestep
        else:
            # ATR and ISSM return tensor directly
            output = self.model(x)
            return output[:, -1]  # Last timestep

    def get_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_evaluate(
    model: ModelWrapper,
    train_data: List[Dict],
    test_data: List[Dict],
    epochs: int = 50,
    lr: float = 0.01,
) -> Tuple[float, float]:
    """
    Train and evaluate model on a task.

    Returns:
        accuracy: Test accuracy (0-1)
        train_time: Training time in seconds
    """
    device = model.device
    task = train_data[0]['task']
    is_goal = task == 'goal'
    is_binary = task == 'collision'

    # Prepare training data - pad each input to 8 features
    train_inputs = []
    for d in train_data:
        inp = d['input']
        if len(inp) < 8:
            inp = inp + [0.0] * (8 - len(inp))
        train_inputs.append(inp[:8])

    train_x = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    # train_x is [batch, 8], need [batch, seq_len, 8]
    # Use input as single timestep repeated, or as sequence
    train_x = train_x.unsqueeze(1).expand(-1, 8, -1)  # [batch, 8, 8]

    if is_goal:
        train_y = torch.tensor(
            [d['target'] for d in train_data],
            dtype=torch.float32,
            device=device
        )
    else:
        train_y = torch.tensor(
            [[d['target']] for d in train_data],
            dtype=torch.float32,
            device=device
        )

    optimizer = optim.Adam(model.model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training
    model.model.train()
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model.forward(train_x)

        if is_goal:
            outputs = outputs[:, :2]
        else:
            outputs = outputs[:, :1]

        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time

    # Evaluation
    model.model.eval()
    correct = 0
    total = len(test_data)

    # Prepare test data
    test_inputs = []
    for d in test_data:
        inp = d['input']
        if len(inp) < 8:
            inp = inp + [0.0] * (8 - len(inp))
        test_inputs.append(inp[:8])
    test_x = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_x = test_x.unsqueeze(1).expand(-1, 8, -1)  # [batch, 8, 8]

    with torch.no_grad():
        outputs = model.forward(test_x)

        for i, d in enumerate(test_data):
            if is_goal:
                pred = outputs[i, :2].cpu().numpy()
                target = d['target']
                # Direction accuracy (cosine similarity > 0.7)
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
                # Relative tolerance (20%) with minimum absolute (0.1)
                tol = max(abs(target) * 0.2, 0.1)
                if abs(pred - target) <= tol:
                    correct += 1

    accuracy = correct / total
    return accuracy, train_time


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    sensory_dim: int = 8
    d_model: int = 32
    bottleneck_dim: int = 16
    output_dim: int = 4
    d_state: int = 16
    train_samples: int = 100
    test_samples: int = 50
    epochs: int = 50
    lr: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_benchmark(config: BenchmarkConfig = None) -> Dict:
    """
    Run full benchmark comparison.

    Returns dictionary with results for each model and task.
    """
    if config is None:
        config = BenchmarkConfig()

    device = torch.device(config.device)
    print(f"\nRunning benchmark on {device}")
    print("=" * 70)

    # Create models with matched parameters
    models = {
        'AnimaATR': AnimaATR(
            sensory_dim=config.sensory_dim,
            d_model=config.d_model,
            bottleneck_dim=config.bottleneck_dim,
            output_dim=config.output_dim,
        ).to(device),
        'AnimaISSM': AnimaISSM(
            sensory_dim=config.sensory_dim,
            d_model=config.d_model,
            bottleneck_dim=config.bottleneck_dim,
            output_dim=config.output_dim,
            d_state=config.d_state,
        ).to(device),
        'AnimaEvolved': AnimaEvolved(
            sensory_dim=config.sensory_dim,
            d_model=config.d_model,
            bottleneck_dim=config.bottleneck_dim,
            output_dim=config.output_dim,
            d_state=config.d_state,
        ).to(device),
    }

    # Print parameter counts
    print("\nModel Parameters:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,}")

    # Define tasks
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

    # Task categories
    reasoning_tasks = ['sequence', 'pattern', 'conditional', 'analogy']
    physics_tasks = ['projectile', 'collision', 'goal', 'momentum']
    memory_tasks = ['delay', 'copy']

    results = {name: {} for name in models.keys()}

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    for task_name, generator in tasks.items():
        print(f"\n--- {task_name.upper()} ---")

        # Generate data
        train_data = generator(config.train_samples)
        test_data = generator(config.test_samples)

        for model_name, model in models.items():
            # Reset seeds for fair comparison
            torch.manual_seed(42)

            # Re-initialize model weights
            model._init_weights() if hasattr(model, '_init_weights') else None

            wrapper = ModelWrapper(model, model_name)

            accuracy, train_time = train_evaluate(
                wrapper,
                train_data,
                test_data,
                epochs=config.epochs,
                lr=config.lr,
            )

            results[model_name][task_name] = {
                'accuracy': accuracy,
                'train_time': train_time,
            }

            print(f"  {model_name}: {accuracy*100:.1f}% ({train_time:.2f}s)")

    # Compute averages
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name in models.keys():
        reasoning_avg = np.mean([
            results[model_name][t]['accuracy'] for t in reasoning_tasks
        ])
        physics_avg = np.mean([
            results[model_name][t]['accuracy'] for t in physics_tasks
        ])
        memory_avg = np.mean([
            results[model_name][t]['accuracy'] for t in memory_tasks
        ])
        overall_avg = np.mean([
            results[model_name][t]['accuracy'] for t in tasks.keys()
        ])

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
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    header = f"{'Task':<15} | {'ATR':>8} | {'ISSM':>8} | {'Evolved':>8} | {'Winner':<12}"
    print(header)
    print("-" * len(header))

    for task_name in tasks.keys():
        atr_acc = results['AnimaATR'][task_name]['accuracy'] * 100
        issm_acc = results['AnimaISSM'][task_name]['accuracy'] * 100
        evolved_acc = results['AnimaEvolved'][task_name]['accuracy'] * 100

        scores = {'ATR': atr_acc, 'ISSM': issm_acc, 'Evolved': evolved_acc}
        winner = max(scores, key=scores.get)
        if max(scores.values()) - min(scores.values()) < 2:
            winner = "Tie"

        print(f"{task_name:<15} | {atr_acc:>7.1f}% | {issm_acc:>7.1f}% | {evolved_acc:>7.1f}% | {winner:<12}")

    print("-" * len(header))

    # Category averages
    for cat, cat_tasks in [('Reasoning', reasoning_tasks), ('Physics', physics_tasks), ('Memory', memory_tasks)]:
        atr_avg = np.mean([results['AnimaATR'][t]['accuracy'] for t in cat_tasks]) * 100
        issm_avg = np.mean([results['AnimaISSM'][t]['accuracy'] for t in cat_tasks]) * 100
        evolved_avg = np.mean([results['AnimaEvolved'][t]['accuracy'] for t in cat_tasks]) * 100

        scores = {'ATR': atr_avg, 'ISSM': issm_avg, 'Evolved': evolved_avg}
        winner = max(scores, key=scores.get)

        print(f"{cat + ' Avg':<15} | {atr_avg:>7.1f}% | {issm_avg:>7.1f}% | {evolved_avg:>7.1f}% | {winner:<12}")

    print("-" * len(header))

    atr_overall = results['AnimaATR']['overall_avg'] * 100
    issm_overall = results['AnimaISSM']['overall_avg'] * 100
    evolved_overall = results['AnimaEvolved']['overall_avg'] * 100

    scores = {'ATR': atr_overall, 'ISSM': issm_overall, 'Evolved': evolved_overall}
    winner = max(scores, key=scores.get)

    print(f"{'OVERALL':<15} | {atr_overall:>7.1f}% | {issm_overall:>7.1f}% | {evolved_overall:>7.1f}% | {winner:<12}")

    return results


if __name__ == "__main__":
    results = run_benchmark()

    # Save results
    import json

    # Convert to serializable format
    serializable = {}
    for model_name, model_results in results.items():
        serializable[model_name] = {}
        for key, value in model_results.items():
            if isinstance(value, dict):
                serializable[model_name][key] = {
                    'accuracy': float(value['accuracy']),
                    'train_time': float(value['train_time']),
                }
            else:
                serializable[model_name][key] = float(value)

    with open('EVOLVED_BENCHMARK_RESULTS.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print("\n\nResults saved to EVOLVED_BENCHMARK_RESULTS.json")
