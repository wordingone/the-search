"""
Benchmark: Perfected ANIMA vs Vanilla Transformer
=================================================

Tests the theoretically-derived ANIMA against baseline transformer
on all 8 benchmark tasks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Import perfected ANIMA
from anima.core.anima_perfected import ANIMA, ANIMAConfig


class VanillaTransformer(nn.Module):
    """Minimal transformer for fair comparison."""

    def __init__(self, input_dim=8, hidden_dim=32, output_dim=4, num_heads=2, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [batch, seq, input_dim]
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# BENCHMARK TASKS
# ============================================

class ReasoningBenchmark:
    """Reasoning tasks: sequence, pattern, conditional, analogy."""

    @staticmethod
    def generate_sequence(n_samples=100, seq_len=10):
        """Predict next in arithmetic sequence."""
        data = []
        for _ in range(n_samples):
            start = random.randint(1, 10)
            step = random.randint(1, 5)
            seq = [start + i * step for i in range(seq_len + 1)]
            # Normalize
            seq_norm = [x / 100.0 for x in seq]
            data.append({
                'input': seq_norm[:-1],
                'target': seq_norm[-1],
                'raw_target': seq[-1]
            })
        return data

    @staticmethod
    def generate_pattern(n_samples=100, seq_len=12):
        """Recognize repeating pattern."""
        data = []
        for _ in range(n_samples):
            pattern_len = random.randint(2, 4)
            pattern = [random.randint(1, 9) for _ in range(pattern_len)]
            seq = (pattern * ((seq_len // pattern_len) + 2))[:seq_len + 1]
            seq_norm = [x / 10.0 for x in seq]
            data.append({
                'input': seq_norm[:-1],
                'target': seq_norm[-1],
                'raw_target': seq[-1]
            })
        return data

    @staticmethod
    def generate_conditional(n_samples=100, seq_len=8):
        """If-then reasoning: if X > 5, output X*2, else X+1."""
        data = []
        for _ in range(n_samples):
            seq = [random.randint(1, 10) for _ in range(seq_len)]
            last = seq[-1]
            target = last * 2 if last > 5 else last + 1
            seq_norm = [x / 20.0 for x in seq]
            data.append({
                'input': seq_norm,
                'target': target / 20.0,
                'raw_target': target
            })
        return data

    @staticmethod
    def generate_analogy(n_samples=100):
        """A:B :: C:? pattern completion."""
        data = []
        for _ in range(n_samples):
            a = random.randint(1, 5)
            b = a * 2  # Relationship: double
            c = random.randint(1, 5)
            d = c * 2  # Same relationship
            seq = [a / 10.0, b / 10.0, c / 10.0, 0.0]  # 0 is placeholder
            data.append({
                'input': seq,
                'target': d / 10.0,
                'raw_target': d
            })
        return data


class PhysicsBenchmark:
    """Physics tasks: projectile, collision, goal, momentum."""

    @staticmethod
    def generate_projectile(n_samples=100, seq_len=10):
        """Predict projectile landing position."""
        data = []
        for _ in range(n_samples):
            v0 = random.uniform(5, 15)
            angle = random.uniform(30, 60)
            g = 9.8
            # Simplified: landing position
            rad = angle * 3.14159 / 180
            landing = (v0 ** 2) * (2 * rad) / g  # Simplified physics
            # Generate trajectory snippet
            t_values = [i * 0.1 for i in range(seq_len)]
            positions = [v0 * t * 0.1 for t in range(seq_len)]  # Simplified x positions
            pos_norm = [p / 20.0 for p in positions]
            data.append({
                'input': pos_norm,
                'target': min(landing / 20.0, 1.0),
                'raw_target': landing
            })
        return data

    @staticmethod
    def generate_collision(n_samples=100, seq_len=8):
        """Predict if two objects will collide."""
        data = []
        for _ in range(n_samples):
            # Object 1: position and velocity
            x1, v1 = random.uniform(0, 5), random.uniform(0.5, 2)
            # Object 2: position and velocity
            x2, v2 = random.uniform(5, 10), random.uniform(-2, -0.5)
            # Will they collide? (simplified: if paths cross)
            will_collide = 1.0 if (x2 - x1) / (v1 - v2) > 0 else 0.0
            seq = [x1/10, v1/2, x2/10, v2/2] + [0.0] * (seq_len - 4)
            data.append({
                'input': seq[:seq_len],
                'target': will_collide,
                'raw_target': int(will_collide)
            })
        return data

    @staticmethod
    def generate_goal(n_samples=100, seq_len=10):
        """Navigate toward goal position."""
        data = []
        for _ in range(n_samples):
            # Current position
            pos = [random.uniform(-5, 5), random.uniform(-5, 5)]
            # Goal position
            goal = [random.uniform(-5, 5), random.uniform(-5, 5)]
            # Optimal action: direction to goal
            dx = goal[0] - pos[0]
            dy = goal[1] - pos[1]
            dist = max((dx**2 + dy**2)**0.5, 0.01)
            action = [dx/dist, dy/dist]  # Unit vector to goal
            seq = [pos[0]/10, pos[1]/10, goal[0]/10, goal[1]/10] + [0.0] * (seq_len - 4)
            data.append({
                'input': seq[:seq_len],
                'target': action,
                'raw_target': action
            })
        return data

    @staticmethod
    def generate_momentum(n_samples=100, seq_len=8):
        """Predict momentum after collision."""
        data = []
        for _ in range(n_samples):
            m1, v1 = random.uniform(1, 5), random.uniform(1, 5)
            m2, v2 = random.uniform(1, 5), random.uniform(-5, -1)
            # Conservation of momentum (inelastic)
            v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
            seq = [m1/5, v1/5, m2/5, v2/5] + [0.0] * (seq_len - 4)
            data.append({
                'input': seq[:seq_len],
                'target': (v_final + 5) / 10,  # Normalize to [0, 1]
                'raw_target': v_final
            })
        return data


def train_and_evaluate(model, train_data, test_data, task_name, epochs=100, lr=0.01):
    """Train model and evaluate on test data."""
    device = next(model.parameters()).device

    # Prepare data
    is_goal_task = task_name == 'goal'

    train_inputs = torch.tensor([[d['input']] for d in train_data], dtype=torch.float32, device=device)
    if is_goal_task:
        train_targets = torch.tensor([d['target'] for d in train_data], dtype=torch.float32, device=device)
    else:
        train_targets = torch.tensor([d['target'] for d in train_data], dtype=torch.float32, device=device).unsqueeze(-1)

    test_inputs = torch.tensor([[d['input']] for d in test_data], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        if isinstance(model, ANIMA):
            # ANIMA processes sequences
            outputs = []
            for i in range(train_inputs.shape[0]):
                model.reset(1, device)
                for t in range(train_inputs.shape[2]):
                    obs = train_inputs[i, 0, t:t+1].unsqueeze(0).expand(-1, 8)
                    if obs.shape[-1] < 8:
                        obs = torch.cat([obs, torch.zeros(1, 8 - obs.shape[-1], device=device)], dim=-1)
                    result = model.step(obs)
                if is_goal_task:
                    outputs.append(result['action'][:, :2])  # 2D direction
                else:
                    outputs.append(result['action'][:, :1])
            outputs = torch.cat(outputs, dim=0)
        else:
            # Transformer
            # Expand input to 8 dims
            expanded = train_inputs.expand(-1, -1, 8) if train_inputs.shape[-1] < 8 else train_inputs[:, :, :8]
            out = model(expanded)
            if is_goal_task:
                outputs = out[:, -1, :2]
            else:
                outputs = out[:, -1, :1]

        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = len(test_data)

    with torch.no_grad():
        for i, d in enumerate(test_data):
            inp = torch.tensor([d['input']], dtype=torch.float32, device=device).unsqueeze(0)

            if isinstance(model, ANIMA):
                model.reset(1, device)
                for t in range(inp.shape[2]):
                    obs = inp[0, 0, t:t+1].unsqueeze(0).expand(-1, 8)
                    if obs.shape[-1] < 8:
                        obs = torch.cat([obs, torch.zeros(1, 8 - obs.shape[-1], device=device)], dim=-1)
                    result = model.step(obs)
                if is_goal_task:
                    pred = result['action'][0, :2].cpu().numpy()
                else:
                    pred = result['action'][0, 0].item()
            else:
                expanded = inp.expand(-1, -1, 8) if inp.shape[-1] < 8 else inp[:, :, :8]
                out = model(expanded)
                if is_goal_task:
                    pred = out[0, -1, :2].cpu().numpy()
                else:
                    pred = out[0, -1, 0].item()

            # Check correctness
            if is_goal_task:
                target = d['target']
                # Direction accuracy: dot product > 0.8
                dot = pred[0] * target[0] + pred[1] * target[1]
                pred_norm = max((pred[0]**2 + pred[1]**2)**0.5, 0.01)
                target_norm = max((target[0]**2 + target[1]**2)**0.5, 0.01)
                cos_sim = dot / (pred_norm * target_norm)
                if cos_sim > 0.7:
                    correct += 1
            elif task_name == 'collision':
                # Binary classification
                if (pred > 0.5) == (d['target'] > 0.5):
                    correct += 1
            else:
                # Regression: within 20% tolerance
                target = d['target']
                if abs(pred - target) < max(0.2 * abs(target), 0.1):
                    correct += 1

    return correct / total


def run_benchmark():
    """Run full benchmark comparison."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create models
    anima_config = ANIMAConfig()
    anima = ANIMA(anima_config).to(device)
    transformer = VanillaTransformer().to(device)

    print(f"\nANIMA parameters: {anima.count_parameters()}")
    print(f"Transformer parameters: {transformer.count_parameters()}")

    # Verify ANIMA constraints
    constraints = anima.verify_type_constraints()
    print(f"\nANIMA Type Constraints: {constraints}")

    results = {
        'ANIMA-Perfected': {
            'params': anima.count_parameters(),
            'reasoning': {},
            'physics': {}
        },
        'VanillaTransformer': {
            'params': transformer.count_parameters(),
            'reasoning': {},
            'physics': {}
        }
    }

    # Reasoning tasks
    reasoning_tasks = [
        ('sequence', ReasoningBenchmark.generate_sequence),
        ('pattern', ReasoningBenchmark.generate_pattern),
        ('conditional', ReasoningBenchmark.generate_conditional),
        ('analogy', ReasoningBenchmark.generate_analogy),
    ]

    print("\n" + "="*60)
    print("REASONING TASKS")
    print("="*60)

    for task_name, generator in reasoning_tasks:
        print(f"\n--- {task_name.upper()} ---")

        # Fresh models for each task
        anima = ANIMA(anima_config).to(device)
        transformer = VanillaTransformer().to(device)

        train_data = generator(n_samples=200)
        test_data = generator(n_samples=100)

        anima_score = train_and_evaluate(anima, train_data, test_data, task_name)
        transformer_score = train_and_evaluate(transformer, train_data, test_data, task_name)

        results['ANIMA-Perfected']['reasoning'][task_name] = {
            'score': anima_score,
            'correct': int(anima_score * 100),
            'total': 100,
            'percent': f"{anima_score * 100:.2f}%"
        }
        results['VanillaTransformer']['reasoning'][task_name] = {
            'score': transformer_score,
            'correct': int(transformer_score * 100),
            'total': 100,
            'percent': f"{transformer_score * 100:.2f}%"
        }

        print(f"ANIMA: {anima_score * 100:.1f}%  |  Transformer: {transformer_score * 100:.1f}%")

    # Physics tasks
    physics_tasks = [
        ('projectile', PhysicsBenchmark.generate_projectile),
        ('collision', PhysicsBenchmark.generate_collision),
        ('goal', PhysicsBenchmark.generate_goal),
        ('momentum', PhysicsBenchmark.generate_momentum),
    ]

    print("\n" + "="*60)
    print("PHYSICS TASKS")
    print("="*60)

    for task_name, generator in physics_tasks:
        print(f"\n--- {task_name.upper()} ---")

        # Fresh models for each task
        anima = ANIMA(anima_config).to(device)
        transformer = VanillaTransformer().to(device)

        train_data = generator(n_samples=200)
        test_data = generator(n_samples=100)

        anima_score = train_and_evaluate(anima, train_data, test_data, task_name)
        transformer_score = train_and_evaluate(transformer, train_data, test_data, task_name)

        results['ANIMA-Perfected']['physics'][task_name] = {
            'score': anima_score,
            'correct': int(anima_score * 100),
            'total': 100,
            'percent': f"{anima_score * 100:.2f}%"
        }
        results['VanillaTransformer']['physics'][task_name] = {
            'score': transformer_score,
            'correct': int(transformer_score * 100),
            'total': 100,
            'percent': f"{transformer_score * 100:.2f}%"
        }

        print(f"ANIMA: {anima_score * 100:.1f}%  |  Transformer: {transformer_score * 100:.1f}%")

    # Calculate averages
    for model_name in results:
        reasoning_scores = [v['score'] for v in results[model_name]['reasoning'].values()]
        physics_scores = [v['score'] for v in results[model_name]['physics'].values()]

        results[model_name]['reasoning_avg'] = sum(reasoning_scores) / len(reasoning_scores)
        results[model_name]['physics_avg'] = sum(physics_scores) / len(physics_scores)
        results[model_name]['overall'] = (results[model_name]['reasoning_avg'] + results[model_name]['physics_avg']) / 2

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Model':<25} {'Params':<10} {'Reasoning':<12} {'Physics':<12} {'Overall':<12}")
    print("-" * 71)

    for model_name in results:
        r = results[model_name]
        print(f"{model_name:<25} {r['params']:<10} {r['reasoning_avg']*100:>10.2f}% {r['physics_avg']*100:>10.2f}% {r['overall']*100:>10.2f}%")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'perfected_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_benchmark()
