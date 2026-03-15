"""
Quick Benchmark: Perfected ANIMA vs Vanilla Transformer
=======================================================

Streamlined version for faster results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np

# Set seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from anima.core.anima_perfected import ANIMA, ANIMAConfig


class VanillaTransformer(nn.Module):
    """Minimal transformer."""

    def __init__(self, input_dim=8, hidden_dim=32, output_dim=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=64,
            dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.transformer(h)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def generate_sequence_data(n=50, seq_len=8):
    """Arithmetic sequence prediction."""
    data = []
    for _ in range(n):
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        seq = [(start + i * step) / 100.0 for i in range(seq_len + 1)]
        data.append({'input': seq[:-1], 'target': seq[-1]})
    return data


def generate_pattern_data(n=50, seq_len=8):
    """Pattern recognition."""
    data = []
    for _ in range(n):
        plen = random.randint(2, 4)
        pattern = [random.randint(1, 9) / 10.0 for _ in range(plen)]
        seq = (pattern * 5)[:seq_len + 1]
        data.append({'input': seq[:-1], 'target': seq[-1]})
    return data


def generate_conditional_data(n=50, seq_len=8):
    """If-then reasoning."""
    data = []
    for _ in range(n):
        seq = [random.randint(1, 10) / 20.0 for _ in range(seq_len)]
        last = seq[-1] * 20
        target = (last * 2 if last > 5 else last + 1) / 20.0
        data.append({'input': seq, 'target': target})
    return data


def generate_analogy_data(n=50):
    """A:B::C:? with doubling relationship."""
    data = []
    for _ in range(n):
        a = random.randint(1, 5) / 10.0
        b = a * 2
        c = random.randint(1, 5) / 10.0
        d = c * 2
        data.append({'input': [a, b, c, 0], 'target': d})
    return data


def generate_projectile_data(n=50, seq_len=8):
    """Projectile trajectory."""
    data = []
    for _ in range(n):
        v0 = random.uniform(5, 15)
        positions = [v0 * t * 0.1 / 20.0 for t in range(seq_len)]
        landing = v0 * 0.8 / 20.0
        data.append({'input': positions, 'target': min(landing, 1.0)})
    return data


def generate_collision_data(n=50, seq_len=8):
    """Collision prediction (binary)."""
    data = []
    for _ in range(n):
        x1, v1 = random.uniform(0, 0.5), random.uniform(0.1, 0.3)
        x2, v2 = random.uniform(0.5, 1), random.uniform(-0.3, -0.1)
        will_collide = 1.0 if v1 > -v2 else 0.0
        seq = [x1, v1, x2, v2] + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': will_collide, 'binary': True})
    return data


def generate_goal_data(n=50, seq_len=8):
    """Goal-directed navigation."""
    data = []
    for _ in range(n):
        pos = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        goal = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        dist = max((dx**2 + dy**2)**0.5, 0.01)
        action = [dx/dist, dy/dist]
        seq = pos + goal + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': action, 'goal': True})
    return data


def generate_momentum_data(n=50, seq_len=8):
    """Momentum conservation."""
    data = []
    for _ in range(n):
        m1, v1 = random.uniform(0.2, 1), random.uniform(0.2, 1)
        m2, v2 = random.uniform(0.2, 1), random.uniform(-1, -0.2)
        v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
        seq = [m1, v1, m2, v2] + [0.0] * (seq_len - 4)
        data.append({'input': seq, 'target': (v_final + 1) / 2})
    return data


def train_evaluate(model, train_data, test_data, task_name, epochs=50, lr=0.01):
    """Train and evaluate."""
    device = next(model.parameters()).device
    is_goal = task_name == 'goal'
    is_binary = task_name == 'collision'

    # Prepare data
    train_x = torch.tensor([d['input'] for d in train_data], dtype=torch.float32, device=device)
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

        if isinstance(model, ANIMA):
            outputs = []
            for i in range(len(train_data)):
                model.reset(1, device)
                inp = train_x[i]
                # Pad to 8 dims
                if len(inp) < 8:
                    inp = torch.cat([inp, torch.zeros(8 - len(inp), device=device)])
                inp = inp[:8].unsqueeze(0)
                result = model.step(inp)
                if is_goal:
                    outputs.append(result['action'][:, :2])
                else:
                    outputs.append(result['action'][:, :1])
            outputs = torch.cat(outputs, dim=0)
        else:
            # Transformer
            x = train_x.unsqueeze(1)  # [batch, 1, seq]
            # Pad to 8
            if x.shape[-1] < 8:
                x = torch.cat([x, torch.zeros(x.shape[0], 1, 8 - x.shape[-1], device=device)], dim=-1)
            out = model(x)
            if is_goal:
                outputs = out[:, 0, :2]
            else:
                outputs = out[:, 0, :1]

        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = len(test_data)

    with torch.no_grad():
        for d in test_data:
            inp = torch.tensor(d['input'], dtype=torch.float32, device=device)
            if len(inp) < 8:
                inp = torch.cat([inp, torch.zeros(8 - len(inp), device=device)])
            inp = inp[:8]

            if isinstance(model, ANIMA):
                model.reset(1, device)
                result = model.step(inp.unsqueeze(0))
                if is_goal:
                    pred = result['action'][0, :2].cpu().numpy()
                else:
                    pred = result['action'][0, 0].item()
            else:
                out = model(inp.unsqueeze(0).unsqueeze(0))
                if is_goal:
                    pred = out[0, 0, :2].cpu().numpy()
                else:
                    pred = out[0, 0, 0].item()

            # Check accuracy
            if is_goal:
                target = d['target']
                dot = pred[0]*target[0] + pred[1]*target[1]
                pn = max((pred[0]**2 + pred[1]**2)**0.5, 0.01)
                tn = max((target[0]**2 + target[1]**2)**0.5, 0.01)
                if dot/(pn*tn) > 0.7:
                    correct += 1
            elif is_binary:
                if (pred > 0.5) == (d['target'] > 0.5):
                    correct += 1
            else:
                target = d['target']
                if abs(pred - target) < max(0.2 * abs(target), 0.1):
                    correct += 1

    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    tasks = [
        ('sequence', generate_sequence_data, 'reasoning'),
        ('pattern', generate_pattern_data, 'reasoning'),
        ('conditional', generate_conditional_data, 'reasoning'),
        ('analogy', generate_analogy_data, 'reasoning'),
        ('projectile', generate_projectile_data, 'physics'),
        ('collision', generate_collision_data, 'physics'),
        ('goal', generate_goal_data, 'physics'),
        ('momentum', generate_momentum_data, 'physics'),
    ]

    anima_scores = {'reasoning': [], 'physics': []}
    transformer_scores = {'reasoning': [], 'physics': []}

    print("\n" + "="*60)
    print("PERFECTED ANIMA vs VANILLA TRANSFORMER")
    print("="*60)

    for task_name, generator, category in tasks:
        print(f"\n{task_name.upper()}: ", end="", flush=True)

        # Fresh models
        anima = ANIMA(ANIMAConfig()).to(device)
        transformer = VanillaTransformer().to(device)

        train = generator(n=100)
        test = generator(n=50)

        a_score = train_evaluate(anima, train, test, task_name)
        t_score = train_evaluate(transformer, train, test, task_name)

        anima_scores[category].append(a_score)
        transformer_scores[category].append(t_score)

        print(f"ANIMA {a_score*100:.1f}% | Transformer {t_score*100:.1f}%")

    # Summary
    anima_params = ANIMA(ANIMAConfig()).count_parameters()
    transformer_params = VanillaTransformer().count_parameters()

    a_reasoning = sum(anima_scores['reasoning']) / 4
    a_physics = sum(anima_scores['physics']) / 4
    a_overall = (a_reasoning + a_physics) / 2

    t_reasoning = sum(transformer_scores['reasoning']) / 4
    t_physics = sum(transformer_scores['physics']) / 4
    t_overall = (t_reasoning + t_physics) / 2

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\n{'Model':<22} {'Params':<8} {'Reasoning':<12} {'Physics':<12} {'Overall':<12}")
    print("-"*66)
    print(f"{'ANIMA-Perfected':<22} {anima_params:<8} {a_reasoning*100:>10.2f}% {a_physics*100:>10.2f}% {a_overall*100:>10.2f}%")
    print(f"{'VanillaTransformer':<22} {transformer_params:<8} {t_reasoning*100:>10.2f}% {t_physics*100:>10.2f}% {t_overall*100:>10.2f}%")

    improvement = a_overall - t_overall
    print(f"\nImprovement: +{improvement*100:.2f}pp")

    # Save
    results = {
        'ANIMA-Perfected': {
            'params': anima_params,
            'reasoning_avg': a_reasoning,
            'physics_avg': a_physics,
            'overall': a_overall
        },
        'VanillaTransformer': {
            'params': transformer_params,
            'reasoning_avg': t_reasoning,
            'physics_avg': t_physics,
            'overall': t_overall
        },
        'improvement_pp': improvement * 100
    }

    with open(os.path.join(os.path.dirname(__file__), 'perfected_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    main()
