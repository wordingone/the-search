"""
Generation 2 Anima Benchmark

Validates the three new architectures against the same 14 synthetic tasks
used to derive their design principles.

Target Performance (based on causal analysis):
- AnimaHierarchical: +20pp sequence, +15pp delay, +10pp projectile
- AnimaInduction: +40pp pattern, +30pp analogy, +50pp associative
- AnimaModular: 96% logic (V2-level) AND 84% memory (Anima-level)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Import Generation 2 architectures
from anima.core import AnimaHierarchical, AnimaInduction, AnimaModular

# Import legacy for comparison (from archive)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'archive', 'legacy', 'anima_core_v1'))
try:
    from anima_evolved_v2 import AnimaEvolvedV2
    from anima_evolved_router import AnimaEvolvedRouter
    from anima import Anima
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    print("Warning: Legacy models not available for comparison")


# ============================================================================
# SYNTHETIC TASK GENERATORS (Same as comprehensive benchmark)
# ============================================================================

def generate_sequence_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear sequence extrapolation: predict next number."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        start = torch.randint(0, 50, (1,)).item()
        step = torch.randint(1, 5, (1,)).item()
        for t in range(seq_len):
            val = start + t * step
            X[i, t, 0] = val / 100.0
            if t > 0:
                Y[i, t, 0] = val / 100.0
    return X, Y

def generate_pattern_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pattern completion: ABAB... -> predict next."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        a, b = torch.rand(1).item(), torch.rand(1).item()
        for t in range(seq_len):
            val = a if t % 2 == 0 else b
            X[i, t, 0] = val
            if t > 1:
                Y[i, t, 0] = a if t % 2 == 0 else b
    return X, Y

def generate_conditional_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Threshold: if last > 0.5, output 2x; else output x+0.1."""
    X = torch.rand(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        for t in range(seq_len):
            if t > 0:
                if X[i, t-1, 0] > 0.5:
                    Y[i, t, 0] = 2 * X[i, t, 0]
                else:
                    Y[i, t, 0] = X[i, t, 0] + 0.1
    return X, Y

def generate_analogy_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """A:B::C:? where relationship is linear transformation."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        a = torch.rand(1).item()
        b = torch.rand(1).item()
        scale = b - a if a != 0 else 1
        for t in range(seq_len):
            if t % 4 == 0:
                X[i, t, 0] = a
            elif t % 4 == 1:
                X[i, t, 0] = b
            elif t % 4 == 2:
                c = torch.rand(1).item()
                X[i, t, 0] = c
            else:
                Y[i, t, 0] = X[i, t-1, 0] + scale
    return X, Y

def generate_xor_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """XOR of last two binary inputs."""
    X = (torch.rand(n_samples, seq_len, 8) > 0.5).float()
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        for t in range(2, seq_len):
            xor_result = int(X[i, t-1, 0].item()) ^ int(X[i, t-2, 0].item())
            Y[i, t, 0] = float(xor_result)
    return X, Y

def generate_implication_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Logical implication: A -> B."""
    X = (torch.rand(n_samples, seq_len, 8) > 0.5).float()
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        for t in range(1, seq_len):
            a = X[i, t-1, 0].item() > 0.5
            b = X[i, t, 0].item() > 0.5
            Y[i, t, 0] = float((not a) or b)
    return X, Y

def generate_projectile_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projectile motion: y = y0 + v0*t - 0.5*g*t^2."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        y0 = torch.rand(1).item() * 10
        v0 = torch.rand(1).item() * 5
        g = 0.5
        for t in range(seq_len):
            X[i, t, 0] = t / seq_len
            y = y0 + v0 * t - 0.5 * g * t * t
            Y[i, t, 0] = max(0, y) / 20.0
    return X, Y

def generate_collision_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two objects collide when positions match."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        x1, v1 = torch.rand(1).item(), torch.rand(1).item() * 0.2
        x2, v2 = torch.rand(1).item() + 0.5, -torch.rand(1).item() * 0.2
        collision_t = None
        for t in range(seq_len):
            p1 = x1 + v1 * t
            p2 = x2 + v2 * t
            X[i, t, 0] = p1
            X[i, t, 1] = p2
            if collision_t is None and abs(p1 - p2) < 0.1:
                collision_t = t
            Y[i, t, 0] = 1.0 if collision_t is not None and t >= collision_t else 0.0
    return X, Y

def generate_momentum_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conservation of momentum: m1v1 + m2v2 = const."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        m1, m2 = torch.rand(1).item() + 0.5, torch.rand(1).item() + 0.5
        v1, v2 = torch.rand(1).item(), torch.rand(1).item()
        total_momentum = m1 * v1 + m2 * v2
        for t in range(seq_len):
            X[i, t, 0] = m1
            X[i, t, 1] = v1 + 0.1 * t
            X[i, t, 2] = m2
            X[i, t, 3] = v2
            Y[i, t, 0] = total_momentum / 5.0
    return X, Y

def generate_gravity_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse square law: F = G*m1*m2/r^2."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        m1, m2 = torch.rand(1).item() + 0.1, torch.rand(1).item() + 0.1
        for t in range(seq_len):
            r = 0.5 + t * 0.1
            X[i, t, 0] = m1
            X[i, t, 1] = m2
            X[i, t, 2] = r
            F = m1 * m2 / (r * r)
            Y[i, t, 0] = min(1.0, F)
    return X, Y

def generate_delay_task(n_samples: int = 500, seq_len: int = 16, delay: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Output input from delay steps ago."""
    X = torch.rand(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        for t in range(delay, seq_len):
            Y[i, t, 0] = X[i, t-delay, 0]
    return X, Y

def generate_copy_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copy first half to second half."""
    X = torch.rand(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    half = seq_len // 2
    for i in range(n_samples):
        for t in range(half, seq_len):
            Y[i, t, 0] = X[i, t - half, 0]
    return X, Y

def generate_counting_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Count occurrences of value > 0.5."""
    X = torch.rand(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        count = 0
        for t in range(seq_len):
            if X[i, t, 0] > 0.5:
                count += 1
            Y[i, t, 0] = count / seq_len
    return X, Y

def generate_associative_task(n_samples: int = 500, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Key-value lookup: remember associations."""
    X = torch.zeros(n_samples, seq_len, 8)
    Y = torch.zeros(n_samples, seq_len, 4)
    for i in range(n_samples):
        keys = torch.randint(0, 4, (4,))
        values = torch.rand(4)
        for t in range(seq_len):
            if t < 4:
                X[i, t, keys[t].item()] = 1.0
                X[i, t, 4] = values[t].item()
            else:
                query_idx = torch.randint(0, 4, (1,)).item()
                X[i, t, query_idx] = 1.0
                for j in range(4):
                    if keys[j].item() == query_idx:
                        Y[i, t, 0] = values[j].item()
                        break
    return X, Y


# ============================================================================
# MECHANISTIC INTERPRETABILITY PROBES
# ============================================================================

def compute_induction_score(model: nn.Module, seq_len: int = 32) -> float:
    """Measure in-context learning capability via AB...AB pattern."""
    model.eval()
    with torch.no_grad():
        # Create AB...AB pattern
        X = torch.zeros(1, seq_len, 8)
        pattern = torch.rand(4, 8)
        for t in range(seq_len):
            X[0, t] = pattern[t % 4]

        # Get outputs
        result = model(X)
        output = result['output']

        # Measure similarity between positions that should match
        scores = []
        for t in range(4, seq_len):
            expected_match = t - 4
            similarity = F.cosine_similarity(
                output[0, t].unsqueeze(0),
                output[0, expected_match].unsqueeze(0)
            )
            scores.append(similarity.item())

        return np.mean(scores) if scores else 0.0


def compute_superposition_score(model: nn.Module) -> float:
    """Measure feature packing via random projection interference."""
    model.eval()
    with torch.no_grad():
        # Generate random feature directions
        n_features = 20
        features = torch.randn(n_features, 8)
        features = F.normalize(features, dim=-1)

        # Process each feature
        outputs = []
        for f in features:
            X = f.unsqueeze(0).unsqueeze(0).expand(1, 8, -1)
            result = model(X)
            outputs.append(result['output'][0, -1])

        outputs = torch.stack(outputs)

        # Compute interference: how much do features overlap?
        gram = outputs @ outputs.T
        off_diagonal = gram - torch.diag(torch.diag(gram))
        superposition = torch.abs(off_diagonal).mean().item()

        return superposition


def compute_gate_sharpness(model: nn.Module) -> float:
    """Measure gate activation sharpness (bimodality)."""
    model.eval()
    gate_activations = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor) and o.dim() >= 2:
                    gate_activations.append(o.detach())
        elif isinstance(output, torch.Tensor):
            gate_activations.append(output.detach())

    # Register hooks on sigmoid layers
    handles = []
    for name, module in model.named_modules():
        if 'gate' in name.lower() or 'alpha' in name.lower() or 'beta' in name.lower():
            handles.append(module.register_forward_hook(hook))

    # Forward pass
    with torch.no_grad():
        X = torch.rand(10, 16, 8)
        model(X)

    # Remove hooks
    for h in handles:
        h.remove()

    if not gate_activations:
        return 0.0

    # Compute bimodality: how close are activations to 0 or 1?
    all_acts = torch.cat([g.flatten() for g in gate_activations])
    distance_to_extreme = torch.min(all_acts, 1 - all_acts)
    sharpness = 1.0 - 2 * distance_to_extreme.mean().item()

    return max(0, sharpness)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.003,
    batch_size: int = 32,
) -> List[float]:
    """Train model and return loss history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            result = model(X_batch)
            output = result['output']
            loss = criterion(output, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        losses.append(epoch_loss / len(loader))

    return losses


def evaluate_accuracy(
    model: nn.Module,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    threshold: float = 0.15,
) -> float:
    """Evaluate accuracy on test set."""
    model.eval()
    with torch.no_grad():
        result = model(X_test)
        output = result['output']

        # Only evaluate positions with non-zero targets
        mask = (Y_test.abs().sum(dim=-1) > 0.01)
        if mask.sum() == 0:
            return 0.0

        errors = torch.abs(output - Y_test)
        correct = (errors < threshold).float()
        accuracy = (correct * mask.unsqueeze(-1)).sum() / (mask.sum() * Y_test.shape[-1])

        return accuracy.item() * 100


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark():
    """Run full benchmark on Generation 2 architectures."""
    print("=" * 80)
    print("GENERATION 2 ANIMA BENCHMARK")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Task generators
    tasks = {
        'sequence': generate_sequence_task,
        'pattern': generate_pattern_task,
        'conditional': generate_conditional_task,
        'analogy': generate_analogy_task,
        'xor': generate_xor_task,
        'implication': generate_implication_task,
        'projectile': generate_projectile_task,
        'collision': generate_collision_task,
        'momentum': generate_momentum_task,
        'gravity': generate_gravity_task,
        'delay': generate_delay_task,
        'copy': generate_copy_task,
        'counting': generate_counting_task,
        'associative': generate_associative_task,
    }

    # Models to benchmark
    models = {
        'Hierarchical': lambda: AnimaHierarchical(sensory_dim=8, d_model=32, bottleneck_dim=16, output_dim=4, d_state=16, d_meta=8),
        'Induction': lambda: AnimaInduction(sensory_dim=8, d_model=32, bottleneck_dim=16, output_dim=4, d_state=16, lookback=4),
        'Modular': lambda: AnimaModular(sensory_dim=8, d_model=32, bottleneck_dim=16, output_dim=4, d_state=16),
    }

    # Add legacy models if available
    if LEGACY_AVAILABLE:
        models['EvolvedV2 (legacy)'] = lambda: AnimaEvolvedV2(sensory_dim=8, d_model=32, bottleneck_dim=16, output_dim=4, d_state=16)
        models['Router (legacy)'] = lambda: AnimaEvolvedRouter(sensory_dim=8, d_model=32, bottleneck_dim=16, output_dim=4, d_state=16)
        models['Anima (legacy)'] = lambda: Anima(sensory_dim=8, d_model=32, bottleneck_dim=16, output_dim=4)

    # Results storage
    results = {name: {} for name in models}
    mi_scores = {name: {} for name in models}

    # Print parameter counts
    print("Model Parameters:")
    print("-" * 40)
    for name, model_fn in models.items():
        model = model_fn()
        n_params = model.get_num_params()
        print(f"  {name}: {n_params:,}")
    print()

    # Run benchmarks
    for task_name, task_fn in tasks.items():
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_name.upper()}")
        print('=' * 60)

        # Generate data
        X, Y = task_fn()
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        for model_name, model_fn in models.items():
            print(f"\n  {model_name}:")

            # Create and train model
            model = model_fn()
            losses = train_model(model, X_train, Y_train, epochs=100)

            # Evaluate
            accuracy = evaluate_accuracy(model, X_test, Y_test)
            results[model_name][task_name] = accuracy

            print(f"    Accuracy: {accuracy:.1f}%")
            print(f"    Final loss: {losses[-1]:.4f}")

    # Compute MI scores
    print("\n" + "=" * 60)
    print("MECHANISTIC INTERPRETABILITY SCORES")
    print("=" * 60)

    for model_name, model_fn in models.items():
        model = model_fn()
        # Quick train on mixed tasks
        X_mix = torch.cat([generate_pattern_task()[0], generate_delay_task()[0]], dim=0)
        Y_mix = torch.cat([generate_pattern_task()[1], generate_delay_task()[1]], dim=0)
        train_model(model, X_mix, Y_mix, epochs=50)

        induction = compute_induction_score(model)
        superposition = compute_superposition_score(model)
        sharpness = compute_gate_sharpness(model)

        mi_scores[model_name] = {
            'induction': induction,
            'superposition': superposition,
            'gate_sharpness': sharpness,
        }

        print(f"\n  {model_name}:")
        print(f"    Induction Score: {induction:.3f}")
        print(f"    Superposition: {superposition:.3f}")
        print(f"    Gate Sharpness: {sharpness:.3f}")

    # Compute category scores
    print("\n" + "=" * 60)
    print("CATEGORY PERFORMANCE")
    print("=" * 60)

    categories = {
        'Logic': ['conditional', 'xor', 'implication'],
        'Physics': ['projectile', 'collision', 'momentum', 'gravity'],
        'Memory': ['delay', 'copy', 'counting', 'associative'],
        'ICL': ['sequence', 'pattern', 'analogy'],
    }

    category_scores = {name: {} for name in models}
    for model_name in models:
        for cat_name, cat_tasks in categories.items():
            scores = [results[model_name].get(t, 0) for t in cat_tasks]
            category_scores[model_name][cat_name] = np.mean(scores)

    # Print category table
    print(f"\n{'Model':<20} | {'Logic':>8} | {'Physics':>8} | {'Memory':>8} | {'ICL':>8} | {'Overall':>8}")
    print("-" * 80)
    for model_name in models:
        overall = np.mean(list(results[model_name].values()))
        print(f"{model_name:<20} | {category_scores[model_name]['Logic']:>7.1f}% | "
              f"{category_scores[model_name]['Physics']:>7.1f}% | "
              f"{category_scores[model_name]['Memory']:>7.1f}% | "
              f"{category_scores[model_name]['ICL']:>7.1f}% | "
              f"{overall:>7.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'task_results': results,
        'category_scores': category_scores,
        'mi_scores': mi_scores,
    }

    output_path = os.path.join(os.path.dirname(__file__), 'generation2_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    return results, category_scores, mi_scores


if __name__ == '__main__':
    run_benchmark()
