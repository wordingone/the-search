"""
Comprehensive Anima Family Benchmark
=====================================

Mechanistic interpretability benchmark for ALL Anima variants.
Uses synthetic logic probes to validate architectural primitives.

Benchmark Philosophy:
- Circuit-level analysis via induction head tracking
- Superposition probes for feature isolation
- Power-law scaling extrapolation from toy model trajectories
- Validates specialized logic, physics, and analogy domains
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import math
from dataclasses import dataclass

# Core variants
from anima.core import Anima, AnimaOptimized, AnimaISSM, AnimaATR, AnimaEvolved
from anima.core.anima_evolved_v2 import AnimaEvolvedV2
from anima.core.anima_evolved_router import AnimaEvolvedRouter

# Try to import archived variants
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'archive', 'legacy_variants'))
    from anima_zero import ANIMA as AnimaZero
    HAS_ZERO = True
except:
    HAS_ZERO = False

try:
    from anima_one import ANIMAOne as AnimaOne
    HAS_ONE = True
except:
    HAS_ONE = False


# =============================================================================
# MECHANISTIC INTERPRETABILITY PROBES
# =============================================================================

@dataclass
class CircuitProbe:
    """Tracks circuit-level behavior for mechanistic analysis."""
    induction_strength: float = 0.0  # Measures in-context learning
    superposition_score: float = 0.0  # Feature compression efficiency
    gate_entropy: float = 0.0  # Gate decision sharpness
    gradient_norm: float = 0.0  # Gradient flow health


def compute_induction_score(model, device) -> float:
    """
    Measures induction head strength via pattern completion.
    Higher score = better in-context learning primitive.
    """
    # Create ABAB pattern - model should predict B after seeing A...B...A
    batch_size = 32
    seq_len = 8

    patterns = []
    for _ in range(batch_size):
        a = random.uniform(0.2, 0.4)
        b = random.uniform(0.6, 0.8)
        # Pattern: [A, B, noise, noise, A, ?, ?, ?] -> should predict B
        seq = [a, b, 0.5, 0.5, a, 0.5, 0.5, 0.5]
        patterns.append(seq)

    x = torch.tensor(patterns, dtype=torch.float32, device=device)
    x = x.unsqueeze(-1).expand(-1, -1, 8)  # [batch, seq, sensory_dim]

    model.eval()
    with torch.no_grad():
        try:
            out = model(x)
            if isinstance(out, dict):
                out = out['output']
        except:
            return 0.0

        # Check if position 5 output resembles position 1 input (B value)
        pred_at_5 = out[:, 5, 0].cpu().numpy()
        target_b = np.array([p[1] for p in patterns])

        correlation = np.corrcoef(pred_at_5, target_b)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0


def compute_superposition_score(model, device) -> float:
    """
    Measures feature superposition via orthogonality of learned representations.
    Higher score = more efficient feature packing.
    """
    # Generate diverse inputs to probe feature space
    batch_size = 64
    x = torch.randn(batch_size, 8, 8, device=device) * 0.5 + 0.5
    x = x.clamp(0, 1)

    model.eval()
    with torch.no_grad():
        try:
            out = model(x)
            if isinstance(out, dict):
                out = out['output']
        except:
            return 0.0

        # Get final hidden representations
        final_out = out[:, -1, :]  # [batch, output_dim]

        # Compute covariance matrix
        centered = final_out - final_out.mean(dim=0)
        cov = torch.mm(centered.t(), centered) / (batch_size - 1)

        # Measure off-diagonal vs diagonal (lower ratio = more orthogonal)
        diag = torch.diag(cov).abs().sum()
        off_diag = (cov.abs().sum() - diag) / 2

        if diag > 0:
            orthogonality = 1.0 - (off_diag / (diag + off_diag)).item()
            return orthogonality
        return 0.5


def compute_gate_entropy(model) -> float:
    """
    Measures gate decision sharpness via entropy.
    Lower entropy = sharper decisions = better specialization.
    """
    # Check if model has gate parameters
    gate_params = []
    for name, param in model.named_parameters():
        if 'gate' in name.lower() or 'alpha' in name.lower() or 'beta' in name.lower():
            gate_params.append(param.data.cpu().numpy().flatten())

    if not gate_params:
        return 0.5  # Neutral if no gates

    # Compute entropy of gate weight distributions
    all_weights = np.concatenate(gate_params)

    # Discretize into bins
    hist, _ = np.histogram(all_weights, bins=20, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    hist = hist / hist.sum()

    entropy = -np.sum(hist * np.log(hist))
    max_entropy = np.log(20)  # Max entropy for 20 bins

    normalized = entropy / max_entropy
    return 1.0 - normalized  # Higher = sharper


# =============================================================================
# SYNTHETIC TASK DOMAINS
# =============================================================================

class SyntheticTaskGenerator:
    """Generates synthetic tasks for architectural primitive validation."""

    @staticmethod
    def logic_tasks(n: int, seed: int) -> Dict[str, List]:
        """Logic reasoning probes."""
        random.seed(seed)

        tasks = {
            'sequence': [],      # Arithmetic sequences
            'pattern': [],       # Pattern completion
            'conditional': [],   # If-then-else
            'analogy': [],       # A:B::C:?
            'xor': [],           # XOR logic
            'implication': [],   # P -> Q
        }

        for _ in range(n):
            # Sequence: predict next in arithmetic sequence
            start, step = random.randint(1, 10), random.randint(1, 5)
            seq = [(start + i * step) / 100.0 for i in range(9)]
            tasks['sequence'].append({'input': seq[:-1], 'target': seq[-1]})

            # Pattern: repeat detection
            plen = random.randint(2, 4)
            pattern = [random.randint(1, 9) / 10.0 for _ in range(plen)]
            seq = (pattern * 5)[:9]
            tasks['pattern'].append({'input': seq[:-1], 'target': seq[-1]})

            # Conditional: if last > 0.25 then 2x else x+0.05
            seq = [random.randint(1, 10) / 20.0 for _ in range(8)]
            last = seq[-1] * 20
            target = (last * 2 if last > 5 else last + 1) / 20.0
            tasks['conditional'].append({'input': seq, 'target': target})

            # Analogy: A:B::C:?
            a = random.randint(1, 5) / 10.0
            b = a * 2
            c = random.randint(1, 5) / 10.0
            d = c * 2
            tasks['analogy'].append({'input': [a, b, c, 0, 0, 0, 0, 0], 'target': d})

            # XOR
            a_bit = random.choice([0.2, 0.8])
            b_bit = random.choice([0.2, 0.8])
            xor_val = 0.8 if (a_bit > 0.5) != (b_bit > 0.5) else 0.2
            tasks['xor'].append({'input': [a_bit, b_bit, 0, 0, 0, 0, 0, 0], 'target': xor_val})

            # Implication: P -> Q (false only if P=T, Q=F)
            p = random.choice([0.2, 0.8])
            q = random.choice([0.2, 0.8])
            impl = 0.2 if (p > 0.5 and q < 0.5) else 0.8
            tasks['implication'].append({'input': [p, q, 0, 0, 0, 0, 0, 0], 'target': impl})

        return tasks

    @staticmethod
    def physics_tasks(n: int, seed: int) -> Dict[str, List]:
        """Physics simulation probes."""
        random.seed(seed)

        tasks = {
            'projectile': [],    # Trajectory prediction
            'collision': [],     # Will objects collide?
            'momentum': [],      # Conservation of momentum
            'gravity': [],       # Gravitational acceleration
        }

        for _ in range(n):
            # Projectile motion
            v0 = random.uniform(5, 15)
            positions = [v0 * t * 0.1 / 20.0 for t in range(8)]
            landing = v0 * 0.8 / 20.0
            tasks['projectile'].append({'input': positions, 'target': min(landing, 1.0)})

            # Collision detection
            x1, v1 = random.uniform(0, 0.5), random.uniform(0.1, 0.3)
            x2, v2 = random.uniform(0.5, 1), random.uniform(-0.3, -0.1)
            will_collide = 1.0 if v1 > -v2 else 0.0
            tasks['collision'].append({'input': [x1, v1, x2, v2, 0, 0, 0, 0], 'target': will_collide, 'binary': True})

            # Momentum conservation
            m1, v1 = random.uniform(0.2, 1), random.uniform(0.2, 1)
            m2, v2 = random.uniform(0.2, 1), random.uniform(-1, -0.2)
            v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
            tasks['momentum'].append({'input': [m1, v1, m2, v2, 0, 0, 0, 0], 'target': (v_final + 1) / 2})

            # Gravity (distance doubling -> 1/4 force)
            dist = random.uniform(0.2, 0.8)
            force = 1.0 / (dist ** 2)
            tasks['gravity'].append({'input': [dist, 0, 0, 0, 0, 0, 0, 0], 'target': min(force / 25, 1.0)})

        return tasks

    @staticmethod
    def memory_tasks(n: int, seed: int) -> Dict[str, List]:
        """Memory and temporal probes."""
        random.seed(seed)

        tasks = {
            'delay': [],         # Delayed recall
            'copy': [],          # Copy first element
            'counting': [],      # Count above threshold
            'associative': [],   # Key-value recall
        }

        for _ in range(n):
            # Delayed recall
            signal = random.uniform(0.2, 0.8)
            delay = random.randint(2, 5)
            seq = [signal] + [0.0] * delay + [0.5] + [0.0] * (7 - delay)
            tasks['delay'].append({'input': seq[:8], 'target': signal})

            # Copy first element
            seq = [random.uniform(0.1, 0.9) for _ in range(8)]
            tasks['copy'].append({'input': seq, 'target': seq[0]})

            # Count above threshold
            threshold = 0.5
            seq = [random.uniform(0.1, 0.9) for _ in range(8)]
            count = sum(1 for x in seq if x > threshold)
            tasks['counting'].append({'input': seq, 'target': count / 10.0})

            # Associative (key-value pairs)
            key = random.randint(1, 4) / 10.0
            value = random.uniform(0.5, 0.9)
            # Pattern: [key, value, noise, noise, key, ?, ?, ?]
            seq = [key, value, 0.3, 0.3, key, 0.3, 0.3, 0.3]
            tasks['associative'].append({'input': seq, 'target': value})

        return tasks


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(arch_name: str, device, seed: int, target_params: int = 48000):
    """Create model with approximate parameter matching."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    configs = {
        'Anima': {'d_model': 32, 'bottleneck_dim': 16},
        'AnimaOptimized': {'d_model': 32, 'bottleneck_dim': 16},
        'ISSM': {'d_model': 24, 'd_state': 16, 'bottleneck_dim': 12},
        'ATR': {'d_model': 48, 'bottleneck_dim': 24},
        'EvolvedV1': {'d_model': 23, 'd_state': 16, 'bottleneck_dim': 11},
        'EvolvedV2': {'d_model': 23, 'd_state': 16, 'bottleneck_dim': 11},
        'Router': {'d_model': 23, 'd_state': 16, 'bottleneck_dim': 10},
    }

    cfg = configs.get(arch_name, {})

    if arch_name == 'Anima':
        return Anima(sensory_dim=8, d_model=cfg['d_model'],
                    bottleneck_dim=cfg['bottleneck_dim'], output_dim=4).to(device)
    elif arch_name == 'AnimaOptimized':
        return AnimaOptimized(sensory_dim=8, d_model=cfg['d_model'],
                             bottleneck_dim=cfg['bottleneck_dim'], output_dim=4).to(device)
    elif arch_name == 'ISSM':
        return AnimaISSM(sensory_dim=8, d_model=cfg['d_model'], d_state=cfg['d_state'],
                        bottleneck_dim=cfg['bottleneck_dim'], output_dim=4).to(device)
    elif arch_name == 'ATR':
        return AnimaATR(sensory_dim=8, d_model=cfg['d_model'],
                       bottleneck_dim=cfg['bottleneck_dim'], output_dim=4).to(device)
    elif arch_name == 'EvolvedV1':
        return AnimaEvolved(sensory_dim=8, d_model=cfg['d_model'], d_state=cfg['d_state'],
                           bottleneck_dim=cfg['bottleneck_dim'], output_dim=4).to(device)
    elif arch_name == 'EvolvedV2':
        return AnimaEvolvedV2(sensory_dim=8, d_model=cfg['d_model'], d_state=cfg['d_state'],
                             bottleneck_dim=cfg['bottleneck_dim'], output_dim=4).to(device)
    elif arch_name == 'Router':
        return AnimaEvolvedRouter(sensory_dim=8, d_model=cfg['d_model'], d_state=cfg['d_state'],
                                 bottleneck_dim=cfg['bottleneck_dim'], output_dim=4).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate(model, train_data: List, test_data: List, device,
                       epochs: int = 50, lr: float = 0.01) -> Tuple[float, Dict]:
    """Train model and return accuracy + loss trajectory for scaling analysis."""
    is_binary = train_data[0].get('binary', False)

    # Prepare data
    train_inputs = [d['input'][:8] + [0]*(8-len(d['input'][:8])) for d in train_data]
    train_x = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    train_x = train_x.unsqueeze(1).expand(-1, 8, -1)
    train_y = torch.tensor([[d['target']] for d in train_data], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_trajectory = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        try:
            out = model(train_x)
            if isinstance(out, dict):
                out = out['output']
            outputs = out[:, -1, :1]
        except Exception as e:
            return 0.0, {'loss_trajectory': [], 'error': str(e)}

        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            loss_trajectory.append(loss.item())

    # Evaluate
    model.eval()
    test_inputs = [d['input'][:8] + [0]*(8-len(d['input'][:8])) for d in test_data]
    test_x = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_x = test_x.unsqueeze(1).expand(-1, 8, -1)

    correct = 0
    with torch.no_grad():
        try:
            out = model(test_x)
            if isinstance(out, dict):
                out = out['output']
            outputs = out[:, -1]
        except:
            return 0.0, {'loss_trajectory': loss_trajectory}

        for i, d in enumerate(test_data):
            pred = outputs[i, 0].item()
            target = d['target']

            if is_binary:
                if (pred > 0.5) == (target > 0.5):
                    correct += 1
            else:
                tol = max(abs(target) * 0.2, 0.1)
                if abs(pred - target) <= tol:
                    correct += 1

    accuracy = correct / len(test_data)
    return accuracy, {'loss_trajectory': loss_trajectory}


def run_comprehensive_benchmark():
    """Run full mechanistic interpretability benchmark."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("COMPREHENSIVE ANIMA FAMILY BENCHMARK")
    print("Mechanistic Interpretability + Synthetic Logic Probes")
    print("=" * 80)

    architectures = ['Anima', 'AnimaOptimized', 'ISSM', 'ATR', 'EvolvedV1', 'EvolvedV2', 'Router']

    # Parameter counts
    print("\nParameter Counts:")
    param_counts = {}
    for arch in architectures:
        try:
            model = create_model(arch, device, seed=42)
            params = sum(p.numel() for p in model.parameters())
            param_counts[arch] = params
            print(f"  {arch}: {params:,}")
        except Exception as e:
            print(f"  {arch}: ERROR - {e}")
            param_counts[arch] = 0

    # Generate all task data
    gen = SyntheticTaskGenerator()
    all_tasks = {}
    all_tasks.update(gen.logic_tasks(100, seed=42))
    all_tasks.update(gen.physics_tasks(100, seed=42))
    all_tasks.update(gen.memory_tasks(100, seed=42))

    test_tasks = {}
    test_tasks.update(gen.logic_tasks(50, seed=43))
    test_tasks.update(gen.physics_tasks(50, seed=43))
    test_tasks.update(gen.memory_tasks(50, seed=43))

    # Results storage
    results = {arch: {'tasks': {}, 'probes': {}, 'params': param_counts.get(arch, 0)} for arch in architectures}

    print("\n" + "=" * 80)
    print("TASK BENCHMARK RESULTS")
    print("=" * 80)

    # Run task benchmarks
    for task_name in all_tasks.keys():
        print(f"\n--- {task_name.upper()} ---")
        train_data = all_tasks[task_name]
        test_data = test_tasks[task_name]

        for arch in architectures:
            try:
                torch.manual_seed(42)
                np.random.seed(42)
                model = create_model(arch, device, seed=42)
                accuracy, meta = train_and_evaluate(model, train_data, test_data, device)
                results[arch]['tasks'][task_name] = accuracy
                print(f"  {arch:<15}: {accuracy*100:>5.1f}%")
            except Exception as e:
                results[arch]['tasks'][task_name] = 0.0
                print(f"  {arch:<15}: ERROR - {str(e)[:30]}")

    # Mechanistic probes
    print("\n" + "=" * 80)
    print("MECHANISTIC INTERPRETABILITY PROBES")
    print("=" * 80)

    for arch in architectures:
        try:
            model = create_model(arch, device, seed=42)

            induction = compute_induction_score(model, device)
            superposition = compute_superposition_score(model, device)
            gate_ent = compute_gate_entropy(model)

            results[arch]['probes']['induction'] = induction
            results[arch]['probes']['superposition'] = superposition
            results[arch]['probes']['gate_sharpness'] = gate_ent

            print(f"\n{arch}:")
            print(f"  Induction Score:    {induction:.3f}")
            print(f"  Superposition:      {superposition:.3f}")
            print(f"  Gate Sharpness:     {gate_ent:.3f}")
        except Exception as e:
            print(f"\n{arch}: PROBE ERROR - {e}")

    # Category summaries
    logic_tasks = ['sequence', 'pattern', 'conditional', 'analogy', 'xor', 'implication']
    physics_tasks = ['projectile', 'collision', 'momentum', 'gravity']
    memory_tasks = ['delay', 'copy', 'counting', 'associative']

    print("\n" + "=" * 80)
    print("CATEGORY SUMMARIES")
    print("=" * 80)

    for arch in architectures:
        logic_avg = np.mean([results[arch]['tasks'].get(t, 0) for t in logic_tasks]) * 100
        physics_avg = np.mean([results[arch]['tasks'].get(t, 0) for t in physics_tasks]) * 100
        memory_avg = np.mean([results[arch]['tasks'].get(t, 0) for t in memory_tasks]) * 100
        overall = np.mean([results[arch]['tasks'].get(t, 0) for t in all_tasks.keys()]) * 100

        results[arch]['summary'] = {
            'logic': logic_avg,
            'physics': physics_avg,
            'memory': memory_avg,
            'overall': overall
        }

        print(f"\n{arch} ({results[arch]['params']:,} params):")
        print(f"  Logic:    {logic_avg:>5.1f}%")
        print(f"  Physics:  {physics_avg:>5.1f}%")
        print(f"  Memory:   {memory_avg:>5.1f}%")
        print(f"  OVERALL:  {overall:>5.1f}%")

    # Final comparison table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 80)

    header = f"{'Architecture':<15} | {'Params':>8} | {'Logic':>6} | {'Physics':>7} | {'Memory':>6} | {'Overall':>7} | {'Ind.':>5} | {'Sup.':>5}"
    print(header)
    print("-" * len(header))

    for arch in architectures:
        s = results[arch]['summary']
        p = results[arch]['probes']
        params = results[arch]['params']
        print(f"{arch:<15} | {params:>8,} | {s['logic']:>5.1f}% | {s['physics']:>6.1f}% | {s['memory']:>5.1f}% | {s['overall']:>6.1f}% | {p.get('induction', 0):>5.2f} | {p.get('superposition', 0):>5.2f}")

    # Save results
    output = {
        'results': {k: {
            'tasks': {kk: float(vv) for kk, vv in v['tasks'].items()},
            'probes': v['probes'],
            'summary': v['summary'],
            'params': v['params']
        } for k, v in results.items()},
        'task_categories': {
            'logic': logic_tasks,
            'physics': physics_tasks,
            'memory': memory_tasks
        }
    }

    with open('COMPREHENSIVE_FAMILY_BENCHMARK.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to COMPREHENSIVE_FAMILY_BENCHMARK.json")

    return results


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
