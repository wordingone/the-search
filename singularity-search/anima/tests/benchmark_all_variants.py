"""
ANIMA Full Variant Benchmark
============================

Benchmarks ALL ANIMA variants against the official standard.

This script:
1. Loads each variant
2. Scales to 25k params (when possible)
3. Runs the 8-task benchmark
4. Records results to JSON

Rate-limited to prevent GPU memory issues.
"""

import sys
import os
import gc
import time
import json
import random
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Benchmark standard
from anima.eval.benchmark_standard import (
    BenchmarkStandard,
    OFFICIAL_TRANSFORMER_BASELINE,
    get_standard,
)

# Set seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STANDARD = get_standard()


# =============================================================================
# TASK GENERATORS (from benchmark standard)
# =============================================================================

def gen_sequence(n=100):
    data = []
    for _ in range(n):
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        seq = [(start + i * step) / 100.0 for i in range(9)]
        data.append({'input': seq[:-1], 'target': seq[-1]})
    return data

def gen_pattern(n=100):
    data = []
    for _ in range(n):
        plen = random.randint(2, 4)
        pattern = [random.randint(1, 9) / 10.0 for _ in range(plen)]
        seq = (pattern * 5)[:9]
        data.append({'input': seq[:-1], 'target': seq[-1]})
    return data

def gen_conditional(n=100):
    data = []
    for _ in range(n):
        seq = [random.randint(1, 10) / 20.0 for _ in range(8)]
        last = seq[-1] * 20
        target = (last * 2 if last > 5 else last + 1) / 20.0
        data.append({'input': seq, 'target': target})
    return data

def gen_analogy(n=100):
    data = []
    for _ in range(n):
        a = random.randint(1, 5) / 10.0
        c = random.randint(1, 5) / 10.0
        data.append({'input': [a, a*2, c, 0, 0, 0, 0, 0], 'target': c*2})
    return data

def gen_projectile(n=100):
    data = []
    for _ in range(n):
        v0 = random.uniform(5, 15)
        positions = [v0 * t * 0.1 / 20.0 for t in range(8)]
        landing = min(v0 * 0.8 / 20.0, 1.0)
        data.append({'input': positions, 'target': landing})
    return data

def gen_collision(n=100):
    data = []
    for _ in range(n):
        x1, v1 = random.uniform(0, 0.5), random.uniform(0.1, 0.3)
        x2, v2 = random.uniform(0.5, 1), random.uniform(-0.3, -0.1)
        collide = 1.0 if v1 > -v2 else 0.0
        seq = [x1, v1, x2, v2] + [0.0] * 4
        data.append({'input': seq, 'target': collide, 'binary': True})
    return data

def gen_goal(n=100):
    data = []
    for _ in range(n):
        pos = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        goal = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        dist = max((dx**2 + dy**2)**0.5, 0.01)
        data.append({'input': pos + goal + [0]*4, 'target': [dx/dist, dy/dist], 'goal': True})
    return data

def gen_momentum(n=100):
    data = []
    for _ in range(n):
        m1, v1 = random.uniform(0.2, 1), random.uniform(0.2, 1)
        m2, v2 = random.uniform(0.2, 1), random.uniform(-1, -0.2)
        v_f = (m1*v1 + m2*v2) / (m1 + m2)
        data.append({'input': [m1, v1, m2, v2] + [0]*4, 'target': (v_f + 1) / 2})
    return data


TASKS = [
    ('sequence', gen_sequence, 'reasoning'),
    ('pattern', gen_pattern, 'reasoning'),
    ('conditional', gen_conditional, 'reasoning'),
    ('analogy', gen_analogy, 'reasoning'),
    ('projectile', gen_projectile, 'physics'),
    ('collision', gen_collision, 'physics'),
    ('goal', gen_goal, 'physics'),
    ('momentum', gen_momentum, 'physics'),
]


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_eval_model(model: nn.Module, task_name: str, gen_fn, epochs: int = 50, lr: float = 0.01) -> float:
    """Train and evaluate a model on a single task."""
    device = next(model.parameters()).device
    is_goal = task_name == 'goal'
    is_binary = task_name == 'collision'

    # Generate data
    train_data = gen_fn(STANDARD.train_samples)
    test_data = gen_fn(STANDARD.test_samples)

    train_x = torch.tensor([d['input'] for d in train_data], dtype=torch.float32, device=device)
    if is_goal:
        train_y = torch.tensor([d['target'] for d in train_data], dtype=torch.float32, device=device)
    else:
        train_y = torch.tensor([[d['target']] for d in train_data], dtype=torch.float32, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        if hasattr(model, 'reset'):
            # ANIMA-style step-by-step
            outputs = []
            for i in range(len(train_data)):
                model.reset(1, device)
                inp = train_x[i]
                if len(inp) < 8:
                    inp = torch.cat([inp, torch.zeros(8 - len(inp), device=device)])
                result = model.step(inp[:8].unsqueeze(0))
                if is_goal:
                    outputs.append(result['action'][:, :2])
                else:
                    outputs.append(result['action'][:, :1])
            outputs = torch.cat(outputs, dim=0)
        else:
            # Standard forward
            x = train_x
            if x.shape[-1] < 8:
                x = torch.cat([x, torch.zeros(x.shape[0], 8 - x.shape[-1], device=device)], -1)
            out = model(x)
            if out.dim() == 3:
                out = out[:, -1]  # Take last timestep
            if is_goal:
                outputs = out[:, :2]
            else:
                outputs = out[:, :1]

        loss = criterion(outputs, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for d in test_data:
            inp = torch.tensor(d['input'], dtype=torch.float32, device=device)
            if len(inp) < 8:
                inp = torch.cat([inp, torch.zeros(8 - len(inp), device=device)])
            inp = inp[:8]

            if hasattr(model, 'reset'):
                model.reset(1, device)
                result = model.step(inp.unsqueeze(0))
                pred = result['action'][0, :2].cpu().numpy() if is_goal else result['action'][0, 0].item()
            else:
                out = model(inp.unsqueeze(0))
                if out.dim() == 3:
                    out = out[:, -1]
                pred = out[0, :2].cpu().numpy() if is_goal else out[0, 0].item()

            # Check accuracy
            if is_goal:
                target = d['target']
                dot = pred[0]*target[0] + pred[1]*target[1]
                pn = max((pred[0]**2 + pred[1]**2)**0.5, 0.01)
                tn = max((target[0]**2 + target[1]**2)**0.5, 0.01)
                if dot/(pn*tn) > STANDARD.direction_cosine_threshold:
                    correct += 1
            elif is_binary:
                if (pred > STANDARD.binary_threshold) == (d['target'] > 0.5):
                    correct += 1
            else:
                tgt = d['target']
                tol = max(STANDARD.regression_tolerance * abs(tgt), STANDARD.regression_min_tolerance)
                if abs(pred - tgt) < tol:
                    correct += 1

    return correct / len(test_data)


def benchmark_variant(model: nn.Module, variant_name: str) -> Dict[str, Any]:
    """Run full benchmark on a variant."""
    results = {
        'name': variant_name,
        'params': sum(p.numel() for p in model.parameters()),
        'reasoning': {},
        'physics': {},
        'timestamp': datetime.now().isoformat(),
    }

    print(f"  Benchmarking {variant_name} ({results['params']:,} params)...")

    for task_name, gen_fn, category in TASKS:
        try:
            # Fresh model state
            if hasattr(model, 'reset'):
                model.reset(1, DEVICE)

            acc = train_eval_model(model, task_name, gen_fn)
            results[category][task_name] = float(acc)
            print(f"    {task_name}: {acc*100:.1f}%")
        except Exception as e:
            print(f"    {task_name}: ERROR - {e}")
            results[category][task_name] = None

    # Compute averages
    reasoning_scores = [v for v in results['reasoning'].values() if v is not None]
    physics_scores = [v for v in results['physics'].values() if v is not None]

    results['reasoning_avg'] = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0
    results['physics_avg'] = sum(physics_scores) / len(physics_scores) if physics_scores else 0
    results['overall'] = (results['reasoning_avg'] + results['physics_avg']) / 2

    # Comparison to transformer baseline
    results['vs_transformer'] = results['overall'] - OFFICIAL_TRANSFORMER_BASELINE['overall']

    print(f"    Overall: {results['overall']*100:.1f}% (vs Transformer: {results['vs_transformer']*100:+.1f}pp)")

    return results


# =============================================================================
# VARIANT LOADERS
# =============================================================================

def load_variant(variant_name: str) -> Optional[nn.Module]:
    """Load a variant model, scaling to ~25k params when possible."""
    try:
        if variant_name == "Anima (Base)":
            from anima.core.base import Anima, AnimaConfig
            # Scale to ~25k params
            config = AnimaConfig()
            for d in range(8, 64):
                config.world_dim = d
                config.internal_dim = d
                config.time_dim = max(4, d // 4)
                model = Anima(config)
                if sum(p.numel() for p in model.parameters()) >= STANDARD.target_params * 0.9:
                    return model.to(DEVICE)
            return Anima(config).to(DEVICE)

        elif variant_name == "AnimaV2 (Core)":
            from anima.core.anima_v2 import AnimaV2, AnimaV2Config
            config = AnimaV2Config()
            return AnimaV2(config).to(DEVICE)

        elif variant_name == "AnimaV3 (Collective)":
            from anima.core.anima_v3 import AnimaV3, AnimaV3Config
            config = AnimaV3Config()
            config.num_agents = 2  # Reduce for param budget
            return AnimaV3(config).to(DEVICE)

        elif variant_name == "AnimaV4 (Minimal)":
            from anima.core.anima_v4 import AnimaV4, AnimaV4Config
            config = AnimaV4Config()
            return AnimaV4(config).to(DEVICE)

        elif variant_name == "AnimaTelos (V4-Telos)":
            from anima.core.anima_v4_telos import AnimaTelos, AnimaTelosConfig
            config = AnimaTelosConfig()
            return AnimaTelos(config).to(DEVICE)

        elif variant_name == "AnimaV5 (Temporal)":
            from anima.core.anima_v5 import AnimaV5, AnimaV5Config
            config = AnimaV5Config()
            return AnimaV5(config).to(DEVICE)

        elif variant_name == "ANIMA-Zero":
            from anima.core.anima_zero import ANIMA, ANIMAConfig
            # Scale to 25k
            config = ANIMAConfig()
            for d in range(8, 64):
                config.world_dim = d
                config.internal_dim = d
                config.action_dim = d
                model = ANIMA(config)
                params = sum(p.numel() for p in model.parameters())
                if params >= STANDARD.target_params * 0.95:
                    return model.to(DEVICE)
            return ANIMA(config).to(DEVICE)

        elif variant_name == "ANIMA-One":
            from anima.core.anima_one import ANIMA1, ANIMA1Config
            config = ANIMA1Config()
            return ANIMA1(config).to(DEVICE)

        elif variant_name == "ANIMA-Two":
            from anima.core.anima_two import ANIMATwo, ANIMATwoConfig
            config = ANIMATwoConfig()
            return ANIMATwo(config).to(DEVICE)

        elif variant_name == "AnimaMortal":
            from anima.variants.mortal import AnimaMortal, AnimaMortalConfig
            config = AnimaMortalConfig()
            return AnimaMortal(config).to(DEVICE)

        elif variant_name == "AnimaMetamorphic":
            from anima.variants.metamorphic import AnimaMetamorphic, AnimaMetamorphicConfig
            config = AnimaMetamorphicConfig()
            return AnimaMetamorphic(config).to(DEVICE)

        elif variant_name == "AnimaCollective":
            from anima.variants.collective import AnimaCollective, AnimaCollectiveConfig
            config = AnimaCollectiveConfig()
            return AnimaCollective(config).to(DEVICE)

        elif variant_name == "AnimaNeuroplastic":
            from anima.variants.neuroplastic import AnimaNeuroplastic, AnimaNeuroplasticConfig
            config = AnimaNeuroplasticConfig()
            return AnimaNeuroplastic(config).to(DEVICE)

        elif variant_name == "AnimaAdaptive":
            from anima.variants.v2_hybrids import AnimaAdaptive, AnimaAdaptiveConfig
            config = AnimaAdaptiveConfig()
            return AnimaAdaptive(config).to(DEVICE)

        elif variant_name == "AnimaResonant":
            from anima.variants.v2_hybrids import AnimaResonant, AnimaResonantConfig
            config = AnimaResonantConfig()
            return AnimaResonant(config).to(DEVICE)

        elif variant_name == "AnimaPhoenix":
            from anima.variants.v2_hybrids import AnimaPhoenix, AnimaPhoenixConfig
            config = AnimaPhoenixConfig()
            return AnimaPhoenix(config).to(DEVICE)

        elif variant_name == "AnimaPressured":
            from anima.variants.v2_hybrids import AnimaPressured, AnimaPressuredConfig
            config = AnimaPressuredConfig()
            return AnimaPressured(config).to(DEVICE)

        elif variant_name == "Transformer (Baseline)":
            from anima.eval.benchmark_standard import StandardTransformer
            return StandardTransformer().to(DEVICE)

        else:
            print(f"  Unknown variant: {variant_name}")
            return None

    except Exception as e:
        print(f"  Error loading {variant_name}: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

# All variants to benchmark
ALL_VARIANTS = [
    # Baselines
    "Transformer (Baseline)",

    # Core variants (chronological)
    "Anima (Base)",
    "AnimaV2 (Core)",
    "AnimaV3 (Collective)",
    "AnimaV4 (Minimal)",
    "AnimaTelos (V4-Telos)",
    "AnimaV5 (Temporal)",

    # Perfected variants (V(N)-V(T)-Phi theorem)
    "ANIMA-Zero",
    "ANIMA-One",
    "ANIMA-Two",

    # Experimental variants
    "AnimaMortal",
    "AnimaMetamorphic",
    "AnimaCollective",
    "AnimaNeuroplastic",

    # V2 Hybrids
    "AnimaAdaptive",
    "AnimaResonant",
    "AnimaPhoenix",
    "AnimaPressured",
]


def run_all_benchmarks(rate_limit_seconds: float = 2.0) -> Dict[str, Any]:
    """Run benchmarks on all variants with rate limiting."""
    print("=" * 70)
    print("ANIMA FULL VARIANT BENCHMARK")
    print(f"Standard: v{STANDARD.version}")
    print(f"Target Params: {STANDARD.target_params:,}")
    print(f"Device: {DEVICE}")
    print(f"Rate Limit: {rate_limit_seconds}s between variants")
    print("=" * 70)
    print()

    all_results = {
        'benchmark_version': STANDARD.version,
        'target_params': STANDARD.target_params,
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'variants': {},
    }

    for i, variant_name in enumerate(ALL_VARIANTS):
        print(f"\n[{i+1}/{len(ALL_VARIANTS)}] {variant_name}")
        print("-" * 50)

        # Load model
        model = load_variant(variant_name)

        if model is None:
            all_results['variants'][variant_name] = {'error': 'Failed to load'}
            continue

        try:
            # Run benchmark
            results = benchmark_variant(model, variant_name)
            all_results['variants'][variant_name] = results

        except Exception as e:
            print(f"  Benchmark failed: {e}")
            traceback.print_exc()
            all_results['variants'][variant_name] = {'error': str(e)}

        finally:
            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Rate limit
        if i < len(ALL_VARIANTS) - 1:
            print(f"  Waiting {rate_limit_seconds}s...")
            time.sleep(rate_limit_seconds)

    # Save results
    output_path = Path(__file__).parent / 'all_variants_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Variant':<25} {'Params':<10} {'Reasoning':<12} {'Physics':<12} {'Overall':<12} {'vs TF':<10}")
    print("-" * 81)

    for name, res in all_results['variants'].items():
        if 'error' in res:
            print(f"{name:<25} {'ERROR':<10}")
        else:
            tf_diff = res.get('vs_transformer', 0) * 100
            print(f"{name:<25} {res['params']:<10,} {res['reasoning_avg']*100:<11.1f}% "
                  f"{res['physics_avg']*100:<11.1f}% {res['overall']*100:<11.1f}% "
                  f"{tf_diff:+.1f}pp")

    return all_results


if __name__ == '__main__':
    results = run_all_benchmarks(rate_limit_seconds=1.0)
