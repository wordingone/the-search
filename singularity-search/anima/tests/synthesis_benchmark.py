"""
ANIMA-Synthesis Benchmark
=========================

Benchmarks ANIMA-Synthesis against all previous variants at exact parameter parity.
"""

import sys
import os
import gc
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import from strict benchmark
from anima.tests.strict_benchmark import (
    TransformerBaseline,
    ANIMAZeroExact,
    ANIMAOneExact,
    ANIMATwoExact,
    count_params,
    verify_param_match,
    TASKS,
    train_eval,
    PARAM_TOLERANCE,
)

# Import ANIMA-Synthesis
from anima.architectures.anima_synthesis import ANIMASynthesisExact


def run_synthesis_benchmark():
    """Run benchmark including ANIMA-Synthesis."""
    print("=" * 70)
    print("ANIMA-SYNTHESIS BENCHMARK")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()

    # Get target params from Transformer
    transformer = TransformerBaseline()
    target_params = count_params(transformer)
    transformer = transformer.to(DEVICE)
    print(f"Target params: {target_params:,}")
    print()

    # Create all models
    print("Creating models...")
    models_specs = {
        'Transformer': lambda: TransformerBaseline().to(DEVICE),
        'ANIMA-Zero': lambda: ANIMAZeroExact(target_params).to(DEVICE),
        'ANIMA-One': lambda: ANIMAOneExact(target_params).to(DEVICE),
        'ANIMA-Two': lambda: ANIMATwoExact(target_params).to(DEVICE),
        'ANIMA-Synthesis': lambda: ANIMASynthesisExact(target_params).to(DEVICE),
    }

    # Verify param counts
    valid = True
    for name, create_fn in models_specs.items():
        model = create_fn()
        valid &= verify_param_match(model, target_params, name)
        del model

    if not valid:
        print("\n[ERROR] Parameter mismatch!")
        return None

    print("\n[OK] All models have matching parameters!")
    print()

    # Run benchmark
    print("Running benchmark...")
    print("-" * 70)

    results = {
        'target_params': target_params,
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }

    for model_name, create_fn in models_specs.items():
        print(f"\n{model_name}:")
        model_results = {
            'reasoning': {},
            'physics': {},
        }

        for task_name, gen_fn, category in TASKS:
            # Fresh model each task
            model = create_fn()
            acc = train_eval(model, task_name, gen_fn)
            model_results[category][task_name] = acc
            print(f"  {task_name}: {acc*100:.1f}%")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Averages
        model_results['reasoning_avg'] = sum(model_results['reasoning'].values()) / 4
        model_results['physics_avg'] = sum(model_results['physics'].values()) / 4
        model_results['overall'] = (model_results['reasoning_avg'] + model_results['physics_avg']) / 2

        results['models'][model_name] = model_results

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS (ALL MODELS: EXACTLY {target_params:,} PARAMS)")
    print("=" * 70)
    print(f"\n{'Model':<18} {'Reasoning':<12} {'Physics':<12} {'Overall':<12}")
    print("-" * 54)

    tf_overall = results['models']['Transformer']['overall']
    for name, res in results['models'].items():
        diff = (res['overall'] - tf_overall) * 100
        diff_str = f"({diff:+.1f}pp)" if name != 'Transformer' else "(baseline)"
        print(f"{name:<18} {res['reasoning_avg']*100:<11.1f}% {res['physics_avg']*100:<11.1f}% "
              f"{res['overall']*100:<11.1f}% {diff_str}")

    # Find best
    best_name = max(results['models'].keys(), key=lambda k: results['models'][k]['overall'])
    best_overall = results['models'][best_name]['overall']
    print(f"\nBEST: {best_name} with {best_overall*100:.1f}% overall")

    # Save
    output_path = Path(__file__).parent / 'synthesis_benchmark_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_synthesis_benchmark()
