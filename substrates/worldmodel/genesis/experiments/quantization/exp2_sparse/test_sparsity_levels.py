"""
Experiment 2: Sparsity - Error Accumulation at Different Sparsity Levels

Question: Does sparsity reduce quantization error accumulation in scatter_add?

Setup:
- Field with controllable sparsity (1%, 5%, 10%, 25%, 50%, 100%)
- Accumulation under different precisions (FP32, FP16, INT8 simulation)

Test Protocol:
1. Create fields with varying occupancy levels
2. Run 1000 scatter_add operations (simulating training)
3. Measure accumulated error vs FP32 reference

Success Criterion:
- Q8 error < 1% at sparsity < 5%
- Q8 error > 10% at sparsity > 50%
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from genesis.experiments.quantization.config import ExperimentConfig, MetricThresholds
from genesis.experiments.quantization.metrics import (
    quantization_error,
    scatter_add_error,
    MetricsTracker,
)
from genesis.experiments.quantization.utils import (
    fake_quant,
    create_sparse_field,
    clear_memory,
)


def measure_scatter_add_accumulation(
    sparsity_pct: float,
    field_size: int,
    channels: int,
    num_iterations: int,
    dtype_test: torch.dtype,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Measure error accumulation in repeated scatter_add operations.

    This simulates the field update loop in Genesis training.
    """
    G = field_size
    B = 1  # Single batch for cleaner measurement

    # Initialize fields
    field_fp32 = torch.zeros(B, channels, G, G, G, device=device, dtype=torch.float32)
    field_test = torch.zeros(B, channels, G, G, G, device=device, dtype=dtype_test)

    errors_per_iter = []

    for iteration in range(num_iterations):
        # Create sparse update
        indices, features, positions = create_sparse_field(
            sparsity_pct=sparsity_pct,
            field_size=G,
            channels=channels,
            batch_size=B,
            device=device,
        )

        # Scatter add pattern from field_model.py
        N = features.shape[1]
        pos_int = positions.long()  # (B, N, 3)

        # Compute linear indices
        linear_idx = (pos_int[:, :, 0] * G * G +
                      pos_int[:, :, 1] * G +
                      pos_int[:, :, 2])  # (B, N)

        batch_offset = torch.arange(B, device=device).view(B, 1) * (G ** 3)
        flat_idx = (linear_idx + batch_offset).flatten()  # (B*N,)

        # Flatten features
        features_flat = features.flatten(0, 1)  # (B*N, C)

        # FP32 accumulation
        field_fp32_flat = field_fp32.permute(0, 2, 3, 4, 1).reshape(B * G**3, channels)
        idx_expanded = flat_idx.unsqueeze(-1).expand(-1, channels)
        field_fp32_flat.scatter_add_(0, idx_expanded, features_flat.float())
        field_fp32 = field_fp32_flat.reshape(B, G, G, G, channels).permute(0, 4, 1, 2, 3)

        # Test dtype accumulation
        field_test_flat = field_test.permute(0, 2, 3, 4, 1).reshape(B * G**3, channels)
        features_test = features_flat.to(dtype_test)
        field_test_flat.scatter_add_(0, idx_expanded, features_test)
        field_test = field_test_flat.reshape(B, G, G, G, channels).permute(0, 4, 1, 2, 3)

        # Measure error every 100 iterations
        if iteration % 100 == 99:
            error = quantization_error(field_fp32, field_test.float(), metric="relative")
            errors_per_iter.append(error)

    return {
        "final_error": quantization_error(field_fp32, field_test.float(), metric="relative"),
        "max_error": quantization_error(field_fp32, field_test.float(), metric="max"),
        "errors_per_iter": errors_per_iter,
        "sparsity_pct": sparsity_pct,
    }


def simulate_int8_scatter_add(
    sparsity_pct: float,
    field_size: int,
    channels: int,
    num_iterations: int,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Simulate INT8 scatter_add using fake quantization.

    Real INT8 scatter_add doesn't exist in PyTorch, so we simulate
    by quantizing inputs/outputs each iteration.
    """
    G = field_size
    B = 1

    # Initialize fields
    field_fp32 = torch.zeros(B, channels, G, G, G, device=device, dtype=torch.float32)
    field_q8 = torch.zeros(B, channels, G, G, G, device=device, dtype=torch.float32)

    errors_per_iter = []

    for iteration in range(num_iterations):
        # Create sparse update
        indices, features, positions = create_sparse_field(
            sparsity_pct=sparsity_pct,
            field_size=G,
            channels=channels,
            batch_size=B,
            device=device,
        )

        N = features.shape[1]
        pos_int = positions.long()

        linear_idx = (pos_int[:, :, 0] * G * G +
                      pos_int[:, :, 1] * G +
                      pos_int[:, :, 2])

        batch_offset = torch.arange(B, device=device).view(B, 1) * (G ** 3)
        flat_idx = (linear_idx + batch_offset).flatten()

        features_flat = features.flatten(0, 1)

        # FP32 accumulation
        field_fp32_flat = field_fp32.permute(0, 2, 3, 4, 1).reshape(B * G**3, channels)
        idx_expanded = flat_idx.unsqueeze(-1).expand(-1, channels)
        field_fp32_flat.scatter_add_(0, idx_expanded, features_flat)
        field_fp32 = field_fp32_flat.reshape(B, G, G, G, channels).permute(0, 4, 1, 2, 3)

        # Q8 simulation: quantize features before scatter, quantize field after
        features_q8 = fake_quant(features_flat, bits=8)
        field_q8_flat = field_q8.permute(0, 2, 3, 4, 1).reshape(B * G**3, channels)
        field_q8_flat.scatter_add_(0, idx_expanded, features_q8)
        field_q8 = field_q8_flat.reshape(B, G, G, G, channels).permute(0, 4, 1, 2, 3)

        # Re-quantize accumulated field (simulates INT8 storage)
        field_q8 = fake_quant(field_q8, bits=8)

        if iteration % 100 == 99:
            error = quantization_error(field_fp32, field_q8, metric="relative")
            errors_per_iter.append(error)

    return {
        "final_error": quantization_error(field_fp32, field_q8, metric="relative"),
        "max_error": quantization_error(field_fp32, field_q8, metric="max"),
        "errors_per_iter": errors_per_iter,
        "sparsity_pct": sparsity_pct,
    }


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the sparsity experiment."""
    torch.manual_seed(config.seed)
    clear_memory()

    sparsity_levels = [1, 5, 10, 25, 50, 100]  # Percentage
    num_iterations = 1000

    results = {
        "config": {
            "field_size": config.field_size,
            "channels": config.dim,
            "num_iterations": num_iterations,
        },
        "fp16_results": {},
        "q8_results": {},
    }

    print("Testing FP16 scatter_add accumulation...")
    for sparsity in sparsity_levels:
        print(f"  Sparsity {sparsity}%...")
        result = measure_scatter_add_accumulation(
            sparsity_pct=sparsity,
            field_size=config.field_size,
            channels=config.dim,
            num_iterations=num_iterations,
            dtype_test=torch.float16,
            device=config.device,
        )
        results["fp16_results"][sparsity] = result
        print(f"    Final error: {result['final_error']:.6f}")

    print("\nTesting Q8 (simulated) scatter_add accumulation...")
    for sparsity in sparsity_levels:
        print(f"  Sparsity {sparsity}%...")
        result = simulate_int8_scatter_add(
            sparsity_pct=sparsity,
            field_size=config.field_size,
            channels=config.dim,
            num_iterations=num_iterations,
            device=config.device,
        )
        results["q8_results"][sparsity] = result
        print(f"    Final error: {result['final_error']:.6f}")

    # Summary
    results["summary"] = {
        "fp16": {str(s): results["fp16_results"][s]["final_error"] for s in sparsity_levels},
        "q8": {str(s): results["q8_results"][s]["final_error"] for s in sparsity_levels},
    }

    return results


def evaluate_hypothesis(results: Dict, thresholds: MetricThresholds) -> Dict:
    """Evaluate experiment results against hypothesis."""
    q8_5pct_error = results["q8_results"][5]["final_error"]
    q8_50pct_error = results["q8_results"][50]["final_error"]

    # Hypothesis: Q8 error < 1% at 5% sparsity, > 10% at 50% sparsity
    sparse_pass = q8_5pct_error < thresholds.sparse_error_threshold
    dense_fail_expected = q8_50pct_error > 0.10

    passed = sparse_pass and dense_fail_expected

    return {
        "passed": passed,
        "q8_error_at_5pct_sparsity": q8_5pct_error,
        "q8_error_at_50pct_sparsity": q8_50pct_error,
        "threshold_sparse": thresholds.sparse_error_threshold,
        "sparse_criterion_met": sparse_pass,
        "dense_fails_as_expected": dense_fail_expected,
        "interpretation": (
            "PASS: Sparsity reduces quantization error accumulation"
            if passed else
            f"MIXED: Sparse={sparse_pass}, Dense fails={dense_fail_expected}"
        ),
    }


def main():
    """Run experiment 2."""
    config = ExperimentConfig(
        dim=64,
        field_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    thresholds = MetricThresholds()

    print("=" * 60)
    print("Experiment 2: Sparsity Levels")
    print("Question: Does sparsity reduce scatter_add error accumulation?")
    print("=" * 60)

    results = run_experiment(config)
    evaluation = evaluate_hypothesis(results, thresholds)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nQ8 Error by Sparsity Level:")
    for sparsity in [1, 5, 10, 25, 50, 100]:
        error = results["q8_results"][sparsity]["final_error"]
        marker = "<-- TARGET" if sparsity == 5 else ""
        print(f"  {sparsity:3d}%: {error:.6f} {marker}")

    print(f"\n{evaluation['interpretation']}")

    # Save results
    results_path = Path(__file__).parent / "results" / "exp2_results.json"
    results_path.parent.mkdir(exist_ok=True)

    # Convert non-serializable items
    serializable_results = {
        "config": results["config"],
        "summary": results["summary"],
        "evaluation": evaluation,
    }
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results, evaluation


if __name__ == "__main__":
    main()
