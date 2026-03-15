"""
Experiment 4: Bounded Activations - Sigmoid/Tanh vs GELU/ReLU

Question: Do bounded activations (sigmoid, tanh) enable quantization where unbounded (GELU) fails?

Setup:
- Minimal FFN with different activations
- Same architecture, different activation functions

Test Protocol:
1. Train each activation FP32 for 500 steps
2. Apply fake quantization, train Q8 for 500 steps
3. Measure activation range, gradient magnitude, loss

Success Criterion:
- Bounded (sigmoid, tanh, hardtanh): Q8 loss < 2x FP32
- Unbounded (GELU, ReLU): Q8 loss > 5x FP32 or diverges
"""

import torch
import torch.nn as nn
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
    activation_stats,
    loss_ratio,
    MetricsTracker,
)
from genesis.experiments.quantization.utils import (
    fake_quant,
    clear_memory,
)


ACTIVATION_FUNCTIONS = {
    "gelu": nn.GELU(),           # Unbounded
    "relu": nn.ReLU(),           # Unbounded positive
    "sigmoid": nn.Sigmoid(),     # [0, 1]
    "tanh": nn.Tanh(),           # [-1, 1]
    "hardtanh": nn.Hardtanh(),   # [-1, 1], piecewise linear
}


class TestFFN(nn.Module):
    """Minimal FFN for testing activation functions."""

    def __init__(self, dim: int, hidden_dim: int, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.act = ACTIVATION_FUNCTIONS[activation]
        self.activation_name = activation

        # Store activations for analysis
        self.last_activations = None

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear1(x)
        h = self.act(h)
        self.last_activations = h.detach()  # Store for analysis
        return self.linear2(h)


class TestModel(nn.Module):
    """Multi-layer model with configurable activation."""

    def __init__(self, config: ExperimentConfig, activation: str):
        super().__init__()
        self.layers = nn.ModuleList([
            TestFFN(config.dim, config.hidden_dim, activation)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = x + layer(self.norm(x))
        return x

    def get_activation_stats(self) -> List[Dict[str, float]]:
        """Get activation statistics from all layers."""
        stats = []
        for layer in self.layers:
            if layer.last_activations is not None:
                stats.append(activation_stats(layer.last_activations))
        return stats


def train_step(
    model: TestModel,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    target: Tensor,
    use_quant: bool = False,
    bits: int = 8,
) -> Tuple[float, List[Dict[str, float]]]:
    """Single training step with optional quantization."""
    optimizer.zero_grad()

    if use_quant:
        # Quantize weights
        with torch.no_grad():
            for param in model.parameters():
                param.data = fake_quant(param.data, bits)
        # Quantize input
        x = fake_quant(x, bits)

    out = model(x)

    if use_quant:
        out = fake_quant(out, bits)

    loss = F.mse_loss(out, target)

    # Check for divergence
    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
        return float("inf"), []

    loss.backward()

    if use_quant:
        # Quantize gradients
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = fake_quant(param.grad, bits)

    optimizer.step()

    return loss.item(), model.get_activation_stats()


def create_synthetic_data(config: ExperimentConfig) -> Tuple[Tensor, Tensor]:
    """Create synthetic data for training."""
    B = config.batch_size
    D = config.dim
    device = config.device

    x = torch.randn(B, D, device=device)
    # Target: nonlinear transformation
    target = torch.sin(x * 2) + torch.cos(x)

    return x, target


def run_single_activation_test(
    activation: str,
    config: ExperimentConfig,
) -> Dict:
    """Test a single activation function."""
    torch.manual_seed(config.seed)

    results = {
        "activation": activation,
        "fp32": {"losses": [], "activation_ranges": []},
        "q8": {"losses": [], "activation_ranges": []},
    }

    # Phase 1: FP32 training
    model = TestModel(config, activation).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for step in range(config.num_steps):
        x, target = create_synthetic_data(config)
        loss, act_stats = train_step(model, optimizer, x, target, use_quant=False)
        results["fp32"]["losses"].append(loss)
        if act_stats:
            results["fp32"]["activation_ranges"].append(
                (act_stats[0]["min"], act_stats[0]["max"])
            )

    # Phase 2: Q8 training
    model_q8 = TestModel(config, activation).to(config.device)
    model_q8.load_state_dict(model.state_dict())
    optimizer_q8 = torch.optim.AdamW(model_q8.parameters(), lr=config.learning_rate)

    diverged = False
    for step in range(config.num_steps):
        x, target = create_synthetic_data(config)
        loss, act_stats = train_step(model_q8, optimizer_q8, x, target, use_quant=True, bits=config.bits)

        if loss == float("inf"):
            diverged = True
            results["q8"]["losses"].append(1e6)  # Mark as diverged
        else:
            results["q8"]["losses"].append(loss)
            if act_stats:
                results["q8"]["activation_ranges"].append(
                    (act_stats[0]["min"], act_stats[0]["max"])
                )

    # Compute summary
    fp32_final = sum(results["fp32"]["losses"][-20:]) / 20
    q8_final = sum(results["q8"]["losses"][-20:]) / 20 if not diverged else float("inf")

    results["summary"] = {
        "fp32_final_loss": fp32_final,
        "q8_final_loss": q8_final,
        "q8_fp32_ratio": q8_final / (fp32_final + 1e-10) if not diverged else float("inf"),
        "diverged": diverged,
        "bounded": activation in ["sigmoid", "tanh", "hardtanh"],
    }

    # Get typical activation range
    if results["fp32"]["activation_ranges"]:
        ranges = results["fp32"]["activation_ranges"]
        results["summary"]["typical_act_min"] = sum(r[0] for r in ranges[-10:]) / 10
        results["summary"]["typical_act_max"] = sum(r[1] for r in ranges[-10:]) / 10

    return results


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the bounded activation experiment."""
    clear_memory()

    results = {
        "config": {
            "dim": config.dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_steps": config.num_steps,
            "bits": config.bits,
        },
        "activations": {},
    }

    for activation in ACTIVATION_FUNCTIONS.keys():
        print(f"\nTesting {activation.upper()}...")
        act_results = run_single_activation_test(activation, config)
        results["activations"][activation] = act_results

        summary = act_results["summary"]
        if summary["diverged"]:
            print(f"  DIVERGED")
        else:
            print(f"  FP32 loss: {summary['fp32_final_loss']:.6f}")
            print(f"  Q8 loss: {summary['q8_final_loss']:.6f}")
            print(f"  Ratio: {summary['q8_fp32_ratio']:.2f}x")
            if "typical_act_min" in summary:
                print(f"  Activation range: [{summary['typical_act_min']:.2f}, {summary['typical_act_max']:.2f}]")

    return results


def evaluate_hypothesis(results: Dict, thresholds: MetricThresholds) -> Dict:
    """Evaluate experiment results against hypothesis."""
    bounded_acts = ["sigmoid", "tanh", "hardtanh"]
    unbounded_acts = ["gelu", "relu"]

    bounded_results = []
    unbounded_results = []

    for act, data in results["activations"].items():
        summary = data["summary"]
        ratio = summary["q8_fp32_ratio"]
        diverged = summary["diverged"]

        if act in bounded_acts:
            bounded_results.append({
                "activation": act,
                "ratio": ratio,
                "diverged": diverged,
                "pass": ratio < thresholds.bounded_loss_ratio and not diverged,
            })
        else:
            unbounded_results.append({
                "activation": act,
                "ratio": ratio,
                "diverged": diverged,
                # For unbounded, we EXPECT failure (divergence or high ratio)
                "fails_as_expected": diverged or ratio > 5.0,
            })

    bounded_pass = all(r["pass"] for r in bounded_results)
    unbounded_fail = all(r["fails_as_expected"] for r in unbounded_results)

    passed = bounded_pass and unbounded_fail

    return {
        "passed": passed,
        "bounded_results": bounded_results,
        "unbounded_results": unbounded_results,
        "bounded_pass": bounded_pass,
        "unbounded_fail_as_expected": unbounded_fail,
        "threshold": thresholds.bounded_loss_ratio,
        "interpretation": (
            "PASS: Bounded activations enable Q8, unbounded fail as expected"
            if passed else
            f"MIXED: bounded_pass={bounded_pass}, unbounded_fail={unbounded_fail}"
        ),
    }


def main():
    """Run experiment 4."""
    config = ExperimentConfig(
        dim=64,
        hidden_dim=256,
        num_layers=4,
        batch_size=32,
        num_steps=500,
        bits=8,
        learning_rate=1e-4,
    )
    thresholds = MetricThresholds()

    print("=" * 60)
    print("Experiment 4: Bounded Activations")
    print("Question: Do bounded activations enable Q8 training?")
    print("=" * 60)

    results = run_experiment(config)
    evaluation = evaluate_hypothesis(results, thresholds)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nBounded Activations (should PASS):")
    for r in evaluation["bounded_results"]:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['activation']:10s}: ratio={r['ratio']:.2f}x [{status}]")

    print("\nUnbounded Activations (should FAIL):")
    for r in evaluation["unbounded_results"]:
        status = "FAIL (expected)" if r["fails_as_expected"] else "PASS (unexpected)"
        ratio_str = "diverged" if r["diverged"] else f"ratio={r['ratio']:.2f}x"
        print(f"  {r['activation']:10s}: {ratio_str} [{status}]")

    print(f"\n{evaluation['interpretation']}")

    # Save results
    results_path = Path(__file__).parent / "results" / "exp4_results.json"
    results_path.parent.mkdir(exist_ok=True)

    # Filter out non-serializable items
    serializable = {
        "config": results["config"],
        "summary": {
            act: data["summary"]
            for act, data in results["activations"].items()
        },
        "evaluation": evaluation,
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results, evaluation


if __name__ == "__main__":
    main()
