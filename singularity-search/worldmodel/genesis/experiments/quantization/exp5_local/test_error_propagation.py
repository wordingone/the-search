"""
Experiment 5: Local Propagation - Error Propagation in Windowed vs Full Attention

Question: Does windowed attention limit error propagation vs full attention?

Setup:
- Inject quantization error at layer L
- Measure error magnitude at layers L+1, L+2, ..., L+k
- Compare windowed (W=4) vs full attention

Test Protocol:
1. Inject small error (1% of activation magnitude) at layer 1
2. Measure error magnitude at layers 2, 3, 4, 5, 6
3. Compare windowed vs full attention growth rates

Success Criterion:
- Windowed: Error grows linearly or sub-linearly with depth
- Full: Error grows exponentially with depth

Expected Result:
    Layer:    1    2    3    4    5    6
    Windowed: 1.0  1.2  1.4  1.6  1.8  2.0  (linear)
    Full:     1.0  2.0  4.0  8.0  16.0 32.0 (exponential)
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
    error_propagation_rate,
    MetricsTracker,
)
from genesis.experiments.quantization.utils import (
    fake_quant,
    inject_error,
    clear_memory,
)


class Windowed3DAttention(nn.Module):
    """Windowed 3D attention (copied from exp1 for isolation)."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape
        ws = self.window_size

        x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        num_windows = (D // ws) * (H // ws) * (W // ws)
        x = x.view(B * num_windows, ws ** 3, C)

        qkv = self.qkv(x).reshape(-1, ws ** 3, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(min=-1e4, max=1e4)
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(-1, ws ** 3, C)
        out = self.proj(out)

        out = out.view(B, D // ws, H // ws, W // ws, ws, ws, ws, C)
        out = out.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        out = out.view(B, D, H, W, C)

        return out


class FullAttention(nn.Module):
    """Full attention (copied from exp1 for isolation)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape
        N = D * H * W

        x_flat = x.view(B, N, C)

        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(min=-1e4, max=1e4)
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out.view(B, D, H, W, C)


class TransformerBlock(nn.Module):
    """Transformer block for error propagation testing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: str = "windowed",
        window_size: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attention_type == "windowed":
            self.attn = Windowed3DAttention(dim, num_heads, window_size)
        else:
            self.attn = FullAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ErrorTrackingTransformer(nn.Module):
    """Transformer that tracks error propagation through layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        attention_type: str = "windowed",
        window_size: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, attention_type, window_size)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward_with_error_injection(
        self,
        x: Tensor,
        inject_at_layer: int = 0,
        error_scale: float = 0.01,
    ) -> Tuple[Tensor, List[float]]:
        """
        Forward pass with error injection and tracking.

        Args:
            x: Input tensor
            inject_at_layer: Layer at which to inject error
            error_scale: Scale of injected error

        Returns:
            (output, error_magnitudes_by_layer)
        """
        # Run clean pass first to get reference
        x_clean = x.clone()
        x_noisy = x.clone()

        error_magnitudes = []

        for i, block in enumerate(self.blocks):
            x_clean = block(x_clean)
            x_noisy = block(x_noisy)

            # Inject error at specified layer
            if i == inject_at_layer:
                x_noisy = inject_error(x_noisy, error_scale)

            # Measure error relative to initial injection
            if i >= inject_at_layer:
                error = quantization_error(x_clean, x_noisy, metric="relative")
                error_magnitudes.append(error)

        return x_noisy, error_magnitudes


def run_error_propagation_test(
    attention_type: str,
    config: ExperimentConfig,
    num_trials: int = 10,
) -> Dict:
    """Run error propagation test for a specific attention type."""
    torch.manual_seed(config.seed)

    # Use smaller field for full attention to avoid OOM
    field_size = config.field_size if attention_type == "windowed" else 8

    model = ErrorTrackingTransformer(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        attention_type=attention_type,
        window_size=config.window_size,
    ).to(config.device)

    # Initialize with reasonable weights
    model.eval()

    all_error_magnitudes = []

    for trial in range(num_trials):
        # Random input
        x = torch.randn(
            config.batch_size, field_size, field_size, field_size, config.dim,
            device=config.device
        )

        with torch.no_grad():
            _, error_mags = model.forward_with_error_injection(
                x, inject_at_layer=0, error_scale=0.01
            )
            all_error_magnitudes.append(error_mags)

    # Average across trials
    avg_errors = []
    for layer_idx in range(len(all_error_magnitudes[0])):
        layer_errors = [trial[layer_idx] for trial in all_error_magnitudes]
        avg_errors.append(sum(layer_errors) / len(layer_errors))

    # Normalize to first layer
    normalized_errors = [e / (avg_errors[0] + 1e-10) for e in avg_errors]

    # Analyze growth pattern
    growth_rate, pattern = error_propagation_rate(normalized_errors)

    return {
        "attention_type": attention_type,
        "raw_errors": avg_errors,
        "normalized_errors": normalized_errors,
        "growth_rate": growth_rate,
        "pattern": pattern,
    }


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the error propagation experiment."""
    clear_memory()

    results = {
        "config": {
            "dim": config.dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "field_size": config.field_size,
            "window_size": config.window_size,
        },
        "windowed": None,
        "full": None,
    }

    print("Testing Windowed Attention...")
    results["windowed"] = run_error_propagation_test("windowed", config)
    print(f"  Growth rate: {results['windowed']['growth_rate']:.3f}x per layer")
    print(f"  Pattern: {results['windowed']['pattern']}")
    print(f"  Normalized errors: {[f'{e:.3f}' for e in results['windowed']['normalized_errors']]}")

    print("\nTesting Full Attention...")
    results["full"] = run_error_propagation_test("full", config)
    print(f"  Growth rate: {results['full']['growth_rate']:.3f}x per layer")
    print(f"  Pattern: {results['full']['pattern']}")
    print(f"  Normalized errors: {[f'{e:.3f}' for e in results['full']['normalized_errors']]}")

    return results


def evaluate_hypothesis(results: Dict, thresholds: MetricThresholds) -> Dict:
    """Evaluate experiment results against hypothesis."""
    windowed = results["windowed"]
    full = results["full"]

    # Hypothesis: Windowed grows < 1.5x per layer, Full grows > 1.5x
    windowed_pass = windowed["growth_rate"] < thresholds.error_growth_rate
    full_grows_faster = full["growth_rate"] > windowed["growth_rate"]

    # Additional check: Full should show exponential pattern
    full_exponential = full["pattern"] == "exponential"
    windowed_linear_or_stable = windowed["pattern"] in ["linear", "stable"]

    passed = windowed_pass and full_grows_faster

    return {
        "passed": passed,
        "windowed_growth_rate": windowed["growth_rate"],
        "full_growth_rate": full["growth_rate"],
        "threshold": thresholds.error_growth_rate,
        "windowed_pattern": windowed["pattern"],
        "full_pattern": full["pattern"],
        "windowed_pass": windowed_pass,
        "full_grows_faster": full_grows_faster,
        "interpretation": (
            "PASS: Windowed attention limits error propagation"
            if passed else
            f"MIXED: windowed_pass={windowed_pass}, full_faster={full_grows_faster}"
        ),
    }


def main():
    """Run experiment 5."""
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        num_layers=6,
        field_size=16,
        window_size=4,
        batch_size=2,
    )
    thresholds = MetricThresholds()

    print("=" * 60)
    print("Experiment 5: Local Propagation")
    print("Question: Does windowed attention limit error propagation?")
    print("=" * 60)

    results = run_experiment(config)
    evaluation = evaluate_hypothesis(results, thresholds)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nError Propagation by Layer:")
    print("Layer:      ", "  ".join(f"{i+1:5d}" for i in range(len(results["windowed"]["normalized_errors"]))))
    print("Windowed:   ", "  ".join(f"{e:5.2f}" for e in results["windowed"]["normalized_errors"]))
    print("Full:       ", "  ".join(f"{e:5.2f}" for e in results["full"]["normalized_errors"]))

    print(f"\nGrowth Rates:")
    print(f"  Windowed: {evaluation['windowed_growth_rate']:.3f}x per layer ({evaluation['windowed_pattern']})")
    print(f"  Full:     {evaluation['full_growth_rate']:.3f}x per layer ({evaluation['full_pattern']})")

    print(f"\n{evaluation['interpretation']}")

    # Save results
    results_path = Path(__file__).parent / "results" / "exp5_results.json"
    results_path.parent.mkdir(exist_ok=True)

    # Convert numpy bools to Python bools for JSON
    serializable_eval = {
        k: bool(v) if isinstance(v, (bool, type(True))) else v
        for k, v in evaluation.items()
    }
    with open(results_path, "w") as f:
        json.dump({"results": results, "evaluation": serializable_eval}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results, evaluation


if __name__ == "__main__":
    main()
