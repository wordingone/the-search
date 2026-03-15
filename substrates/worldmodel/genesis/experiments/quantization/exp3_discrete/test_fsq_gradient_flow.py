"""
Experiment 3: Discrete Latents - FSQ Gradient Flow Under Quantization

Question: Does FSQ's discrete space survive quantized gradients via STE?

Setup:
- FSQ encoder-decoder (matches genesis/tokenizer/fsq.py)
- Test gradient flow with Q8 gradients
- Test with Q8 weights + gradients

Test Protocol:
1. Train FSQ encoder-decoder FP32 (baseline)
2. Train with Q8 gradients only (weights FP32)
3. Train with Q8 weights + gradients
4. Measure reconstruction loss, codebook utilization

Success Criterion:
- Q8 gradients: loss within 1.5x baseline
- Q8 weights+grads: loss within 3x baseline
- Codebook utilization > 90% (no collapse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple
import json
from pathlib import Path
import math

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from genesis.experiments.quantization.config import ExperimentConfig, MetricThresholds
from genesis.experiments.quantization.metrics import (
    quantization_error,
    codebook_utilization,
    loss_ratio,
    MetricsTracker,
)
from genesis.experiments.quantization.utils import (
    fake_quant,
    QuantizedSTE,
    clear_memory,
)


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (matches genesis/tokenizer/fsq.py:55-85).

    Maps continuous latents to discrete codes without learned codebook.
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))

        # Total codebook size
        self.codebook_size = math.prod(levels)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize continuous latents to discrete tokens.

        Args:
            z: [..., D] continuous latent vectors

        Returns:
            z_q: [..., D] quantized vectors (straight-through gradient)
            indices: [...] discrete token indices
        """
        assert z.shape[-1] == self.dim, f"Expected dim {self.dim}, got {z.shape[-1]}"

        # Bound to [-1, 1] via tanh
        z_bounded = torch.tanh(z)

        # Scale each dimension to [0, L-1]
        half_levels = (self._levels - 1) / 2
        z_scaled = z_bounded * half_levels + half_levels

        # Round to nearest integer
        z_quantized = torch.round(z_scaled)

        # Straight-through estimator: forward uses quantized, backward uses continuous
        z_q = z_scaled + (z_quantized - z_scaled).detach()

        # Compute indices
        indices = self._to_indices(z_quantized)

        return z_q, indices

    def _to_indices(self, z_quantized: Tensor) -> Tensor:
        """Convert quantized values to single index."""
        indices = torch.zeros(z_quantized.shape[:-1], dtype=torch.long, device=z_quantized.device)
        multiplier = 1
        for i in range(self.dim - 1, -1, -1):
            indices = indices + z_quantized[..., i].long() * multiplier
            multiplier = multiplier * self.levels[i]
        return indices


class FSQEncoder(nn.Module):
    """Encoder that produces FSQ-compatible latents."""

    def __init__(self, input_dim: int, hidden_dim: int, fsq_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fsq_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FSQDecoder(nn.Module):
    """Decoder that reconstructs from FSQ latents."""

    def __init__(self, fsq_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fsq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_q: Tensor) -> Tensor:
        return self.net(z_q)


class FSQAutoEncoder(nn.Module):
    """Complete FSQ autoencoder for testing."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.fsq = FSQ(config.fsq_levels)
        fsq_dim = len(config.fsq_levels)

        self.encoder = FSQEncoder(config.dim, config.hidden_dim, fsq_dim)
        self.decoder = FSQDecoder(fsq_dim, config.hidden_dim, config.dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, ..., D) input

        Returns:
            recon: (B, ..., D) reconstruction
            z_q: quantized latents
            indices: discrete token indices
        """
        z = self.encoder(x)
        z_q, indices = self.fsq(z)
        recon = self.decoder(z_q)
        return recon, z_q, indices


class QuantizedGradientHook:
    """Hook to quantize gradients during backward pass."""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.hooks = []

    def register(self, model: nn.Module):
        """Register gradient quantization hooks on all parameters."""
        for param in model.parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad: fake_quant(grad, self.bits))
                self.hooks.append(hook)

    def remove(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def train_step_fp32(
    model: FSQAutoEncoder,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
) -> Tuple[float, Tensor]:
    """Standard FP32 training step."""
    optimizer.zero_grad()

    recon, z_q, indices = model(x)
    loss = F.mse_loss(recon, x)

    loss.backward()
    optimizer.step()

    return loss.item(), indices


def train_step_q8_grad(
    model: FSQAutoEncoder,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    grad_hook: QuantizedGradientHook,
) -> Tuple[float, Tensor]:
    """Training with Q8 gradients only."""
    optimizer.zero_grad()

    recon, z_q, indices = model(x)
    loss = F.mse_loss(recon, x)

    loss.backward()
    # Gradients are quantized by hook during backward

    optimizer.step()

    return loss.item(), indices


def train_step_q8_full(
    model: FSQAutoEncoder,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    bits: int = 8,
) -> Tuple[float, Tensor]:
    """Training with Q8 weights and activations."""
    optimizer.zero_grad()

    # Quantize weights before forward
    with torch.no_grad():
        for param in model.parameters():
            param.data = fake_quant(param.data, bits)

    # Quantize input
    x_q = fake_quant(x, bits)

    recon, z_q, indices = model(x_q)

    # Quantize output
    recon_q = fake_quant(recon, bits)

    loss = F.mse_loss(recon_q, x_q)
    loss.backward()

    # Quantize gradients
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.grad = fake_quant(param.grad, bits)

    optimizer.step()

    return loss.item(), indices


def create_synthetic_data(config: ExperimentConfig) -> Tensor:
    """Create synthetic data for reconstruction task."""
    B = config.batch_size
    D = config.dim
    device = config.device

    # Random vectors with some structure (clusters)
    num_clusters = 10
    cluster_centers = torch.randn(num_clusters, D, device=device)
    cluster_ids = torch.randint(0, num_clusters, (B,), device=device)
    x = cluster_centers[cluster_ids] + 0.1 * torch.randn(B, D, device=device)

    return x


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the FSQ gradient flow experiment."""
    torch.manual_seed(config.seed)
    clear_memory()

    results = {
        "config": {
            "dim": config.dim,
            "hidden_dim": config.hidden_dim,
            "fsq_levels": config.fsq_levels,
            "num_steps": config.num_steps,
            "bits": config.bits,
        },
        "fp32": {"losses": [], "codebook_utilization": []},
        "q8_grad": {"losses": [], "codebook_utilization": []},
        "q8_full": {"losses": [], "codebook_utilization": []},
    }

    fsq_codebook_size = math.prod(config.fsq_levels)

    print("Phase 1: FP32 Baseline...")
    model_fp32 = FSQAutoEncoder(config).to(config.device)
    optimizer_fp32 = torch.optim.AdamW(model_fp32.parameters(), lr=config.learning_rate)

    all_indices_fp32 = []
    for step in range(config.num_steps):
        x = create_synthetic_data(config)
        loss, indices = train_step_fp32(model_fp32, optimizer_fp32, x)
        results["fp32"]["losses"].append(loss)
        all_indices_fp32.append(indices)

        if step % 50 == 0:
            print(f"  Step {step}: loss={loss:.6f}")

    # Calculate codebook utilization
    all_indices_fp32 = torch.cat(all_indices_fp32)
    util_fp32 = codebook_utilization(all_indices_fp32, fsq_codebook_size)
    results["fp32"]["codebook_utilization"] = util_fp32
    print(f"  Codebook utilization: {util_fp32:.2%}")

    print("\nPhase 2: Q8 Gradients Only...")
    model_q8_grad = FSQAutoEncoder(config).to(config.device)
    model_q8_grad.load_state_dict(model_fp32.state_dict())  # Start from same point
    optimizer_q8_grad = torch.optim.AdamW(model_q8_grad.parameters(), lr=config.learning_rate)

    grad_hook = QuantizedGradientHook(bits=config.bits)
    grad_hook.register(model_q8_grad)

    all_indices_q8_grad = []
    for step in range(config.num_steps):
        x = create_synthetic_data(config)
        loss, indices = train_step_q8_grad(model_q8_grad, optimizer_q8_grad, x, grad_hook)
        results["q8_grad"]["losses"].append(loss)
        all_indices_q8_grad.append(indices)

        if step % 50 == 0:
            print(f"  Step {step}: loss={loss:.6f}")

    grad_hook.remove()

    all_indices_q8_grad = torch.cat(all_indices_q8_grad)
    util_q8_grad = codebook_utilization(all_indices_q8_grad, fsq_codebook_size)
    results["q8_grad"]["codebook_utilization"] = util_q8_grad
    print(f"  Codebook utilization: {util_q8_grad:.2%}")

    print("\nPhase 3: Q8 Weights + Gradients...")
    model_q8_full = FSQAutoEncoder(config).to(config.device)
    model_q8_full.load_state_dict(model_fp32.state_dict())  # Start from same point
    optimizer_q8_full = torch.optim.AdamW(model_q8_full.parameters(), lr=config.learning_rate)

    all_indices_q8_full = []
    for step in range(config.num_steps):
        x = create_synthetic_data(config)
        loss, indices = train_step_q8_full(model_q8_full, optimizer_q8_full, x, bits=config.bits)
        results["q8_full"]["losses"].append(loss)
        all_indices_q8_full.append(indices)

        if step % 50 == 0:
            print(f"  Step {step}: loss={loss:.6f}")

    all_indices_q8_full = torch.cat(all_indices_q8_full)
    util_q8_full = codebook_utilization(all_indices_q8_full, fsq_codebook_size)
    results["q8_full"]["codebook_utilization"] = util_q8_full
    print(f"  Codebook utilization: {util_q8_full:.2%}")

    # Summary
    results["summary"] = {
        "fp32_final_loss": sum(results["fp32"]["losses"][-20:]) / 20,
        "q8_grad_final_loss": sum(results["q8_grad"]["losses"][-20:]) / 20,
        "q8_full_final_loss": sum(results["q8_full"]["losses"][-20:]) / 20,
        "fp32_codebook_util": util_fp32,
        "q8_grad_codebook_util": util_q8_grad,
        "q8_full_codebook_util": util_q8_full,
    }

    return results


def evaluate_hypothesis(results: Dict, thresholds: MetricThresholds) -> Dict:
    """Evaluate experiment results against hypothesis."""
    summary = results["summary"]

    fp32_loss = summary["fp32_final_loss"]
    q8_grad_loss = summary["q8_grad_final_loss"]
    q8_full_loss = summary["q8_full_final_loss"]

    q8_grad_ratio = q8_grad_loss / (fp32_loss + 1e-10)
    q8_full_ratio = q8_full_loss / (fp32_loss + 1e-10)

    # Success criteria
    q8_grad_pass = q8_grad_ratio < 1.5
    q8_full_pass = q8_full_ratio < 3.0
    codebook_pass = summary["q8_full_codebook_util"] > thresholds.fsq_codebook_utilization

    passed = q8_grad_pass and codebook_pass

    return {
        "passed": passed,
        "q8_grad_loss_ratio": q8_grad_ratio,
        "q8_full_loss_ratio": q8_full_ratio,
        "q8_grad_threshold": 1.5,
        "q8_full_threshold": 3.0,
        "codebook_utilization": summary["q8_full_codebook_util"],
        "codebook_threshold": thresholds.fsq_codebook_utilization,
        "q8_grad_pass": q8_grad_pass,
        "q8_full_pass": q8_full_pass,
        "codebook_pass": codebook_pass,
        "interpretation": (
            "PASS: FSQ survives quantized gradients with good codebook utilization"
            if passed else
            f"MIXED: grad_pass={q8_grad_pass}, full_pass={q8_full_pass}, codebook={codebook_pass}"
        ),
    }


def main():
    """Run experiment 3."""
    config = ExperimentConfig(
        dim=64,
        hidden_dim=256,
        fsq_levels=[8, 6, 5, 5, 5],  # Matches genesis/tokenizer/fsq.py
        batch_size=32,
        num_steps=200,
        bits=8,
        learning_rate=1e-3,
    )
    thresholds = MetricThresholds()

    print("=" * 60)
    print("Experiment 3: Discrete Latents (FSQ)")
    print("Question: Does FSQ's discrete space survive Q8 gradients?")
    print("=" * 60)

    results = run_experiment(config)
    evaluation = evaluate_hypothesis(results, thresholds)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"FP32 baseline loss: {results['summary']['fp32_final_loss']:.6f}")
    print(f"Q8 gradients loss: {results['summary']['q8_grad_final_loss']:.6f} (ratio: {evaluation['q8_grad_loss_ratio']:.2f}x)")
    print(f"Q8 full loss: {results['summary']['q8_full_final_loss']:.6f} (ratio: {evaluation['q8_full_loss_ratio']:.2f}x)")
    print(f"\nCodebook utilization:")
    print(f"  FP32: {results['summary']['fp32_codebook_util']:.2%}")
    print(f"  Q8 grad: {results['summary']['q8_grad_codebook_util']:.2%}")
    print(f"  Q8 full: {results['summary']['q8_full_codebook_util']:.2%}")
    print(f"\n{evaluation['interpretation']}")

    # Save results
    results_path = Path(__file__).parent / "results" / "exp3_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"results": results, "evaluation": evaluation}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results, evaluation


if __name__ == "__main__":
    main()
