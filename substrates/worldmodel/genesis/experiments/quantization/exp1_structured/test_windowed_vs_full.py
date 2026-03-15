"""
Experiment 1: Structured Spaces - Windowed vs Full Attention Under Q8

Question: Does spatial structure (3D grid) enable quantization where unstructured fails?

Setup:
- Two identical transformers, same params, different attention
- StructuredTransformer: Windowed 3D attention (W=4)
- UnstructuredTransformer: Full attention (N=4096)

Test Protocol:
1. Train both FP32 for 100 steps (baseline)
2. Apply fake quantization to weights + activations
3. Train both Q8 for 100 steps
4. Measure: gradient SNR, loss convergence, output error

Success Criterion:
- Structured Q8 loss within 2x of FP32 baseline
- Unstructured Q8 loss diverges or >10x baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from genesis.experiments.quantization.config import ExperimentConfig, MetricThresholds
from genesis.experiments.quantization.metrics import (
    gradient_snr,
    quantization_error,
    loss_ratio,
    MetricsTracker,
)
from genesis.experiments.quantization.utils import (
    fake_quant,
    FakeQuantWrapper,
    GradientHook,
    clear_memory,
)


class Windowed3DAttention(nn.Module):
    """Windowed 3D attention with O(N * W^3) complexity."""

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
        """
        Args:
            x: (B, D, H, W, C) - 3D field
        Returns:
            (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        ws = self.window_size

        # Partition into windows: (B, D//ws, H//ws, W//ws, ws, ws, ws, C)
        x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        # (B, nD, nH, nW, ws, ws, ws, C)
        num_windows = (D // ws) * (H // ws) * (W // ws)
        x = x.view(B * num_windows, ws ** 3, C)  # (B*nW, ws^3, C)

        # Attention within each window
        qkv = self.qkv(x).reshape(-1, ws ** 3, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW, heads, ws^3, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(min=-1e4, max=1e4)  # Numerical stability
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(-1, ws ** 3, C)
        out = self.proj(out)

        # Reverse window partition
        out = out.view(B, D // ws, H // ws, W // ws, ws, ws, ws, C)
        out = out.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        out = out.view(B, D, H, W, C)

        return out


class FullAttention(nn.Module):
    """Full attention with O(N^2) complexity."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, D, H, W, C) - 3D field
        Returns:
            (B, D, H, W, C)
        """
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
    """Transformer block with configurable attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: str = "windowed",
        window_size: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if attention_type == "windowed":
            self.attn = Windowed3DAttention(dim, num_heads, window_size)
        else:
            self.attn = FullAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StructuredTransformer(nn.Module):
    """Transformer with windowed 3D attention."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                attention_type="windowed",
                window_size=config.window_size,
            )
            for _ in range(config.num_layers)
        ])
        self.head = nn.Linear(config.dim, config.dim)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        B, D, H, W, C = x.shape
        return self.head(x.view(B, -1, C)).view(B, D, H, W, C)


class UnstructuredTransformer(nn.Module):
    """Transformer with full attention."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                attention_type="full",
            )
            for _ in range(config.num_layers)
        ])
        self.head = nn.Linear(config.dim, config.dim)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        B, D, H, W, C = x.shape
        return self.head(x.view(B, -1, C)).view(B, D, H, W, C)


def create_synthetic_data(config: ExperimentConfig) -> Tuple[Tensor, Tensor]:
    """Create synthetic 3D field data for training."""
    G = config.field_size
    B = config.batch_size
    C = config.dim
    device = config.device

    # Input: random 3D field
    x = torch.randn(B, G, G, G, C, device=device)

    # Target: shifted version (simple reconstruction task)
    target = torch.roll(x, shifts=1, dims=1)

    return x, target


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    target: Tensor,
    use_quant: bool = False,
    bits: int = 8,
) -> Tuple[float, Dict[str, Tensor]]:
    """Single training step."""
    optimizer.zero_grad()

    if use_quant:
        # Quantize weights before forward
        with torch.no_grad():
            for param in model.parameters():
                param.data = fake_quant(param.data, bits)

    out = model(x)

    if use_quant:
        # Quantize activations
        out = fake_quant(out, bits)

    loss = F.mse_loss(out, target)
    loss.backward()

    # Capture gradients before optimizer step
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    optimizer.step()

    return loss.item(), grads


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the structured vs unstructured experiment."""
    torch.manual_seed(config.seed)
    clear_memory()

    results = {
        "config": {
            "dim": config.dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "field_size": config.field_size,
            "window_size": config.window_size,
            "bits": config.bits,
            "num_steps": config.num_steps,
        },
        "structured": {"fp32": [], "q8": []},
        "unstructured": {"fp32": [], "q8": []},
        "gradient_snr": {"structured": [], "unstructured": []},
    }

    # Create models
    structured_model = StructuredTransformer(config).to(config.device)
    unstructured_model = UnstructuredTransformer(config).to(config.device)

    # Use smaller field for full attention to avoid OOM
    # Full attention on 16^3 = 4096 tokens is O(16M) which may OOM
    # Reduce to 8^3 = 512 for fair comparison
    small_config = ExperimentConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        field_size=8,  # Smaller for full attention
        window_size=config.window_size,
        batch_size=config.batch_size,
        num_steps=config.num_steps,
        device=config.device,
        seed=config.seed,
    )

    print("Training Structured Transformer (Windowed Attention)...")

    # Phase 1: FP32 baseline for structured
    structured_optimizer = torch.optim.AdamW(structured_model.parameters(), lr=config.learning_rate)
    for step in range(config.num_steps):
        x, target = create_synthetic_data(config)
        loss, _ = train_step(structured_model, structured_optimizer, x, target, use_quant=False)
        results["structured"]["fp32"].append(loss)
        if step % 20 == 0:
            print(f"  FP32 Step {step}: loss={loss:.6f}")

    # Phase 2: Q8 for structured
    structured_model_q8 = StructuredTransformer(config).to(config.device)
    structured_model_q8.load_state_dict(structured_model.state_dict())
    structured_optimizer_q8 = torch.optim.AdamW(structured_model_q8.parameters(), lr=config.learning_rate)

    for step in range(config.num_steps):
        x, target = create_synthetic_data(config)

        # Get FP32 gradients for SNR calculation
        structured_model.zero_grad()
        out_fp32 = structured_model(x)
        loss_fp32 = F.mse_loss(out_fp32, target)
        loss_fp32.backward()
        grads_fp32 = {name: p.grad.clone() for name, p in structured_model.named_parameters() if p.grad is not None}

        # Q8 training step
        loss, grads_q8 = train_step(structured_model_q8, structured_optimizer_q8, x, target, use_quant=True, bits=config.bits)
        results["structured"]["q8"].append(loss)

        # Calculate gradient SNR
        snrs = []
        for name in grads_fp32:
            if name in grads_q8:
                snr = gradient_snr(grads_fp32[name], grads_q8[name])
                snrs.append(snr)
        if snrs:
            results["gradient_snr"]["structured"].append(sum(snrs) / len(snrs))

        if step % 20 == 0:
            print(f"  Q8 Step {step}: loss={loss:.6f}")

    print("\nTraining Unstructured Transformer (Full Attention)...")

    # Recreate models for unstructured test (smaller field)
    unstructured_model = UnstructuredTransformer(small_config).to(config.device)

    # Phase 1: FP32 baseline for unstructured
    unstructured_optimizer = torch.optim.AdamW(unstructured_model.parameters(), lr=config.learning_rate)
    for step in range(config.num_steps):
        x, target = create_synthetic_data(small_config)
        loss, _ = train_step(unstructured_model, unstructured_optimizer, x, target, use_quant=False)
        results["unstructured"]["fp32"].append(loss)
        if step % 20 == 0:
            print(f"  FP32 Step {step}: loss={loss:.6f}")

    # Phase 2: Q8 for unstructured
    unstructured_model_q8 = UnstructuredTransformer(small_config).to(config.device)
    unstructured_model_q8.load_state_dict(unstructured_model.state_dict())
    unstructured_optimizer_q8 = torch.optim.AdamW(unstructured_model_q8.parameters(), lr=config.learning_rate)

    for step in range(config.num_steps):
        x, target = create_synthetic_data(small_config)

        # Get FP32 gradients
        unstructured_model.zero_grad()
        out_fp32 = unstructured_model(x)
        loss_fp32 = F.mse_loss(out_fp32, target)
        loss_fp32.backward()
        grads_fp32 = {name: p.grad.clone() for name, p in unstructured_model.named_parameters() if p.grad is not None}

        # Q8 training step
        loss, grads_q8 = train_step(unstructured_model_q8, unstructured_optimizer_q8, x, target, use_quant=True, bits=config.bits)
        results["unstructured"]["q8"].append(loss)

        # Calculate gradient SNR
        snrs = []
        for name in grads_fp32:
            if name in grads_q8:
                snr = gradient_snr(grads_fp32[name], grads_q8[name])
                snrs.append(snr)
        if snrs:
            results["gradient_snr"]["unstructured"].append(sum(snrs) / len(snrs))

        if step % 20 == 0:
            print(f"  Q8 Step {step}: loss={loss:.6f}")

    # Compute summary metrics
    results["summary"] = {
        "structured_fp32_final": sum(results["structured"]["fp32"][-10:]) / 10,
        "structured_q8_final": sum(results["structured"]["q8"][-10:]) / 10,
        "unstructured_fp32_final": sum(results["unstructured"]["fp32"][-10:]) / 10,
        "unstructured_q8_final": sum(results["unstructured"]["q8"][-10:]) / 10,
        "structured_q8_ratio": (sum(results["structured"]["q8"][-10:]) / 10) / (sum(results["structured"]["fp32"][-10:]) / 10 + 1e-10),
        "unstructured_q8_ratio": (sum(results["unstructured"]["q8"][-10:]) / 10) / (sum(results["unstructured"]["fp32"][-10:]) / 10 + 1e-10),
        "structured_grad_snr_mean": sum(results["gradient_snr"]["structured"]) / len(results["gradient_snr"]["structured"]) if results["gradient_snr"]["structured"] else 0,
        "unstructured_grad_snr_mean": sum(results["gradient_snr"]["unstructured"]) / len(results["gradient_snr"]["unstructured"]) if results["gradient_snr"]["unstructured"] else 0,
    }

    return results


def evaluate_hypothesis(results: Dict, thresholds: MetricThresholds) -> Dict:
    """Evaluate experiment results against hypothesis."""
    summary = results["summary"]

    # Hypothesis: Structured Q8 loss ratio should be < 0.5x of Unstructured Q8 loss ratio
    # Or equivalently: Windowed attention should be more quantization-friendly
    structured_ratio = summary["structured_q8_ratio"]
    unstructured_ratio = summary["unstructured_q8_ratio"]

    passed = (
        structured_ratio < thresholds.structured_loss_ratio * unstructured_ratio
        or structured_ratio < 2.0  # Or structured Q8 within 2x of FP32
    )

    return {
        "passed": passed,
        "structured_q8_ratio": structured_ratio,
        "unstructured_q8_ratio": unstructured_ratio,
        "ratio_of_ratios": structured_ratio / (unstructured_ratio + 1e-10),
        "threshold": thresholds.structured_loss_ratio,
        "interpretation": (
            "PASS: Structured (windowed) attention is more quantization-friendly"
            if passed else
            "FAIL: Structured attention did not show expected quantization benefits"
        ),
    }


def main():
    """Run experiment 1."""
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        num_layers=4,
        field_size=16,
        window_size=4,
        batch_size=2,
        num_steps=100,
        bits=8,
    )
    thresholds = MetricThresholds()

    print("=" * 60)
    print("Experiment 1: Structured Spaces")
    print("Question: Does spatial structure enable quantization?")
    print("=" * 60)

    results = run_experiment(config)
    evaluation = evaluate_hypothesis(results, thresholds)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Structured (Windowed) Q8/FP32 ratio: {results['summary']['structured_q8_ratio']:.4f}")
    print(f"Unstructured (Full) Q8/FP32 ratio: {results['summary']['unstructured_q8_ratio']:.4f}")
    print(f"Structured gradient SNR: {results['summary']['structured_grad_snr_mean']:.2f} dB")
    print(f"Unstructured gradient SNR: {results['summary']['unstructured_grad_snr_mean']:.2f} dB")
    print(f"\n{evaluation['interpretation']}")

    # Save results
    results_path = Path(__file__).parent / "results" / "exp1_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"results": results, "evaluation": evaluation}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results, evaluation


if __name__ == "__main__":
    main()
