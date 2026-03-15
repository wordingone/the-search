"""
Final Integration Test: Full Quantized Training with All Conditions

Tests the complete hypothesis:
"Fully quantized training for Genesis is plausible only in structured,
sparse, or discrete latent spaces with bounded activations and local propagation"

After sub-problems pass, this test combines ALL conditions:
- Structured: 3D voxel grid
- Sparse: <5% occupancy
- Discrete: FSQ tokenizer
- Bounded: HardTanh activations
- Local: Windowed attention (W=4)

Test Protocol:
1. Train FP32 for 1000 steps (baseline)
2. Train Q8 (fake quant) for 1000 steps
3. Train mixed precision (Q8 weights, FP32 accumulators) for 1000 steps
4. Train INT8 with FP32 sensitive ops (LayerNorm, softmax)

Success Criteria:
| Config | Loss vs Baseline | Verdict |
|--------|------------------|---------|
| FP32 | 1.0x | Baseline |
| Q8 full | < 3x | PASS (hypothesis confirmed) |
| Q8 full | > 10x | FAIL (hypothesis needs refinement) |
| Mixed | < 1.5x | Expected (practical target) |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional
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
    create_sparse_field,
    clear_memory,
)


class FSQ(nn.Module):
    """Finite Scalar Quantization (matches genesis/tokenizer/fsq.py)."""

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))
        self.codebook_size = math.prod(levels)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        assert z.shape[-1] == self.dim
        z_bounded = torch.tanh(z)
        half_levels = (self._levels - 1) / 2
        z_scaled = z_bounded * half_levels + half_levels
        z_quantized = torch.round(z_scaled)
        z_q = z_scaled + (z_quantized - z_scaled).detach()
        indices = self._to_indices(z_quantized)
        return z_q, indices

    def _to_indices(self, z_quantized: Tensor) -> Tensor:
        indices = torch.zeros(z_quantized.shape[:-1], dtype=torch.long, device=z_quantized.device)
        multiplier = 1
        for i in range(self.dim - 1, -1, -1):
            indices = indices + z_quantized[..., i].long() * multiplier
            multiplier = multiplier * self.levels[i]
        return indices


class BoundedFFN(nn.Module):
    """FFN with bounded activations (HardTanh instead of GELU)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.act = nn.Hardtanh()  # Bounded: [-1, 1]

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.act(self.linear1(x)))


class WindowedAttention3D(nn.Module):
    """Windowed 3D attention for local propagation."""

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

        # Window partition
        x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        num_windows = (D // ws) * (H // ws) * (W // ws)
        x = x.view(B * num_windows, ws ** 3, C)

        # Attention
        qkv = self.qkv(x).reshape(-1, ws ** 3, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(min=-1e4, max=1e4)
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn @ v).transpose(1, 2).reshape(-1, ws ** 3, C)
        out = self.proj(out)

        # Window reverse
        out = out.view(B, D // ws, H // ws, W // ws, ws, ws, ws, C)
        out = out.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return out.view(B, D, H, W, C)


class QuantizationFriendlyBlock(nn.Module):
    """
    Transformer block designed for quantization:
    - Windowed attention (local)
    - Bounded activations (HardTanh)
    - LayerNorm (kept in FP32 for mixed precision)
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention3D(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = BoundedFFN(dim, dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SparseEncoder(nn.Module):
    """Encoder that outputs sparse indices and features."""

    def __init__(self, dim: int, hidden_dim: int, fsq_levels: List[int]):
        super().__init__()
        self.fsq_dim = len(fsq_levels)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Hardtanh(),
            nn.Linear(hidden_dim, self.fsq_dim),
        )
        self.fsq = FSQ(fsq_levels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, N, D) sparse features

        Returns:
            z_q: (B, N, fsq_dim) quantized features
            indices: (B, N) discrete codes
        """
        z = self.net(x)
        return self.fsq(z)


class SparseDecoder(nn.Module):
    """Decoder that reconstructs from sparse FSQ latents."""

    def __init__(self, fsq_dim: int, hidden_dim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fsq_dim, hidden_dim),
            nn.Hardtanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_q: Tensor) -> Tensor:
        return self.net(z_q)


class QuantizationFriendlyModel(nn.Module):
    """
    Model satisfying ALL quantization-friendly conditions:
    - Structured: 3D voxel grid
    - Sparse: Operates on sparse indices
    - Discrete: FSQ tokenizer
    - Bounded: HardTanh activations
    - Local: Windowed attention (W=4)
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        fsq_dim = len(config.fsq_levels)

        # Sparse encoder -> discrete latents
        self.encoder = SparseEncoder(config.dim, config.hidden_dim, config.fsq_levels)

        # Project FSQ latents to field channels
        self.fsq_proj = nn.Linear(fsq_dim, config.dim)

        # Windowed propagator
        self.propagator = nn.ModuleList([
            QuantizationFriendlyBlock(config.dim, config.num_heads, config.window_size)
            for _ in range(config.num_layers)
        ])

        # Sparse decoder
        self.decoder = SparseDecoder(config.dim, config.hidden_dim, config.dim)

    def forward(
        self,
        sparse_features: Tensor,
        sparse_positions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            sparse_features: (B, N, D) sparse input features
            sparse_positions: (B, N, 3) positions in 3D grid

        Returns:
            recon: (B, N, D) reconstructed features
            z_q: quantized FSQ latents
            indices: discrete codes
        """
        B, N, D = sparse_features.shape
        G = self.config.field_size

        # Encode to discrete latents
        z_q, indices = self.encoder(sparse_features)

        # Project to field channels
        features = self.fsq_proj(z_q)  # (B, N, dim)

        # Scatter to 3D field for propagation
        pos_int = sparse_positions.long().clamp(0, G - 1)
        linear_idx = (pos_int[:, :, 0] * G * G +
                      pos_int[:, :, 1] * G +
                      pos_int[:, :, 2])

        # Create dense field
        field = torch.zeros(B, G, G, G, D, device=sparse_features.device)
        batch_offset = torch.arange(B, device=field.device).view(B, 1, 1) * (G ** 3)
        flat_idx = (linear_idx + batch_offset.squeeze(-1)).flatten()
        idx_expanded = flat_idx.unsqueeze(-1).expand(-1, D)
        field_flat = field.reshape(B * G ** 3, D)
        field_flat.scatter_add_(0, idx_expanded, features.flatten(0, 1))
        field = field_flat.reshape(B, G, G, G, D)

        # Propagate through windowed transformer
        for block in self.propagator:
            field = block(field)

        # Gather back sparse features
        features_out = torch.zeros(B, N, D, device=field.device)
        field_flat = field.reshape(B * G ** 3, D)
        gathered = field_flat[flat_idx]  # (B*N, D)
        features_out = gathered.view(B, N, D)

        # Decode
        recon = self.decoder(features_out)

        return recon, z_q, indices


def create_sparse_data(config: ExperimentConfig) -> Tuple[Tensor, Tensor]:
    """Create sparse input data."""
    B = config.batch_size
    G = config.field_size
    D = config.dim
    device = config.device

    # ~5% sparsity
    n_active = max(1, int(G ** 3 * config.default_sparsity))

    positions_list = []
    for _ in range(B):
        idx = torch.randperm(G ** 3, device=device)[:n_active]
        z = idx % G
        y = (idx // G) % G
        x = idx // (G * G)
        pos = torch.stack([x, y, z], dim=-1).float()
        positions_list.append(pos)

    positions = torch.stack(positions_list)  # (B, N, 3)
    features = torch.randn(B, n_active, D, device=device)

    return features, positions


def train_step(
    model: QuantizationFriendlyModel,
    optimizer: torch.optim.Optimizer,
    features: Tensor,
    positions: Tensor,
    quant_mode: str = "fp32",
    bits: int = 8,
) -> float:
    """
    Training step with different quantization modes.

    Args:
        quant_mode: "fp32", "q8_full", "mixed", or "q8_sensitive"
    """
    optimizer.zero_grad()

    if quant_mode == "q8_full":
        # Quantize everything
        with torch.no_grad():
            for param in model.parameters():
                param.data = fake_quant(param.data, bits)
        features = fake_quant(features, bits)

    elif quant_mode == "mixed":
        # Quantize weights but keep accumulators FP32
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" not in name:  # Keep LayerNorm in FP32
                    param.data = fake_quant(param.data, bits)

    elif quant_mode == "q8_sensitive":
        # Quantize non-sensitive ops only
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "norm" not in name and "softmax" not in name:
                    param.data = fake_quant(param.data, bits)

    recon, z_q, indices = model(features, positions)

    if quant_mode == "q8_full":
        recon = fake_quant(recon, bits)

    loss = F.mse_loss(recon, features)

    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
        return float("inf")

    loss.backward()

    if quant_mode in ["q8_full"]:
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = fake_quant(param.grad, bits)

    optimizer.step()

    return loss.item()


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the full integration experiment."""
    torch.manual_seed(config.seed)
    clear_memory()

    num_steps = config.num_steps
    fsq_codebook_size = math.prod(config.fsq_levels)

    results = {
        "config": {
            "dim": config.dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "field_size": config.field_size,
            "window_size": config.window_size,
            "fsq_levels": config.fsq_levels,
            "sparsity": config.default_sparsity,
            "num_steps": num_steps,
            "bits": config.bits,
        },
        "modes": {},
    }

    modes = ["fp32", "q8_full", "mixed", "q8_sensitive"]

    for mode in modes:
        print(f"\nTraining {mode.upper()}...")
        model = QuantizationFriendlyModel(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        losses = []
        all_indices = []

        for step in range(num_steps):
            features, positions = create_sparse_data(config)
            loss = train_step(model, optimizer, features, positions, quant_mode=mode, bits=config.bits)
            losses.append(loss)

            # Track codebook usage
            with torch.no_grad():
                _, _, indices = model(features, positions)
                all_indices.append(indices)

            if step % 200 == 0:
                print(f"  Step {step}: loss={loss:.6f}")

            if loss == float("inf"):
                print(f"  DIVERGED at step {step}")
                break

        # Compute metrics
        all_indices = torch.cat(all_indices)
        util = codebook_utilization(all_indices, fsq_codebook_size)

        final_loss = sum(losses[-50:]) / 50 if len(losses) >= 50 else sum(losses) / len(losses)
        diverged = any(l == float("inf") for l in losses)

        results["modes"][mode] = {
            "losses": losses,
            "final_loss": final_loss,
            "codebook_utilization": util,
            "diverged": diverged,
        }

        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Codebook utilization: {util:.2%}")

    return results


def evaluate_hypothesis(results: Dict, thresholds: MetricThresholds) -> Dict:
    """Evaluate against the main hypothesis."""
    fp32_loss = results["modes"]["fp32"]["final_loss"]
    q8_full_loss = results["modes"]["q8_full"]["final_loss"]
    mixed_loss = results["modes"]["mixed"]["final_loss"]

    q8_full_ratio = q8_full_loss / (fp32_loss + 1e-10)
    mixed_ratio = mixed_loss / (fp32_loss + 1e-10)

    # Main hypothesis: Q8 full < 3x baseline
    q8_full_pass = (
        not results["modes"]["q8_full"]["diverged"] and
        q8_full_ratio < thresholds.full_q8_loss_ratio
    )

    # Mixed should be < 1.5x (practical target)
    mixed_pass = (
        not results["modes"]["mixed"]["diverged"] and
        mixed_ratio < 1.5
    )

    # Codebook should not collapse
    codebook_pass = results["modes"]["q8_full"]["codebook_utilization"] > 0.5

    hypothesis_confirmed = q8_full_pass and codebook_pass

    return {
        "hypothesis_confirmed": hypothesis_confirmed,
        "q8_full_ratio": q8_full_ratio,
        "mixed_ratio": mixed_ratio,
        "q8_full_pass": q8_full_pass,
        "mixed_pass": mixed_pass,
        "codebook_pass": codebook_pass,
        "threshold": thresholds.full_q8_loss_ratio,
        "interpretation": (
            "HYPOTHESIS CONFIRMED: Fully quantized training is plausible with all conditions"
            if hypothesis_confirmed else
            "HYPOTHESIS NEEDS REFINEMENT: Full Q8 training did not meet criteria"
        ),
    }


def main():
    """Run the final integration test."""
    config = ExperimentConfig(
        dim=64,
        hidden_dim=256,
        num_heads=4,
        num_layers=4,
        field_size=16,
        window_size=4,
        fsq_levels=[8, 6, 5, 5, 5],
        default_sparsity=0.05,
        batch_size=4,
        num_steps=1000,
        bits=8,
        learning_rate=1e-4,
    )
    thresholds = MetricThresholds()

    print("=" * 70)
    print("FINAL INTEGRATION TEST: Full Quantized Training")
    print("=" * 70)
    print("\nConditions being tested:")
    print("  - Structured: 3D voxel grid (16^3)")
    print("  - Sparse: 5% occupancy")
    print("  - Discrete: FSQ tokenizer [8,6,5,5,5]")
    print("  - Bounded: HardTanh activations")
    print("  - Local: Windowed attention (W=4)")
    print("=" * 70)

    results = run_experiment(config)
    evaluation = evaluate_hypothesis(results, thresholds)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\n| Mode         | Final Loss | Ratio vs FP32 | Verdict |")
    print("|--------------|------------|---------------|---------|")
    for mode in ["fp32", "q8_full", "mixed", "q8_sensitive"]:
        data = results["modes"][mode]
        loss = data["final_loss"]
        ratio = loss / (results["modes"]["fp32"]["final_loss"] + 1e-10)
        status = "BASELINE" if mode == "fp32" else ("DIVERGED" if data["diverged"] else "OK")
        print(f"| {mode:12s} | {loss:10.6f} | {ratio:13.2f}x | {status:7s} |")

    print(f"\nCodebook Utilization: {results['modes']['q8_full']['codebook_utilization']:.2%}")
    print(f"\n{evaluation['interpretation']}")

    # Save results
    results_path = Path(__file__).parent / "results" / "integration_results.json"
    results_path.parent.mkdir(exist_ok=True)

    # Filter for JSON serialization
    serializable = {
        "config": results["config"],
        "summary": {
            mode: {
                "final_loss": data["final_loss"],
                "codebook_utilization": data["codebook_utilization"],
                "diverged": data["diverged"],
            }
            for mode, data in results["modes"].items()
        },
        "evaluation": evaluation,
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results, evaluation


if __name__ == "__main__":
    main()
