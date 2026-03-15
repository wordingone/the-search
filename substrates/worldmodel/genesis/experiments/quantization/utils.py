"""
Quantization utilities for hypothesis testing.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import gc


def fake_quant(
    x: Tensor,
    bits: int = 8,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> Tensor:
    """
    Fake quantization: quantize then dequantize.

    Simulates INT8 quantization without actually using INT8 storage.
    Used for training with quantization-aware behavior.

    Args:
        x: Input tensor
        bits: Number of bits (default 8)
        symmetric: Use symmetric quantization (default True)
        per_channel: Quantize per-channel vs per-tensor
        channel_dim: Channel dimension for per-channel quantization

    Returns:
        Fake-quantized tensor (same dtype as input)
    """
    if per_channel:
        # Move channel dim to front for easier processing
        x_t = x.transpose(0, channel_dim)
        shape = x_t.shape
        x_flat = x_t.reshape(shape[0], -1)

        # Per-channel scale
        if symmetric:
            scale = x_flat.abs().max(dim=1, keepdim=True)[0] / (2 ** (bits - 1) - 1)
        else:
            x_min = x_flat.min(dim=1, keepdim=True)[0]
            x_max = x_flat.max(dim=1, keepdim=True)[0]
            scale = (x_max - x_min) / (2 ** bits - 1)

        scale = scale.clamp(min=1e-10)

        if symmetric:
            x_q = (x_flat / scale).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
            x_dq = x_q * scale
        else:
            x_q = ((x_flat - x_min) / scale).round().clamp(0, 2 ** bits - 1)
            x_dq = x_q * scale + x_min

        x_dq = x_dq.reshape(shape).transpose(0, channel_dim)
        return x_dq
    else:
        # Per-tensor quantization
        if symmetric:
            scale = x.abs().max() / (2 ** (bits - 1) - 1)
            scale = scale.clamp(min=1e-10)
            x_q = (x / scale).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
            return x_q * scale
        else:
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / (2 ** bits - 1)
            scale = scale.clamp(min=1e-10)
            x_q = ((x - x_min) / scale).round().clamp(0, 2 ** bits - 1)
            return x_q * scale + x_min


def fake_quant_tensor(x: Tensor, bits: int = 8) -> Tuple[Tensor, Tensor]:
    """
    Fake quantization with scale factor returned.

    Args:
        x: Input tensor
        bits: Number of bits

    Returns:
        (quantized_tensor, scale)
    """
    scale = x.abs().max() / (2 ** (bits - 1) - 1)
    scale = scale.clamp(min=1e-10)
    x_q = (x / scale).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    return x_q * scale, scale


class QuantizedSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for quantized gradients.

    Forward: Apply quantization
    Backward: Pass gradients through unchanged (optionally quantized)
    """

    @staticmethod
    def forward(ctx, x: Tensor, bits: int = 8, quant_grad: bool = False):
        ctx.bits = bits
        ctx.quant_grad = quant_grad
        return fake_quant(x, bits)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.quant_grad:
            # Quantize gradients too (for testing gradient quantization)
            return fake_quant(grad_output, ctx.bits), None, None
        else:
            # Standard STE: pass gradients unchanged
            return grad_output, None, None


def apply_fake_quant(x: Tensor, bits: int = 8) -> Tensor:
    """Functional wrapper for QuantizedSTE."""
    return QuantizedSTE.apply(x, bits, False)


def apply_fake_quant_with_grad(x: Tensor, bits: int = 8) -> Tensor:
    """Apply fake quantization to both forward and backward pass."""
    return QuantizedSTE.apply(x, bits, True)


class FakeQuantWrapper(nn.Module):
    """
    Wrapper to apply fake quantization to a module's weights and activations.
    """

    def __init__(
        self,
        module: nn.Module,
        bits: int = 8,
        quant_weights: bool = True,
        quant_activations: bool = True,
    ):
        super().__init__()
        self.module = module
        self.bits = bits
        self.quant_weights = quant_weights
        self.quant_activations = quant_activations

    def forward(self, x: Tensor) -> Tensor:
        # Quantize activations if enabled
        if self.quant_activations:
            x = fake_quant(x, self.bits)

        # Temporarily quantize weights
        if self.quant_weights:
            original_weights = {}
            for name, param in self.module.named_parameters():
                if "weight" in name:
                    original_weights[name] = param.data.clone()
                    param.data = fake_quant(param.data, self.bits)

        # Forward pass
        out = self.module(x)

        # Restore original weights
        if self.quant_weights:
            for name, param in self.module.named_parameters():
                if name in original_weights:
                    param.data = original_weights[name]

        return out


def quantize_model_weights(model: nn.Module, bits: int = 8) -> None:
    """
    Apply fake quantization to all model weights in-place.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.data = fake_quant(param.data, bits)


def create_sparse_field(
    sparsity_pct: float,
    field_size: int = 16,
    channels: int = 64,
    batch_size: int = 4,
    device: str = "cuda",
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Create a sparse field with specified occupancy.

    Args:
        sparsity_pct: Percentage of active voxels (0-100)
        field_size: Size of cubic field
        channels: Feature channels
        batch_size: Batch size
        device: Device

    Returns:
        (indices, features, positions) tuple
    """
    G = field_size
    total_voxels = G ** 3
    n_active = max(1, int(total_voxels * sparsity_pct / 100))

    # Random active voxel indices per batch
    indices_list = []
    positions_list = []
    for _ in range(batch_size):
        idx = torch.randperm(total_voxels, device=device)[:n_active]
        indices_list.append(idx)

        # Convert to 3D positions
        z = idx % G
        y = (idx // G) % G
        x = idx // (G * G)
        pos = torch.stack([x, y, z], dim=-1).float()
        positions_list.append(pos)

    indices = torch.stack(indices_list)  # (B, N)
    positions = torch.stack(positions_list)  # (B, N, 3)

    # Random features
    features = torch.randn(batch_size, n_active, channels, device=device)

    return indices, features, positions


def measure_memory_usage() -> float:
    """
    Get current GPU memory usage in MB.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class GradientHook:
    """
    Hook to capture gradients for analysis.
    """

    def __init__(self):
        self.gradients = {}

    def save_gradient(self, name: str):
        """Create hook function for a specific parameter."""
        def hook(grad):
            self.gradients[name] = grad.clone()
        return hook

    def register(self, model: nn.Module) -> None:
        """Register hooks on all parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(self.save_gradient(name))

    def get(self, name: str) -> Optional[Tensor]:
        """Get captured gradient."""
        return self.gradients.get(name)

    def clear(self) -> None:
        """Clear captured gradients."""
        self.gradients.clear()


def inject_error(x: Tensor, error_scale: float = 0.01) -> Tensor:
    """
    Inject quantization-like error into tensor.

    Used for error propagation analysis.

    Args:
        x: Input tensor
        error_scale: Error magnitude relative to activation magnitude

    Returns:
        Tensor with injected error
    """
    noise = torch.randn_like(x) * error_scale * x.abs().mean()
    return x + noise
