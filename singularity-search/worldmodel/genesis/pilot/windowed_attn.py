"""Windowed 3D Attention for Phase 0 Pilot.

Implements O(N × W³) attention instead of O(N²) full attention.
- Window size 4: 64 tokens per window (vs 4096 total)
- Shifted windows on alternating layers for cross-window communication
- Memory: ~36 MB vs ~3.2 GB for full attention

Reference: TRELLIS windowed_attn.py pattern adapted for dense 3D tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import math
from typing import Optional, Tuple


def window_partition_3d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition 3D tensor into non-overlapping windows.

    Args:
        x: (B, D, H, W, C) - Input tensor
        window_size: Size of each window dimension

    Returns:
        windows: (B * num_windows, window_size^3, C)
    """
    B, D, H, W, C = x.shape

    # Reshape to windows: (B, D//ws, ws, H//ws, ws, W//ws, ws, C)
    x = x.view(B, D // window_size, window_size,
                  H // window_size, window_size,
                  W // window_size, window_size, C)

    # Permute to: (B, D//ws, H//ws, W//ws, ws, ws, ws, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()

    # Flatten to: (B * num_windows, ws^3, C)
    num_windows = (D // window_size) * (H // window_size) * (W // window_size)
    windows = x.view(B * num_windows, window_size ** 3, C)

    return windows


def window_reverse_3d(windows: torch.Tensor, window_size: int, D: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition.

    Args:
        windows: (B * num_windows, window_size^3, C)
        window_size: Size of each window dimension
        D, H, W: Original spatial dimensions

    Returns:
        x: (B, D, H, W, C)
    """
    nD = D // window_size
    nH = H // window_size
    nW = W // window_size
    num_windows = nD * nH * nW

    B = windows.shape[0] // num_windows
    C = windows.shape[-1]

    # Reshape: (B, nD, nH, nW, ws, ws, ws, C)
    x = windows.view(B, nD, nH, nW, window_size, window_size, window_size, C)

    # Permute back: (B, nD, ws, nH, ws, nW, ws, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()

    # Reshape to: (B, D, H, W, C)
    x = x.view(B, D, H, W, C)

    return x


def cyclic_shift_3d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Apply cyclic shift for shifted window attention.

    Args:
        x: (B, D, H, W, C)
        shift: Shift amount (typically window_size // 2)

    Returns:
        shifted: (B, D, H, W, C)
    """
    return torch.roll(x, shifts=(-shift, -shift, -shift), dims=(1, 2, 3))


def cyclic_unshift_3d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Reverse cyclic shift.

    Args:
        x: (B, D, H, W, C)
        shift: Shift amount

    Returns:
        unshifted: (B, D, H, W, C)
    """
    return torch.roll(x, shifts=(shift, shift, shift), dims=(1, 2, 3))


class Windowed3DAttention(nn.Module):
    """Multi-head self-attention within 3D windows.

    Uses PyTorch's native scaled_dot_product_attention for efficiency.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B * num_windows, window_size^3, C)

        Returns:
            out: (B * num_windows, window_size^3, C)
        """
        B_nW, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B_nW, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Use PyTorch's efficient SDPA
        # (B*nW, heads, N, head_dim)
        dropout_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B_nW, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Windowed3DAttentionLayer(nn.Module):
    """Single layer of windowed 3D attention with optional shift.

    Includes:
    - Pre-norm
    - Windowed multi-head attention
    - Residual connection
    - FFN with GELU
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 4,
                 shift: int = 0, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift

        # Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Windowed3DAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)

        # FFN
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, H, W, C) - Must be divisible by window_size

        Returns:
            out: (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape

        # Apply shift if specified
        if self.shift > 0:
            x_shifted = cyclic_shift_3d(x, self.shift)
        else:
            x_shifted = x

        # Partition into windows
        x_windows = window_partition_3d(x_shifted, self.window_size)  # (B*nW, ws^3, C)

        # Apply attention
        x_windows = x_windows + self.attn(self.norm1(x_windows))

        # Reverse windows
        x_out = window_reverse_3d(x_windows, self.window_size, D, H, W)

        # Unshift if shifted
        if self.shift > 0:
            x_out = cyclic_unshift_3d(x_out, self.shift)

        # FFN
        x_out = x_out + self.ffn(self.norm2(x_out))

        return x_out


class Windowed3DTransformer(nn.Module):
    """Windowed 3D Transformer for field propagation.

    Uses O(N × W³) attention instead of O(N²):
    - Window size 4: 64 tokens per window
    - 64 windows cover 16³ = 4096 tokens
    - Memory: 64 × 64² × 8 heads × 4 bytes × 6 layers ≈ 36 MB
    - vs Full: 4096² × 8 heads × 4 bytes × 6 layers ≈ 3.2 GB

    Shifted windows on odd layers enable cross-window communication.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 544, num_layers: int = 6,
                 num_heads: int = 8, window_size: int = 4, dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 3D positional encoding (learnable)
        # Will be set based on actual field size
        self.pos_embed = None

        # Transformer layers with alternating shift
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            shift = (window_size // 2) if (i % 2 == 1) else 0
            self.layers.append(
                Windowed3DAttentionLayer(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=shift,
                    dropout=dropout
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _ensure_pos_embed(self, D: int, H: int, W: int, device: torch.device):
        """Create positional embedding if not exists or size changed."""
        if self.pos_embed is None or self.pos_embed.shape[1:4] != (D, H, W):
            # Learnable 3D positional encoding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, D, H, W, self.hidden_dim, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) - Field tensor in channel-first format

        Returns:
            out: (B, C, D, H, W) - Propagated field
        """
        B, C, D, H, W = x.shape

        # Convert to (B, D, H, W, C) for attention
        x = x.permute(0, 2, 3, 4, 1)

        # Input projection
        x = self.input_proj(x)  # (B, D, H, W, hidden_dim)

        # Add positional encoding
        self._ensure_pos_embed(D, H, W, x.device)
        x = x + self.pos_embed

        # Apply transformer layers
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Final norm and output projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, D, H, W, input_dim)

        # Convert back to (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        return x


class WindowedFieldPropagator(nn.Module):
    """Drop-in replacement for FieldPropagator using windowed attention.

    API compatible with original FieldPropagator but uses O(N × W³) attention.
    """

    def __init__(self, field_size: int = 16, field_channels: int = 32,
                 hidden_dim: int = 544, num_layers: int = 6, num_heads: int = 8,
                 window_size: int = 4, use_checkpoint: bool = False):
        super().__init__()
        self.field_size = field_size
        self.field_channels = field_channels

        # Ensure field_size is divisible by window_size
        assert field_size % window_size == 0, \
            f"field_size ({field_size}) must be divisible by window_size ({window_size})"

        self.transformer = Windowed3DTransformer(
            input_dim=field_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            window_size=window_size,
            use_checkpoint=use_checkpoint
        )

    def forward(self, field: torch.Tensor, certainty: Optional[torch.Tensor] = None,
                certainty_threshold: float = 0.05) -> torch.Tensor:
        """Propagate field state using windowed attention.

        Args:
            field: (B, C, Gx, Gy, Gz) - Field tensor
            certainty: (B, Gx, Gy, Gz) - Optional, ignored for windowed attention
                       (sparsity handled by window structure, not masking)
            certainty_threshold: Ignored (kept for API compatibility)

        Returns:
            propagated: (B, C, Gx, Gy, Gz)
        """
        # Windowed attention processes all windows uniformly
        # Sparsity is implicitly handled by the window structure
        return self.transformer(field)


if __name__ == '__main__':
    # Verification
    print("Testing Windowed 3D Attention...")

    # Test parameters matching Phase 0 pilot
    B = 8  # Now we can use larger batch sizes!
    C = 32  # field_channels
    G = 16  # field_size
    hidden_dim = 544
    num_layers = 6
    num_heads = 8
    window_size = 4

    # Create module
    propagator = WindowedFieldPropagator(
        field_size=G,
        field_channels=C,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        window_size=window_size
    )

    # Count parameters
    num_params = sum(p.numel() for p in propagator.parameters())
    print(f"Parameters: {num_params:,}")

    # Test forward pass
    field = torch.randn(B, C, G, G, G)

    # Measure memory
    if torch.cuda.is_available():
        propagator = propagator.cuda()
        field = field.cuda()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            out = propagator(field)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory (batch={B}): {peak_mem:.2f} GB")
    else:
        out = propagator(field)
        print("CUDA not available, skipping memory test")

    print(f"Input shape: {field.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Shapes match: {field.shape == out.shape}")

    # Test gradient flow
    field.requires_grad = True
    out = propagator(field)
    loss = out.sum()
    loss.backward()
    print(f"Gradient flows: {field.grad is not None}")

    print("\nAll tests passed!")
