"""3D Rotary Position Embedding for spatiotemporal data."""

import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Tuple, Optional


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding.

    Extends standard RoPE to handle temporal + spatial dimensions.
    Dimensions are split:
    - 1/3 for temporal position (t)
    - 1/3 for height position (h)
    - 1/3 for width position (w)

    Reference: V-JEPA 2, HunyuanVideo 1.5
    """

    def __init__(
        self,
        dim: int,
        max_temporal: int = 64,
        max_spatial: int = 64,
        base: float = 10000.0,
        temporal_fraction: float = 0.333,
    ):
        """
        Args:
            dim: Total embedding dimension (must be divisible by 2)
            max_temporal: Maximum temporal positions
            max_spatial: Maximum spatial positions (H and W)
            base: Base for frequency computation
            temporal_fraction: Fraction of dims for temporal (rest split between H/W)
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"

        self.dim = dim
        self.max_temporal = max_temporal
        self.max_spatial = max_spatial
        self.base = base

        # Dimension allocation
        self.t_dim = int(dim * temporal_fraction)
        self.t_dim = self.t_dim - (self.t_dim % 2)  # Make even
        remaining = dim - self.t_dim
        self.h_dim = remaining // 2
        self.h_dim = self.h_dim - (self.h_dim % 2)  # Make even
        self.w_dim = dim - self.t_dim - self.h_dim

        # Precompute frequency bands
        self.register_buffer("_t_freqs", self._compute_freqs(self.t_dim, base))
        self.register_buffer("_h_freqs", self._compute_freqs(self.h_dim, base))
        self.register_buffer("_w_freqs", self._compute_freqs(self.w_dim, base))

        # Cache for position embeddings
        self._cache: dict = {}

    def _compute_freqs(self, dim: int, base: float) -> Tensor:
        """Compute frequency bands for RoPE."""
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return freqs

    def _get_rotary_embedding(
        self,
        positions: Tensor,
        freqs: Tensor,
        dim: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute cos/sin embeddings for given positions.

        Args:
            positions: [B, ..., 1] position indices
            freqs: [dim//2] frequency bands

        Returns:
            cos, sin: [B, ..., dim] rotary embeddings
        """
        # positions: [B, ...] -> [B, ..., 1]
        # freqs: [dim//2] -> [1, ..., dim//2]
        angles = positions.unsqueeze(-1) * freqs.view(*([1] * positions.dim()), -1)
        # angles: [B, ..., dim//2]

        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [B, ..., dim]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)

        return cos, sin

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims of x."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(
        self,
        x: Tensor,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply 3D rotary position embedding.

        Args:
            x: [B, T, H, W, D] input tensor
            positions: Optional [B, T, H, W, 3] position tensor (t, h, w)
                      If None, positions are inferred from shape.

        Returns:
            x_rotated: [B, T, H, W, D] with position information
        """
        B, T, H, W, D = x.shape
        assert D == self.dim, f"Expected dim {self.dim}, got {D}"

        # Generate positions if not provided
        if positions is None:
            t_pos = torch.arange(T, device=x.device).view(1, T, 1, 1).expand(B, T, H, W)
            h_pos = torch.arange(H, device=x.device).view(1, 1, H, 1).expand(B, T, H, W)
            w_pos = torch.arange(W, device=x.device).view(1, 1, 1, W).expand(B, T, H, W)
        else:
            t_pos = positions[..., 0]
            h_pos = positions[..., 1]
            w_pos = positions[..., 2]

        # Compute rotary embeddings for each dimension
        cos_t, sin_t = self._get_rotary_embedding(t_pos, self._t_freqs, self.t_dim)
        cos_h, sin_h = self._get_rotary_embedding(h_pos, self._h_freqs, self.h_dim)
        cos_w, sin_w = self._get_rotary_embedding(w_pos, self._w_freqs, self.w_dim)

        # Concatenate
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        # Apply rotation
        x_rotated = x * cos + self._rotate_half(x) * sin

        return x_rotated

    def forward_qk(
        self,
        q: Tensor,
        k: Tensor,
        positions_q: Optional[Tensor] = None,
        positions_k: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE to query and key tensors separately.

        Useful when Q and K have different positions (e.g., causal attention).
        """
        q_rotated = self.forward(q, positions_q)
        k_rotated = self.forward(k, positions_k)
        return q_rotated, k_rotated


class RoPE1D(nn.Module):
    """
    Standard 1D RoPE for sequence models.

    Used as building block or for simpler attention patterns.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)

        # Precompute for common sequence lengths
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        positions = torch.arange(seq_len)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)
        self.register_buffer("_cos_cached", torch.cos(angles).repeat_interleave(2, dim=-1))
        self.register_buffer("_sin_cached", torch.sin(angles).repeat_interleave(2, dim=-1))

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Args:
            x: [B, L, D] input
            offset: Position offset (for KV cache)

        Returns:
            x_rotated: [B, L, D]
        """
        seq_len = x.shape[1]
        cos = self._cos_cached[offset:offset + seq_len]
        sin = self._sin_cached[offset:offset + seq_len]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rotated = torch.stack((
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x2 * cos[..., 1::2] + x1 * sin[..., 1::2],
        ), dim=-1).flatten(-2)

        return x_rotated
