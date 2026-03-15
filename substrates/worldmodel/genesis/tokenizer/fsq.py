"""Finite Scalar Quantization (FSQ) for discrete tokenization."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import math


class FSQ(nn.Module):
    """
    Finite Scalar Quantization - no codebook, no collapse.

    Instead of learning a codebook like VQ-VAE, FSQ divides each dimension
    into a fixed number of levels and rounds to the nearest level.

    Vocabulary size = prod(levels)

    Reference: TinyWorlds, Mentzer et al. "Finite Scalar Quantization"
    """

    def __init__(self, levels: List[int] = [8, 6, 5, 5, 5]):
        """
        Args:
            levels: Number of quantization levels per dimension.
                    Default [8, 6, 5, 5, 5] gives vocab = 6000.
        """
        super().__init__()
        self.levels = levels
        self.dim = len(levels)

        # Register levels as buffer for device movement
        self.register_buffer(
            "_levels",
            torch.tensor(levels, dtype=torch.float32)
        )

        # Precompute multipliers for index computation
        multipliers = [1]
        for L in levels[:-1]:
            multipliers.append(multipliers[-1] * L)
        self.register_buffer(
            "_multipliers",
            torch.tensor(multipliers, dtype=torch.int64)
        )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        result = 1
        for L in self.levels:
            result *= L
        return result

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
        # z_scaled[..., i] in [0, levels[i] - 1]
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
        """Convert quantized values to single indices."""
        # z_quantized: [..., D] with values in {0, 1, ..., L_i - 1}
        z_int = z_quantized.long()

        # Compute index = sum(z_i * multiplier_i)
        indices = (z_int * self._multipliers).sum(dim=-1)
        return indices

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Convert indices back to quantized codes."""
        # indices: [...]
        # output: [..., D]
        codes = []
        remaining = indices.clone()

        for i, L in enumerate(self.levels):
            codes.append(remaining % L)
            remaining = remaining // L

        return torch.stack(codes, dim=-1).float()

    def codes_to_latent(self, codes: Tensor) -> Tensor:
        """Convert quantized codes back to latent space (inverse of forward)."""
        # codes: [..., D] in {0, ..., L_i - 1}
        # output: [..., D] in [-1, 1]
        half_levels = (self._levels - 1) / 2
        z_scaled = codes.float()
        z_bounded = (z_scaled - half_levels) / half_levels
        return z_bounded

    def codes_to_indices(self, codes: Tensor) -> Tensor:
        """Convert quantized codes back to indices."""
        return self._to_indices(codes)


class FSQEmbedding(nn.Module):
    """
    Embedding layer that maps FSQ indices to continuous vectors.

    This is more memory-efficient than a full embedding table when
    vocab size is large (e.g., 6000+).
    """

    def __init__(self, levels: List[int], embed_dim: int):
        """
        Args:
            levels: FSQ level configuration
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.fsq = FSQ(levels)
        self.embed_dim = embed_dim

        # Per-level embeddings (much smaller than full vocab embedding)
        self.level_embeds = nn.ModuleList([
            nn.Embedding(L, embed_dim // len(levels))
            for L in levels
        ])

        # Final projection
        self.proj = nn.Linear(
            (embed_dim // len(levels)) * len(levels),
            embed_dim
        )

    def forward(self, indices: Tensor) -> Tensor:
        """
        Args:
            indices: [...] discrete token indices

        Returns:
            embeddings: [..., embed_dim] continuous embeddings
        """
        # Convert indices to per-dimension codes
        codes = self.fsq.indices_to_codes(indices).long()

        # Embed each dimension separately
        embeds = []
        for i, embed_layer in enumerate(self.level_embeds):
            embeds.append(embed_layer(codes[..., i]))

        # Concatenate and project
        concat = torch.cat(embeds, dim=-1)
        return self.proj(concat)
