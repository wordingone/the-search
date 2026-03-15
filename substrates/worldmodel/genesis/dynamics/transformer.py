"""Dynamics transformer with KV-cache and windowed attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
import math
from einops import rearrange


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to input tensor."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class KVCache:
    """KV-cache for efficient autoregressive generation."""

    def __init__(self):
        self.k_cache: Optional[Tensor] = None
        self.v_cache: Optional[Tensor] = None

    def update(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        return self.k_cache, self.v_cache

    def get_seq_len(self) -> int:
        if self.k_cache is None:
            return 0
        return self.k_cache.shape[2]

    def clear(self):
        self.k_cache = None
        self.v_cache = None


class WindowedAttention(nn.Module):
    """Multi-head attention with sliding window for O(N*W) complexity."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rotary = RotaryEmbedding(head_dim)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        B, T, D = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None and kv_cache.k_cache is not None:
            offset = kv_cache.get_seq_len()
        else:
            offset = 0

        cos, sin = self.rotary(q, offset + T)
        q = apply_rotary_emb(q, cos[offset:offset+T], sin[offset:offset+T])
        k = apply_rotary_emb(k, cos[offset:offset+T], sin[offset:offset+T])

        if use_cache:
            if kv_cache is None:
                kv_cache = KVCache()
            k, v = kv_cache.update(k, v)

        S = k.shape[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal + window mask
        mask = torch.ones(T, S, device=x.device, dtype=torch.bool)

        for i in range(T):
            mask[i, offset + i + 1:] = False

        for i in range(T):
            start = max(0, offset + i - self.window_size + 1)
            mask[i, :start] = False

        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out(out)

        return out, kv_cache


class TransformerBlock(nn.Module):
    """Transformer block with windowed attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim, num_heads, head_dim, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        attn_out, kv_cache = self.attn(self.norm1(x), kv_cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, kv_cache


class DynamicsTransformer(nn.Module):
    """Autoregressive transformer for latent dynamics prediction.

    Features:
    - Windowed attention for O(N*W) complexity
    - KV-cache for efficient generation
    - Rotary position embeddings
    - Action conditioning
    """

    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        action_dim: int = 32,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.latent_proj = nn.Linear(latent_dim, latent_dim)

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=latent_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        latents: Tensor,
        actions: Optional[Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[List[KVCache]]]:
        """
        Args:
            latents: [B, T, D] latent tokens from tokenizer
            actions: [B, T, action_dim] action inputs (optional)
            kv_caches: list of KV caches per layer
            use_cache: whether to use/update caches

        Returns:
            output: [B, T, D] predicted next latents
            kv_caches: updated caches if use_cache=True
        """
        B, T, D = latents.shape

        x = self.latent_proj(latents)

        if actions is not None:
            action_emb = self.action_embed(actions)
            x = x + action_emb

        if use_cache and kv_caches is None:
            kv_caches = [None] * self.num_layers

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, cache = block(x, cache, use_cache)
            new_caches.append(cache)

        x = self.norm(x)
        output = self.output_proj(x)

        if use_cache:
            return output, new_caches
        return output, None

    def generate(
        self,
        initial_latents: Tensor,
        actions: Optional[Tensor] = None,
        num_steps: int = 16,
    ) -> Tensor:
        """Autoregressive generation with KV-cache."""
        B = initial_latents.shape[0]
        device = initial_latents.device

        generated = [initial_latents]
        output, kv_caches = self.forward(initial_latents, use_cache=True)

        next_latent = output[:, -1:, :]

        for step in range(num_steps):
            action = None
            if actions is not None:
                action = actions[:, step:step+1, :]

            output, kv_caches = self.forward(
                next_latent, action, kv_caches, use_cache=True
            )
            next_latent = output
            generated.append(next_latent)

        return torch.cat(generated, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
