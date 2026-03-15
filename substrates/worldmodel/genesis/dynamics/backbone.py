"""Autoregressive Transformer Backbone for dynamics prediction."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from einops import rearrange

from genesis.config import DynamicsConfig
from genesis.dynamics.rope import RoPE3D
from genesis.dynamics.attention import SlidingTileAttention


class SwiGLU(nn.Module):
    """SwiGLU activation: swish(xW) * (xV)."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w = nn.Linear(in_features, hidden_features, bias=False)
        self.v = nn.Linear(in_features, hidden_features, bias=False)
        self.out = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(nn.functional.silu(self.w(x)) * self.v(x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    """
    Transformer block with sliding tile attention and SwiGLU MLP.

    Pre-norm architecture with residual connections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        tile_size: int = 8,
        global_heads: int = 8,
        rope: Optional[RoPE3D] = None,
    ):
        super().__init__()

        # Attention
        self.norm1 = RMSNorm(dim)
        self.attn = SlidingTileAttention(
            dim=dim,
            num_heads=num_heads,
            tile_size=tile_size,
            global_heads=global_heads,
            dropout=dropout,
            rope=rope,
        )

        # MLP
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio * 2 / 3)  # SwiGLU uses 2/3 ratio
        self.mlp = SwiGLU(dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # KV cache for efficient inference
        self._kv_cache: Optional[Tuple[Tensor, Tensor]] = None

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tensor:
        """
        Args:
            x: [B, T, H, W, D] input
            mask: Optional attention mask
            positions: Optional position tensor for RoPE
            use_cache: Whether to use/update KV cache

        Returns:
            out: [B, T, H, W, D]
        """
        # Attention with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask, positions))

        # MLP with residual
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x

    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self._kv_cache = None


class DynamicsBackbone(nn.Module):
    """
    Autoregressive Transformer for latent dynamics prediction.

    Takes latent history and action history, predicts next latent features.
    Uses 3D RoPE for position encoding and sliding tile attention for efficiency.
    """

    def __init__(self, config: DynamicsConfig, latent_channels: int = 5):
        super().__init__()
        self.config = config

        # Input projections (latent_channels → hidden_dim)
        self.latent_proj = nn.Linear(latent_channels, config.hidden_dim)

        # Action embedding projection
        self.action_dim = getattr(config, 'action_dim', 8)  # LAM default is 8 dims
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # Position encoding (shared across all attention layers)
        self.rope = RoPE3D(
            dim=config.hidden_dim,  # Full hidden dim, split internally for 3D
            max_temporal=config.context_length,
            max_spatial=64,  # Max spatial resolution
            base=config.rope.base,
            temporal_fraction=config.rope.temporal_fraction,
        )

        # Transformer blocks (with shared RoPE)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                tile_size=config.attention.tile_size,
                global_heads=config.attention.global_heads,
                rope=self.rope,
            )
            for _ in range(config.layers)
        ])

        # Output normalization
        self.norm_out = RMSNorm(config.hidden_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        latent_history: Tensor,
        action_history: Tensor,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict next latent features from history.

        Args:
            latent_history: [B, T, H, W, C] past latent frames
            action_history: [B, T, A] past action embeddings
            positions: Optional [B, T, H, W, 3] position tensor for RoPE (t, h, w)

        Returns:
            features: [B, H, W, D] features for DeltaV prediction
        """
        B, T, H, W, C = latent_history.shape

        # Shape validation
        assert action_history.shape[0] == B, f"Batch mismatch: latent {B} vs action {action_history.shape[0]}"
        assert action_history.shape[1] == T, f"Time mismatch: latent {T} vs action {action_history.shape[1]}"

        # Project latents
        x = self.latent_proj(latent_history)

        # Add action conditioning via FiLM-style modulation
        action_embed = self.action_embed(action_history)  # [B, T, D]
        action_embed = action_embed.unsqueeze(2).unsqueeze(2)  # [B, T, 1, 1, D]
        x = x + action_embed.expand(-1, -1, H, W, -1)

        # Generate positions if not provided
        if positions is None:
            device = x.device
            t_pos = torch.arange(T, device=device).view(1, T, 1, 1).expand(B, T, H, W)
            h_pos = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, T, H, W)
            w_pos = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, T, H, W)
            positions = torch.stack([t_pos, h_pos, w_pos], dim=-1)  # [B, T, H, W, 3]

        # Transformer blocks with RoPE positions
        for block in self.blocks:
            x = block(x, positions=positions)

        # Output normalization
        x = self.norm_out(x)

        # Take last timestep for next-frame prediction
        features = x[:, -1]  # [B, H, W, D]

        return features

    def forward_with_kv_cache(
        self,
        latent_t: Tensor,
        action_t: Tensor,
        kv_cache: Optional[list] = None,
        cache_timestep: int = 0,
    ) -> Tuple[Tensor, list]:
        """
        Efficient inference with KV cache.

        Only processes the new timestep, reusing cached keys/values.
        Manages sliding window for context_length timesteps.

        Args:
            latent_t: [B, 1, H, W, C] current latent frame
            action_t: [B, 1, A] current action
            kv_cache: List of cached tensors per layer, each entry is
                      Dict with 'latent_cache': [B, T_cache, H, W, D]
            cache_timestep: Current timestep index for position encoding

        Returns:
            features: [B, H, W, D] features for current step
            new_cache: Updated KV cache with sliding window management
        """
        B, _, H, W, C = latent_t.shape
        device = latent_t.device
        context_length = self.config.context_length

        # Project current frame
        x = self.latent_proj(latent_t)  # [B, 1, H, W, D]

        # Add action conditioning
        action_embed = self.action_embed(action_t)  # [B, 1, D]
        action_embed = action_embed.unsqueeze(2).unsqueeze(2)  # [B, 1, 1, 1, D]
        x = x + action_embed.expand(-1, -1, H, W, -1)

        # Generate position for current timestep
        t_pos = torch.full((B, 1, H, W), cache_timestep, device=device)
        h_pos = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        w_pos = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        positions = torch.stack([t_pos, h_pos, w_pos], dim=-1)  # [B, 1, H, W, 3]

        # Initialize cache if needed
        if kv_cache is None:
            kv_cache = [{'latent_cache': None} for _ in range(len(self.blocks))]

        new_cache = []

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i]

            # Concatenate with cached history for attention
            if layer_cache['latent_cache'] is not None:
                cached_latents = layer_cache['latent_cache']
                # Concatenate current with history
                x_with_history = torch.cat([cached_latents, x], dim=1)  # [B, T_cache+1, H, W, D]

                # Build positions for full sequence
                T_cache = cached_latents.shape[1]
                t_pos_full = torch.arange(cache_timestep - T_cache, cache_timestep + 1, device=device)
                t_pos_full = t_pos_full.view(1, -1, 1, 1).expand(B, -1, H, W)
                h_pos_full = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, T_cache + 1, H, W)
                w_pos_full = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, T_cache + 1, H, W)
                positions_full = torch.stack([t_pos_full, h_pos_full, w_pos_full], dim=-1)
            else:
                x_with_history = x
                positions_full = positions

            # Run block with full history
            x_out = block(x_with_history, positions=positions_full)

            # Extract only the current timestep output
            x = x_out[:, -1:, :, :, :]  # [B, 1, H, W, D]

            # Update cache with sliding window
            if layer_cache['latent_cache'] is not None:
                # Append and apply sliding window
                updated_cache = torch.cat([layer_cache['latent_cache'], x], dim=1)
                # Keep only last context_length - 1 frames (leave room for next)
                if updated_cache.shape[1] > context_length - 1:
                    updated_cache = updated_cache[:, -(context_length - 1):, :, :, :]
            else:
                updated_cache = x

            new_cache.append({'latent_cache': updated_cache})

        # Output normalization
        x = self.norm_out(x)
        features = x[:, 0]  # [B, H, W, D]

        return features, new_cache

    def clear_kv_cache(self) -> None:
        """Clear KV cache in all blocks."""
        for block in self.blocks:
            block.clear_cache()

    @property
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
