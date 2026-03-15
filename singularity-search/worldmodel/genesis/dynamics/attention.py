"""Sliding tile attention for efficient spatiotemporal processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, TYPE_CHECKING
from einops import rearrange, repeat

if TYPE_CHECKING:
    from genesis.dynamics.rope import RoPE3D


class SlidingTileAttention(nn.Module):
    """
    Sliding Tile Attention for efficient spatiotemporal processing.

    Combines:
    1. Local attention within tiles (captures fine details)
    2. Global attention on tile representatives (captures long-range)

    This reduces complexity from O(N²) to O(N × tile_size + num_tiles²)

    Reference: HunyuanVideo SSTA (Selective and Sliding Tile Attention)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        tile_size: int = 8,
        global_heads: int = 8,
        dropout: float = 0.0,
        rope: Optional["RoPE3D"] = None,
    ):
        """
        Args:
            dim: Model dimension
            num_heads: Number of attention heads for local attention
            tile_size: Size of local attention tiles
            global_heads: Number of heads for global attention (subset)
            dropout: Attention dropout
            rope: Optional RoPE3D for positional encoding (applied to Q/K)
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.tile_size = tile_size
        self.global_heads = global_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert global_heads <= num_heads, "global_heads must be <= num_heads"

        # Local attention projections
        self.qkv_local = nn.Linear(dim, 3 * dim)
        self.out_local = nn.Linear(dim, dim)

        # Global attention projections (fewer heads)
        global_dim = self.head_dim * global_heads
        self.qkv_global = nn.Linear(dim, 3 * global_dim)
        self.out_global = nn.Linear(global_dim, dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Learnable scale
        self.scale = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))

    def set_rope(self, rope: "RoPE3D") -> None:
        """Set the RoPE instance for position encoding."""
        self.rope = rope

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply sliding tile attention.

        Args:
            x: [B, T, H, W, D] input tensor
            mask: Optional attention mask
            positions: Optional [B, T, H, W, 3] position tensor for RoPE

        Returns:
            out: [B, T, H, W, D] attended tensor
        """
        B, T, H, W, D = x.shape

        # Local attention within tiles
        local_out = self._local_attention(x, positions)

        # Global attention on tile representatives
        global_out = self._global_attention(x)

        # Combine (simple addition, could use gating)
        out = local_out + global_out

        return out

    def _local_attention(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Attention within spatial tiles at each timestep."""
        B, T, H, W, D = x.shape

        # Pad if necessary to make H, W divisible by tile_size
        pad_h = (self.tile_size - H % self.tile_size) % self.tile_size
        pad_w = (self.tile_size - W % self.tile_size) % self.tile_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            _, _, H_pad, W_pad, _ = x.shape
        else:
            H_pad, W_pad = H, W

        # Reshape to tiles: [B, T, num_tiles_h, num_tiles_w, tile_size, tile_size, D]
        num_tiles_h = H_pad // self.tile_size
        num_tiles_w = W_pad // self.tile_size

        x_tiles = rearrange(
            x,
            'b t (nh th) (nw tw) d -> (b t nh nw) (th tw) d',
            th=self.tile_size, tw=self.tile_size,
        )

        # Compute Q, K, V
        qkv = self.qkv_local(x_tiles)  # [(B*T*tiles), tile_size², 3D]
        qkv = rearrange(qkv, 'n s (three h d) -> three n h s d',
                       three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K if available
        if self.rope is not None:
            # Reshape for RoPE: [N, heads, seq, head_dim] -> apply per-head
            # RoPE expects [B, T, H, W, D] so we need to reshape appropriately
            # For local attention within tiles, we apply position encoding per tile
            n_batch = B * T * num_tiles_h * num_tiles_w
            seq_len = self.tile_size * self.tile_size

            # Reshape Q and K to [n_batch, tile_h, tile_w, heads * head_dim] for RoPE
            q_for_rope = rearrange(q, 'n h s d -> n s (h d)')
            k_for_rope = rearrange(k, 'n h s d -> n s (h d)')

            # Apply simplified RoPE (1D position encoding for tiles)
            # Note: Full 3D RoPE would require tracking tile positions
            q_for_rope = q_for_rope.view(n_batch, self.tile_size, self.tile_size, -1)
            k_for_rope = k_for_rope.view(n_batch, self.tile_size, self.tile_size, -1)

            # Use 2D subset of RoPE (spatial only within tiles)
            # Generate local positions for the tile
            device = q.device
            h_pos = torch.arange(self.tile_size, device=device).view(1, self.tile_size, 1).expand(n_batch, -1, self.tile_size)
            w_pos = torch.arange(self.tile_size, device=device).view(1, 1, self.tile_size).expand(n_batch, self.tile_size, -1)
            t_pos = torch.zeros_like(h_pos)  # Temporal handled separately

            # Create position tensor [n_batch, tile_h, tile_w, 3]
            positions_local = torch.stack([t_pos, h_pos, w_pos], dim=-1)

            # Apply RoPE using forward_qk for separate Q/K handling
            # Expand to match RoPE expected shape [B, T, H, W, D]
            q_reshaped = q_for_rope.unsqueeze(1)  # [n_batch, 1, tile_h, tile_w, D]
            k_reshaped = k_for_rope.unsqueeze(1)
            pos_reshaped = positions_local.unsqueeze(1)

            q_rotated, k_rotated = self.rope.forward_qk(q_reshaped, k_reshaped, pos_reshaped, pos_reshaped)

            # Reshape back to attention format
            q = rearrange(q_rotated.squeeze(1).view(n_batch, seq_len, self.num_heads, self.head_dim),
                         'n s h d -> n h s d')
            k = rearrange(k_rotated.squeeze(1).view(n_batch, seq_len, self.num_heads, self.head_dim),
                         'n s h d -> n h s d')

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Handle numerical stability: clamp before softmax
        attn = attn.clamp(min=-1e4, max=1e4)
        attn = F.softmax(attn, dim=-1)

        # Handle NaN from all-masked positions
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = attn @ v  # [N, heads, tile_size², head_dim]
        out = rearrange(out, 'n h s d -> n s (h d)')

        # Project and reshape back
        out = self.out_local(out)
        out = rearrange(
            out,
            '(b t nh nw) (th tw) d -> b t (nh th) (nw tw) d',
            b=B, t=T, nh=num_tiles_h, nw=num_tiles_w,
            th=self.tile_size, tw=self.tile_size,
        )

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W, :]

        return out

    def _global_attention(self, x: Tensor) -> Tensor:
        """Attention between tile representatives."""
        B, T, H, W, D = x.shape

        # Pool tiles to get representatives
        # Each tile represented by its mean
        pad_h = (self.tile_size - H % self.tile_size) % self.tile_size
        pad_w = (self.tile_size - W % self.tile_size) % self.tile_size

        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        else:
            x_padded = x

        # Pool to tile representatives: [B, T, num_tiles_h, num_tiles_w, D]
        x_pooled = rearrange(
            x_padded,
            'b t (nh th) (nw tw) d -> b t nh nw th tw d',
            th=self.tile_size, tw=self.tile_size,
        ).mean(dim=(-3, -2))  # Average over tile

        num_tiles_h = x_pooled.shape[2]
        num_tiles_w = x_pooled.shape[3]

        # Flatten spatial for global attention: [B, T * num_tiles, D]
        x_flat = rearrange(x_pooled, 'b t nh nw d -> b (t nh nw) d')

        # Compute Q, K, V with fewer heads
        qkv = self.qkv_global(x_flat)
        qkv = rearrange(qkv, 'b s (three h d) -> three b h s d',
                       three=3, h=self.global_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal mask for temporal causality
        seq_len = T * num_tiles_h * num_tiles_w
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        # Attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Numerical stability: clamp before softmax to avoid overflow
        attn = attn.clamp(min=-1e4, max=1e4)
        attn = F.softmax(attn, dim=-1)

        # Handle NaN from all-masked rows
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.out_global(out)

        # Reshape and expand back to original resolution
        out = rearrange(out, 'b (t nh nw) d -> b t nh nw d',
                       t=T, nh=num_tiles_h, nw=num_tiles_w)

        # Expand tiles back (repeat for each position in tile)
        out = repeat(out, 'b t nh nw d -> b t (nh th) (nw tw) d',
                    th=self.tile_size, tw=self.tile_size)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W, :]

        return out


class CausalSelfAttention(nn.Module):
    """
    Standard causal self-attention for comparison or simpler cases.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, L, D] input sequence
            mask: Optional [L, L] attention mask

        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b l (three h d) -> three b h l d',
                       three=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out(out)

        return out
