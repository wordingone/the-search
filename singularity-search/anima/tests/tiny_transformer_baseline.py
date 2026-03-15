"""
T I N Y  T R A N S F O R M E R  B A S E L I N E
================================================

A minimal transformer model with ~103k parameters to match Anima.

This serves as a direct comparison: same parameter budget,
standard transformer architecture vs Anima's W/I/T architecture.

Architecture choices to hit ~103k params:
- Embedding dim: 64
- Heads: 4
- Layers: 2
- FFN hidden: 128
- Vocab/state dim: 16

This is EXTREMELY small for a transformer - typical "tiny" models
start at 10M+ parameters. We're 100x smaller.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
import math


class TinyTransformerConfig:
    """Configuration for tiny transformer."""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


class TinyTransformer(nn.Module):
    """
    Minimal transformer for comparison with Anima.

    Target: ~103k parameters to match Anima variants.
    """

    def __init__(self, config: Optional[TinyTransformerConfig] = None):
        super().__init__()

        if config is None:
            config = TinyTransformerConfig()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.output_dim)

        # For stateful operation (like Anima)
        self.hidden_state = None
        self.step_count = 0
        self.history: List[torch.Tensor] = []
        self.max_history = config.max_seq_len

    def reset(self):
        """Reset internal state."""
        self.hidden_state = None
        self.step_count = 0
        self.history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence input.
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = ~mask  # Invert for attention

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection
        return self.output_proj(x)

    def step(self, observation: torch.Tensor) -> Dict:
        """
        Single step interface to match Anima API.
        observation: (batch, input_dim) or (batch, 1, input_dim)
        """
        self.step_count += 1

        # Ensure correct shape
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)  # (batch, 1, input_dim)

        # Add to history
        self.history.append(observation)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Concatenate history
        x = torch.cat(self.history, dim=1)  # (batch, seq_len, input_dim)

        # Forward pass
        output = self.forward(x)

        # Take last position output as action
        action = output[:, -1, :]  # (batch, output_dim)

        return {
            'alive': True,
            'action': action,
            'step': self.step_count,
        }

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_matched_transformer(sensory_dim: int = 16, action_dim: int = 16,
                                target_params: int = 103520) -> TinyTransformer:
    """
    Create a transformer with approximately the target parameter count.

    We'll iterate to find the right d_model/d_ff combination.
    """

    # Start with baseline config
    config = TinyTransformerConfig(
        input_dim=sensory_dim,
        output_dim=action_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
    )

    model = TinyTransformer(config)
    actual_params = model.parameters_count()

    print(f"Initial config: d_model={config.d_model}, d_ff={config.d_ff}, "
          f"n_layers={config.n_layers}, params={actual_params:,}")

    # Adjust to get closer to target
    # Try different configurations
    best_config = config
    best_diff = abs(actual_params - target_params)

    for d_model in [48, 56, 64, 72, 80]:
        for d_ff in [96, 128, 160, 192]:
            for n_layers in [2, 3]:
                if d_model % 4 != 0:  # Must be divisible by n_heads
                    continue

                test_config = TinyTransformerConfig(
                    input_dim=sensory_dim,
                    output_dim=action_dim,
                    d_model=d_model,
                    n_heads=4,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    max_seq_len=32,
                )

                test_model = TinyTransformer(test_config)
                test_params = test_model.parameters_count()
                diff = abs(test_params - target_params)

                if diff < best_diff:
                    best_diff = diff
                    best_config = test_config

    model = TinyTransformer(best_config)
    print(f"Final config: d_model={best_config.d_model}, d_ff={best_config.d_ff}, "
          f"n_layers={best_config.n_layers}, params={model.parameters_count():,}")
    print(f"Target: {target_params:,}, Diff: {best_diff:,}")

    return model


if __name__ == '__main__':
    # Test the transformer
    print("Creating matched transformer...")
    model = create_matched_transformer(sensory_dim=16, action_dim=16, target_params=103520)

    print(f"\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel():,}")

    # Test step interface
    print("\nTesting step interface...")
    model.reset()

    for i in range(5):
        obs = torch.randn(1, 16)
        result = model.step(obs)
        print(f"  Step {i+1}: action shape = {result['action'].shape}")
