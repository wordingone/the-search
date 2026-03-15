"""Slot Attention module for object-centric representation.

Based on: "Object-Centric Learning with Slot Attention" (Locatello et al., 2020)

Key insight: Slots compete for input features via softmax normalization over slots.
This creates a natural binding of features to objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlotAttention(nn.Module):
    """Slot Attention module.

    Takes input features and produces K slot vectors, each representing an object.
    Slots persist across time, providing object permanence.
    """

    def __init__(
        self,
        num_slots: int = 4,
        slot_dim: int = 64,
        input_dim: int = 128,
        num_iterations: int = 3,
        hidden_dim: int = 128,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            num_slots: Number of object slots (K)
            slot_dim: Dimension of each slot vector
            input_dim: Dimension of input features
            num_iterations: Number of attention iterations for slot refinement
            hidden_dim: Hidden dimension for MLPs
            epsilon: Small constant for numerical stability
        """
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Layer norms
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Attention projections
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # GRU for slot update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )

        # Scale factor for attention
        self.scale = slot_dim ** -0.5

    def _init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize slot vectors with learned Gaussian.

        Returns:
            slots: (B, K, slot_dim)
        """
        # Sample from learned Gaussian
        mu = self.slot_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        return slots

    def forward(
        self,
        inputs: torch.Tensor,
        slots: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (B, N, input_dim) - Input features (e.g., from CNN encoder)
            slots: (B, K, slot_dim) - Previous slots (for temporal persistence)
                   If None, initialize fresh slots.

        Returns:
            slots: (B, K, slot_dim) - Updated slot vectors
            attn_weights: (B, K, N) - Attention weights (which inputs each slot attends to)
        """
        B, N, _ = inputs.shape
        device = inputs.device

        # Initialize slots if not provided
        if slots is None:
            slots = self._init_slots(B, device)

        # Normalize inputs
        inputs = self.norm_input(inputs)

        # Project inputs to keys and values (done once)
        k = self.project_k(inputs)  # (B, N, slot_dim)
        v = self.project_v(inputs)  # (B, N, slot_dim)

        # Iterative slot refinement
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention: slots query inputs
            q = self.project_q(slots)  # (B, K, slot_dim)

            # Compute attention scores
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale  # (B, K, N)

            # Softmax over SLOTS (not over inputs)
            # This makes slots compete for features
            attn_weights = F.softmax(attn_logits, dim=1)  # (B, K, N)

            # Weighted mean of values (normalize by sum of weights per slot)
            attn_weights_norm = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + self.epsilon)
            updates = torch.einsum('bkn,bnd->bkd', attn_weights_norm, v)  # (B, K, slot_dim)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

            # MLP refinement with residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SlotDecoder(nn.Module):
    """Decode slot vectors back to spatial features.

    Uses broadcast decoder: each slot generates a full spatial map,
    then maps are combined via attention weights.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        output_dim: int = 128,
        spatial_size: int = 8,
        hidden_dim: int = 128,
    ):
        """
        Args:
            slot_dim: Dimension of slot vectors
            output_dim: Dimension of output features per position
            spatial_size: Output spatial resolution (H = W = spatial_size)
            hidden_dim: Hidden dimension for decoder MLP
        """
        super().__init__()

        self.spatial_size = spatial_size

        # Positional encoding for spatial positions
        self.register_buffer(
            'pos_embed',
            self._create_positional_encoding(spatial_size ** 2, slot_dim)
        )

        # Broadcast decoder MLP (applied to each position independently)
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim + 1),  # +1 for alpha mask
        )

    def _create_positional_encoding(self, num_positions: int, dim: int) -> torch.Tensor:
        """Create learned positional encoding."""
        pos = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos[:, 0::2] = torch.sin(position * div_term[:dim//2])
        pos[:, 1::2] = torch.cos(position * div_term[:dim//2])
        return pos

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: (B, K, slot_dim)

        Returns:
            output: (B, N, output_dim) where N = spatial_size^2
        """
        B, K, _ = slots.shape
        N = self.spatial_size ** 2

        # Broadcast slots to all positions: (B, K, N, slot_dim)
        slots_broadcast = slots.unsqueeze(2).expand(-1, -1, N, -1)

        # Add positional encoding
        pos = self.pos_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, N, slot_dim)
        slots_pos = slots_broadcast + pos

        # Decode each slot at each position
        decoded = self.decoder(slots_pos)  # (B, K, N, output_dim + 1)

        # Split into features and alpha
        features = decoded[..., :-1]  # (B, K, N, output_dim)
        alpha = decoded[..., -1:]     # (B, K, N, 1)

        # Softmax over slots for alpha (competitive reconstruction)
        alpha = F.softmax(alpha, dim=1)

        # Weighted combination
        output = (features * alpha).sum(dim=1)  # (B, N, output_dim)

        return output

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test slot attention
    print("Testing SlotAttention...")

    slot_attn = SlotAttention(
        num_slots=4,
        slot_dim=64,
        input_dim=128,
        num_iterations=3,
    )

    print(f"SlotAttention params: {slot_attn.count_parameters():,}")

    # Forward pass
    inputs = torch.randn(2, 64, 128)  # (B, N, input_dim)
    slots, attn = slot_attn(inputs)

    print(f"Input shape: {inputs.shape}")
    print(f"Slots shape: {slots.shape}")
    print(f"Attention shape: {attn.shape}")

    # Test temporal persistence
    slots2, attn2 = slot_attn(inputs, slots=slots)
    print(f"Slots after 2nd call: {slots2.shape}")

    # Test decoder
    print("\nTesting SlotDecoder...")

    decoder = SlotDecoder(
        slot_dim=64,
        output_dim=128,
        spatial_size=8,
    )

    print(f"SlotDecoder params: {decoder.count_parameters():,}")

    output = decoder(slots)
    print(f"Decoder output shape: {output.shape}")

    print("\nAll tests passed!")
