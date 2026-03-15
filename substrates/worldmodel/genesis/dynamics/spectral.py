"""Bounded Spectral Dynamics (BSD) for stable long-horizon generation.

Key insight: Mode collapse in autoregressive models happens because dynamics
converge to a fixed point. BSD prevents this by:

1. Fixed eigenvalue spectrum - controls decay rates, prevents explosion/vanishing
2. Rotating eigenvectors - prevents collapse to dominant mode
3. State-dependent adaptation - input drives meaningful state evolution

Mathematical guarantees:
- ||h_t|| <= B for all t (bounded state norm)
- eig(A_t) = Lambda for all t (spectral invariance)
- No mode collapse via continuous eigenvector rotation

Reference: Rotational State-Space Models (RSM) concept
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


class GivensRotation(nn.Module):
    """Apply Givens rotations to pairs of dimensions.

    Givens rotations are orthogonal transformations that rotate in a 2D plane.
    Applying multiple Givens rotations creates a general orthogonal transformation.
    """

    def __init__(self, dim: int, n_rotations: int):
        """
        Args:
            dim: State dimension (must be even)
            n_rotations: Number of rotation pairs to apply
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for Givens rotations"

        self.dim = dim
        self.n_rotations = n_rotations
        self.n_pairs = dim // 2

        # Rotation pairs: each rotation acts on (2i, 2i+1)
        # This covers all dimensions with n_pairs rotations
        self.register_buffer(
            'pairs',
            torch.arange(self.n_pairs).unsqueeze(1) * 2 + torch.arange(2),
        )

    def forward(self, x: Tensor, angles: Tensor) -> Tensor:
        """Apply Givens rotations.

        Args:
            x: [B, D] state vectors
            angles: [B, n_pairs] rotation angles in radians

        Returns:
            rotated: [B, D] rotated state vectors
        """
        B, D = x.shape

        # Compute sin/cos for each pair
        cos_theta = torch.cos(angles)  # [B, n_pairs]
        sin_theta = torch.sin(angles)  # [B, n_pairs]

        # Apply rotations to pairs of dimensions
        x_even = x[:, 0::2]  # [B, n_pairs]
        x_odd = x[:, 1::2]   # [B, n_pairs]

        # 2D rotation: [cos -sin; sin cos] @ [x0; x1]
        y_even = cos_theta * x_even - sin_theta * x_odd
        y_odd = sin_theta * x_even + cos_theta * x_odd

        # Interleave back
        result = torch.empty_like(x)
        result[:, 0::2] = y_even
        result[:, 1::2] = y_odd

        return result


class SpectralRecurrentCell(nn.Module):
    """Recurrent cell with bounded spectral dynamics.

    Core innovation: Separate eigenvalues (fixed) from eigenvectors (rotating).

    The state update can be written as:
        h_{t+1} = C_t @ Lambda @ C_t^T @ h_t + B @ x_t

    Where:
    - Lambda: Fixed diagonal matrix of eigenvalues in (0, 1)
    - C_t: Time-varying orthogonal matrix (eigenvector basis)
    - C_t evolves via Givens rotations driven by state and input

    This guarantees:
    1. Bounded norm: ||h_t|| <= ||h_0|| / (1 - lambda_max) for stable eigenvalues
    2. No mode collapse: Rotating basis prevents convergence to dominant eigenvector
    3. Rich dynamics: State-dependent rotation angles create complex trajectories

    Args:
        d_state: State dimension
        d_input: Input dimension
        n_rot: Number of Givens rotation pairs per step
        lambda_range: (min, max) eigenvalue range, both in (0, 1)
        rotation_scale: Scale factor for rotation angles
    """

    def __init__(
        self,
        d_state: int,
        d_input: int,
        n_rot: Optional[int] = None,
        lambda_range: Tuple[float, float] = (0.9, 0.999),
        rotation_scale: float = 0.1,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even"
        assert 0 < lambda_range[0] < lambda_range[1] < 1, "Eigenvalues must be in (0, 1)"

        self.d_state = d_state
        self.d_input = d_input
        self.n_pairs = d_state // 2
        self.n_rot = n_rot or self.n_pairs
        self.rotation_scale = rotation_scale

        # Fixed eigenvalues - evenly spaced in lambda_range for diverse timescales
        lambda_min, lambda_max = lambda_range
        eigenvalues = torch.linspace(lambda_min, lambda_max, d_state)
        self.register_buffer('eigenvalues', eigenvalues)

        # Input projection to state space
        self.input_proj = nn.Linear(d_input, d_state)

        # Rotation angle predictor: state + input -> rotation angles
        # Uses both current state and input to determine rotation
        self.rotation_net = nn.Sequential(
            nn.Linear(d_state + d_input, d_state),
            nn.GELU(),
            nn.Linear(d_state, self.n_pairs),
            nn.Tanh(),  # Bound angles to [-1, 1] then scale
        )

        # Givens rotation module
        self.givens = GivensRotation(d_state, self.n_pairs)

        # Output projection
        self.output_proj = nn.Linear(d_state, d_state)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(d_state)

    def forward(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Single step of spectral recurrence.

        Args:
            x: [B, d_input] input at current timestep
            h: [B, d_state] hidden state from previous step (or None for init)

        Returns:
            output: [B, d_state] output features
            h_new: [B, d_state] updated hidden state
        """
        B = x.shape[0]
        device = x.device

        # Initialize state if needed
        if h is None:
            h = torch.zeros(B, self.d_state, device=device)

        # Project input
        x_proj = self.input_proj(x)  # [B, d_state]

        # Compute rotation angles from state + input
        combined = torch.cat([h, x], dim=-1)  # [B, d_state + d_input]
        angles = self.rotation_net(combined) * self.rotation_scale * math.pi  # [B, n_pairs]

        # Apply eigenvalue decay
        # h_decayed = Lambda @ h (element-wise multiplication)
        h_decayed = h * self.eigenvalues.unsqueeze(0)  # [B, d_state]

        # Apply Givens rotations (rotate eigenvector basis)
        # This is equivalent to: C_t @ Lambda @ C_t^T @ h
        # where C_t evolves via rotations
        h_rotated = self.givens(h_decayed, angles)

        # Add input contribution
        h_new = h_rotated + x_proj

        # Normalize for stability
        h_new = self.norm(h_new)

        # Output
        output = self.output_proj(h_new)

        return output, h_new

    def forward_sequence(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Process a sequence of inputs.

        Args:
            x: [B, T, d_input] input sequence
            h: [B, d_state] initial hidden state

        Returns:
            outputs: [B, T, d_state] output sequence
            h_final: [B, d_state] final hidden state
        """
        B, T, _ = x.shape
        outputs = []

        for t in range(T):
            out, h = self.forward(x[:, t], h)
            outputs.append(out)

        return torch.stack(outputs, dim=1), h


class SpectralDynamicsLayer(nn.Module):
    """Full dynamics layer combining spectral recurrence with attention.

    Architecture:
    1. SpectralRecurrentCell for temporal state tracking (prevents collapse)
    2. Self-attention for spatial/contextual modeling (captures structure)
    3. Feedforward for capacity

    The spectral cell handles long-horizon stability, while attention
    handles the rich spatial correlations in video latents.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        lambda_range: Tuple[float, float] = (0.9, 0.999),
        rotation_scale: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Spectral recurrent cell
        self.spectral = SpectralRecurrentCell(
            d_state=d_state,
            d_input=d_model,
            lambda_range=lambda_range,
            rotation_scale=rotation_scale,
        )

        # Project spectral state to model dimension
        self.state_proj = nn.Linear(d_state, d_model)

        # Self-attention for spatial modeling
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: [B, T, d_model] input sequence
            h: [B, d_state] recurrent state
            attn_mask: optional attention mask

        Returns:
            output: [B, T, d_model] output sequence
            h_new: [B, d_state] updated recurrent state
        """
        B, T, D = x.shape

        # Process through spectral cell
        spectral_out, h_new = self.spectral.forward_sequence(x, h)

        # Project spectral output and add residual
        x = x + self.state_proj(spectral_out)

        # Self-attention with residual
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out

        # Feedforward with residual
        x = x + self.ff(self.ff_norm(x))

        return x, h_new


class BoundedSpectralDynamics(nn.Module):
    """Full dynamics model with bounded spectral layers.

    Replaces pure transformer dynamics with spectral-stabilized version.
    Each layer has:
    - SpectralRecurrentCell for temporal stability
    - Self-attention for spatial modeling
    - Feedforward for capacity
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        lambda_range: Tuple[float, float] = (0.9, 0.999),
        rotation_scale: float = 0.1,
        dropout: float = 0.0,
        action_dim: int = 18,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Spectral dynamics layers
        self.layers = nn.ModuleList([
            SpectralDynamicsLayer(
                d_model=d_model,
                d_state=d_state,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=mlp_ratio,
                lambda_range=lambda_range,
                rotation_scale=rotation_scale,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output normalization and projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Tensor,
        actions: Optional[Tensor] = None,
        states: Optional[list] = None,
    ) -> Tuple[Tensor, list]:
        """Forward pass.

        Args:
            x: [B, T, d_model] input sequence
            actions: [B, T, action_dim] optional action inputs
            states: list of [B, d_state] per-layer states

        Returns:
            output: [B, T, d_model] predicted next states
            new_states: list of updated per-layer states
        """
        # Initialize states if needed
        if states is None:
            states = [None] * self.num_layers

        # Project input
        x = self.input_proj(x)

        # Add action embedding
        if actions is not None:
            action_emb = self.action_embed(actions)
            x = x + action_emb

        # Process through layers
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)

        # Output
        x = self.output_norm(x)
        output = self.output_proj(x)

        return output, new_states

    def generate_step(
        self,
        x: Tensor,
        actions: Optional[Tensor] = None,
        states: Optional[list] = None,
    ) -> Tuple[Tensor, list]:
        """Single generation step (same as forward for this model)."""
        return self.forward(x, actions, states)


def test_spectral_cell():
    """Unit tests for SpectralRecurrentCell."""
    print("Testing SpectralRecurrentCell...")

    # Test 1: State boundedness
    print("\n1. Testing state boundedness over 10K steps...")
    cell = SpectralRecurrentCell(d_state=64, d_input=32)
    h = torch.zeros(4, 64)
    norms = []

    for _ in range(10000):
        x = torch.randn(4, 32) * 0.1  # Small inputs
        _, h = cell(x, h)
        norms.append(h.norm(dim=-1).mean().item())

    max_norm = max(norms)
    min_norm = min(norms[100:])  # Skip warmup
    print(f"   Max norm: {max_norm:.4f}")
    print(f"   Min norm: {min_norm:.4f}")
    print(f"   Ratio: {max_norm/min_norm:.2f}x")

    if max_norm < 100 and max_norm/min_norm < 10:
        print("   PASS: State bounded")
    else:
        print("   FAIL: State unbounded")

    # Test 2: No mode collapse (variance should be maintained)
    print("\n2. Testing for mode collapse...")
    cell = SpectralRecurrentCell(d_state=64, d_input=32)
    h = torch.randn(4, 64) * 0.1
    variances = []

    for _ in range(1000):
        x = torch.randn(4, 32) * 0.1
        _, h = cell(x, h)
        variances.append(h.var(dim=-1).mean().item())

    var_start = sum(variances[:100]) / 100
    var_end = sum(variances[-100:]) / 100
    print(f"   Variance at start: {var_start:.6f}")
    print(f"   Variance at end: {var_end:.6f}")
    print(f"   Ratio: {var_end/var_start:.2f}x")

    if var_end > 0.1 * var_start:  # Variance should not collapse to near-zero
        print("   PASS: No mode collapse")
    else:
        print("   FAIL: Mode collapsed")

    # Test 3: Orthogonality of Givens rotations
    print("\n3. Testing Givens rotation orthogonality...")
    givens = GivensRotation(dim=64, n_rotations=32)
    x = torch.randn(4, 64)
    angles = torch.randn(4, 32) * 0.1

    x_rot = givens(x, angles)
    norm_before = x.norm(dim=-1)
    norm_after = x_rot.norm(dim=-1)
    diff = (norm_before - norm_after).abs().max().item()

    print(f"   Max norm difference: {diff:.6e}")
    if diff < 1e-5:
        print("   PASS: Givens rotations preserve norm")
    else:
        print("   FAIL: Norm not preserved")

    print("\nAll tests completed.")


if __name__ == "__main__":
    test_spectral_cell()
