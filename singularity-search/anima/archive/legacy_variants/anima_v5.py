"""
Anima V5: Temporal Horizon Architecture
========================================

Formal System: S = (V, tau, F, phi, H)

where:
  V = {W, I}                    -- World State, Internal State
  tau: V -> {S, M}              -- Type assignment (State, Memory)
  F: V x E -> V                 -- Evolution function
  phi: I x H -> A               -- Action from internal state + horizon
  H = (H_short, H_long, alpha)  -- Temporal Horizon Structure

Key Innovation:
  Goal is NOT a separate pathway (like V4-Telos) but integrated into
  temporal processing through H. This enhances ALL tasks:
  - Reasoning tasks (sequence, pattern, analogy): H_long captures structure
  - Reactive tasks (collision, projectile): H_short provides immediacy
  - Navigation tasks (goal-seeking): Both horizons combined

Critical Property: grad(phi)/grad(H) != 0
  Horizon causally influences action selection.

Parameter Budget: ~25k (under 100k constraint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class AnimaV5Config:
    """Configuration for AnimaV5 temporal horizon architecture."""

    # Dimensions
    sensory_dim: int = 8          # Input observation dimension
    world_dim: int = 32           # W state dimension
    internal_dim: int = 32        # I state dimension
    horizon_dim: int = 16         # H dimension (both short and long)
    action_dim: int = 4           # Output action dimension

    # Temporal parameters
    short_decay: float = 0.7      # Fast decay for short horizon
    long_decay: float = 0.95      # Slow decay for long horizon
    horizon_history: int = 10     # Steps to compress for long horizon

    # Architecture
    hidden_dim: int = 32          # Hidden layer size
    use_adaptive_alpha: bool = True   # Learn alpha (V5-B) vs fixed (V5-A)
    use_adaptive_decay: bool = False  # Learn decay rates (V5-C)

    # Memory
    memory_slots: int = 4         # Number of memory slots

    # Stability
    max_state_norm: float = 10.0  # Maximum state norm before reset

    def __post_init__(self):
        """Validate configuration."""
        assert self.sensory_dim > 0
        assert self.world_dim > 0
        assert self.internal_dim > 0
        assert self.horizon_dim > 0
        assert 0 < self.short_decay < 1
        assert 0 < self.long_decay < 1
        assert self.short_decay < self.long_decay  # Short decays faster


class AnimaV5(nn.Module):
    """
    Anima V5: Temporal Horizon Architecture

    Formal System: S = (V, tau, F, phi, H)

    V = {W, I} with types:
      W -> S (State): What IS in the world
      I -> M (Memory): What WAS experienced

    H = (H_short, H_long, alpha) where:
      H_short: Short-term horizon (~1-5 steps, fast decay)
      H_long:  Long-term horizon (~10-100 steps, slow integration)
      alpha:   Learned balance per context

    Key property: H enhances ALL tasks via temporal context:
    - Short horizon: immediate reactive context (collision, projectile)
    - Long horizon: planning/pattern context (sequence, analogy, goal)
    - Alpha: system learns optimal balance per task type
    """

    def __init__(self, config: Optional[AnimaV5Config] = None):
        super().__init__()
        self.config = config or AnimaV5Config()

        # ===== WORLD STATE (W) =====
        # Encodes external world observations
        self.world_encoder = nn.Linear(
            self.config.sensory_dim,
            self.config.world_dim
        )
        self.world_gru = nn.GRUCell(
            self.config.world_dim,
            self.config.world_dim
        )

        # World predictor (for prediction error)
        self.world_predictor = nn.Sequential(
            nn.Linear(self.config.world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.world_dim)
        )

        # ===== INTERNAL STATE (I) =====
        # Self-model updated from prediction error
        self.internal_gru = nn.GRUCell(
            self.config.world_dim,  # Error signal dimension
            self.config.internal_dim
        )

        # ===== TEMPORAL HORIZONS (H) =====
        # Short horizon: fast decay, recent emphasis
        self.short_horizon_gru = nn.GRUCell(
            self.config.world_dim,
            self.config.horizon_dim
        )

        # Long horizon: slow integration, structural emphasis
        self.long_horizon_gru = nn.GRUCell(
            self.config.horizon_dim,  # Compresses short horizon history
            self.config.horizon_dim
        )

        # Compression for long horizon
        self.horizon_compressor = nn.Linear(
            self.config.horizon_dim * self.config.horizon_history,
            self.config.horizon_dim
        )

        # ===== HORIZON BALANCE (alpha) =====
        if self.config.use_adaptive_alpha:
            # Learned alpha network
            self.alpha_net = nn.Sequential(
                nn.Linear(
                    self.config.internal_dim + self.config.horizon_dim * 2,
                    self.config.hidden_dim
                ),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            # Fixed alpha = 0.5
            self.register_buffer('fixed_alpha', torch.tensor(0.5))

        # ===== ADAPTIVE DECAY (V5-C only) =====
        if self.config.use_adaptive_decay:
            self.decay_net = nn.Sequential(
                nn.Linear(self.config.internal_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, 2),
                nn.Sigmoid()  # Output in [0, 1]
            )

        # ===== ACTION HEAD (phi) =====
        # phi: I x H -> A
        self.action_head = nn.Sequential(
            nn.Linear(
                self.config.internal_dim + self.config.horizon_dim,
                self.config.hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_dim)
        )

        # ===== MEMORY =====
        # Separate world and internal memory
        self.register_buffer(
            'world_memory',
            torch.zeros(1, self.config.memory_slots, self.config.world_dim)
        )
        self.register_buffer(
            'internal_memory',
            torch.zeros(1, self.config.memory_slots, self.config.internal_dim)
        )

        # ===== STATE INITIALIZATION =====
        self.reset_state()

    def reset_state(self):
        """Reset all dynamic state."""
        device = next(self.parameters()).device

        # Core states
        self.W = torch.zeros(1, self.config.world_dim, device=device)
        self.I = torch.zeros(1, self.config.internal_dim, device=device)

        # Temporal horizons
        self.H_short = torch.zeros(1, self.config.horizon_dim, device=device)
        self.H_long = torch.zeros(1, self.config.horizon_dim, device=device)

        # Short horizon history for long horizon compression
        self.short_history: List[torch.Tensor] = []

        # Current alpha value
        self.alpha = torch.tensor(0.5, device=device)

        # Tracking
        self.step_count = 0
        self.last_error = 0.0

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action."""
        result = self.step(observation)
        return result['action']

    def step(self, observation: torch.Tensor) -> Dict[str, Any]:
        """
        Single step of temporal horizon processing.

        Process:
        1. ENCODE: Observation -> encoded representation
        2. PREDICT: World prediction from current state
        3. ERROR: Compute prediction error
        4. UPDATE_W: World state from observation
        5. UPDATE_I: Internal state from error
        6. UPDATE_H_SHORT: Short horizon (fast)
        7. UPDATE_H_LONG: Long horizon (slow, from compressed history)
        8. COMPUTE_ALPHA: Horizon balance
        9. COMBINE_H: Combined horizon
        10. ACT: Action from I and combined H

        Returns:
            Dict with action, states, horizons, alpha, error
        """
        # Ensure observation has batch dimension
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Handle device
        device = next(self.parameters()).device
        observation = observation.to(device)

        # Initialize states if needed
        if self.W.device != device:
            self.reset_state()

        # Check for state explosion
        if self.W.norm() > self.config.max_state_norm:
            self.reset_state()

        # ===== 1. ENCODE =====
        obs_encoded = self.world_encoder(observation)

        # ===== 2. PREDICT =====
        world_prediction = self.world_predictor(self.W)

        # ===== 3. ERROR =====
        prediction_error = obs_encoded - world_prediction
        error_magnitude = prediction_error.pow(2).mean()
        self.last_error = error_magnitude.item()

        # ===== 4. UPDATE_W (World state from observation) =====
        self.W = self.world_gru(obs_encoded, self.W)

        # ===== 5. UPDATE_I (Internal state from error) =====
        # Error-weighted input: high error -> stronger signal
        error_signal = prediction_error * (1 + error_magnitude)
        self.I = self.internal_gru(error_signal, self.I)

        # ===== 6. UPDATE_H_SHORT (Short horizon - fast) =====
        if self.config.use_adaptive_decay:
            # Learned decay rates
            decays = self.decay_net(self.I)
            short_decay = 0.5 + 0.4 * decays[0, 0]  # Range [0.5, 0.9]
            long_decay = 0.8 + 0.19 * decays[0, 1]  # Range [0.8, 0.99]
        else:
            short_decay = self.config.short_decay
            long_decay = self.config.long_decay

        # Short horizon integrates recent observations quickly
        self.H_short = self.short_horizon_gru(obs_encoded, self.H_short)
        # Apply decay (GRU handles gating, but we add explicit decay)
        self.H_short = short_decay * self.H_short + (1 - short_decay) * obs_encoded[:, :self.config.horizon_dim]

        # ===== 7. UPDATE_H_LONG (Long horizon - slow) =====
        # Accumulate short horizon history
        self.short_history.append(self.H_short.clone().detach())

        if len(self.short_history) >= self.config.horizon_history:
            # Compress history into long horizon
            history_stack = torch.cat(self.short_history[-self.config.horizon_history:], dim=-1)
            compressed = self.horizon_compressor(history_stack)

            # Update long horizon with compressed history
            self.H_long = self.long_horizon_gru(compressed, self.H_long)
            # Apply slow decay
            self.H_long = long_decay * self.H_long + (1 - long_decay) * compressed

            # Keep only recent history (sliding window)
            self.short_history = self.short_history[-(self.config.horizon_history // 2):]

        # ===== 8. COMPUTE_ALPHA (Horizon balance) =====
        if self.config.use_adaptive_alpha:
            alpha_input = torch.cat([self.I, self.H_short, self.H_long], dim=-1)
            self.alpha = self.alpha_net(alpha_input).squeeze(-1)
        else:
            self.alpha = self.fixed_alpha

        # ===== 9. COMBINE_H (Combined horizon) =====
        # H = alpha * H_short + (1 - alpha) * H_long
        H_combined = self.alpha * self.H_short + (1 - self.alpha) * self.H_long

        # ===== 10. ACT (Action from I and H) =====
        # phi: I x H -> A
        action_input = torch.cat([self.I, H_combined], dim=-1)
        action = self.action_head(action_input)

        # Update step count
        self.step_count += 1

        return {
            'action': action,
            'world': self.W.clone(),
            'internal': self.I.clone(),
            'H_short': self.H_short.clone(),
            'H_long': self.H_long.clone(),
            'H_combined': H_combined.clone(),
            'alpha': self.alpha.clone() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'error': error_magnitude,
            'alive': True,
            'step': self.step_count
        }

    def get_horizon_state(self) -> Dict[str, Any]:
        """Get current horizon state for analysis."""
        return {
            'H_short': self.H_short.clone(),
            'H_long': self.H_long.clone(),
            'alpha': self.alpha.clone() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            'history_length': len(self.short_history)
        }


class AnimaV5Fixed(AnimaV5):
    """
    V5-A: Fixed Horizons

    alpha = 0.5 (fixed, no learning)
    Tests if horizon structure alone helps.
    """

    def __init__(self, config: Optional[AnimaV5Config] = None):
        if config is None:
            config = AnimaV5Config(use_adaptive_alpha=False, use_adaptive_decay=False)
        else:
            config.use_adaptive_alpha = False
            config.use_adaptive_decay = False
        super().__init__(config)


class AnimaV5Adaptive(AnimaV5):
    """
    V5-B: Adaptive Horizons (Main variant)

    alpha learned per-context, decay rates fixed.
    Tests if alpha adaptation helps.
    """

    def __init__(self, config: Optional[AnimaV5Config] = None):
        if config is None:
            config = AnimaV5Config(use_adaptive_alpha=True, use_adaptive_decay=False)
        else:
            config.use_adaptive_alpha = True
            config.use_adaptive_decay = False
        super().__init__(config)


class AnimaV5Full(AnimaV5):
    """
    V5-C: Full Adaptive

    alpha learned per-context, decay rates also learned.
    Maximum flexibility.
    """

    def __init__(self, config: Optional[AnimaV5Config] = None):
        if config is None:
            config = AnimaV5Config(use_adaptive_alpha=True, use_adaptive_decay=True)
        else:
            config.use_adaptive_alpha = True
            config.use_adaptive_decay = True
        super().__init__(config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===== QUICK TEST =====
if __name__ == '__main__':
    print("AnimaV5 Temporal Horizon Architecture")
    print("=" * 50)

    # Test all variants
    for name, cls in [('V5-A (Fixed)', AnimaV5Fixed),
                       ('V5-B (Adaptive)', AnimaV5Adaptive),
                       ('V5-C (Full)', AnimaV5Full)]:
        model = cls()
        params = count_parameters(model)
        print(f"\n{name}: {params:,} parameters")

        # Test step
        obs = torch.randn(1, 8)
        for i in range(5):
            result = model.step(obs)

        print(f"  Action shape: {result['action'].shape}")
        print(f"  Alpha: {result['alpha'].item() if isinstance(result['alpha'], torch.Tensor) else result['alpha']:.3f}")
        print(f"  Error: {result['error'].item():.4f}")
        print(f"  H_short norm: {result['H_short'].norm().item():.3f}")
        print(f"  H_long norm: {result['H_long'].norm().item():.3f}")

    print("\n" + "=" * 50)
    print("V5 architecture ready for benchmarking!")
