"""Latent Action Model - unsupervised action extraction from video."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from genesis.config import LAMConfig
from genesis.tokenizer.fsq import FSQ


class ResBlock2d(nn.Module):
    """2D Residual block."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + x


class LatentActionModel(nn.Module):
    """
    Unsupervised Latent Action Model (LAM).

    Learns to extract discrete action codes from frame transitions
    WITHOUT any action labels. The key insight from Genie:

    1. Given consecutive latent frames z_t and z_{t+1}
    2. Infer what "action" caused the transition
    3. Train by reconstruction: can we predict z_{t+1} from z_t + action?

    Actions are quantized to discrete codes via FSQ.
    """

    def __init__(self, latent_channels: int, config: LAMConfig):
        super().__init__()
        self.config = config
        self.latent_channels = latent_channels

        # Action encoder: takes concatenated consecutive latents
        hidden_ch = config.hidden_channels

        self.encoder = nn.Sequential(
            # Input: [z_t, z_{t+1}] concatenated
            nn.Conv2d(latent_channels * 2, hidden_ch[0], 3, 1, 1),
            ResBlock2d(hidden_ch[0]),
            ResBlock2d(hidden_ch[0]),
            nn.Conv2d(hidden_ch[0], hidden_ch[1], 3, 2, 1),  # Downsample
            ResBlock2d(hidden_ch[1]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Action head: produces continuous action before quantization
        action_dim = len(config.fsq_levels)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_ch[1], hidden_ch[1]),
            nn.GELU(),
            nn.Linear(hidden_ch[1], action_dim),
        )

        # FSQ quantization for discrete actions
        self.fsq = FSQ(config.fsq_levels)

        # Variance tracking for regularization
        self.register_buffer("_action_variance", torch.tensor(1.0))

    @property
    def num_actions(self) -> int:
        """Number of possible discrete actions."""
        return self.fsq.vocab_size

    def forward(
        self,
        z_t: Tensor,
        z_t1: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Infer latent action from frame transition.

        Args:
            z_t: [B, C, H, W] current latent frame
            z_t1: [B, C, H, W] next latent frame

        Returns:
            action_continuous: [B, A] continuous action (pre-quantization)
            action_quantized: [B, A] quantized action (straight-through)
            action_indices: [B] discrete action indices
        """
        # Concatenate consecutive frames
        z_cat = torch.cat([z_t, z_t1], dim=1)

        # Encode to features
        features = self.encoder(z_cat)

        # Predict continuous action
        action_continuous = self.action_head(features)

        # Quantize
        action_quantized, action_indices = self.fsq(action_continuous)

        # Update variance tracking (for regularization)
        with torch.no_grad():
            var = action_continuous.var(dim=0).mean()
            self._action_variance = 0.99 * self._action_variance + 0.01 * var

        return action_continuous, action_quantized, action_indices

    def compute_variance_loss(self, action_continuous: Tensor) -> Tensor:
        """
        Variance regularization to prevent action collapse.

        Without this, the model might map all transitions to the same action.
        We want diverse actions that capture different types of transitions.

        Target variance ~0.1 (after tanh, values in [-1, 1])
        """
        target_var = 0.1
        actual_var = action_continuous.var(dim=0).mean()
        return (actual_var - target_var).pow(2) * self.config.variance_loss_weight

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Convert action indices back to continuous codes."""
        return self.fsq.indices_to_codes(indices)


class ActionConditionedPredictor(nn.Module):
    """
    Predict z_{t+1} from z_t and action.

    Used during training to verify that inferred actions are meaningful:
    if predict(z_t, inferred_action) ≈ z_{t+1}, the action captures the transition.
    """

    def __init__(self, latent_channels: int, action_dim: int):
        super().__init__()

        # FiLM conditioning: action modulates latent features
        self.action_to_scale = nn.Linear(action_dim, latent_channels)
        self.action_to_shift = nn.Linear(action_dim, latent_channels)

        # Prediction network
        hidden_ch = latent_channels * 4  # Use larger hidden dim
        # Find valid group count (must divide hidden_ch)
        for g in [8, 4, 2, 1]:
            if hidden_ch % g == 0:
                num_groups = g
                break
        self.predictor = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_ch, 3, 1, 1),
            nn.GroupNorm(num_groups, hidden_ch),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.GroupNorm(num_groups, hidden_ch),
            nn.SiLU(),
            nn.Conv2d(hidden_ch, latent_channels, 3, 1, 1),
        )

    def forward(self, z_t: Tensor, action: Tensor) -> Tensor:
        """
        Predict next latent frame.

        Args:
            z_t: [B, C, H, W] current latent
            action: [B, A] action embedding

        Returns:
            z_t1_pred: [B, C, H, W] predicted next latent
        """
        # FiLM conditioning
        scale = self.action_to_scale(action)[:, :, None, None]  # [B, C, 1, 1]
        shift = self.action_to_shift(action)[:, :, None, None]

        # Modulate and predict
        z_conditioned = z_t * (1 + scale) + shift
        z_t1_pred = self.predictor(z_conditioned) + z_t  # Residual

        return z_t1_pred
