"""Keyboard/mouse action encoder."""

import torch
import torch.nn as nn
from torch import Tensor

from genesis.config import ActionEncoderConfig


class ActionEncoder(nn.Module):
    """
    Encode keyboard and mouse inputs to latent action embedding.

    Keyboard: 6 binary inputs (WASD + Space + Shift) -> embedding lookup
    Mouse: 2 continuous values (dx, dy) -> MLP

    Combined via fusion layer.
    """

    def __init__(self, config: ActionEncoderConfig):
        super().__init__()
        self.config = config

        # Keyboard embedding: 2^6 = 64 possible combinations
        num_keyboard_combinations = 2 ** config.keyboard_keys
        self.keyboard_embed = nn.Embedding(num_keyboard_combinations, config.action_dim)

        # Mouse MLP
        self.mouse_mlp = nn.Sequential(
            nn.Linear(config.mouse_dims, config.action_dim // 2),
            nn.GELU(),
            nn.Linear(config.action_dim // 2, config.action_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.action_dim * 2, config.action_dim),
            nn.LayerNorm(config.action_dim),
        )

        # Binary multipliers for keyboard index
        self.register_buffer(
            "_kb_multipliers",
            2 ** torch.arange(config.keyboard_keys, dtype=torch.long)
        )

    def forward(
        self,
        keyboard: Tensor,
        mouse: Tensor,
    ) -> Tensor:
        """
        Encode user inputs to action embedding.

        Args:
            keyboard: [B, 6] binary tensor (W, A, S, D, Space, Shift)
            mouse: [B, 2] continuous tensor (dx, dy)

        Returns:
            action: [B, action_dim] latent action embedding
        """
        # Convert binary keyboard to index
        kb_idx = (keyboard.long() * self._kb_multipliers).sum(dim=-1)
        kb_embed = self.keyboard_embed(kb_idx)

        # Encode mouse
        mouse_embed = self.mouse_mlp(mouse)

        # Fuse
        combined = torch.cat([kb_embed, mouse_embed], dim=-1)
        action = self.fusion(combined)

        return action

    def encode_keyboard_only(self, keyboard: Tensor) -> Tensor:
        """Encode keyboard without mouse input."""
        kb_idx = (keyboard.long() * self._kb_multipliers).sum(dim=-1)
        return self.keyboard_embed(kb_idx)

    def encode_discrete(self, action_id: Tensor) -> Tensor:
        """Encode discrete action ID (0-63) directly."""
        return self.keyboard_embed(action_id)


class ContinuousActionEncoder(nn.Module):
    """
    Alternative encoder for fully continuous action space.

    Used when actions are not discrete keyboard/mouse but
    continuous control vectors (e.g., joystick, velocity commands).
    """

    def __init__(self, input_dim: int, action_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
            nn.LayerNorm(action_dim),
        )

    def forward(self, continuous_action: Tensor) -> Tensor:
        """
        Args:
            continuous_action: [B, input_dim] continuous control vector

        Returns:
            action: [B, action_dim] latent action embedding
        """
        return self.mlp(continuous_action)
