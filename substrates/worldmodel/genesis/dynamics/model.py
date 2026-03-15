"""Latent dynamics model - combines tokenizer with dynamics transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict
from einops import rearrange

from genesis.dynamics.transformer import DynamicsTransformer, KVCache


class LatentDynamicsModel(nn.Module):
    """World model that operates in latent space.

    Architecture:
    1. Tokenizer encodes video to latent tokens
    2. Dynamics transformer predicts next latent tokens
    3. Tokenizer decodes latent tokens back to video

    For training: predict next latent, compute loss in latent space
    For generation: use KV-cache for efficient autoregressive rollout
    """

    def __init__(
        self,
        # Latent dimensions (must match tokenizer output)
        latent_channels: int = 8,
        latent_height: int = 8,
        latent_width: int = 8,
        # Dynamics transformer
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # Action
        action_dim: int = 32,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width

        # Flatten latent spatial dimensions for transformer
        latent_dim = latent_channels * latent_height * latent_width

        # Project flattened latent to transformer dimension
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)

        # Dynamics transformer
        self.dynamics = DynamicsTransformer(
            latent_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            action_dim=action_dim,
        )

    def flatten_latents(self, latents: Tensor) -> Tensor:
        """Flatten spatial dimensions: [B, T, C, H, W] -> [B, T, C*H*W]"""
        B, T = latents.shape[:2]
        return latents.reshape(B, T, -1)

    def unflatten_latents(self, latents: Tensor) -> Tensor:
        """Unflatten spatial dimensions: [B, T, C*H*W] -> [B, T, C, H, W]"""
        B, T = latents.shape[:2]
        return latents.reshape(B, T, self.latent_channels, self.latent_height, self.latent_width)

    def forward(
        self,
        latents: Tensor,
        actions: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict next latent tokens.

        Args:
            latents: [B, T, C, H, W] latent tokens from tokenizer
            actions: [B, T, action_dim] action inputs (optional)

        Returns:
            pred_latents: [B, T, C, H, W] predicted next latents (shifted by 1)
        """
        B, T, C, H, W = latents.shape

        # Flatten spatial dims
        x = self.flatten_latents(latents)  # [B, T, C*H*W]

        # Project to hidden dim
        x = self.latent_to_hidden(x)  # [B, T, hidden_dim]

        # Dynamics prediction
        x, _ = self.dynamics(x, actions, use_cache=False)  # [B, T, hidden_dim]

        # Project back to latent dim
        x = self.hidden_to_latent(x)  # [B, T, C*H*W]

        # Unflatten
        pred_latents = self.unflatten_latents(x)  # [B, T, C, H, W]

        return pred_latents

    def generate(
        self,
        initial_latents: Tensor,
        actions: Optional[Tensor] = None,
        num_steps: int = 16,
    ) -> Tensor:
        """Autoregressive generation with KV-cache.

        Args:
            initial_latents: [B, T_init, C, H, W] initial context
            actions: [B, num_steps, action_dim] actions for generation
            num_steps: number of frames to generate

        Returns:
            generated: [B, T_init + num_steps, C, H, W] full sequence
        """
        B = initial_latents.shape[0]
        device = initial_latents.device

        # Flatten and project initial context
        x = self.flatten_latents(initial_latents)
        x = self.latent_to_hidden(x)

        # Process initial context
        output, kv_caches = self.dynamics(x, use_cache=True)
        next_hidden = output[:, -1:, :]  # [B, 1, hidden_dim]

        # Store generated latents
        generated_hidden = [x]

        # Generate step by step
        for step in range(num_steps):
            action = None
            if actions is not None and step < actions.shape[1]:
                action = actions[:, step:step+1, :]

            output, kv_caches = self.dynamics(
                next_hidden, action, kv_caches, use_cache=True
            )
            next_hidden = output  # [B, 1, hidden_dim]
            generated_hidden.append(next_hidden)

        # Concatenate all hidden states
        all_hidden = torch.cat(generated_hidden, dim=1)  # [B, T_init + num_steps, hidden_dim]

        # Project back to latent space
        all_latents = self.hidden_to_latent(all_hidden)

        # Unflatten
        return self.unflatten_latents(all_latents)

    def compute_loss(
        self,
        latents: Tensor,
        actions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute training loss.

        Args:
            latents: [B, T, C, H, W] ground truth latent sequence
            actions: [B, T-1, action_dim] actions (one less than latents)

        Returns:
            dict with loss components
        """
        # Input: all but last frame
        input_latents = latents[:, :-1]  # [B, T-1, C, H, W]

        # Target: all but first frame
        target_latents = latents[:, 1:]  # [B, T-1, C, H, W]

        # Predict
        pred_latents = self.forward(input_latents, actions)

        # MSE loss in latent space
        mse_loss = F.mse_loss(pred_latents, target_latents)

        # L1 loss for sparsity
        l1_loss = F.l1_loss(pred_latents, target_latents)

        total_loss = mse_loss + 0.1 * l1_loss

        return {
            "total": total_loss,
            "mse": mse_loss,
            "l1": l1_loss,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SlotLatentDynamicsModel(nn.Module):
    """Latent dynamics model with slot attention for object permanence.

    Adds slot attention layer between tokenizer and dynamics for explicit
    object-level representation.
    """

    def __init__(
        self,
        # Latent dimensions
        latent_channels: int = 8,
        latent_height: int = 8,
        latent_width: int = 8,
        # Slot attention
        num_slots: int = 8,
        slot_dim: int = 64,
        slot_iters: int = 3,
        slot_decay: float = 0.95,
        slot_norm_mode: str = 'decay',  # 'decay', 'layernorm', or 'none'
        # Dynamics
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 256,
        # Action
        action_dim: int = 32,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slot_decay = slot_decay
        self.slot_norm_mode = slot_norm_mode

        # Slot prior normalization (alternative to decay)
        if slot_norm_mode == 'layernorm':
            self.slot_prior_norm = nn.LayerNorm(slot_dim)
        else:
            self.slot_prior_norm = None

        # Latent dimensions
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        latent_dim = latent_channels * latent_height * latent_width

        # Project latent tokens to slot query dimension (per spatial location)
        # BUG FIX: Was Linear(latent_dim, slot_dim) which created identical tokens
        # Now projects each spatial location independently: [B, H*W, C] -> [B, H*W, slot_dim]
        self.latent_proj = nn.Linear(latent_channels, slot_dim)

        # 2D positional encoding for spatial tokens
        # Helps slots learn locality and spatial binding
        self.pos_embed = nn.Parameter(torch.randn(1, latent_height * latent_width, slot_dim) * 0.02)

        # Slot attention components
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.1)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        self.slot_attn_norm = nn.LayerNorm(slot_dim)
        self.slot_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_v = nn.Linear(slot_dim, slot_dim, bias=False)

        self.slot_gru = nn.GRUCell(slot_dim, slot_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )
        self.slot_norm = nn.LayerNorm(slot_dim)

        # Dynamics on slots
        self.dynamics = DynamicsTransformer(
            latent_dim=num_slots * slot_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
            action_dim=action_dim,
        )

        # Decode slots back to latent
        self.slot_decoder = nn.Sequential(
            nn.Linear(num_slots * slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Output normalization to match encoder distribution
        # LayerNorm normalizes to std~1.0, then scale maps to encoder std
        # Encoder outputs std ~0.35-0.40 at 256x256 with adaptive channels (16ch)
        # Previously was 8.0 which produced 23x overshoot -> black frames after decode
        self.latent_output_norm = nn.LayerNorm(latent_dim)
        self.latent_scale = nn.Parameter(torch.ones(1) * 1.0)  # Match encoder std (~0.96 at 720p)

    def slot_attention(
        self,
        inputs: Tensor,
        slots: Optional[Tensor] = None,
    ) -> Tensor:
        """Single-frame slot attention.

        Args:
            inputs: [B, N, D] flattened latent tokens
            slots: [B, K, D] previous slots (for temporal continuity)

        Returns:
            slots: [B, K, D] updated slots
        """
        B, N, D = inputs.shape

        # Initialize slots
        if slots is None:
            mu = self.slot_mu.expand(B, -1, -1)
            sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(sigma)
        else:
            # Normalize previous slots based on mode
            if self.slot_norm_mode == 'decay':
                # Original: exponential decay for bounded memory
                slots = slots * self.slot_decay
            elif self.slot_norm_mode == 'layernorm':
                # Alternative: LayerNorm to bound without information loss
                slots = self.slot_prior_norm(slots)
            elif self.slot_norm_mode == 'layernorm_noise':
                # LayerNorm + noise injection to prevent fixed-point collapse
                slots = self.slot_prior_norm(slots)
                if self.training or True:  # Always add noise (inference too for diversity)
                    noise_scale = 0.01  # Small noise to break symmetry
                    slots = slots + noise_scale * torch.randn_like(slots)
            elif self.slot_norm_mode == 'clip':
                # Clip slot norms to max value (allow growth up to threshold)
                # Per-slot norm clip - allows slots to grow independently
                max_norm = 1000.0  # Higher threshold to allow more growth
                slot_norms = torch.norm(slots, dim=-1, keepdim=True)  # [B, K, 1]
                scale = torch.clamp(max_norm / (slot_norms + 1e-6), max=1.0)
                slots = slots * scale
            # else 'none': no normalization (for ablation)

        # Iterative attention
        inputs = self.slot_attn_norm(inputs)

        for _ in range(3):  # slot_iters
            slots_prev = slots

            q = self.slot_q(slots)  # [B, K, D]
            k = self.slot_k(inputs)  # [B, N, D]
            v = self.slot_v(inputs)  # [B, N, D]

            # Attention: slots attend to inputs
            attn = torch.einsum("bkd,bnd->bkn", q, k) / (D ** 0.5)
            attn = F.softmax(attn, dim=1)  # Normalize over slots

            # Weighted sum of values
            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            # GRU update
            slots = self.slot_gru(
                updates.reshape(B * self.num_slots, -1),
                slots_prev.reshape(B * self.num_slots, -1),
            ).reshape(B, self.num_slots, -1)

            # MLP
            slots = slots + self.slot_mlp(self.slot_norm(slots))

        return slots

    def forward(
        self,
        latents: Tensor,
        actions: Optional[Tensor] = None,
        prev_slots: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with slot attention.

        Args:
            latents: [B, T, C, H, W] latent tokens
            actions: [B, T, action_dim] actions
            prev_slots: [B, K, D] slots from previous sequence

        Returns:
            pred_latents: [B, T, C, H, W] predicted latents
            slots: [B, K, D] final slots
        """
        B, T, C, H, W = latents.shape

        all_slots = []
        slots = prev_slots

        # Process each timestep with slot attention
        for t in range(T):
            # BUG FIX: Create UNIQUE tokens per spatial location (was identical copies)
            # Old: [B, C*H*W] -> [B, slot_dim] -> expand -> [B, H*W, slot_dim] (all identical!)
            # New: [B, C, H, W] -> [B, H*W, C] -> [B, H*W, slot_dim] (unique per location)
            frame_latent = latents[:, t]  # [B, C, H, W]
            frame_latent = frame_latent.permute(0, 2, 3, 1)  # [B, H, W, C]
            frame_latent = frame_latent.reshape(B, H * W, C)  # [B, H*W, C]
            frame_tokens = self.latent_proj(frame_latent)  # [B, H*W, slot_dim] - unique per location!

            # Add 2D positional encoding for spatial awareness
            frame_tokens = frame_tokens + self.pos_embed

            # Slot attention
            slots = self.slot_attention(frame_tokens, slots)
            all_slots.append(slots)

        # Stack slots over time: [B, T, K*D]
        slot_sequence = torch.stack(all_slots, dim=1)
        slot_sequence = slot_sequence.reshape(B, T, -1)

        # Dynamics on slot representation
        pred_slots, _ = self.dynamics(slot_sequence, actions, use_cache=False)

        # Decode to latent space
        pred_latents = self.slot_decoder(pred_slots)  # [B, T, C*H*W]
        # Normalize and scale to match encoder distribution (std ~7.8)
        pred_latents = self.latent_output_norm(pred_latents)
        pred_latents = pred_latents * self.latent_scale
        pred_latents = pred_latents.reshape(B, T, C, H, W)

        return pred_latents, slots

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SlotBSDDynamicsModel(nn.Module):
    """Latent dynamics model with slot attention + Bounded Spectral Dynamics.

    Combines:
    1. Slot attention for object-level representation (prevents mode collapse)
    2. BSD for temporal dynamics (bounded state, diverse trajectories)

    This addresses the mode collapse issue observed in pure transformer dynamics.
    """

    def __init__(
        self,
        # Latent dimensions
        latent_channels: int = 8,
        latent_height: int = 8,
        latent_width: int = 8,
        # Slot attention
        num_slots: int = 8,
        slot_dim: int = 64,
        slot_iters: int = 3,
        slot_decay: float = 0.95,
        # BSD dynamics
        d_state: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        hidden_dim: int = 512,
        mlp_ratio: float = 4.0,
        lambda_range: tuple = (0.9, 0.999),
        rotation_scale: float = 0.1,
        dropout: float = 0.0,
        # Action
        action_dim: int = 18,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slot_decay = slot_decay
        self.d_state = d_state

        # Latent dimensions
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        latent_dim = latent_channels * latent_height * latent_width

        # Project latent tokens to slot query dimension (per spatial location)
        # BUG FIX: Was Linear(latent_dim, slot_dim) which created identical tokens
        # Now projects each spatial location independently: [B, H*W, C] -> [B, H*W, slot_dim]
        self.latent_proj = nn.Linear(latent_channels, slot_dim)

        # 2D positional encoding for spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(1, latent_height * latent_width, slot_dim) * 0.02)

        # Slot attention components (same as SlotLatentDynamicsModel)
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.1)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        self.slot_attn_norm = nn.LayerNorm(slot_dim)
        self.slot_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_v = nn.Linear(slot_dim, slot_dim, bias=False)

        self.slot_gru = nn.GRUCell(slot_dim, slot_dim)
        self.slot_mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )
        self.slot_norm = nn.LayerNorm(slot_dim)

        # BSD dynamics instead of transformer
        from genesis.dynamics.spectral import BoundedSpectralDynamics

        slot_hidden_dim = num_slots * slot_dim
        self.dynamics = BoundedSpectralDynamics(
            d_model=slot_hidden_dim,
            d_state=d_state,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            lambda_range=lambda_range,
            rotation_scale=rotation_scale,
            dropout=dropout,
            action_dim=action_dim,
        )

        # Decode slots back to latent
        self.slot_decoder = nn.Sequential(
            nn.Linear(slot_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Output normalization to match encoder distribution (same as SlotLatentDynamicsModel)
        self.latent_output_norm = nn.LayerNorm(latent_dim)
        self.latent_scale = nn.Parameter(torch.ones(1) * 1.0)

    def slot_attention(
        self,
        inputs: Tensor,
        slots: Optional[Tensor] = None,
    ) -> Tensor:
        """Single-frame slot attention (same as SlotLatentDynamicsModel)."""
        B, N, D = inputs.shape

        # Initialize slots
        if slots is None:
            mu = self.slot_mu.expand(B, -1, -1)
            sigma = self.slot_log_sigma.exp().expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(sigma)
        else:
            # Decay previous slots for bounded memory
            slots = slots * self.slot_decay

        # Iterative attention
        inputs = self.slot_attn_norm(inputs)

        for _ in range(3):  # slot_iters
            slots_prev = slots

            q = self.slot_q(slots)  # [B, K, D]
            k = self.slot_k(inputs)  # [B, N, D]
            v = self.slot_v(inputs)  # [B, N, D]

            # Attention: slots attend to inputs
            attn = torch.einsum("bkd,bnd->bkn", q, k) / (D ** 0.5)
            attn = F.softmax(attn, dim=1)  # Normalize over slots

            # Weighted sum of values
            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            # GRU update
            slots = self.slot_gru(
                updates.reshape(B * self.num_slots, -1),
                slots_prev.reshape(B * self.num_slots, -1),
            ).reshape(B, self.num_slots, -1)

            # MLP
            slots = slots + self.slot_mlp(self.slot_norm(slots))

        return slots

    def forward(
        self,
        latents: Tensor,
        actions: Optional[Tensor] = None,
        prev_slots: Optional[Tensor] = None,
        bsd_states: Optional[list] = None,
    ) -> Tuple[Tensor, Tensor, list]:
        """Forward pass with slot attention + BSD.

        Args:
            latents: [B, T, C, H, W] latent tokens
            actions: [B, T, action_dim] actions
            prev_slots: [B, K, D] slots from previous sequence
            bsd_states: list of BSD layer states

        Returns:
            pred_latents: [B, T, C, H, W] predicted latents
            slots: [B, K, D] final slots
            bsd_states: updated BSD states
        """
        B, T, C, H, W = latents.shape

        all_slots = []
        slots = prev_slots

        # Process each timestep with slot attention
        for t in range(T):
            # BUG FIX: Create UNIQUE tokens per spatial location (was identical copies)
            # Old: [B, C*H*W] -> [B, slot_dim] -> expand -> [B, H*W, slot_dim] (all identical!)
            # New: [B, C, H, W] -> [B, H*W, C] -> [B, H*W, slot_dim] (unique per location)
            frame_latent = latents[:, t]  # [B, C, H, W]
            frame_latent = frame_latent.permute(0, 2, 3, 1)  # [B, H, W, C]
            frame_latent = frame_latent.reshape(B, H * W, C)  # [B, H*W, C]
            frame_tokens = self.latent_proj(frame_latent)  # [B, H*W, slot_dim] - unique per location!

            # Add 2D positional encoding for spatial awareness
            frame_tokens = frame_tokens + self.pos_embed

            # Slot attention
            slots = self.slot_attention(frame_tokens, slots)
            all_slots.append(slots)

        # Stack slots over time: [B, T, K*D]
        slot_sequence = torch.stack(all_slots, dim=1)
        slot_sequence = slot_sequence.reshape(B, T, -1)

        # BSD dynamics on slot representation
        pred_slots, new_bsd_states = self.dynamics(slot_sequence, actions, bsd_states)

        # Decode to latent space
        pred_latents = self.slot_decoder(pred_slots)  # [B, T, C*H*W]
        # Normalize and scale to match encoder distribution (std ~0.35-0.40)
        pred_latents = self.latent_output_norm(pred_latents)
        pred_latents = pred_latents * self.latent_scale
        pred_latents = pred_latents.reshape(B, T, C, H, W)

        return pred_latents, slots, new_bsd_states

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
