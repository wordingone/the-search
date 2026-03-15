"""Genesis: Complete world model combining tokenizer + dynamics + generation.

Target: Sub-1B parameters, 720p @ 24 FPS, unlimited horizon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from einops import rearrange


@dataclass
class GenesisConfig:
    """Configuration for Genesis world model."""
    # Resolution
    image_size: int = 64
    num_frames: int = 16

    # Tokenizer
    tokenizer_type: str = "motion"  # "motion" or "vae"
    keyframe_interval: int = 8
    keyframe_channels: int = 8
    motion_channels: int = 4
    residual_channels: int = 4
    use_fsq: bool = False  # Enable FSQ discrete quantization (prevents corruption)

    # Dynamics
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64
    window_size: int = 256
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Slots
    use_slots: bool = True
    num_slots: int = 8
    slot_dim: int = 64
    slot_decay: float = 0.95
    slot_norm_mode: str = 'decay'  # 'decay', 'layernorm', or 'none'

    # BSD (Bounded Spectral Dynamics) for long-horizon stability
    use_bsd: bool = False  # Use BSD instead of transformer for dynamics
    bsd_d_state: int = 256  # BSD hidden state dimension
    bsd_lambda_range: tuple = (0.9, 0.999)  # Eigenvalue range
    bsd_rotation_scale: float = 0.1  # Rotation angle scale

    # Action
    action_dim: int = 18
    continuous_actions: bool = False

    # Memory optimization
    gradient_checkpoint: bool = False  # Trade ~15% speed for ~30% memory reduction


class Genesis(nn.Module):
    """Genesis World Model.

    Complete pipeline:
    1. Encode video to latent tokens (motion-aware tokenizer)
    2. Process with slot attention for object permanence
    3. Predict dynamics with transformer + KV-cache
    4. Decode latent tokens back to video

    Supports:
    - Action-conditioned generation
    - Unlimited horizon via bounded slot attention
    - Real-time inference via KV-cache
    """

    def __init__(self, config: GenesisConfig = None):
        super().__init__()
        if config is None:
            config = GenesisConfig()
        self.config = config

        # Build tokenizer
        if config.tokenizer_type == "motion":
            from genesis.tokenizer.motion import MotionAwareTokenizer
            # Let tokenizer compute adaptive channels based on resolution
            # This maintains constant bits/pixel across resolutions:
            # - 64x64: 8 channels (4.0 bpp)
            # - 256x256: 16 channels (2.0 bpp, improved from 1.0 bpp)
            # - 512x512: 32 channels (2.0 bpp)
            self.tokenizer = MotionAwareTokenizer(
                in_channels=3,
                hidden_channels=128,  # Increased from 64 for 720p decoder capacity
                keyframe_interval=config.keyframe_interval,
                image_size=config.image_size,
                adaptive_channels=True,  # Enable adaptive channel scaling
                use_fsq=config.use_fsq,  # Discrete quantization prevents corruption
            )
            # Read computed channels from tokenizer (not config!)
            self.latent_channels = self.tokenizer.latent_channels
            self.latent_height = self.tokenizer.latent_size
            self.latent_width = self.tokenizer.latent_size
        else:
            from genesis.tokenizer import VideoTokenizer
            from genesis.config import TokenizerConfig
            tok_config = TokenizerConfig()
            # Adaptive downsampling for VAE tokenizer
            if config.image_size <= 64:
                tok_config.spatial_downsample = 8
            else:
                tok_config.spatial_downsample = max(8, config.image_size // 16)
            self.tokenizer = VideoTokenizer(tok_config)
            self.latent_channels = tok_config.latent_channels
            self.latent_height = config.image_size // tok_config.spatial_downsample
            self.latent_width = config.image_size // tok_config.spatial_downsample

        # Build dynamics model
        if config.use_bsd:
            # Use Bounded Spectral Dynamics for long-horizon stability
            from genesis.dynamics.model import SlotBSDDynamicsModel
            self.dynamics = SlotBSDDynamicsModel(
                latent_channels=self.latent_channels,
                latent_height=self.latent_height,
                latent_width=self.latent_width,
                num_slots=config.num_slots,
                slot_dim=config.slot_dim,
                slot_decay=config.slot_decay,
                d_state=config.bsd_d_state,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                mlp_ratio=config.mlp_ratio,
                lambda_range=config.bsd_lambda_range,
                rotation_scale=config.bsd_rotation_scale,
                dropout=config.dropout,
                action_dim=config.action_dim,
            )
            self._use_bsd = True
        elif config.use_slots:
            from genesis.dynamics.model import SlotLatentDynamicsModel
            self.dynamics = SlotLatentDynamicsModel(
                latent_channels=self.latent_channels,
                latent_height=self.latent_height,
                latent_width=self.latent_width,
                num_slots=config.num_slots,
                slot_dim=config.slot_dim,
                slot_decay=config.slot_decay,
                slot_norm_mode=config.slot_norm_mode,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                window_size=config.window_size,
                action_dim=config.action_dim,
            )
            self._use_bsd = False
        else:
            from genesis.dynamics.model import LatentDynamicsModel
            self.dynamics = LatentDynamicsModel(
                latent_channels=self.latent_channels,
                latent_height=self.latent_height,
                latent_width=self.latent_width,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                window_size=config.window_size,
                action_dim=config.action_dim,
            )
            self._use_bsd = False

        # Action encoder
        if config.continuous_actions:
            self.action_encoder = nn.Sequential(
                nn.Linear(config.action_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.action_dim),
            )
        else:
            self.action_encoder = nn.Embedding(config.action_dim, config.action_dim)

        # State for generation
        self.prev_slots = None
        self.kv_caches = None
        self.bsd_states = None  # BSD layer states

    def encode(self, video: Tensor) -> Tuple[Tensor, List[dict]]:
        """Encode video to latent tokens.

        Args:
            video: [B, T, C, H, W] input video

        Returns:
            latents: [B, T, C', H', W'] latent tokens
            intermediates: encoding metadata
        """
        if self.config.tokenizer_type == "motion":
            latents, intermediates = self.tokenizer.encode(video)
        else:
            out = self.tokenizer(video)
            latents = out['codes']
            intermediates = [{'type': 'vae'}]
        return latents, intermediates

    def decode(
        self,
        latents: Tensor,
        start_frame_idx: int = 0,
        prev_frame: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode latent tokens to video.

        Args:
            latents: [B, T, C', H', W'] latent tokens
            start_frame_idx: Frame index that latents[:, 0] corresponds to.
                Set to 1 when decoding dynamics-predicted latents.
            prev_frame: [B, C, H, W] previous decoded frame for P-frame context.

        Returns:
            video: [B, T, C, H, W] decoded video
        """
        H = self.config.image_size
        W = self.config.image_size

        if self.config.tokenizer_type == "motion":
            video = self.tokenizer.decode(
                latents, (H, W),
                start_frame_idx=start_frame_idx,
                prev_frame=prev_frame,
            )
        else:
            video = self.tokenizer.decode(latents)
        return video

    def encode_action(self, actions: Tensor) -> Tensor:
        """Encode actions.

        Args:
            actions: [B, T] discrete or [B, T, D] continuous

        Returns:
            encoded: [B, T, action_dim]
        """
        if self.config.continuous_actions:
            return self.action_encoder(actions)
        else:
            if actions.dim() == 2:
                return self.action_encoder(actions)
            else:
                return actions

    def forward(
        self,
        video: Tensor,
        actions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Full forward pass for training.

        Args:
            video: [B, T, C, H, W] input video
            actions: [B, T-1] or [B, T-1, D] actions

        Returns:
            dict with predictions and losses
        """
        B, T, C, H, W = video.shape
        use_gc = self.config.gradient_checkpoint and self.training

        # Encode video to latent
        if use_gc:
            # Gradient checkpointing: recompute encode during backward to save memory.
            # encode() returns (latents, intermediates) but checkpoint() expects
            # tensor-only returns. Use a wrapper that returns only the latent tensor.
            def _encode_only(v):
                lat, _ = self.encode(v)
                return lat
            latents = grad_checkpoint(_encode_only, video, use_reentrant=False)
            intermediates = []  # Not needed for loss computation
        else:
            latents, intermediates = self.encode(video)

        # Encode actions
        if actions is not None:
            actions = self.encode_action(actions)

        # Dynamics prediction
        if self._use_bsd:
            # BSD model returns (pred, slots, bsd_states)
            pred_latents, slots, _ = self.dynamics(latents[:, :-1], actions)
        elif self.config.use_slots:
            pred_latents, slots = self.dynamics(latents[:, :-1], actions)
        else:
            pred_latents = self.dynamics(latents[:, :-1], actions)
            slots = None

        # Compute loss in latent space
        target_latents = latents[:, 1:]
        latent_loss = F.mse_loss(pred_latents, target_latents)

        # Tokenizer roundtrip loss: encode -> decode on first 2 frames (I-frame + P-frame)
        # Bug #4 Fix B: Extended to [:2] to supervise P-frame residual decoder
        # Without this, training pushes the tokenizer into degenerate latents
        # that dynamics can predict easily but the decoder cannot reconstruct
        if use_gc:
            tok_recon = grad_checkpoint(
                self.decode, latents[:, :2], use_reentrant=False
            )
        else:
            tok_recon = self.decode(latents[:, :2])
        tok_recon_loss = F.mse_loss(tok_recon, video[:, :2])

        # Get decoded frame 0 for P-frame context in dynamics decode
        # Bug #6 Fix: pred_latents predict frames 1..T-1, which are ALL P-frames
        # (unless a keyframe boundary is crossed). The decoder must know the correct
        # frame indices so it uses P-frame decode (warp + residual) instead of
        # I-frame decode (keyframe channels through Sigmoid).
        prev_decoded_frame = tok_recon[:, 0].detach()  # Frame 0 from tokenizer roundtrip

        # Decode dynamics predictions with correct frame type context
        if use_gc:
            def _decode_pred(pred_lat, prev_f):
                return self.decode(pred_lat, start_frame_idx=1, prev_frame=prev_f)
            pred_video = grad_checkpoint(
                _decode_pred, pred_latents, prev_decoded_frame,
                use_reentrant=False,
            )
        else:
            pred_video = self.decode(
                pred_latents, start_frame_idx=1, prev_frame=prev_decoded_frame,
            )
        target_video = video[:, 1:]

        # Dynamics -> decode reconstruction loss
        recon_loss = F.mse_loss(pred_video, target_video)

        return {
            "pred_video": pred_video,
            "target_video": target_video,
            "pred_latents": pred_latents,
            "target_latents": target_latents,
            "latent_loss": latent_loss,
            "recon_loss": recon_loss,
            "tok_recon_loss": tok_recon_loss,
            "total_loss": latent_loss + 1.0 * recon_loss + 0.5 * tok_recon_loss,
            "slots": slots,
            "intermediates": intermediates,
        }

    def reset_state(self):
        """Reset generation state."""
        self.prev_slots = None
        self.kv_caches = None
        self.bsd_states = None

    @torch.no_grad()
    def generate(
        self,
        initial_video: Tensor,
        actions: Tensor,
        num_frames: int = 16,
    ) -> Tensor:
        """Generate video autoregressively.

        Args:
            initial_video: [B, T_init, C, H, W] context frames
            actions: [B, num_frames] or [B, num_frames, D] actions
            num_frames: number of frames to generate

        Returns:
            generated: [B, T_init + num_frames, C, H, W] full video
        """
        self.eval()
        B, T_init = initial_video.shape[:2]
        device = initial_video.device

        # Encode initial context
        latents, _ = self.encode(initial_video)

        # Encode actions
        if actions is not None:
            actions = self.encode_action(actions)

        # Simple autoregressive generation
        generated_latents = [latents]
        T_context = latents.shape[1]

        # Use full context for first prediction
        current_context = latents

        for step in range(num_frames):
            # Get action for this step (if available)
            action_idx = min(step, actions.shape[1] - 1) if actions is not None else 0
            action = actions[:, action_idx:action_idx+1] if actions is not None else None

            # Expand action to match context length if needed
            if action is not None and current_context.shape[1] > 1:
                action = action.expand(-1, current_context.shape[1], -1)

            # Predict next latent
            if self._use_bsd:
                # BSD model returns (pred, slots, bsd_states)
                pred, self.prev_slots, self.bsd_states = self.dynamics(
                    current_context, action, self.prev_slots, self.bsd_states
                )
                next_latent = pred[:, -1:]
            elif self.config.use_slots:
                pred, self.prev_slots = self.dynamics(current_context, action, self.prev_slots)
                next_latent = pred[:, -1:]
            else:
                pred = self.dynamics(current_context, action)
                next_latent = pred[:, -1:]

            generated_latents.append(next_latent)

            # Update context (use sliding window to bound memory)
            window = min(T_context + step + 1, self.config.window_size // 4)
            current_context = torch.cat([current_context, next_latent], dim=1)[:, -window:]

        all_latents = torch.cat(generated_latents, dim=1)

        # Decode all latents
        generated_video = self.decode(all_latents)

        return generated_video

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get parameter count by component."""
        return {
            "tokenizer": sum(p.numel() for p in self.tokenizer.parameters()),
            "dynamics": sum(p.numel() for p in self.dynamics.parameters()),
            "action_encoder": sum(p.numel() for p in self.action_encoder.parameters()),
            "total": self.count_parameters(),
        }


def create_genesis(
    size: str = "small",
    image_size: int = 64,
    use_slots: bool = True,
    slot_norm_mode: str = "decay",  # Default to decay for training
) -> Genesis:
    """Factory function to create Genesis models.

    Args:
        size: "tiny", "small", "medium", "large"
        image_size: input resolution
        use_slots: whether to use slot attention
        slot_norm_mode: slot normalization mode ("decay", "layernorm", "clip", "none")

    Training/Inference Workflow:
        - TRAINING: Use slot_norm_mode="decay" (default) to learn diversity generation.
          The decay forces the model to actively regenerate content each step.
        - INFERENCE: Override with slot_norm_mode="clip" at test time for stability.
          Clip preserves learned diversity while preventing state explosion.

        Training with "clip" from scratch causes mode collapse because the model
        converges to a trivial fixed-point solution.

    Returns:
        Genesis model
    """
    configs = {
        "tiny": GenesisConfig(
            image_size=image_size,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            head_dim=64,
            num_slots=4,
            use_slots=use_slots,
            slot_norm_mode=slot_norm_mode,
        ),
        "small": GenesisConfig(
            image_size=image_size,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            head_dim=64,
            num_slots=8,
            use_slots=use_slots,
            slot_norm_mode=slot_norm_mode,
        ),
        "medium": GenesisConfig(
            image_size=image_size,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            num_slots=12,
            use_slots=use_slots,
            slot_norm_mode=slot_norm_mode,
        ),
        "large": GenesisConfig(
            image_size=image_size,
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            head_dim=64,
            num_slots=16,
            use_slots=use_slots,
            slot_norm_mode=slot_norm_mode,
        ),
    }

    config = configs.get(size, configs["small"])
    return Genesis(config)
