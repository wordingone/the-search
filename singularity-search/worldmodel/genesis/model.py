"""Genesis: Complete world model assembly."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

from genesis.config import GenesisConfig
from genesis.tokenizer.decoder import VideoTokenizer
from genesis.tokenizer.fsq import FSQ
from genesis.action.encoder import ActionEncoder
from genesis.action.lam import LatentActionModel, ActionConditionedPredictor
from genesis.conditioning.text import TextConditioner, TextConditionerStub
from genesis.conditioning.image import ImageConditioner, ImageConditionerStub
from genesis.conditioning.initializer import WorldInitializer
from genesis.dynamics.backbone import DynamicsBackbone
from genesis.deltav.predictor import DeltaVPredictor, DeltaV
from genesis.memory.ovoxel import OVoxelMemory
from genesis.render.rasterizer import VoxelRasterizer, Camera


class Genesis(nn.Module):
    """
    Genesis: Continuous Interactive World Model.

    Combines all components into a unified model:
    1. Video Tokenizer - encode/decode video to latent
    2. Action System - keyboard/mouse encoding + unsupervised LAM
    3. Conditioning - text/image for world initialization
    4. Dynamics Backbone - autoregressive transformer
    5. DeltaV Predictor - sparse voxel updates
    6. OVoxel Memory - persistent 3D state
    7. Renderer - CUDA rasterization
    """

    def __init__(self, config: Optional[GenesisConfig] = None, use_stubs: bool = False):
        """
        Args:
            config: Model configuration (uses defaults if None)
            use_stubs: Use stub implementations for text/image conditioning
        """
        super().__init__()
        self.config = config or GenesisConfig()

        # Video Tokenizer
        self.tokenizer = VideoTokenizer(self.config.tokenizer)

        # Action System
        self.action_encoder = ActionEncoder(self.config.action.encoder)
        self.lam = LatentActionModel(
            latent_channels=self.config.tokenizer.latent_channels,
            config=self.config.action.lam,
        )
        self.action_predictor = ActionConditionedPredictor(
            latent_channels=self.config.tokenizer.latent_channels,
            action_dim=len(self.config.action.lam.fsq_levels),
        )

        # Conditioning
        if use_stubs:
            self.text_conditioner = TextConditionerStub(self.config.conditioning.text.proj_dim)
            self.image_conditioner = ImageConditionerStub(self.config.conditioning.image.proj_dim)
        else:
            self.text_conditioner = TextConditioner(
                model_name=self.config.conditioning.text.model,
                hidden_dim=self.config.conditioning.text.hidden_dim,
                proj_dim=self.config.conditioning.text.proj_dim,
            )
            self.image_conditioner = ImageConditioner(
                model_name=self.config.conditioning.image.model,
                hidden_dim=self.config.conditioning.image.hidden_dim,
                proj_dim=self.config.conditioning.image.proj_dim,
                freeze=self.config.conditioning.image.freeze,
            )

        self.world_initializer = WorldInitializer(
            config=self.config.conditioning.initializer,
            memory_config=self.config.memory,
            conditioning_dim=self.config.conditioning.text.proj_dim,
        )

        # Dynamics Backbone
        self.dynamics = DynamicsBackbone(
            self.config.dynamics,
            latent_channels=self.config.tokenizer.latent_channels,
        )

        # DeltaV Predictor
        self.deltav = DeltaVPredictor(self.config.deltav)

        # Renderer
        self.renderer = VoxelRasterizer(self.config.render)

        # Runtime state (not nn.Module, managed separately)
        self._memory: Optional[OVoxelMemory] = None
        self._latent_history: Optional[deque] = None
        self._action_history: Optional[deque] = None

    @property
    def memory(self) -> Optional[OVoxelMemory]:
        """Current OVoxel memory state."""
        return self._memory

    def initialize_world(
        self,
        text: Optional[List[str]] = None,
        images: Optional[Tensor] = None,
        video: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> OVoxelMemory:
        """
        Initialize world from conditioning.

        Args:
            text: Text descriptions for initialization
            images: [B, 3, H, W] reference images
            video: [B, T, 3, H, W] bootstrap video
            device: Target device

        Returns:
            Initialized OVoxelMemory
        """
        if device is None:
            device = next(self.parameters()).device

        # Get conditioning
        if text is not None:
            conditioning = self.text_conditioner(text, device)
        elif images is not None:
            conditioning = self.image_conditioner(images, device)
        elif video is not None:
            # Use tokenizer to get latent, then use as conditioning
            latent = self.tokenizer.encode(video.to(device))
            # Pool over time and space
            conditioning = latent.mean(dim=(1, 2, 3)).unsqueeze(1)  # [B, 1, C]
            # Pad to expected conditioning dim
            conditioning = nn.functional.pad(
                conditioning,
                (0, self.config.conditioning.text.proj_dim - conditioning.shape[-1]),
            )
        else:
            # Empty world
            self._memory = OVoxelMemory(self.config.memory, device)
            self._latent_history = deque(maxlen=self.config.dynamics.context_length)
            self._action_history = deque(maxlen=self.config.dynamics.context_length)
            return self._memory

        # Initialize world
        self._memory = self.world_initializer(conditioning, device)
        self._latent_history = deque(maxlen=self.config.dynamics.context_length)
        self._action_history = deque(maxlen=self.config.dynamics.context_length)

        return self._memory

    def step(
        self,
        keyboard: Optional[Tensor] = None,
        mouse: Optional[Tensor] = None,
        action_embedding: Optional[Tensor] = None,
    ) -> Tuple[DeltaV, Tensor]:
        """
        Single simulation step.

        Args:
            keyboard: [B, 6] binary keyboard state
            mouse: [B, 2] continuous mouse delta
            action_embedding: [B, A] pre-computed action (overrides keyboard/mouse)

        Returns:
            delta_v: Sparse voxel updates
            features: Dynamics features (for debugging)
        """
        if self._memory is None:
            raise RuntimeError("World not initialized. Call initialize_world() first.")

        device = next(self.parameters()).device
        B = 1  # Default batch size

        # Encode action
        if action_embedding is not None:
            action = action_embedding
        elif keyboard is not None and mouse is not None:
            action = self.action_encoder(keyboard.to(device), mouse.to(device))
        else:
            # Default: no action
            action = torch.zeros(B, self.config.action.encoder.action_dim, device=device)

        self._action_history.append(action)

        # Get latent history tensor with consistent shape
        H = W = 16  # Default latent spatial size
        C = self.config.dynamics.hidden_dim

        if len(self._latent_history) == 0:
            # Bootstrap with zeros [B, 1, H, W, C]
            latent_tensor = torch.zeros(
                B, 1, H, W, C, device=device
            )
        else:
            # Stack history frames: each frame is [B, H, W, D]
            # Result: [B, T, H, W, D]
            history_list = list(self._latent_history)
            latent_tensor = torch.stack(history_list, dim=1)

        # Get action history tensor [B, T, A]
        action_tensor = torch.stack(list(self._action_history), dim=1)

        # Shape assertions at component boundaries
        assert latent_tensor.dim() == 5, f"Expected [B, T, H, W, C], got shape {latent_tensor.shape}"
        assert action_tensor.dim() == 3, f"Expected [B, T, A], got shape {action_tensor.shape}"
        assert latent_tensor.shape[0] == action_tensor.shape[0], "Batch size mismatch"
        assert latent_tensor.shape[1] == action_tensor.shape[1], f"Time mismatch: latent {latent_tensor.shape[1]} vs action {action_tensor.shape[1]}"

        # Dynamics forward
        features = self.dynamics(latent_tensor, action_tensor)  # [B, H, W, D]

        # Validate output shape
        assert features.dim() == 4, f"Expected [B, H, W, D], got shape {features.shape}"
        assert features.shape[:1] == (B,), f"Batch mismatch: expected {B}, got {features.shape[0]}"

        # Predict deltas
        delta_v = self.deltav(features)

        # Apply to memory
        self._memory.apply_deltas(delta_v)

        # Update latent history: store [B, H, W, D] tensor directly
        # (not unsqueezed - stack handles the time dimension)
        self._latent_history.append(features)

        return delta_v, features

    def render(
        self,
        camera: Camera,
    ) -> Tensor:
        """
        Render current world state.

        Args:
            camera: Camera parameters

        Returns:
            image: [3, H, W] rendered RGB image
        """
        if self._memory is None:
            raise RuntimeError("World not initialized.")

        coords, features = self._memory.get_all_voxels()
        return self.renderer(coords, features, camera)

    def forward(
        self,
        video: Tensor,
        actions: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass for training.

        Args:
            video: [B, T, 3, H, W] input video
            actions: [B, T-1, A] action embeddings (optional)

        Returns:
            dict with outputs for loss computation
        """
        B, T, C, H, W = video.shape
        device = video.device

        # Encode video
        latent = self.tokenizer.encode(video)  # [B, T', H', W', C]

        # Infer latent actions if not provided
        if actions is None:
            actions = []
            for t in range(latent.shape[1] - 1):
                z_t = latent[:, t].permute(0, 3, 1, 2)  # [B, C, H, W]
                z_t1 = latent[:, t + 1].permute(0, 3, 1, 2)
                _, action_q, _ = self.lam(z_t, z_t1)
                actions.append(action_q)

            # Handle single-frame video (no actions to infer)
            if len(actions) == 0:
                # Create dummy action tensor
                action_dim = len(self.config.action.lam.fsq_levels)
                actions = torch.zeros(B, 0, action_dim, device=device)
            else:
                actions = torch.stack(actions, dim=1)  # [B, T'-1, A]

        # Dynamics prediction
        outputs = {
            'latent': latent,
            'actions': actions,
        }

        # Predict next frames
        predicted_features = []
        num_predictions = min(latent.shape[1] - 1, actions.shape[1]) if actions.shape[1] > 0 else 0

        for t in range(num_predictions):
            history = latent[:, :t + 1]
            action_history = actions[:, :t + 1]

            features = self.dynamics(history, action_history)
            predicted_features.append(features)

        if len(predicted_features) > 0:
            outputs['predicted_features'] = torch.stack(predicted_features, dim=1)
        else:
            # Single frame: no predictions possible
            outputs['predicted_features'] = None

        # Tokenizer reconstruction
        recon = self.tokenizer.decode(latent)
        outputs['reconstruction'] = recon

        return outputs

    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts by component."""
        counts = {}

        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        counts['tokenizer'] = count_params(self.tokenizer)
        counts['action_encoder'] = count_params(self.action_encoder)
        counts['lam'] = count_params(self.lam)
        counts['action_predictor'] = count_params(self.action_predictor)
        counts['text_conditioner'] = count_params(self.text_conditioner)
        counts['image_conditioner'] = count_params(self.image_conditioner)
        counts['world_initializer'] = count_params(self.world_initializer)
        counts['dynamics'] = count_params(self.dynamics)
        counts['deltav'] = count_params(self.deltav)
        counts['total'] = count_params(self)

        return counts

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state': self.state_dict(),
            'config': self.config,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "Genesis":
        """Load model from checkpoint."""
        data = torch.load(path, map_location=device)
        model = cls(config=data['config'])
        model.load_state_dict(data['model_state'])
        if device is not None:
            model = model.to(device)
        return model
