"""Image conditioning via CLIP encoder."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class ImageConditioner(nn.Module):
    """
    Image encoder for world conditioning.

    Uses CLIP to encode reference images into embeddings
    that guide world initialization.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        hidden_dim: int = 768,
        proj_dim: int = 1024,
        freeze: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace CLIP model name
            hidden_dim: CLIP hidden dimension
            proj_dim: Output projection dimension
            freeze: Whether to freeze CLIP weights
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.freeze = freeze

        # Projection layer
        self.proj = nn.Linear(hidden_dim, proj_dim)

        # Lazy load
        self._encoder_loaded = False
        self._encoder = None
        self._processor = None

    def _load_encoder(self):
        """Lazy load CLIP encoder."""
        if self._encoder_loaded:
            return

        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor

            self._processor = CLIPImageProcessor.from_pretrained(self.model_name)
            self._encoder = CLIPVisionModel.from_pretrained(self.model_name)

            if self.freeze:
                self._encoder.requires_grad_(False)

            self._encoder_loaded = True
        except ImportError:
            raise ImportError("transformers package required for ImageConditioner")

    def forward(
        self,
        images: Tensor,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Encode images.

        Args:
            images: [B, 3, H, W] RGB images (normalized to [0, 1])
            device: Target device

        Returns:
            embeddings: [B, num_patches+1, proj_dim] image embeddings
        """
        self._load_encoder()

        if device is None:
            device = next(self.proj.parameters()).device

        self._encoder = self._encoder.to(device)
        images = images.to(device)

        # CLIP expects specific normalization
        # Assuming input is [0, 1], apply CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        images_norm = (images - mean[None, :, None, None]) / std[None, :, None, None]

        # Resize to CLIP's expected size (224x224)
        if images_norm.shape[-2:] != (224, 224):
            images_norm = nn.functional.interpolate(
                images_norm, size=(224, 224), mode='bilinear', align_corners=False
            )

        # Encode
        with torch.no_grad():
            outputs = self._encoder(pixel_values=images_norm)
            hidden_states = outputs.last_hidden_state  # [B, num_patches+1, hidden_dim]

        # Project
        embeddings = self.proj(hidden_states)  # [B, L, proj_dim]

        return embeddings

    def get_pooled(self, images: Tensor, device: Optional[torch.device] = None) -> Tensor:
        """
        Get pooled image embedding (CLS token).

        Args:
            images: [B, 3, H, W] RGB images
            device: Target device

        Returns:
            pooled: [B, proj_dim] pooled embeddings
        """
        embeddings = self.forward(images, device)
        return embeddings[:, 0]  # CLS token


class ImageConditionerStub(nn.Module):
    """
    Stub image conditioner for testing without transformers.
    """

    def __init__(self, proj_dim: int = 1024):
        super().__init__()
        self.proj_dim = proj_dim

    def forward(self, images: Tensor, device: Optional[torch.device] = None) -> Tensor:
        B = images.shape[0]
        L = 197  # ViT-B/16: 14*14 + 1 = 197 tokens
        if device is None:
            device = images.device
        return torch.randn(B, L, self.proj_dim, device=device)

    def get_pooled(self, images: Tensor, device: Optional[torch.device] = None) -> Tensor:
        B = images.shape[0]
        if device is None:
            device = images.device
        return torch.randn(B, self.proj_dim, device=device)
