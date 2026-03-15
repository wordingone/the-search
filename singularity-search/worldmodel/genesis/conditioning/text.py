"""Text conditioning via T5-small encoder."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class TextConditioner(nn.Module):
    """
    Text encoder for world conditioning.

    Uses T5-small to encode text descriptions into embeddings
    that guide world initialization.
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        hidden_dim: int = 512,
        proj_dim: int = 1024,
        max_length: int = 512,
    ):
        """
        Args:
            model_name: HuggingFace model name
            hidden_dim: T5 hidden dimension
            proj_dim: Output projection dimension
            max_length: Maximum sequence length
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.max_length = max_length

        # Projection layer (encoder loaded lazily)
        self.proj = nn.Linear(hidden_dim, proj_dim)

        # Lazy load flag
        self._encoder_loaded = False
        self._encoder = None
        self._tokenizer = None

    def _load_encoder(self):
        """Lazy load T5 encoder."""
        if self._encoder_loaded:
            return

        try:
            from transformers import T5EncoderModel, T5Tokenizer

            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._encoder = T5EncoderModel.from_pretrained(self.model_name)
            self._encoder.requires_grad_(False)  # Freeze encoder
            self._encoder_loaded = True
        except ImportError:
            raise ImportError("transformers package required for TextConditioner")

    def forward(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Encode text descriptions.

        Args:
            texts: List of text descriptions
            device: Target device

        Returns:
            embeddings: [B, L, proj_dim] text embeddings
        """
        self._load_encoder()

        if device is None:
            device = next(self.proj.parameters()).device

        # Tokenize
        tokens = self._tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Encode
        self._encoder = self._encoder.to(device)
        with torch.no_grad():
            outputs = self._encoder(**tokens)
            hidden_states = outputs.last_hidden_state  # [B, L, hidden_dim]

        # Project
        embeddings = self.proj(hidden_states)  # [B, L, proj_dim]

        return embeddings

    def get_pooled(self, texts: List[str], device: Optional[torch.device] = None) -> Tensor:
        """
        Get pooled text embedding (single vector per text).

        Args:
            texts: List of text descriptions
            device: Target device

        Returns:
            pooled: [B, proj_dim] pooled embeddings
        """
        embeddings = self.forward(texts, device)
        return embeddings.mean(dim=1)  # Mean pooling


class TextConditionerStub(nn.Module):
    """
    Stub text conditioner for testing without transformers.

    Returns random embeddings with correct shape.
    """

    def __init__(self, proj_dim: int = 1024):
        super().__init__()
        self.proj_dim = proj_dim

    def forward(self, texts: List[str], device: Optional[torch.device] = None) -> Tensor:
        B = len(texts)
        L = 64  # Fixed sequence length for stub
        if device is None:
            device = torch.device('cpu')
        return torch.randn(B, L, self.proj_dim, device=device)

    def get_pooled(self, texts: List[str], device: Optional[torch.device] = None) -> Tensor:
        B = len(texts)
        if device is None:
            device = torch.device('cpu')
        return torch.randn(B, self.proj_dim, device=device)
