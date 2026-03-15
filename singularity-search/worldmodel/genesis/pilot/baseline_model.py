"""Pilot Baseline Model - Frame-to-frame predictor without field.

Architecture:
- Same CNN backbone as field model
- Same transformer architecture
- NO persistent field, NO temporal stereo depth
- Pure frame pair → next frame prediction

Target: ~25M parameters (matched to field model within 10%)
"""

import torch
import torch.nn as nn
import math


class FrameEncoder(nn.Module):
    """Encode frame pair to features."""

    def __init__(self, in_channels=6, hidden_dim=544):
        super().__init__()
        # Process concatenated frame pair
        # 64×64 → 8×8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Project to transformer dim
        self.proj = nn.Linear(128, hidden_dim)

    def forward(self, frame_prev, frame_curr):
        """Encode frame pair.

        Args:
            frame_prev: (B, 3, H, W)
            frame_curr: (B, 3, H, W)

        Returns:
            features: (B, N, hidden_dim) where N = 64 (8×8)
        """
        # Concatenate frames
        x = torch.cat([frame_prev, frame_curr], dim=1)

        # CNN backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (B, 128, 8, 8)

        # Flatten spatial dims
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project to hidden dim
        x = self.proj(x)

        return x


class TemporalTransformer(nn.Module):
    """Transformer for temporal reasoning."""

    def __init__(self, hidden_dim=544, num_layers=6, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Positional encoding for 8×8 grid
        self.register_buffer('pos_embed', self._create_positional_encoding(64, hidden_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def _create_positional_encoding(self, num_tokens, dim):
        """Create 2D sinusoidal positional encoding."""
        pos = torch.zeros(num_tokens, dim)

        position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)

        return pos

    def forward(self, x):
        """Apply transformer.

        Args:
            x: (B, N, hidden_dim)

        Returns:
            x: (B, N, hidden_dim)
        """
        # Add positional encoding
        x = x + self.pos_embed.unsqueeze(0)

        # Transform
        x = self.transformer(x)

        return x


class FrameDecoder(nn.Module):
    """Decode features to predicted frame."""

    def __init__(self, hidden_dim=544):
        super().__init__()
        # Project back to spatial features
        self.proj = nn.Linear(hidden_dim, 128)

        # Decoder: 8×8 → 64×64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 64×64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Decode features to frame.

        Args:
            x: (B, N, hidden_dim) where N = 64

        Returns:
            frame: (B, 3, 64, 64)
        """
        B, N, _ = x.shape
        H = W = 8  # sqrt(64)

        # Project
        x = self.proj(x)  # (B, 64, 128)

        # Reshape to spatial
        x = x.reshape(B, H, W, 128).permute(0, 3, 1, 2)

        # Decode
        frame = self.decoder(x)

        return frame


class PilotBaselineModel(nn.Module):
    """Baseline frame-to-frame predictor without field.

    Target: ~25M parameters (matched to field model)
    """

    def __init__(self, hidden_dim=544, num_layers=6, num_heads=8):
        super().__init__()
        self.encoder = FrameEncoder(in_channels=6, hidden_dim=hidden_dim)
        self.transformer = TemporalTransformer(hidden_dim, num_layers, num_heads)
        self.decoder = FrameDecoder(hidden_dim)

    def forward(self, frames):
        """Full forward pass for training.

        Args:
            frames: (B, T, 3, H, W)

        Returns:
            predictions: (B, T-2, 3, H, W)
        """
        B, T, _, H, W = frames.shape

        predictions = []

        # Predict frames t+1 from (t-1, t) for t in [1, T-2]
        for t in range(1, T - 1):
            frame_prev = frames[:, t - 1]
            frame_curr = frames[:, t]

            # Encode
            features = self.encoder(frame_prev, frame_curr)

            # Transform
            features = self.transformer(features)

            # Decode
            pred = self.decoder(features)

            predictions.append(pred)

        # Stack: (B, T-2, 3, H, W)
        predictions = torch.stack(predictions, dim=1)

        return predictions

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Verification
    model = PilotBaselineModel()
    print(f"Parameters: {model.count_parameters():,}")

    # Forward pass test
    frames = torch.randn(4, 10, 3, 64, 64)
    with torch.no_grad():
        preds = model(frames)

    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Expected: (4, 8, 3, 64, 64)")
    print(f"Match: {preds.shape == (4, 8, 3, 64, 64)}")
