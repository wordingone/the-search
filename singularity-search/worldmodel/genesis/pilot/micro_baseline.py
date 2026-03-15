"""Micro Baseline: Minimal 2D sequence model for 16x16 @ 8 frames.

This is the control model - no explicit 3D/4D state.
Processes frames as stacked 2D channels, relies on implicit temporal modeling.

Architecture (~65K params - MORE than Genesis, giving it an advantage):
- Encoder: 16x16 grayscale -> 4x4 features (same as Genesis)
- Temporal: All frames stacked channel-wise, processed by 2D convs
- Decoder: 4x4 -> 16x16 predictions (same as Genesis)

The question: Does explicit 4D structure (Genesis) help occlusion recovery
compared to implicit temporal modeling (this baseline)?
"""

import torch
import torch.nn as nn


class MicroBaseline(nn.Module):
    """Minimal 2D sequence model for 16x16 @ 8 frames.

    Same encoder/decoder as Genesis for fair comparison.
    The middle differs: stacks all frames and processes as 2D channels.
    No explicit 3D/4D state - tests whether implicit modeling suffices.
    """

    def __init__(self, channels: int = 16, num_frames: int = 8):
        """Initialize micro baseline model.

        Args:
            channels: Feature channels per frame
            num_frames: Number of input frames
        """
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.pred_frames = num_frames - 2  # Predict frames 2..T-1

        # Encoder: 16x16 grayscale -> 4x4 features
        # Same as Genesis for fair comparison
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=4),
            nn.GELU(),
        )

        # Temporal modeling via stacked 2D convs
        # Input: all frames stacked channel-wise [B, C*T, 4, 4]
        # Output: predicted frames [B, C*(T-2), 4, 4]
        # Match Genesis param count (~28K) for fair comparison
        self.temporal = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * self.pred_frames, kernel_size=3, padding=1),
        )

        # Decoder: 4x4 -> 16x16
        # Same as Genesis for fair comparison
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Forward pass through baseline model.

        Args:
            frames: [B, T, 1, 16, 16] - grayscale sequence

        Returns:
            predictions: [B, T-2, 1, 16, 16] - predicted frames 2..T-1
        """
        B, T, C, H, W = frames.shape

        # Encode all frames
        encoded = []
        for t in range(T):
            enc = self.encoder(frames[:, t])  # [B, channels, 4, 4]
            encoded.append(enc)

        # Stack all frames channel-wise
        stacked = torch.cat(encoded, dim=1)  # [B, channels*T, 4, 4]

        # Process temporally
        out = self.temporal(stacked)  # [B, channels*(T-2), 4, 4]

        # Decode predictions
        predictions = []
        for t in range(self.pred_frames):
            start_ch = t * self.channels
            end_ch = (t + 1) * self.channels
            pred_feat = out[:, start_ch:end_ch, :, :]  # [B, channels, 4, 4]
            pred = self.decoder(pred_feat)  # [B, 1, 16, 16]
            predictions.append(pred)

        return torch.stack(predictions, dim=1)  # [B, T-2, 1, 16, 16]


class MicroBaselineRecurrent(nn.Module):
    """Recurrent baseline variant using ConvGRU-style updates.

    Alternative baseline that processes frames sequentially.
    Still no explicit 3D structure - uses hidden state instead.
    """

    def __init__(self, channels: int = 16, hidden: int = 32):
        super().__init__()
        self.channels = channels
        self.hidden = hidden

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=4),
            nn.GELU(),
        )

        # GRU-style gates (simplified)
        self.gate_z = nn.Conv2d(channels + hidden, hidden, kernel_size=3, padding=1)
        self.gate_r = nn.Conv2d(channels + hidden, hidden, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(channels + hidden, hidden, kernel_size=3, padding=1)

        # Project hidden to output channels
        self.out_proj = nn.Conv2d(hidden, channels, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        device = frames.device

        # Initialize hidden state
        h = torch.zeros(B, self.hidden, 4, 4, device=device)

        predictions = []
        for t in range(T):
            x = self.encoder(frames[:, t])  # [B, channels, 4, 4]
            combined = torch.cat([x, h], dim=1)  # [B, channels+hidden, 4, 4]

            # GRU update
            z = torch.sigmoid(self.gate_z(combined))
            r = torch.sigmoid(self.gate_r(combined))
            h_reset = torch.cat([x, r * h], dim=1)
            h_candidate = torch.tanh(self.candidate(h_reset))
            h = (1 - z) * h + z * h_candidate

            # Predict frames 2 onwards
            if t >= 2:
                out = self.out_proj(h)  # [B, channels, 4, 4]
                pred = self.decoder(out)  # [B, 1, 16, 16]
                predictions.append(pred)

        return torch.stack(predictions, dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Verify model structure and param count
    model = MicroBaseline()
    print(f"MicroBaseline parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 8, 1, 16, 16)  # Batch=2, T=8, C=1, H=16, W=16
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test recurrent variant
    model_rec = MicroBaselineRecurrent()
    print(f"\nMicroBaselineRecurrent parameters: {count_parameters(model_rec):,}")
    y_rec = model_rec(x)
    print(f"Recurrent output shape: {y_rec.shape}")
