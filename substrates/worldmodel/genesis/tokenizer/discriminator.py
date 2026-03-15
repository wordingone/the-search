"""3D Patch Discriminator for video tokenizer training."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class ResBlock3d(nn.Module):
    """3D Residual block for discriminator."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))


class PatchDiscriminator3d(nn.Module):
    """
    3D Patch Discriminator for video.

    Outputs patch-wise real/fake predictions rather than single scalar.
    This provides more stable training and better gradients.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        num_layers: int = 3,
    ):
        super().__init__()

        # Use smaller kernels (3x3x3) for better handling of small inputs
        # Initial conv
        layers = [
            nn.Conv3d(in_channels, channels[0], 3, 2, 1),
            nn.LeakyReLU(0.2),
        ]

        # Downsampling layers
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            stride = 2 if i < num_layers - 1 else 1
            layers.extend([
                nn.Conv3d(in_ch, out_ch, 3, stride, 1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2),
            ])
            in_ch = out_ch

        # Final prediction layer - use 1x1x1 for flexibility with small inputs
        layers.append(nn.Conv3d(in_ch, 1, 1, 1, 0))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, C, H, W] or [B, C, T, H, W] video tensor

        Returns:
            patch_logits: [B, 1, T', H', W'] patch-wise predictions
        """
        # Rearrange to [B, C, T, H, W] if needed
        # Input [B, T, C, H, W] has shape[2] == 3 (channels)
        # Input [B, C, T, H, W] has shape[1] == 3 (channels)
        if x.dim() == 5:
            if x.shape[1] != 3 and x.shape[2] == 3:
                # [B, T, C, H, W] -> [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)

        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better gradient flow.

    Uses multiple discriminators at different resolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_scales: int = 3,
        channels: List[int] = [64, 128, 256, 512],
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            PatchDiscriminator3d(in_channels, channels)
            for _ in range(num_scales)
        ])

        self.downsample = nn.AvgPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0,
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, T, C, H, W] or [B, C, T, H, W] video tensor

        Returns:
            list of patch predictions at each scale
        """
        # Rearrange to [B, C, T, H, W] for pooling if needed
        if x.dim() == 5:
            if x.shape[1] != 3 and x.shape[2] == 3:
                # [B, T, C, H, W] -> [B, C, T, H, W]
                x = x.permute(0, 2, 1, 3, 4)

        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)

        return outputs


def hinge_loss_d(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """Hinge loss for discriminator."""
    real_loss = torch.mean(torch.relu(1.0 - real_logits))
    fake_loss = torch.mean(torch.relu(1.0 + fake_logits))
    return real_loss + fake_loss


def hinge_loss_g(fake_logits: Tensor) -> Tensor:
    """Hinge loss for generator."""
    return -torch.mean(fake_logits)


def multiscale_hinge_loss_d(real_outputs: List[Tensor], fake_outputs: List[Tensor]) -> Tensor:
    """Multi-scale hinge loss for discriminator."""
    loss = 0.0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += hinge_loss_d(real, fake)
    return loss / len(real_outputs)


def multiscale_hinge_loss_g(fake_outputs: List[Tensor]) -> Tensor:
    """Multi-scale hinge loss for generator."""
    loss = 0.0
    for fake in fake_outputs:
        loss += hinge_loss_g(fake)
    return loss / len(fake_outputs)
