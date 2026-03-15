"""3D Causal VAE Encoder for video compression."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from einops import rearrange

from genesis.config import TokenizerConfig


class ResBlock3d(nn.Module):
    """3D Residual block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

        # Skip connection
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample3d(nn.Module):
    """3D downsampling with configurable temporal/spatial strides."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
    ):
        super().__init__()
        stride = (temporal_stride, spatial_stride, spatial_stride)
        padding = (temporal_stride // 2, spatial_stride // 2, spatial_stride // 2)

        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class VideoEncoder(nn.Module):
    """
    3D Causal VAE Encoder.

    Compresses video [B, T, 3, H, W] to latent [B, T//tau, H//16, W//16, C].

    Architecture:
    - 4 downsampling stages
    - ResBlocks at each stage
    - Temporal compression: tau = 4
    - Spatial compression: 16x
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

        # Initial convolution
        self.conv_in = nn.Conv3d(
            config.input_channels,
            config.encoder_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Encoder stages
        self.stages = nn.ModuleList()

        in_ch = config.encoder_channels[0]
        temporal_strides = self._compute_temporal_strides(config.temporal_downsample)
        spatial_strides = self._compute_spatial_strides(config.spatial_downsample)

        for i, (out_ch, depth) in enumerate(zip(
            config.encoder_channels,
            config.encoder_depths,
        )):
            stage = nn.ModuleList()

            # ResBlocks
            for j in range(depth):
                ch = in_ch if j == 0 else out_ch
                stage.append(ResBlock3d(ch, out_ch))

            # Downsampling (except last stage)
            if i < len(config.encoder_channels) - 1:
                next_ch = config.encoder_channels[i + 1]
                stage.append(Downsample3d(
                    out_ch,
                    next_ch,
                    temporal_stride=temporal_strides[i],
                    spatial_stride=spatial_strides[i],
                ))
                in_ch = next_ch
            else:
                in_ch = out_ch

            self.stages.append(stage)

        # Final convolution to latent dimension
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, config.encoder_channels[-1]),
            nn.SiLU(),
            nn.Conv3d(config.encoder_channels[-1], config.latent_channels, 1),
        )

    def _compute_temporal_strides(self, total: int) -> List[int]:
        """Distribute temporal downsampling across stages."""
        # tau=4 -> strides [1, 2, 2, 1]
        if total == 4:
            return [1, 2, 2, 1]
        elif total == 8:
            return [2, 2, 2, 1]
        else:
            return [1, 1, 1, 1]

    def _compute_spatial_strides(self, total: int) -> List[int]:
        """Distribute spatial downsampling across stages."""
        # 16x -> strides [2, 2, 2, 2]
        import math
        num_stages = 4
        stride = int(round(total ** (1 / num_stages)))
        return [stride] * num_stages

    def forward(self, video: Tensor) -> Tensor:
        """
        Encode video to latent representation.

        Args:
            video: [B, T, C, H, W] input video (C=3 RGB)

        Returns:
            latent: [B, T', H', W', latent_channels]
                    where T' = T // tau, H' = H // 16, W' = W // 16
        """
        # Rearrange to 3D conv format: [B, C, T, H, W]
        x = rearrange(video, 'b t c h w -> b c t h w')

        # Initial conv
        x = self.conv_in(x)

        # Encoder stages
        for stage in self.stages:
            for layer in stage:
                x = layer(x)

        # Output projection
        x = self.conv_out(x)

        # Rearrange to [B, T', H', W', C]
        x = rearrange(x, 'b c t h w -> b t h w c')

        return x

    def get_latent_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute output shape given input shape."""
        B, T, C, H, W = input_shape
        T_out = T // self.config.temporal_downsample
        H_out = H // self.config.spatial_downsample
        W_out = W // self.config.spatial_downsample
        return (B, T_out, H_out, W_out, self.config.latent_channels)
