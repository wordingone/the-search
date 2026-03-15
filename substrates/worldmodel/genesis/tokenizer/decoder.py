"""3D Causal VAE Decoder for video reconstruction."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from einops import rearrange

from genesis.config import TokenizerConfig


class ResBlock3d(nn.Module):
    """3D Residual block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.act = nn.SiLU()

        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Upsample3d(nn.Module):
    """3D upsampling with configurable temporal/spatial factors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_factor: int = 1,
        spatial_factor: int = 2,
    ):
        super().__init__()
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor

        self.conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        # Upsample via interpolation
        x = nn.functional.interpolate(
            x,
            scale_factor=(self.temporal_factor, self.spatial_factor, self.spatial_factor),
            mode='trilinear',
            align_corners=False,
        )
        return self.conv(x)


class VideoDecoder(nn.Module):
    """
    3D Causal VAE Decoder.

    Reconstructs video [B, T, 3, H, W] from latent [B, T', H', W', C].

    Architecture mirrors encoder with transposed operations.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

        # Input projection from latent
        self.conv_in = nn.Conv3d(
            config.latent_channels,
            config.decoder_channels[0],
            kernel_size=3,
            padding=1,
        )

        # Decoder stages (reverse of encoder)
        self.stages = nn.ModuleList()

        temporal_factors = self._compute_temporal_factors(config.temporal_downsample)
        spatial_factors = self._compute_spatial_factors(config.spatial_downsample)

        in_ch = config.decoder_channels[0]
        for i, (out_ch, depth) in enumerate(zip(
            config.decoder_channels,
            config.decoder_depths,
        )):
            stage = nn.ModuleList()

            # ResBlocks
            for j in range(depth):
                ch = in_ch if j == 0 else out_ch
                stage.append(ResBlock3d(ch, out_ch))

            # Upsampling (except last stage)
            if i < len(config.decoder_channels) - 1:
                next_ch = config.decoder_channels[i + 1]
                stage.append(Upsample3d(
                    out_ch,
                    next_ch,
                    temporal_factor=temporal_factors[i],
                    spatial_factor=spatial_factors[i],
                ))
                in_ch = next_ch
            else:
                in_ch = out_ch

            self.stages.append(stage)

        # Final convolution to RGB
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, config.decoder_channels[-1]),
            nn.SiLU(),
            nn.Conv3d(config.decoder_channels[-1], config.input_channels, 3, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def _compute_temporal_factors(self, total: int) -> List[int]:
        """Distribute temporal upsampling across stages."""
        if total == 4:
            return [1, 2, 2, 1]
        elif total == 8:
            return [2, 2, 2, 1]
        else:
            return [1, 1, 1, 1]

    def _compute_spatial_factors(self, total: int) -> List[int]:
        """Distribute spatial upsampling across stages."""
        import math
        num_stages = 4
        factor = int(round(total ** (1 / num_stages)))
        return [factor] * num_stages

    def forward(self, latent: Tensor) -> Tensor:
        """
        Decode latent to video.

        Args:
            latent: [B, T', H', W', C] latent representation

        Returns:
            video: [B, T, C, H, W] reconstructed video
        """
        # Rearrange to 3D conv format: [B, C, T, H, W]
        x = rearrange(latent, 'b t h w c -> b c t h w')

        # Input projection
        x = self.conv_in(x)

        # Decoder stages
        for stage in self.stages:
            for layer in stage:
                x = layer(x)

        # Output projection
        x = self.conv_out(x)

        # Rearrange to [B, T, C, H, W]
        x = rearrange(x, 'b c t h w -> b t c h w')

        return x


class VideoTokenizer(nn.Module):
    """
    Complete video tokenizer with encoder, decoder, and FSQ quantization.

    Combines VideoEncoder + FSQ + VideoDecoder for training.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

        from genesis.tokenizer.encoder import VideoEncoder
        from genesis.tokenizer.fsq import FSQ

        self.encoder = VideoEncoder(config)
        self.decoder = VideoDecoder(config)
        self.fsq = FSQ(config.fsq.levels)

    def forward(self, video: Tensor) -> dict:
        """
        Full forward pass for training.

        Args:
            video: [B, T, C, H, W] input video

        Returns:
            dict with:
                - recon: reconstructed video
                - latent: continuous latent
                - codes: quantized codes
                - indices: discrete indices
        """
        # Encode
        latent = self.encoder(video)

        # Quantize
        codes, indices = self.fsq(latent)

        # Decode
        recon = self.decoder(codes)

        return {
            "recon": recon,
            "latent": latent,
            "codes": codes,
            "indices": indices,
        }

    def encode(self, video: Tensor) -> Tensor:
        """Encode video to quantized latent."""
        latent = self.encoder(video)
        codes, _ = self.fsq(latent)
        return codes

    def decode(self, codes: Tensor) -> Tensor:
        """Decode quantized latent to video."""
        return self.decoder(codes)

    def tokenize(self, video: Tensor) -> Tensor:
        """Get discrete token indices."""
        latent = self.encoder(video)
        _, indices = self.fsq(latent)
        return indices
