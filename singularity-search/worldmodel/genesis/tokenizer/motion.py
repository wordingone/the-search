"""Motion-aware video tokenizer with video encoding insights.

Exploits temporal structure like H.264/H.265:
- Motion estimation between frames
- Predictive coding (encode residuals, not raw frames)
- Keyframe structure (I-frames + P-frames)

This reduces what the dynamics model needs to learn.

Resolution Support:
- 64x64: 8x downsample -> 8x8 latent (64 tokens)
- 256x256: 16x downsample -> 16x16 latent (256 tokens)
- Both maintain efficient attention (< 512 tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List
from einops import rearrange, repeat
import math


def compute_spatial_downsample(image_size: int, target_latent_size: int = 16) -> int:
    """Compute spatial downsample factor to achieve target latent size.

    Args:
        image_size: input resolution (e.g., 64, 256, 512)
        target_latent_size: desired latent spatial size (default 16)

    Returns:
        downsample factor (8, 16, 32, etc.)
    """
    return max(4, image_size // target_latent_size)


def compute_latent_channels(
    image_size: int,
    base_channels: int = 8,
    base_resolution: int = 64,
    base_downsample: int = 8,
) -> int:
    """Compute latent channels to maintain constant information density (bits/pixel).

    At higher resolutions with larger downsample factors, we need more channels
    to maintain the same bits-per-pixel ratio as the 64x64 baseline.

    The key insight is that information density = (channels * latent_h * latent_w) / (image_h * image_w)

    At 64x64 with 8x downsample and 8 channels:
        - Latent: 8x8x8 = 512 values
        - Input: 64x64 = 4096 pixels
        - Density: 512/4096 = 0.125 (4.0 bits/pixel at FP32)

    At 256x256 with 16x downsample and 8 channels (CURRENT - BROKEN):
        - Latent: 16x16x8 = 2048 values
        - Input: 256x256 = 65536 pixels
        - Density: 2048/65536 = 0.03125 (1.0 bits/pixel - 4x WORSE!)

    At 256x256 with 16x downsample and 16 channels (FIXED):
        - Latent: 16x16x16 = 4096 values
        - Input: 256x256 = 65536 pixels
        - Density: 4096/65536 = 0.0625 (2.0 bits/pixel - 2x better)

    Formula: channels = base_channels * (current_downsample / base_downsample)

    Args:
        image_size: Input resolution (e.g., 64, 256, 512)
        base_channels: Channels at base resolution (default 8 at 64x64)
        base_resolution: Reference resolution (default 64)
        base_downsample: Downsample factor at base resolution (default 8)

    Returns:
        Scaled channel count to maintain information density
    """
    if image_size <= base_resolution:
        return base_channels

    current_downsample = compute_spatial_downsample(image_size, target_latent_size=16)
    scale = current_downsample / base_downsample
    return int(base_channels * scale)


# =============================================================================
# OPTICAL FLOW ESTIMATION
# =============================================================================

class FlowEstimator(nn.Module):
    """Lightweight optical flow estimation network.

    Estimates dense motion field between two frames.
    Based on PWC-Net / RAFT concepts but simplified for efficiency.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_levels: int = 3,
    ):
        super().__init__()
        self.num_levels = num_levels

        # Feature encoder (shared for both frames)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Cost volume processing
        self.cost_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Flow prediction head
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 2, 3, 1, 1),  # 2 channels for (dx, dy)
        )

    def forward(self, frame1: Tensor, frame2: Tensor) -> Tensor:
        """Estimate optical flow from frame1 to frame2.

        Args:
            frame1: [B, C, H, W] source frame
            frame2: [B, C, H, W] target frame

        Returns:
            flow: [B, 2, H, W] motion field (dx, dy per pixel)
        """
        B, C, H, W = frame1.shape

        # Extract features
        feat1 = self.encoder(frame1)  # [B, C', H/4, W/4]
        feat2 = self.encoder(frame2)

        # Concatenate features (simplified cost volume)
        cost = torch.cat([feat1, feat2], dim=1)
        cost = self.cost_conv(cost)

        # Predict flow at low resolution
        flow_low = self.flow_head(cost)  # [B, 2, H/4, W/4]

        # Upsample flow to full resolution
        flow = F.interpolate(flow_low, size=(H, W), mode='bilinear', align_corners=False)
        flow = flow * 4  # Scale flow values with resolution

        return flow


# =============================================================================
# FRAME WARPING
# =============================================================================

def warp_frame(frame: Tensor, flow: Tensor) -> Tensor:
    """Warp frame using optical flow (backward warping).

    Args:
        frame: [B, C, H, W] source frame
        flow: [B, 2, H, W] motion field

    Returns:
        warped: [B, C, H, W] warped frame
    """
    B, C, H, W = frame.shape

    # Create sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=frame.device, dtype=frame.dtype),
        torch.arange(W, device=frame.device, dtype=frame.dtype),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]

    # Apply flow
    new_grid = grid + flow

    # Normalize to [-1, 1] for grid_sample
    new_grid[:, 0] = 2 * new_grid[:, 0] / (W - 1) - 1  # x
    new_grid[:, 1] = 2 * new_grid[:, 1] / (H - 1) - 1  # y

    # Rearrange for grid_sample: [B, H, W, 2]
    new_grid = new_grid.permute(0, 2, 3, 1)

    # Warp
    warped = F.grid_sample(frame, new_grid, mode='bilinear', padding_mode='border', align_corners=False)

    return warped


# =============================================================================
# MOTION ENCODER/DECODER
# =============================================================================

class MotionEncoder(nn.Module):
    """Encode optical flow compactly."""

    def __init__(self, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, latent_dim, 3, 2, 1),
        )

    def forward(self, flow: Tensor) -> Tensor:
        """Encode flow to compact representation."""
        return self.encoder(flow)


class MotionDecoder(nn.Module):
    """Decode optical flow from compact representation."""

    def __init__(self, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, 2, 4, 2, 1),
        )

    def forward(self, latent: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """Decode latent to flow."""
        flow = self.decoder(latent)
        # Resize to target
        if flow.shape[-2:] != target_size:
            flow = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=False)
        return flow


class ResolutionAwareMotionEncoder(nn.Module):
    """Motion encoder that adapts to input resolution.

    Outputs latent at target spatial size directly (no interpolation needed).
    For 256x256 input with 16x downsample, outputs 16x16 latent directly.
    """

    def __init__(self, hidden_dim: int = 32, latent_dim: int = 4, spatial_downsample: int = 16):
        super().__init__()
        self.spatial_downsample = spatial_downsample
        num_stages = int(math.log2(spatial_downsample))

        layers = []
        c_in = 2  # Flow has 2 channels (dx, dy)
        c_out = hidden_dim

        for i in range(num_stages):
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, 2, 1),
                nn.GroupNorm(8 if c_out >= 8 else c_out, c_out),
                nn.GELU(),
            ])
            c_in = c_out
            c_out = min(c_out * 2, hidden_dim * 4)

        layers.append(nn.Conv2d(c_in, latent_dim, 3, 1, 1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, flow: Tensor) -> Tensor:
        return self.encoder(flow)


class ResolutionAwareMotionDecoder(nn.Module):
    """Motion decoder that adapts to output resolution."""

    def __init__(self, hidden_dim: int = 32, latent_dim: int = 4, spatial_downsample: int = 16):
        super().__init__()
        self.spatial_downsample = spatial_downsample
        num_stages = int(math.log2(spatial_downsample))

        channels = [hidden_dim]
        for i in range(num_stages - 1):
            channels.append(min(channels[-1] * 2, hidden_dim * 4))
        channels = channels[::-1]

        layers = [nn.Conv2d(latent_dim, channels[0], 3, 1, 1), nn.GELU()]
        for i in range(num_stages):
            c_in = channels[i] if i < len(channels) else hidden_dim
            c_out = channels[i + 1] if i + 1 < len(channels) else hidden_dim
            # Upsample+Conv instead of ConvTranspose2d to prevent checkerboard artifacts
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(c_in, c_out, 3, 1, 1),
                nn.GroupNorm(8 if c_out >= 8 else c_out, c_out),
                nn.GELU(),
            ])
        layers.append(nn.Conv2d(hidden_dim, 2, 3, 1, 1))  # Output 2 channels for flow
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: Tensor, target_size: Tuple[int, int]) -> Tensor:
        flow = self.decoder(latent)
        if flow.shape[-2:] != target_size:
            flow = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=False)
        return flow


# =============================================================================
# RESIDUAL ENCODER/DECODER
# =============================================================================

class ResidualEncoder(nn.Module):
    """Encode prediction residuals (what flow can't explain)."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, latent_channels, 3, 2, 1),
        )

    def forward(self, residual: Tensor) -> Tensor:
        """Encode residual to latent."""
        return self.encoder(residual)


class ResidualDecoder(nn.Module):
    """Decode prediction residuals."""

    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1),
        )

    def forward(self, latent: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """Decode latent to residual."""
        residual = self.decoder(latent)
        if residual.shape[-2:] != target_size:
            residual = F.interpolate(residual, size=target_size, mode='bilinear', align_corners=False)
        return residual


class ResolutionAwareResidualEncoder(nn.Module):
    """Residual encoder that adapts to input resolution.

    Outputs latent at target spatial size directly (no interpolation needed).
    For 256x256 input with 16x downsample, outputs 16x16 latent directly.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 4,
        spatial_downsample: int = 16,
    ):
        super().__init__()
        self.spatial_downsample = spatial_downsample
        num_stages = int(math.log2(spatial_downsample))

        layers = []
        c_in = in_channels
        c_out = hidden_channels

        for i in range(num_stages):
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, 2, 1),
                nn.GroupNorm(8 if c_out >= 8 else c_out, c_out),
                nn.GELU(),
            ])
            if i < num_stages - 1:
                layers.extend([nn.Conv2d(c_out, c_out, 3, 1, 1), nn.GELU()])
            c_in = c_out
            c_out = min(c_out * 2, hidden_channels * 4)

        layers.append(nn.Conv2d(c_in, latent_channels, 3, 1, 1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, residual: Tensor) -> Tensor:
        return self.encoder(residual)


class ResolutionAwareResidualDecoder(nn.Module):
    """Residual decoder that adapts to output resolution."""

    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 4,
        spatial_downsample: int = 16,
    ):
        super().__init__()
        self.spatial_downsample = spatial_downsample
        num_stages = int(math.log2(spatial_downsample))

        channels = [hidden_channels]
        for i in range(num_stages - 1):
            channels.append(min(channels[-1] * 2, hidden_channels * 4))
        channels = channels[::-1]

        layers = [nn.Conv2d(latent_channels, channels[0], 3, 1, 1), nn.GELU()]
        for i in range(num_stages):
            c_in = channels[i] if i < len(channels) else hidden_channels
            c_out = channels[i + 1] if i + 1 < len(channels) else hidden_channels
            # Upsample+Conv instead of ConvTranspose2d to prevent checkerboard artifacts
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(c_in, c_out, 3, 1, 1),
                nn.GroupNorm(8 if c_out >= 8 else c_out, c_out),
                nn.GELU(),
            ])
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            nn.Tanh(),  # Bug #4 Fix A: Bound residuals to [-1, 1]
        ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: Tensor, target_size: Tuple[int, int]) -> Tensor:
        residual = self.decoder(latent)
        if residual.shape[-2:] != target_size:
            residual = F.interpolate(residual, size=target_size, mode='bilinear', align_corners=False)
        return residual


# =============================================================================
# KEYFRAME ENCODER (for I-frames)
# =============================================================================

class KeyframeEncoder(nn.Module):
    """Encode keyframes (I-frames) independently."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 8,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, latent_channels, 3, 1, 1),
        )

    def forward(self, frame: Tensor) -> Tensor:
        return self.encoder(frame)


class KeyframeDecoder(nn.Module):
    """Decode keyframes (I-frames)."""

    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 8,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent: Tensor, target_size: Tuple[int, int]) -> Tensor:
        frame = self.decoder(latent)
        if frame.shape[-2:] != target_size:
            frame = F.interpolate(frame, size=target_size, mode='bilinear', align_corners=False)
        return frame


# =============================================================================
# RESOLUTION-AWARE ENCODERS/DECODERS
# =============================================================================

class ResolutionAwareKeyframeEncoder(nn.Module):
    """Keyframe encoder that adapts to input resolution.

    64x64 -> 8x8 latent (8x downsample, 3 stages)
    256x256 -> 16x16 latent (16x downsample, 4 stages)
    512x512 -> 16x16 latent (32x downsample, 5 stages)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 8,
        spatial_downsample: int = 8,
    ):
        super().__init__()
        self.spatial_downsample = spatial_downsample
        num_stages = int(math.log2(spatial_downsample))

        layers = []
        c_in = in_channels
        c_out = hidden_channels

        for i in range(num_stages):
            # Downsample conv
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, 2, 1),
                nn.GroupNorm(8 if c_out >= 8 else c_out, c_out),
                nn.GELU(),
            ])
            # Refinement conv (except last stage)
            if i < num_stages - 1:
                layers.extend([
                    nn.Conv2d(c_out, c_out, 3, 1, 1),
                    nn.GELU(),
                ])
            c_in = c_out
            c_out = min(c_out * 2, hidden_channels * 4)

        # Final projection to latent channels
        layers.append(nn.Conv2d(c_in, latent_channels, 3, 1, 1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, frame: Tensor) -> Tensor:
        return self.encoder(frame)


class ResolutionAwareKeyframeDecoder(nn.Module):
    """Keyframe decoder that adapts to output resolution.

    Mirrors ResolutionAwareKeyframeEncoder.
    """

    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 64,
        latent_channels: int = 8,
        spatial_downsample: int = 8,
    ):
        super().__init__()
        self.spatial_downsample = spatial_downsample
        num_stages = int(math.log2(spatial_downsample))

        # Compute channel progression (reverse of encoder)
        channels = [hidden_channels]
        for i in range(num_stages - 1):
            channels.append(min(channels[-1] * 2, hidden_channels * 4))
        channels = channels[::-1]  # Reverse for decoder

        layers = []
        # Initial projection from latent
        layers.extend([
            nn.Conv2d(latent_channels, channels[0], 3, 1, 1),
            nn.GELU(),
        ])

        for i in range(num_stages):
            c_in = channels[i] if i < len(channels) else hidden_channels
            c_out = channels[i + 1] if i + 1 < len(channels) else hidden_channels

            # Upsample+Conv instead of ConvTranspose2d to prevent checkerboard artifacts
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(c_in, c_out, 3, 1, 1),
                nn.GroupNorm(8 if c_out >= 8 else c_out, c_out),
                nn.GELU(),
            ])

        # Final projection to output channels
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: Tensor, target_size: Tuple[int, int]) -> Tensor:
        frame = self.decoder(latent)
        if frame.shape[-2:] != target_size:
            frame = F.interpolate(frame, size=target_size, mode='bilinear', align_corners=False)
        return frame


# =============================================================================
# MOTION-AWARE VIDEO TOKENIZER
# =============================================================================

class MotionAwareTokenizer(nn.Module):
    """Video tokenizer with motion estimation and predictive coding.

    Structure:
    - Frame 0: Keyframe (I-frame) - encoded independently
    - Frame 1+: P-frames - encoded as motion + residual from previous

    This exploits temporal redundancy like H.264/H.265.

    Resolution Support:
    - 64x64: 8x downsample -> 8x8 latent (64 tokens)
    - 256x256: 16x downsample -> 16x16 latent (256 tokens)

    Latent representation for each frame:
    - Keyframe: [B, keyframe_channels, H/ds, W/ds] where ds = spatial_downsample
    - P-frame: motion_latent + residual_latent at same spatial resolution
    """

    def __init__(
        self,
        in_channels: int = 3,
        keyframe_channels: Optional[int] = None,  # None = compute adaptively
        motion_channels: Optional[int] = None,    # None = compute adaptively
        residual_channels: Optional[int] = None,  # None = compute adaptively
        hidden_channels: int = 128,  # Increased from 64 for 720p decoder capacity
        keyframe_interval: int = 8,  # I-frame every N frames
        image_size: int = 64,  # Input resolution for adaptive downsampling
        adaptive_channels: bool = True,  # Enable resolution-adaptive channel scaling
        use_fsq: bool = False,  # Enable FSQ discrete quantization (prevents corruption)
        fsq_levels: Optional[List[int]] = None,  # FSQ levels (default [8, 6, 5, 5, 5])
    ):
        super().__init__()
        self.keyframe_interval = keyframe_interval
        self.image_size = image_size

        # Compute spatial downsample factor to maintain ~16x16 latent
        # 64x64 -> 8x downsample -> 8x8 (legacy, keep 8 for compatibility)
        # 256x256 -> 16x downsample -> 16x16
        # 512x512 -> 32x downsample -> 16x16
        if image_size <= 64:
            self.spatial_downsample = 8
        else:
            self.spatial_downsample = compute_spatial_downsample(image_size, target_latent_size=16)

        # Actual downsample is 2^num_stages (encoder uses stride-2 convs)
        # log2(45) = 5.49 -> 5 stages -> actual downsample = 32
        num_stages = int(math.log2(self.spatial_downsample))
        actual_downsample = 2 ** num_stages
        self.latent_size = (image_size + actual_downsample - 1) // actual_downsample  # ceil division

        # Compute adaptive channels to maintain information density (bits/pixel)
        # This fixes the 256x256 quality collapse (CLIP-IQA 0.181 -> target 0.50+)
        if adaptive_channels and image_size > 64:
            # Scale channels proportionally with downsample factor
            # 64x64: 8x down, 8 channels -> 4.0 bpp
            # 256x256: 16x down, 16 channels -> 4.0 bpp (previously 8 channels = 1.0 bpp)
            computed_keyframe = compute_latent_channels(image_size, base_channels=8)
            computed_motion = compute_latent_channels(image_size, base_channels=4)
            computed_residual = compute_latent_channels(image_size, base_channels=4)

            # Use computed values if not explicitly provided
            self.keyframe_channels = keyframe_channels if keyframe_channels is not None else computed_keyframe
            self.motion_channels = motion_channels if motion_channels is not None else computed_motion
            self.residual_channels = residual_channels if residual_channels is not None else computed_residual
        else:
            # Legacy fixed channels (64x64 or adaptive_channels=False)
            self.keyframe_channels = keyframe_channels if keyframe_channels is not None else 8
            self.motion_channels = motion_channels if motion_channels is not None else 4
            self.residual_channels = residual_channels if residual_channels is not None else 4

        # Total latent channels per frame
        # P-frames: motion + residual
        # I-frames: keyframe only (padded to same size)
        self.latent_channels = max(self.keyframe_channels, self.motion_channels + self.residual_channels)

        # Flow estimation
        self.flow_estimator = FlowEstimator(in_channels, hidden_channels)

        # Motion encoder/decoder - use resolution-aware version for higher res
        if self.spatial_downsample > 8:
            self.motion_encoder = ResolutionAwareMotionEncoder(
                hidden_channels // 2, self.motion_channels, self.spatial_downsample
            )
            self.motion_decoder = ResolutionAwareMotionDecoder(
                hidden_channels // 2, self.motion_channels, self.spatial_downsample
            )
        else:
            self.motion_encoder = MotionEncoder(hidden_channels // 2, self.motion_channels)
            self.motion_decoder = MotionDecoder(hidden_channels // 2, self.motion_channels)

        # Residual encoder/decoder - use resolution-aware version for higher res
        if self.spatial_downsample > 8:
            self.residual_encoder = ResolutionAwareResidualEncoder(
                in_channels, hidden_channels, self.residual_channels, self.spatial_downsample
            )
            self.residual_decoder = ResolutionAwareResidualDecoder(
                in_channels, hidden_channels, self.residual_channels, self.spatial_downsample
            )
        else:
            self.residual_encoder = ResidualEncoder(in_channels, hidden_channels, self.residual_channels)
            self.residual_decoder = ResidualDecoder(in_channels, hidden_channels, self.residual_channels)

        # Keyframe encoder/decoder - use resolution-aware version for higher res
        if self.spatial_downsample > 8:
            self.keyframe_encoder = ResolutionAwareKeyframeEncoder(
                in_channels, hidden_channels, self.keyframe_channels, self.spatial_downsample
            )
            self.keyframe_decoder = ResolutionAwareKeyframeDecoder(
                in_channels, hidden_channels, self.keyframe_channels, self.spatial_downsample
            )
        else:
            # Legacy 8x downsample for 64x64
            self.keyframe_encoder = KeyframeEncoder(in_channels, hidden_channels, self.keyframe_channels)
            self.keyframe_decoder = KeyframeDecoder(in_channels, hidden_channels, self.keyframe_channels)

        # FSQ discrete quantization (prevents slot attention corruption)
        self.use_fsq = use_fsq
        if use_fsq:
            from genesis.tokenizer.fsq import FSQ
            self.fsq_levels = fsq_levels if fsq_levels is not None else [8, 6, 5, 5, 5]
            self.fsq_dim = len(self.fsq_levels)
            self.fsq = FSQ(self.fsq_levels)
            # Project latent channels to FSQ dim and back
            self.fsq_proj_in = nn.Linear(self.latent_channels, self.fsq_dim)
            self.fsq_proj_out = nn.Linear(self.fsq_dim, self.latent_channels)

    def encode_keyframe(self, frame: Tensor) -> Tensor:
        """Encode a keyframe (I-frame)."""
        latent = self.keyframe_encoder(frame)
        # Pad to match latent_channels if needed
        if latent.shape[1] < self.latent_channels:
            padding = torch.zeros(
                latent.shape[0], self.latent_channels - latent.shape[1],
                latent.shape[2], latent.shape[3],
                device=latent.device, dtype=latent.dtype
            )
            latent = torch.cat([latent, padding], dim=1)
        return latent

    def decode_keyframe(self, latent: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """Decode a keyframe."""
        # Extract keyframe channels
        kf_latent = latent[:, :self.keyframe_channels]
        return self.keyframe_decoder(kf_latent, target_size)

    def encode_pframe(self, frame: Tensor, prev_frame: Tensor) -> Tuple[Tensor, dict]:
        """Encode a P-frame (predicted from previous).

        Returns latent and intermediate values for loss computation.
        """
        B, C, H, W = frame.shape

        # Estimate motion
        flow = self.flow_estimator(prev_frame, frame)

        # Warp previous frame
        predicted = warp_frame(prev_frame, flow)

        # Compute residual
        residual = frame - predicted

        # Encode motion and residual (resolution-aware encoders output correct size directly)
        motion_latent = self.motion_encoder(flow)
        residual_latent = self.residual_encoder(residual)

        # Concatenate to form full latent
        latent = torch.cat([motion_latent, residual_latent], dim=1)

        # Pad if needed
        if latent.shape[1] < self.latent_channels:
            padding = torch.zeros(
                B, self.latent_channels - latent.shape[1],
                latent.shape[2], latent.shape[3],
                device=latent.device, dtype=latent.dtype
            )
            latent = torch.cat([latent, padding], dim=1)

        return latent, {
            'flow': flow,
            'predicted': predicted,
            'residual': residual,
        }

    def decode_pframe(
        self,
        latent: Tensor,
        prev_frame: Tensor,
        target_size: Tuple[int, int],
    ) -> Tensor:
        """Decode a P-frame given previous reconstructed frame."""
        # Split latent
        motion_latent = latent[:, :self.motion_channels]
        residual_latent = latent[:, self.motion_channels:self.motion_channels + self.residual_channels]

        # Decode motion
        flow = self.motion_decoder(motion_latent, target_size)

        # Warp previous frame
        predicted = warp_frame(prev_frame, flow)

        # Decode and add residual
        residual = self.residual_decoder(residual_latent, target_size)
        frame = predicted + residual

        # Clamp to valid range
        frame = torch.clamp(frame, 0, 1)

        return frame

    def encode(self, video: Tensor) -> Tuple[Tensor, List[dict]]:
        """Encode video to latent representation.

        Args:
            video: [B, T, C, H, W] input video

        Returns:
            latents: [B, T, latent_channels, H', W'] latent representation
            intermediates: list of dicts with flow/residual for each frame
        """
        B, T, C, H, W = video.shape

        latents = []
        intermediates = []

        for t in range(T):
            frame = video[:, t]  # [B, C, H, W]

            if t == 0 or t % self.keyframe_interval == 0:
                # Keyframe (I-frame)
                latent = self.encode_keyframe(frame)
                intermediates.append({'type': 'I', 'frame_idx': t})
            else:
                # P-frame
                prev_frame = video[:, t - 1]
                latent, info = self.encode_pframe(frame, prev_frame)
                info['type'] = 'P'
                info['frame_idx'] = t
                intermediates.append(info)

            latents.append(latent)

        # Stack along time dimension
        latents = torch.stack(latents, dim=1)  # [B, T, C, H', W']

        # Apply FSQ quantization if enabled (prevents slot attention corruption)
        if self.use_fsq:
            latents = self.quantize_latents(latents)

        return latents, intermediates

    def quantize_latents(self, latents: Tensor) -> Tensor:
        """Apply FSQ quantization to latents.

        Args:
            latents: [B, T, C, H', W'] continuous latents

        Returns:
            latents: [B, T, C, H', W'] quantized latents (discrete codes)
        """
        B, T, C, Hl, Wl = latents.shape

        # Permute to [B, T, H', W', C] for FSQ (channels last)
        x = latents.permute(0, 1, 3, 4, 2)  # [B, T, H', W', C]

        # Project to FSQ dimension
        x = self.fsq_proj_in(x)  # [B, T, H', W', fsq_dim]

        # Apply FSQ quantization (straight-through gradient)
        x_q, indices = self.fsq(x)  # x_q: [B, T, H', W', fsq_dim]

        # Project back to latent channels
        x_q = self.fsq_proj_out(x_q)  # [B, T, H', W', C]

        # Permute back to [B, T, C, H', W']
        latents_q = x_q.permute(0, 1, 4, 2, 3)

        return latents_q

    def decode(
        self,
        latents: Tensor,
        target_size: Tuple[int, int],
        start_frame_idx: int = 0,
        prev_frame: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode latents back to video.

        Args:
            latents: [B, T, C, H', W'] latent representation
            target_size: (H, W) output resolution
            start_frame_idx: The frame index that latents[:, 0] corresponds to.
                Used to correctly determine I-frame vs P-frame boundaries.
                Default 0 (standard encode->decode). Set to 1 when decoding
                dynamics-predicted latents (which predict frames 1..T-1).
            prev_frame: [B, C, H, W] previous decoded frame for P-frame decode
                when start_frame_idx > 0. Required if the first latent is a P-frame.

        Returns:
            video: [B, T, C, H, W] reconstructed video
        """
        B, T, C, Hl, Wl = latents.shape

        frames = []

        for t in range(T):
            latent = latents[:, t]  # [B, C, H', W']
            actual_frame_idx = start_frame_idx + t

            if actual_frame_idx == 0 or actual_frame_idx % self.keyframe_interval == 0:
                # Keyframe
                frame = self.decode_keyframe(latent, target_size)
            else:
                # P-frame - use previous reconstructed frame
                if frames:
                    pf = frames[-1]
                elif prev_frame is not None:
                    pf = prev_frame
                else:
                    # Fallback: decode as keyframe if no prev_frame available
                    frame = self.decode_keyframe(latent, target_size)
                    frames.append(frame)
                    continue
                frame = self.decode_pframe(latent, pf, target_size)

            frames.append(frame)

        # Stack
        video = torch.stack(frames, dim=1)  # [B, T, C, H, W]

        return video

    def forward(self, video: Tensor) -> dict:
        """Full forward pass for training.

        Args:
            video: [B, T, C, H, W] input video

        Returns:
            dict with latents, reconstruction, intermediates, and optionally indices
        """
        B, T, C, H, W = video.shape

        # Encode (includes FSQ if enabled)
        latents, intermediates = self.encode(video)

        # Decode
        recon = self.decode(latents, (H, W))

        result = {
            'latents': latents,
            'recon': recon,
            'intermediates': intermediates,
        }

        # Add FSQ indices if quantization is enabled
        if self.use_fsq:
            result['fsq_indices'] = self.get_indices(latents)

        return result

    def get_indices(self, latents: Tensor) -> Tensor:
        """Get discrete FSQ indices from latents.

        Args:
            latents: [B, T, C, H', W'] latents (already quantized if use_fsq)

        Returns:
            indices: [B, T, H', W'] discrete token indices
        """
        if not self.use_fsq:
            raise ValueError("FSQ not enabled, cannot get indices")

        B, T, C, Hl, Wl = latents.shape

        # Permute to [B, T, H', W', C]
        x = latents.permute(0, 1, 3, 4, 2)

        # Project to FSQ dimension
        x = self.fsq_proj_in(x)

        # Get indices (not quantized values)
        _, indices = self.fsq(x)

        return indices

    def get_compression_ratio(self, input_shape: Tuple[int, ...]) -> float:
        """Compute compression ratio."""
        B, T, C, H, W = input_shape
        input_size = T * C * H * W

        # Latent size per frame (use actual spatial downsample)
        Hl, Wl = H // self.spatial_downsample, W // self.spatial_downsample
        latent_size = T * self.latent_channels * Hl * Wl

        return input_size / latent_size


# =============================================================================
# MOTION-AWARE LOSS
# =============================================================================

class MotionAwareLoss(nn.Module):
    """Loss function for motion-aware tokenizer."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        flow_smooth_weight: float = 0.1,
        residual_sparse_weight: float = 0.1,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.flow_smooth_weight = flow_smooth_weight
        self.residual_sparse_weight = residual_sparse_weight

    def flow_smoothness_loss(self, flow: Tensor) -> Tensor:
        """Encourage smooth flow (total variation)."""
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

    def residual_sparsity_loss(self, residual: Tensor) -> Tensor:
        """Encourage sparse residuals (L1)."""
        return torch.mean(torch.abs(residual))

    def forward(
        self,
        recon: Tensor,
        target: Tensor,
        intermediates: List[dict],
    ) -> dict:
        """Compute losses."""
        losses = {}

        # Reconstruction loss
        losses['recon'] = F.mse_loss(recon, target) + F.l1_loss(recon, target)

        # Flow smoothness and residual sparsity for P-frames
        flow_loss = 0.0
        residual_loss = 0.0
        num_pframes = 0

        for info in intermediates:
            if info['type'] == 'P':
                flow_loss += self.flow_smoothness_loss(info['flow'])
                residual_loss += self.residual_sparsity_loss(info['residual'])
                num_pframes += 1

        if num_pframes > 0:
            losses['flow_smooth'] = flow_loss / num_pframes
            losses['residual_sparse'] = residual_loss / num_pframes
        else:
            losses['flow_smooth'] = torch.tensor(0.0, device=recon.device)
            losses['residual_sparse'] = torch.tensor(0.0, device=recon.device)

        # Total
        losses['total'] = (
            self.recon_weight * losses['recon'] +
            self.flow_smooth_weight * losses['flow_smooth'] +
            self.residual_sparse_weight * losses['residual_sparse']
        )

        return losses
