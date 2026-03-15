"""2D to 3D lifting module for DeltaV prediction."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class DepthLifting(nn.Module):
    """
    Lift 2D features to 3D by predicting depth distribution.

    For each 2D position (h, w), predicts a distribution over depth bins.
    This allows converting 2D features to sparse 3D voxel predictions.
    """

    def __init__(
        self,
        input_dim: int,
        depth_bins: int = 64,
        hidden_dim: int = 512,
    ):
        """
        Args:
            input_dim: Input feature dimension from dynamics backbone
            depth_bins: Number of discrete depth hypotheses
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.depth_bins = depth_bins

        # MLP to predict depth distribution
        self.depth_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, depth_bins),
        )

        # Learnable depth range
        self.register_buffer(
            "depth_values",
            torch.linspace(0.1, 10.0, depth_bins),  # Near to far
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict depth distribution for each 2D position.

        Args:
            features: [B, H, W, D] 2D feature map

        Returns:
            depth_logits: [B, H, W, depth_bins] unnormalized depth scores
            depth_probs: [B, H, W, depth_bins] softmax probabilities
        """
        depth_logits = self.depth_predictor(features)

        # Numerical stability: subtract max before softmax to prevent overflow
        depth_logits_stable = depth_logits - depth_logits.max(dim=-1, keepdim=True).values
        depth_probs = torch.softmax(depth_logits_stable, dim=-1)

        # Handle NaN (can occur with extreme values)
        depth_probs = torch.nan_to_num(depth_probs, nan=1.0 / self.depth_bins)

        return depth_logits, depth_probs

    def get_expected_depth(self, depth_probs: Tensor) -> Tensor:
        """
        Compute expected depth from probability distribution.

        Args:
            depth_probs: [B, H, W, depth_bins]

        Returns:
            expected_depth: [B, H, W] expected depth values
        """
        return (depth_probs * self.depth_values).sum(dim=-1)

    def sample_depth(self, depth_logits: Tensor, temperature: float = 1.0) -> Tensor:
        """
        Sample depth indices from distribution.

        Args:
            depth_logits: [B, H, W, depth_bins]
            temperature: Sampling temperature

        Returns:
            depth_indices: [B, H, W] sampled depth bin indices
        """
        probs = torch.softmax(depth_logits / temperature, dim=-1)
        return torch.multinomial(
            probs.view(-1, self.depth_bins), 1
        ).view(depth_logits.shape[:-1])


class FeatureLifting(nn.Module):
    """
    Lift 2D features to 3D voxel features.

    Combines depth lifting with feature transformation to produce
    per-voxel features at each depth bin.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth_bins: int = 64,
        hidden_dim: int = 512,
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output per-voxel feature dimension
            depth_bins: Number of depth bins
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.depth_bins = depth_bins
        self.output_dim = output_dim

        # Depth lifting
        self.depth_lift = DepthLifting(input_dim, depth_bins, hidden_dim)

        # Feature transformation per depth
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, depth_bins * output_dim),
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Lift 2D features to 3D.

        Args:
            features: [B, H, W, D_in] 2D features

        Returns:
            voxel_features: [B, H, W, depth_bins, D_out] 3D features
            depth_probs: [B, H, W, depth_bins] depth confidence
        """
        B, H, W, D = features.shape

        # Get depth distribution
        _, depth_probs = self.depth_lift(features)

        # Transform features for each depth
        voxel_features = self.feature_transform(features)
        voxel_features = voxel_features.view(B, H, W, self.depth_bins, self.output_dim)

        # Weight by depth probability (soft assignment)
        voxel_features = voxel_features * depth_probs.unsqueeze(-1)

        return voxel_features, depth_probs


class CoordinateLifting(nn.Module):
    """
    Convert 2D positions + depth to 3D world coordinates.

    Handles the camera projection math to get voxel positions
    in world space from image coordinates and depth.
    """

    def __init__(self, voxel_resolution: int = 256):
        super().__init__()
        self.voxel_resolution = voxel_resolution

    def forward(
        self,
        h_indices: Tensor,
        w_indices: Tensor,
        depth_indices: Tensor,
        H: int,
        W: int,
        depth_bins: int,
    ) -> Tensor:
        """
        Convert image coordinates + depth to voxel coordinates.

        Args:
            h_indices: [N] height indices in image
            w_indices: [N] width indices in image
            depth_indices: [N] depth bin indices
            H, W: Image dimensions
            depth_bins: Number of depth bins

        Returns:
            coords: [N, 3] voxel coordinates (x, y, z)
        """
        # Normalize to [0, resolution-1] with division by zero protection
        x = (w_indices.float() / max(W, 1) * self.voxel_resolution).long()
        y = (h_indices.float() / max(H, 1) * self.voxel_resolution).long()
        z = (depth_indices.float() / max(depth_bins, 1) * self.voxel_resolution).long()

        # Clamp to valid range
        x = x.clamp(0, self.voxel_resolution - 1)
        y = y.clamp(0, self.voxel_resolution - 1)
        z = z.clamp(0, self.voxel_resolution - 1)

        return torch.stack([x, y, z], dim=-1)
