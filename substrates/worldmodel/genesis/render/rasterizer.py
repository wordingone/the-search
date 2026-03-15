"""Voxel rasterizer for rendering OVoxel memory to images.

STUB: This is a CPU-only reference implementation. For production use,
replace with CUDA-accelerated rasterizer from TRELLIS (diffoctreerast).
The current implementation uses simple alpha-blended splatting which is
~100x slower than the CUDA version.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from dataclasses import dataclass

from genesis.config import RenderConfig


@dataclass
class Camera:
    """Camera parameters for rendering."""
    position: Tensor   # [3] world position
    forward: Tensor    # [3] forward direction
    up: Tensor         # [3] up direction
    fov: float         # Field of view in degrees
    aspect: float      # Width / height
    near: float        # Near plane
    far: float         # Far plane

    def get_view_matrix(self) -> Tensor:
        """Compute view matrix."""
        f = self.forward / self.forward.norm()
        r = torch.cross(f, self.up)
        r = r / r.norm()
        u = torch.cross(r, f)

        view = torch.eye(4, device=self.position.device)
        view[0, :3] = r
        view[1, :3] = u
        view[2, :3] = -f
        view[:3, 3] = -torch.stack([
            r.dot(self.position),
            u.dot(self.position),
            -f.dot(self.position),
        ])
        return view

    def get_projection_matrix(self) -> Tensor:
        """Compute projection matrix."""
        import math
        fov_rad = self.fov * math.pi / 180
        f = 1.0 / math.tan(fov_rad / 2)

        proj = torch.zeros(4, 4, device=self.position.device)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = 2 * self.far * self.near / (self.near - self.far)
        proj[3, 2] = -1
        return proj


class VoxelRasterizer(nn.Module):
    """
    Voxel rasterizer for rendering OVoxel memory.

    Renders sparse voxels to 2D images using a simplified
    ray marching / splatting approach.

    For production use, this should be replaced with CUDA implementation
    from TRELLIS (diffoctreerast).
    """

    def __init__(self, config: RenderConfig):
        """
        Args:
            config: Render configuration
        """
        super().__init__()
        self.config = config
        self.width, self.height = config.resolution

    def forward(
        self,
        coords: Tensor,
        features: Tensor,
        camera: Camera,
    ) -> Tensor:
        """
        Render voxels to image.

        Args:
            coords: [N, 3] voxel coordinates (x, y, z)
            features: [N, 7] voxel features (RGB, metallic, roughness, opacity, SDF)
            camera: Camera parameters

        Returns:
            image: [3, H, W] rendered RGB image
        """
        device = coords.device if coords.numel() > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Handle empty voxels
        if coords.numel() == 0:
            bg = torch.tensor(self.config.background, device=device)
            return bg.view(3, 1, 1).expand(3, self.height, self.width)

        # Get view and projection matrices
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        mvp = proj @ view

        # Transform voxels to clip space
        coords_h = torch.cat([
            coords.float(),
            torch.ones(coords.shape[0], 1, device=device)
        ], dim=-1)  # [N, 4]

        clip = coords_h @ mvp.T  # [N, 4]

        # Perspective divide
        ndc = clip[:, :3] / (clip[:, 3:4] + 1e-8)  # [N, 3]

        # Filter visible voxels (in NDC cube [-1, 1])
        visible = (
            (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) &
            (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1) &
            (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)
        )

        if not visible.any():
            bg = torch.tensor(self.config.background, device=device)
            return bg.view(3, 1, 1).expand(3, self.height, self.width)

        ndc = ndc[visible]
        feats = features[visible]

        # Convert to screen coordinates
        screen_x = ((ndc[:, 0] + 1) / 2 * self.width).long()
        screen_y = ((1 - ndc[:, 1]) / 2 * self.height).long()  # Flip Y
        depth = ndc[:, 2]

        # Clamp to valid range
        screen_x = screen_x.clamp(0, self.width - 1)
        screen_y = screen_y.clamp(0, self.height - 1)

        # Sort by depth (back to front for alpha blending)
        sorted_idx = depth.argsort(descending=True)
        screen_x = screen_x[sorted_idx]
        screen_y = screen_y[sorted_idx]
        feats = feats[sorted_idx]

        # Initialize with background
        bg = torch.tensor(self.config.background, device=device)
        image = bg.view(3, 1, 1).expand(3, self.height, self.width).clone()
        alpha_buffer = torch.zeros(self.height, self.width, device=device)

        # Simple splatting (no anti-aliasing)
        rgb = feats[:, :3]
        opacity = feats[:, 5]

        for i in range(len(screen_x)):
            x, y = screen_x[i].item(), screen_y[i].item()
            a = opacity[i].item()

            if alpha_buffer[y, x] < self.config.alpha_threshold:
                # Alpha blending
                old_alpha = alpha_buffer[y, x]
                new_alpha = a + old_alpha * (1 - a)

                if new_alpha > 0:
                    image[:, y, x] = (
                        rgb[i] * a + image[:, y, x] * old_alpha * (1 - a)
                    ) / new_alpha

                alpha_buffer[y, x] = new_alpha

        return image

    def render_batch(
        self,
        coords_list: list,
        features_list: list,
        cameras: list,
    ) -> Tensor:
        """
        Render multiple views.

        Args:
            coords_list: List of [N_i, 3] coordinate tensors
            features_list: List of [N_i, 7] feature tensors
            cameras: List of Camera objects

        Returns:
            images: [B, 3, H, W] rendered images
        """
        images = []
        for coords, features, camera in zip(coords_list, features_list, cameras):
            img = self.forward(coords, features, camera)
            images.append(img)
        return torch.stack(images)


class CUDARasterizerStub(nn.Module):
    """
    Stub for CUDA rasterizer interface.

    In production, this should be replaced with actual CUDA implementation
    from TRELLIS (diffoctreerast).
    """

    def __init__(self, config: RenderConfig):
        super().__init__()
        self.config = config
        self.width, self.height = config.resolution
        self._fallback = VoxelRasterizer(config)

    def forward(
        self,
        coords: Tensor,
        features: Tensor,
        camera: Camera,
    ) -> Tensor:
        """Render using fallback (to be replaced with CUDA)."""
        return self._fallback(coords, features, camera)
