"""Sparse voxel operations."""

import torch
from torch import Tensor
from typing import Tuple, Optional


def morton_encode(coords: Tensor) -> Tensor:
    """
    Encode 3D coordinates to Morton codes (Z-order curve).

    Morton codes interleave bits of x, y, z coordinates,
    providing good spatial locality for cache efficiency.

    Args:
        coords: [N, 3] int32 tensor of (x, y, z) coordinates

    Returns:
        codes: [N] int64 tensor of Morton codes
    """
    x = coords[:, 0].long()
    y = coords[:, 1].long()
    z = coords[:, 2].long()

    # Spread bits (interleave with zeros)
    def spread_bits(v: Tensor) -> Tensor:
        # Spread 10 bits of v to occupy every 3rd bit
        v = v & 0x3FF  # Keep only 10 bits
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    code = spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
    return code


def morton_decode(codes: Tensor) -> Tensor:
    """
    Decode Morton codes back to 3D coordinates.

    Args:
        codes: [N] int64 Morton codes

    Returns:
        coords: [N, 3] int32 (x, y, z) coordinates
    """
    def compact_bits(v: Tensor) -> Tensor:
        # Reverse of spread_bits
        v = v & 0x09249249
        v = (v | (v >> 2)) & 0x030C30C3
        v = (v | (v >> 4)) & 0x0300F00F
        v = (v | (v >> 8)) & 0x030000FF
        v = (v | (v >> 16)) & 0x3FF
        return v

    x = compact_bits(codes)
    y = compact_bits(codes >> 1)
    z = compact_bits(codes >> 2)

    return torch.stack([x, y, z], dim=-1).int()


def sparse_scatter(
    values: Tensor,
    indices: Tensor,
    size: int,
    reduce: str = "mean",
) -> Tensor:
    """
    Scatter values to sparse positions with reduction.

    Args:
        values: [N, C] values to scatter
        indices: [N] linear indices
        size: Output size
        reduce: Reduction mode ("mean", "sum", "max")

    Returns:
        output: [size, C] scattered values
    """
    C = values.shape[1]
    output = torch.zeros(size, C, device=values.device, dtype=values.dtype)

    if reduce == "sum":
        output.scatter_add_(0, indices.unsqueeze(-1).expand(-1, C), values)
    elif reduce == "mean":
        output.scatter_add_(0, indices.unsqueeze(-1).expand(-1, C), values)
        counts = torch.zeros(size, device=values.device)
        counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
        counts = counts.clamp(min=1).unsqueeze(-1)
        output = output / counts
    elif reduce == "max":
        output.scatter_reduce_(
            0, indices.unsqueeze(-1).expand(-1, C), values,
            reduce="amax", include_self=False,
        )
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")

    return output


def sparse_gather(
    values: Tensor,
    indices: Tensor,
) -> Tensor:
    """
    Gather values from sparse positions.

    Args:
        values: [size, C] source values
        indices: [N] linear indices

    Returns:
        output: [N, C] gathered values
    """
    C = values.shape[1]
    return values[indices]


def coords_to_index(
    coords: Tensor,
    resolution: int,
) -> Tensor:
    """
    Convert (x, y, z) coordinates to linear indices.

    Args:
        coords: [N, 3] coordinates
        resolution: Grid resolution

    Returns:
        indices: [N] linear indices
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return x + y * resolution + z * resolution * resolution


def index_to_coords(
    indices: Tensor,
    resolution: int,
) -> Tensor:
    """
    Convert linear indices to (x, y, z) coordinates.

    Args:
        indices: [N] linear indices
        resolution: Grid resolution

    Returns:
        coords: [N, 3] coordinates
    """
    x = indices % resolution
    y = (indices // resolution) % resolution
    z = indices // (resolution * resolution)
    return torch.stack([x, y, z], dim=-1)


def build_coord_hashmap(
    coords: Tensor,
    resolution: int,
) -> dict:
    """
    Build hash map from coordinates to indices.

    Args:
        coords: [N, 3] coordinates
        resolution: Grid resolution

    Returns:
        hashmap: dict mapping (x, y, z) tuples to indices
    """
    hashmap = {}
    for i, (x, y, z) in enumerate(coords.tolist()):
        hashmap[(x, y, z)] = i
    return hashmap


def frustum_cull(
    coords: Tensor,
    features: Tensor,
    camera_pos: Tensor,
    camera_dir: Tensor,
    fov: float = 90.0,
    near: float = 0.1,
    far: float = 100.0,
) -> Tuple[Tensor, Tensor]:
    """
    Cull voxels outside camera frustum.

    Args:
        coords: [N, 3] voxel coordinates
        features: [N, C] voxel features
        camera_pos: [3] camera position
        camera_dir: [3] camera forward direction
        fov: Field of view in degrees
        near, far: Clipping planes

    Returns:
        culled_coords: [M, 3] visible coordinates
        culled_features: [M, C] visible features
    """
    # Vector from camera to voxels
    to_voxel = coords.float() - camera_pos.unsqueeze(0)

    # Distance along view direction
    dist = (to_voxel * camera_dir.unsqueeze(0)).sum(dim=-1)

    # Near/far culling
    mask = (dist >= near) & (dist <= far)

    # FOV culling (simplified: check angle)
    fov_rad = fov * 3.14159 / 180.0
    cos_half_fov = torch.cos(torch.tensor(fov_rad / 2))

    to_voxel_norm = to_voxel / (to_voxel.norm(dim=-1, keepdim=True) + 1e-8)
    cos_angle = (to_voxel_norm * camera_dir.unsqueeze(0)).sum(dim=-1)
    mask = mask & (cos_angle >= cos_half_fov)

    return coords[mask], features[mask]
