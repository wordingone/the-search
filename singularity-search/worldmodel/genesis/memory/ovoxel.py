"""OVoxel Memory: persistent sparse voxel storage."""

import torch
from torch import Tensor
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from genesis.config import MemoryConfig
from genesis.memory.octree import SparseVoxelOctree
from genesis.memory.ops import coords_to_index, build_coord_hashmap


class OVoxelMemory:
    """
    Persistent sparse voxel memory for Genesis world model.

    Stores the 3D world state as sparse voxels with PBR attributes.
    Supports efficient add/modify/remove operations via delta updates.

    Feature layout (7 channels):
    - RGB: 3 channels (diffuse color, linear space)
    - Metallic: 1 channel ([0, 1])
    - Roughness: 1 channel ([0, 1])
    - Opacity: 1 channel ([0, 1], 0 = removed)
    - SDF: 1 channel ([-1, 1], signed distance)
    """

    def __init__(self, config: MemoryConfig, device: torch.device = None):
        """
        Args:
            config: Memory configuration
            device: Target device
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Use float32 on CPU (float16 has limited CPU support), float16 on CUDA
        self.feature_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        # Core storage
        self.coords: Tensor = torch.empty(0, 4, dtype=torch.int32, device=self.device)  # [N, 4]: batch, x, y, z
        self.features: Tensor = torch.empty(0, config.feature_channels, dtype=self.feature_dtype, device=self.device)

        # Spatial index
        self.svo = SparseVoxelOctree(
            max_depth=config.svo.max_depth,
            resolution=config.resolution,
        )

        # Coordinate lookup (for fast modify/remove)
        self._coord_to_idx: Dict[Tuple[int, int, int, int], int] = {}

        # Statistics
        self.frame_count = 0
        self.total_adds = 0
        self.total_modifies = 0
        self.total_removes = 0

    @property
    def num_voxels(self) -> int:
        """Current number of voxels."""
        return self.coords.shape[0]

    @property
    def memory_usage_mb(self) -> float:
        """Approximate memory usage in MB."""
        coord_bytes = self.coords.numel() * 4  # int32
        feature_bytes = self.features.numel() * 2  # float16
        return (coord_bytes + feature_bytes) / (1024 * 1024)

    def apply_deltas(self, delta_v) -> None:
        """
        Apply sparse voxel deltas to memory.

        Args:
            delta_v: DeltaV object with coords, features, op_type
        """
        from genesis.deltav.predictor import DeltaV

        if not isinstance(delta_v, DeltaV):
            raise TypeError("Expected DeltaV object")

        # Separate by operation type
        add_mask = delta_v.op_type == 2
        modify_mask = delta_v.op_type == 1
        remove_mask = delta_v.op_type == 0

        # Process adds
        if add_mask.any():
            self._add_voxels(
                delta_v.coords[add_mask],
                delta_v.features[add_mask],
            )

        # Process modifies
        if modify_mask.any():
            self._modify_voxels(
                delta_v.coords[modify_mask],
                delta_v.features[modify_mask],
            )

        # Process removes (set opacity to 0)
        if remove_mask.any():
            self._remove_voxels(delta_v.coords[remove_mask])

        self.frame_count += 1

        # Periodic pruning
        if self.frame_count % self.config.prune_interval == 0:
            self.prune()

    def _add_voxels(self, coords: Tensor, features: Tensor) -> None:
        """Add new voxels (skip if already exists)."""
        coords = coords.to(self.device)
        features = features.to(self.device, dtype=self.feature_dtype)

        new_coords = []
        new_features = []

        for i in range(coords.shape[0]):
            key = tuple(coords[i].tolist())
            if key not in self._coord_to_idx:
                new_coords.append(coords[i])
                new_features.append(features[i])
                self._coord_to_idx[key] = self.num_voxels + len(new_coords) - 1

        if new_coords:
            new_coords = torch.stack(new_coords)
            new_features = torch.stack(new_features)

            self.coords = torch.cat([self.coords, new_coords], dim=0)
            self.features = torch.cat([self.features, new_features], dim=0)
            self.total_adds += len(new_coords)

            # Invalidate SVO
            self._rebuild_svo()

    def _modify_voxels(self, coords: Tensor, features: Tensor) -> None:
        """Modify existing voxels."""
        coords = coords.to(self.device)
        features = features.to(self.device, dtype=self.feature_dtype)

        for i in range(coords.shape[0]):
            key = tuple(coords[i].tolist())
            if key in self._coord_to_idx:
                idx = self._coord_to_idx[key]
                self.features[idx] = features[i]
                self.total_modifies += 1

    def _remove_voxels(self, coords: Tensor) -> None:
        """Mark voxels for removal by setting opacity to 0."""
        coords = coords.to(self.device)

        for i in range(coords.shape[0]):
            key = tuple(coords[i].tolist())
            if key in self._coord_to_idx:
                idx = self._coord_to_idx[key]
                self.features[idx, 5] = 0  # Opacity channel
                self.total_removes += 1

    def prune(self) -> int:
        """
        Remove voxels with zero opacity.

        Returns:
            Number of voxels pruned
        """
        if self.num_voxels == 0:
            return 0

        # Keep voxels with opacity > threshold
        mask = self.features[:, 5] > self.config.opacity_threshold
        num_pruned = (~mask).sum().item()

        if num_pruned > 0:
            self.coords = self.coords[mask]
            self.features = self.features[mask]

            # Rebuild lookup
            self._rebuild_coord_lookup()
            self._rebuild_svo()

        return num_pruned

    def _rebuild_coord_lookup(self) -> None:
        """Rebuild coordinate to index mapping."""
        self._coord_to_idx = {}
        for i in range(self.num_voxels):
            key = tuple(self.coords[i].tolist())
            self._coord_to_idx[key] = i

    def _rebuild_svo(self) -> None:
        """Rebuild spatial index."""
        if self.num_voxels > 0:
            # Extract xyz (exclude batch)
            xyz = self.coords[:, 1:4]
            self.svo.build(xyz, self.features)

    def query_frustum(
        self,
        camera_pos: Tensor,
        camera_dir: Tensor,
        fov: float = 90.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Query visible voxels for rendering.

        Args:
            camera_pos: [3] camera position
            camera_dir: [3] camera forward direction
            fov: Field of view in degrees
            near, far: Clipping planes

        Returns:
            coords: [M, 3] visible voxel coordinates
            features: [M, 7] visible voxel features
        """
        return self.svo.query_frustum(camera_pos, camera_dir, fov, near, far)

    def get_all_voxels(self) -> Tuple[Tensor, Tensor]:
        """Get all voxels (for rendering without culling)."""
        if self.num_voxels == 0:
            return self.coords, self.features
        return self.coords[:, 1:4], self.features  # Exclude batch dim

    def clear(self) -> None:
        """Clear all voxels."""
        self.coords = torch.empty(0, 4, dtype=torch.int32, device=self.device)
        self.features = torch.empty(0, self.config.feature_channels, dtype=self.feature_dtype, device=self.device)
        self._coord_to_idx = {}
        self.svo = SparseVoxelOctree(
            max_depth=self.config.svo.max_depth,
            resolution=self.config.resolution,
        )

    def save(self, path: str) -> None:
        """Save memory to file."""
        torch.save({
            'coords': self.coords.cpu(),
            'features': self.features.cpu(),
            'frame_count': self.frame_count,
            'stats': {
                'adds': self.total_adds,
                'modifies': self.total_modifies,
                'removes': self.total_removes,
            },
        }, path)

    def load(self, path: str) -> None:
        """Load memory from file."""
        data = torch.load(path)
        self.coords = data['coords'].to(self.device)
        self.features = data['features'].to(self.device)
        self.frame_count = data.get('frame_count', 0)

        stats = data.get('stats', {})
        self.total_adds = stats.get('adds', 0)
        self.total_modifies = stats.get('modifies', 0)
        self.total_removes = stats.get('removes', 0)

        self._rebuild_coord_lookup()
        self._rebuild_svo()

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            'num_voxels': self.num_voxels,
            'memory_mb': self.memory_usage_mb,
            'frame_count': self.frame_count,
            'total_adds': self.total_adds,
            'total_modifies': self.total_modifies,
            'total_removes': self.total_removes,
            'capacity_used': self.num_voxels / max(1, self.config.max_voxels),
        }
