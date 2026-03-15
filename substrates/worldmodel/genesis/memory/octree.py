"""Sparse Voxel Octree for efficient spatial queries.

STUB: This is a reference implementation. For production use, replace with
CUDA-accelerated hierarchical octree from TRELLIS (diffoctreerast).
"""

import torch
from torch import Tensor
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from genesis.memory.ops import morton_encode, morton_decode


@dataclass
class OctreeNode:
    """Octree node representation."""
    level: int
    morton_code: int
    child_mask: int  # 8 bits, one per octant
    data_offset: int  # Offset into data array
    data_count: int   # Number of voxels in this node
    aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    children: List[int] = field(default_factory=list)  # Indices of child nodes


class SparseVoxelOctree:
    """
    Sparse Voxel Octree (SVO) for efficient spatial queries.

    Organizes voxels hierarchically for:
    - Fast frustum culling (O(log N) vs O(N))
    - Level-of-detail rendering
    - Efficient neighbor queries

    Reference: TRELLIS DfsOctree implementation

    NOTE: This is a reference implementation. Production should use CUDA.
    """

    def __init__(
        self,
        max_depth: int = 8,
        resolution: int = 256,
    ):
        """
        Args:
            max_depth: Maximum octree depth (2^max_depth = resolution)
            resolution: Grid resolution
        """
        self.max_depth = max_depth
        self.resolution = resolution

        # Node storage - [num_nodes, 7]: level, morton, child_mask, data_offset, data_count, aabb_idx, parent
        self.nodes: Optional[Tensor] = None
        self.node_count = 0

        # AABB storage - [num_nodes, 6]: min_x, min_y, min_z, max_x, max_y, max_z
        self.aabbs: Optional[Tensor] = None

        # Child indices - [num_nodes, 8]: indices of children (-1 for empty)
        self.children: Optional[Tensor] = None

        # Voxel data storage
        self.coords: Optional[Tensor] = None
        self.features: Optional[Tensor] = None

        # Sorted order for efficient traversal
        self.sorted_indices: Optional[Tensor] = None

        # Node to voxel mapping
        self.node_voxel_ranges: Optional[Tensor] = None  # [num_nodes, 2]: start, end

    def build(self, coords: Tensor, features: Tensor) -> None:
        """
        Build hierarchical octree from voxel coordinates and features.

        Args:
            coords: [N, 3] voxel coordinates
            features: [N, C] voxel features
        """
        if coords.shape[0] == 0:
            self.coords = coords
            self.features = features
            self.nodes = None
            self.aabbs = None
            self.children = None
            self.sorted_indices = torch.empty(0, dtype=torch.long, device=coords.device)
            return

        device = coords.device

        # Compute Morton codes for spatial coherence
        morton_codes = morton_encode(coords)

        # Sort by Morton code (Z-order curve for cache-friendly traversal)
        sorted_indices = morton_codes.argsort()
        self.sorted_indices = sorted_indices
        self.coords = coords[sorted_indices]
        self.features = features[sorted_indices]
        sorted_morton = morton_codes[sorted_indices]

        # Build hierarchical node structure
        self._build_hierarchical_nodes(sorted_morton, device)

    def _build_hierarchical_nodes(self, sorted_morton: Tensor, device: torch.device) -> None:
        """Build hierarchical octree node structure from sorted Morton codes."""
        n_voxels = sorted_morton.shape[0]

        # Estimate max nodes: at most n_voxels leaf nodes + internal nodes
        max_nodes = min(n_voxels * 2, 1_000_000)

        # Initialize storage
        self.nodes = torch.zeros(max_nodes, 5, dtype=torch.long, device=device)
        self.aabbs = torch.zeros(max_nodes, 6, dtype=torch.float32, device=device)
        self.children = torch.full((max_nodes, 8), -1, dtype=torch.long, device=device)
        self.node_voxel_ranges = torch.zeros(max_nodes, 2, dtype=torch.long, device=device)

        # Build recursively using a stack (iterative to avoid Python recursion limit)
        # Stack entries: (node_idx, level, voxel_start, voxel_end, aabb_min, aabb_max)
        self.node_count = 0

        # Root node
        root_idx = self._allocate_node()
        self.nodes[root_idx, 0] = 0  # level
        self.nodes[root_idx, 1] = 0  # morton code
        self.nodes[root_idx, 2] = 0  # child_mask (will be updated)
        self.nodes[root_idx, 3] = 0  # data_offset
        self.nodes[root_idx, 4] = n_voxels  # data_count
        self.aabbs[root_idx] = torch.tensor([0, 0, 0, self.resolution, self.resolution, self.resolution],
                                             dtype=torch.float32, device=device)
        self.node_voxel_ranges[root_idx] = torch.tensor([0, n_voxels], dtype=torch.long, device=device)

        # Build tree iteratively
        stack = [(root_idx, 0, 0, n_voxels)]

        while stack:
            node_idx, level, v_start, v_end = stack.pop()

            if level >= self.max_depth or v_end - v_start <= 8:
                # Leaf node - store as is
                continue

            # Compute AABB center for splitting
            aabb = self.aabbs[node_idx]
            center = (aabb[:3] + aabb[3:]) / 2

            # Partition voxels into 8 octants
            coords_subset = self.coords[v_start:v_end].float()

            octant_masks = [
                (coords_subset[:, 0] < center[0]) & (coords_subset[:, 1] < center[1]) & (coords_subset[:, 2] < center[2]),
                (coords_subset[:, 0] >= center[0]) & (coords_subset[:, 1] < center[1]) & (coords_subset[:, 2] < center[2]),
                (coords_subset[:, 0] < center[0]) & (coords_subset[:, 1] >= center[1]) & (coords_subset[:, 2] < center[2]),
                (coords_subset[:, 0] >= center[0]) & (coords_subset[:, 1] >= center[1]) & (coords_subset[:, 2] < center[2]),
                (coords_subset[:, 0] < center[0]) & (coords_subset[:, 1] < center[1]) & (coords_subset[:, 2] >= center[2]),
                (coords_subset[:, 0] >= center[0]) & (coords_subset[:, 1] < center[1]) & (coords_subset[:, 2] >= center[2]),
                (coords_subset[:, 0] < center[0]) & (coords_subset[:, 1] >= center[1]) & (coords_subset[:, 2] >= center[2]),
                (coords_subset[:, 0] >= center[0]) & (coords_subset[:, 1] >= center[1]) & (coords_subset[:, 2] >= center[2]),
            ]

            child_mask = 0
            current_offset = v_start

            for octant in range(8):
                count = octant_masks[octant].sum().item()
                if count > 0:
                    child_mask |= (1 << octant)

                    # Create child node
                    child_idx = self._allocate_node()
                    self.children[node_idx, octant] = child_idx

                    # Compute child AABB
                    child_min = aabb[:3].clone()
                    child_max = aabb[3:].clone()
                    if octant & 1: child_min[0] = center[0]
                    else: child_max[0] = center[0]
                    if octant & 2: child_min[1] = center[1]
                    else: child_max[1] = center[1]
                    if octant & 4: child_min[2] = center[2]
                    else: child_max[2] = center[2]

                    self.nodes[child_idx, 0] = level + 1
                    self.nodes[child_idx, 3] = current_offset
                    self.nodes[child_idx, 4] = count
                    self.aabbs[child_idx, :3] = child_min
                    self.aabbs[child_idx, 3:] = child_max
                    self.node_voxel_ranges[child_idx] = torch.tensor([current_offset, current_offset + count],
                                                                      dtype=torch.long, device=device)

                    # Add to stack for further subdivision
                    stack.append((child_idx, level + 1, current_offset, current_offset + count))

                    current_offset += count

            self.nodes[node_idx, 2] = child_mask

        # Trim storage to actual size
        self.nodes = self.nodes[:self.node_count]
        self.aabbs = self.aabbs[:self.node_count]
        self.children = self.children[:self.node_count]
        self.node_voxel_ranges = self.node_voxel_ranges[:self.node_count]

    def _allocate_node(self) -> int:
        """Allocate a new node and return its index."""
        idx = self.node_count
        self.node_count += 1
        return idx

    def query_frustum(
        self,
        camera_pos: Tensor,
        camera_dir: Tensor,
        fov: float = 90.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Query visible voxels within camera frustum.

        Uses hierarchical octree traversal for O(log N) culling.

        Args:
            camera_pos: [3] camera position
            camera_dir: [3] camera forward direction
            fov: Field of view in degrees
            near, far: Clipping planes

        Returns:
            visible_coords: [M, 3]
            visible_features: [M, C]
        """
        feature_dim = self.features.shape[1] if self.features is not None and self.features.numel() > 0 else 7

        if self.coords is None or self.coords.shape[0] == 0:
            device = camera_pos.device
            return torch.empty(0, 3, device=device), torch.empty(0, feature_dim, device=device)

        if self.nodes is None or self.node_count == 0:
            # Fallback to brute force if octree not built
            from genesis.memory.ops import frustum_cull
            return frustum_cull(
                self.coords, self.features,
                camera_pos, camera_dir,
                fov, near, far,
            )

        # Compute frustum planes for AABB testing
        frustum_planes = self._compute_frustum_planes(camera_pos, camera_dir, fov, near, far)

        # Hierarchical traversal
        visible_indices = []
        stack = [0]  # Start with root node

        while stack:
            node_idx = stack.pop()

            # Test AABB against frustum
            aabb = self.aabbs[node_idx]
            if not self._aabb_in_frustum(aabb, frustum_planes):
                continue  # Entire subtree culled

            # Get child mask
            child_mask = self.nodes[node_idx, 2].item()

            if child_mask == 0 or self.nodes[node_idx, 0].item() >= self.max_depth:
                # Leaf node - collect all voxels
                v_start = self.node_voxel_ranges[node_idx, 0].item()
                v_end = self.node_voxel_ranges[node_idx, 1].item()
                visible_indices.extend(range(v_start, v_end))
            else:
                # Internal node - traverse children
                for octant in range(8):
                    if child_mask & (1 << octant):
                        child_idx = self.children[node_idx, octant].item()
                        if child_idx >= 0:
                            stack.append(child_idx)

        if not visible_indices:
            device = camera_pos.device
            return torch.empty(0, 3, device=device), torch.empty(0, feature_dim, device=device)

        indices = torch.tensor(visible_indices, dtype=torch.long, device=self.coords.device)
        return self.coords[indices], self.features[indices]

    def _compute_frustum_planes(
        self,
        camera_pos: Tensor,
        camera_dir: Tensor,
        fov: float,
        near: float,
        far: float,
    ) -> Tensor:
        """
        Compute 6 frustum planes for AABB testing.

        Returns:
            planes: [6, 4] plane equations (nx, ny, nz, d) where nx*x + ny*y + nz*z + d = 0
        """
        import math

        device = camera_pos.device
        forward = camera_dir / (camera_dir.norm() + 1e-8)

        # Approximate up vector (assume world up is Y)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        right = torch.cross(forward, world_up)
        if right.norm() < 1e-6:
            world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
            right = torch.cross(forward, world_up)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(right, forward)
        up = up / (up.norm() + 1e-8)

        # Half angles
        fov_rad = fov * math.pi / 180
        half_v = math.tan(fov_rad / 2)
        half_h = half_v  # Assume square aspect

        # Near/far plane centers
        near_center = camera_pos + forward * near
        far_center = camera_pos + forward * far

        planes = torch.zeros(6, 4, device=device)

        # Near plane: normal = forward
        planes[0, :3] = forward
        planes[0, 3] = -torch.dot(forward, near_center)

        # Far plane: normal = -forward
        planes[1, :3] = -forward
        planes[1, 3] = torch.dot(forward, far_center)

        # Left plane
        left_normal = (forward + right * half_h)
        left_normal = left_normal / (left_normal.norm() + 1e-8)
        planes[2, :3] = left_normal
        planes[2, 3] = -torch.dot(left_normal, camera_pos)

        # Right plane
        right_normal = (forward - right * half_h)
        right_normal = right_normal / (right_normal.norm() + 1e-8)
        planes[3, :3] = right_normal
        planes[3, 3] = -torch.dot(right_normal, camera_pos)

        # Top plane
        top_normal = (forward - up * half_v)
        top_normal = top_normal / (top_normal.norm() + 1e-8)
        planes[4, :3] = top_normal
        planes[4, 3] = -torch.dot(top_normal, camera_pos)

        # Bottom plane
        bottom_normal = (forward + up * half_v)
        bottom_normal = bottom_normal / (bottom_normal.norm() + 1e-8)
        planes[5, :3] = bottom_normal
        planes[5, 3] = -torch.dot(bottom_normal, camera_pos)

        return planes

    def _aabb_in_frustum(self, aabb: Tensor, planes: Tensor) -> bool:
        """
        Test if AABB intersects frustum using plane-AABB intersection.

        Args:
            aabb: [6] tensor (min_x, min_y, min_z, max_x, max_y, max_z)
            planes: [6, 4] frustum planes

        Returns:
            True if AABB is (partially) inside frustum
        """
        aabb_min = aabb[:3]
        aabb_max = aabb[3:]

        for i in range(6):
            normal = planes[i, :3]
            d = planes[i, 3]

            # Find the p-vertex (corner most in normal direction)
            p = torch.where(normal > 0, aabb_max, aabb_min)

            # If p-vertex is outside, entire AABB is outside
            if torch.dot(normal, p) + d < 0:
                return False

        return True

    def query_box(
        self,
        min_corner: Tensor,
        max_corner: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Query voxels within axis-aligned bounding box.

        Args:
            min_corner: [3] minimum corner
            max_corner: [3] maximum corner

        Returns:
            coords: [M, 3]
            features: [M, C]
        """
        if self.coords is None or self.coords.shape[0] == 0:
            return torch.empty(0, 3), torch.empty(0, self.features.shape[1] if self.features is not None else 7)

        mask = (
            (self.coords[:, 0] >= min_corner[0]) & (self.coords[:, 0] <= max_corner[0]) &
            (self.coords[:, 1] >= min_corner[1]) & (self.coords[:, 1] <= max_corner[1]) &
            (self.coords[:, 2] >= min_corner[2]) & (self.coords[:, 2] <= max_corner[2])
        )

        return self.coords[mask], self.features[mask]

    def query_neighbors(
        self,
        coord: Tensor,
        radius: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Query neighboring voxels within Manhattan distance.

        Args:
            coord: [3] query coordinate
            radius: Search radius

        Returns:
            coords: [M, 3]
            features: [M, C]
        """
        min_corner = coord - radius
        max_corner = coord + radius
        return self.query_box(min_corner, max_corner)

    def serialize(self) -> bytes:
        """
        Serialize octree to bytes for storage.

        Returns:
            Compressed byte representation
        """
        import io
        import struct

        buffer = io.BytesIO()

        # Header
        buffer.write(b'SVO')  # Magic
        buffer.write(struct.pack('B', 1))  # Version
        buffer.write(struct.pack('I', self.max_depth))
        buffer.write(struct.pack('I', self.resolution))

        if self.coords is not None:
            num_voxels = self.coords.shape[0]
            buffer.write(struct.pack('I', num_voxels))

            # Write coords and features
            buffer.write(self.coords.cpu().numpy().tobytes())
            buffer.write(self.features.cpu().numpy().tobytes())
        else:
            buffer.write(struct.pack('I', 0))

        return buffer.getvalue()

    @classmethod
    def deserialize(cls, data: bytes, device: torch.device = None) -> "SparseVoxelOctree":
        """
        Deserialize octree from bytes.

        Args:
            data: Serialized bytes
            device: Target device

        Returns:
            Reconstructed SparseVoxelOctree
        """
        import io
        import struct
        import numpy as np

        buffer = io.BytesIO(data)

        # Header
        magic = buffer.read(3)
        assert magic == b'SVO', "Invalid SVO file"
        version = struct.unpack('B', buffer.read(1))[0]
        max_depth = struct.unpack('I', buffer.read(4))[0]
        resolution = struct.unpack('I', buffer.read(4))[0]

        octree = cls(max_depth=max_depth, resolution=resolution)

        num_voxels = struct.unpack('I', buffer.read(4))[0]

        if num_voxels > 0:
            coords_bytes = buffer.read(num_voxels * 3 * 4)  # int32
            features_bytes = buffer.read(num_voxels * 7 * 4)  # float32

            coords = np.frombuffer(coords_bytes, dtype=np.int32).reshape(num_voxels, 3)
            features = np.frombuffer(features_bytes, dtype=np.float32).reshape(num_voxels, 7)

            octree.coords = torch.from_numpy(coords.copy())
            octree.features = torch.from_numpy(features.copy())

            if device is not None:
                octree.coords = octree.coords.to(device)
                octree.features = octree.features.to(device)

            octree.build(octree.coords, octree.features)

        return octree
