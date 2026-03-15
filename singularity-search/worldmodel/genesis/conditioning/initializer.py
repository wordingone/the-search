"""World initializer: generate initial OVoxel state from conditioning."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union, List

from genesis.config import InitializerConfig, MemoryConfig
from genesis.memory.ovoxel import OVoxelMemory


class WorldInitializer(nn.Module):
    """
    Generate initial OVoxel world state from text/image conditioning.

    Uses cross-attention to fuse conditioning information,
    then 3D deconvolution to generate dense voxels,
    followed by sparsification to keep top-K confident voxels.
    """

    def __init__(
        self,
        config: InitializerConfig,
        memory_config: MemoryConfig,
        conditioning_dim: int = 1024,
    ):
        """
        Args:
            config: Initializer configuration
            memory_config: Memory configuration
            conditioning_dim: Conditioning embedding dimension
        """
        super().__init__()
        self.config = config
        self.memory_config = memory_config

        # Seed generator
        self.seed = nn.Parameter(torch.randn(1, conditioning_dim))

        # Cross-attention for conditioning fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=conditioning_dim,
            num_heads=16,
            batch_first=True,
        )

        # 3D deconvolution to generate dense voxels
        # From 1x1x1 to output_resolution³
        channels = config.deconv_channels
        self.deconv = nn.Sequential(
            # 1³ -> 4³
            nn.ConvTranspose3d(conditioning_dim, channels[0], 4, 1, 0),
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),

            # 4³ -> 8³
            nn.ConvTranspose3d(channels[0], channels[1], 4, 2, 1),
            nn.GroupNorm(8, channels[1]),
            nn.SiLU(),

            # 8³ -> 16³
            nn.ConvTranspose3d(channels[1], channels[2], 4, 2, 1),
            nn.GroupNorm(8, channels[2]),
            nn.SiLU(),

            # 16³ -> 32³
            nn.ConvTranspose3d(channels[2], channels[3], 4, 2, 1),
            nn.GroupNorm(8, channels[3]),
            nn.SiLU(),

            # 32³ -> 64³
            nn.ConvTranspose3d(channels[3], channels[4], 4, 2, 1),
            nn.GroupNorm(8, channels[4]),
            nn.SiLU(),

            # Output: 7 PBR channels + 1 confidence
            nn.Conv3d(channels[4], 8, 1),
        )

        # PBR activations
        self.rgb_act = nn.Sigmoid()
        self.factor_act = nn.Sigmoid()
        self.sdf_act = nn.Tanh()
        self.conf_act = nn.Sigmoid()

    def forward(
        self,
        conditioning: Tensor,
        device: Optional[torch.device] = None,
    ) -> OVoxelMemory:
        """
        Generate initial world state.

        Args:
            conditioning: [B, L, C] conditioning embeddings (from text or image)
            device: Target device

        Returns:
            memory: OVoxelMemory with initial voxels
        """
        B = conditioning.shape[0]
        if device is None:
            device = conditioning.device

        # Expand seed for batch
        seed = self.seed.expand(B, 1, -1)  # [B, 1, C]

        # Cross-attend to conditioning
        fused, _ = self.cross_attn(seed, conditioning, conditioning)  # [B, 1, C]

        # Reshape for 3D deconv
        fused = fused.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]

        # Generate dense voxels
        dense = self.deconv(fused)  # [B, 8, 64, 64, 64]

        # Split into features and confidence
        features_raw = dense[:, :7]  # [B, 7, 64, 64, 64]
        confidence = self.conf_act(dense[:, 7:8])  # [B, 1, 64, 64, 64]

        # Apply PBR activations
        rgb = self.rgb_act(features_raw[:, :3])
        metallic = self.factor_act(features_raw[:, 3:4])
        roughness = self.factor_act(features_raw[:, 4:5])
        opacity = self.factor_act(features_raw[:, 5:6])
        sdf = self.sdf_act(features_raw[:, 6:7])

        features = torch.cat([rgb, metallic, roughness, opacity, sdf], dim=1)  # [B, 7, 64, 64, 64]

        # Sparsify: keep top-K by confidence
        memory = OVoxelMemory(self.memory_config, device)

        for b in range(B):
            conf_flat = confidence[b, 0].flatten()  # [64*64*64]
            feat_flat = features[b].permute(1, 2, 3, 0).flatten(0, 2)  # [64*64*64, 7]

            # Top-K
            k = min(self.config.top_k_voxels, conf_flat.shape[0])
            topk_conf, topk_idx = conf_flat.topk(k)

            # Filter by threshold
            threshold = 0.1
            valid = topk_conf > threshold
            topk_idx = topk_idx[valid]
            topk_features = feat_flat[topk_idx]

            # Convert indices to coordinates
            res = self.config.output_resolution
            x = topk_idx % res
            y = (topk_idx // res) % res
            z = topk_idx // (res * res)

            coords = torch.stack([
                torch.full_like(x, b),  # batch
                x, y, z
            ], dim=-1).int()

            # Scale coordinates to memory resolution
            scale = self.memory_config.resolution // res
            coords[:, 1:] = coords[:, 1:] * scale

            # Add to memory via direct assignment
            from genesis.deltav.predictor import DeltaV
            delta = DeltaV(
                coords=coords,
                features=topk_features,
                op_type=torch.full((coords.shape[0],), 2, device=device),  # add
                confidence=topk_conf[valid],
            )
            memory.apply_deltas(delta)

        return memory

    def initialize_from_text(
        self,
        text_embeddings: Tensor,
        device: Optional[torch.device] = None,
    ) -> OVoxelMemory:
        """Initialize world from text conditioning."""
        return self.forward(text_embeddings, device)

    def initialize_from_image(
        self,
        image_embeddings: Tensor,
        device: Optional[torch.device] = None,
    ) -> OVoxelMemory:
        """Initialize world from image conditioning."""
        return self.forward(image_embeddings, device)

    def initialize_empty(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> OVoxelMemory:
        """Initialize empty world (no voxels)."""
        return OVoxelMemory(self.memory_config, device)
