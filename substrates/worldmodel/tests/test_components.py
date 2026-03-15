"""Component tests for Genesis."""

import pytest
import torch

# Skip if dependencies not available
pytest.importorskip("torch")


class TestFSQ:
    """Test Finite Scalar Quantization."""

    def test_forward_shape(self):
        from genesis.tokenizer.fsq import FSQ

        fsq = FSQ(levels=[8, 6, 5, 5, 5])
        z = torch.randn(2, 4, 16, 16, 5)  # [B, T, H, W, D]

        z_q, indices = fsq(z)

        assert z_q.shape == z.shape
        assert indices.shape == z.shape[:-1]

    def test_vocab_size(self):
        from genesis.tokenizer.fsq import FSQ

        fsq = FSQ(levels=[8, 6, 5, 5, 5])
        assert fsq.vocab_size == 8 * 6 * 5 * 5 * 5  # 6000

    def test_roundtrip(self):
        from genesis.tokenizer.fsq import FSQ

        fsq = FSQ(levels=[8, 6, 5, 5, 5])
        z = torch.randn(2, 5)

        z_q, indices = fsq(z)
        codes = fsq.indices_to_codes(indices)

        # Verify indices → codes → indices roundtrip
        indices_back = fsq.codes_to_indices(codes)
        assert torch.equal(indices, indices_back)


class TestRoPE3D:
    """Test 3D Rotary Position Embedding."""

    def test_forward_shape(self):
        from genesis.dynamics.rope import RoPE3D

        rope = RoPE3D(dim=64, max_temporal=16, max_spatial=32)
        x = torch.randn(2, 8, 16, 16, 64)  # [B, T, H, W, D]

        x_rotated = rope(x)

        assert x_rotated.shape == x.shape

    def test_dimension_allocation(self):
        from genesis.dynamics.rope import RoPE3D

        rope = RoPE3D(dim=96, temporal_fraction=0.333)

        # Check dimension allocation
        assert rope.t_dim + rope.h_dim + rope.w_dim == 96
        assert abs(rope.t_dim - 32) <= 2  # ~1/3 (allow rounding)


class TestOVoxelMemory:
    """Test OVoxel memory operations."""

    def test_empty_memory(self):
        from genesis.memory.ovoxel import OVoxelMemory
        from genesis.config import MemoryConfig

        config = MemoryConfig()
        memory = OVoxelMemory(config, device=torch.device('cpu'))

        assert memory.num_voxels == 0

    def test_add_voxels(self):
        from genesis.memory.ovoxel import OVoxelMemory
        from genesis.config import MemoryConfig
        from genesis.deltav.predictor import DeltaV

        config = MemoryConfig()
        memory = OVoxelMemory(config, device=torch.device('cpu'))

        # Create delta
        coords = torch.tensor([[0, 10, 20, 30], [0, 11, 21, 31]], dtype=torch.int32)
        features = torch.randn(2, 7)
        delta = DeltaV(
            coords=coords,
            features=features,
            op_type=torch.tensor([2, 2]),  # add
            confidence=torch.ones(2),
        )

        memory.apply_deltas(delta)

        assert memory.num_voxels == 2

    def test_prune(self):
        from genesis.memory.ovoxel import OVoxelMemory
        from genesis.config import MemoryConfig
        from genesis.deltav.predictor import DeltaV

        config = MemoryConfig()
        memory = OVoxelMemory(config, device=torch.device('cpu'))

        # Add voxels with varying opacity
        coords = torch.tensor([[0, 10, 20, 30], [0, 11, 21, 31]], dtype=torch.int32)
        features = torch.tensor([
            [1, 1, 1, 0.5, 0.5, 0.5, 0],  # opacity = 0.5
            [1, 1, 1, 0.5, 0.5, 0.0, 0],  # opacity = 0 (should be pruned)
        ])
        delta = DeltaV(
            coords=coords,
            features=features,
            op_type=torch.tensor([2, 2]),
            confidence=torch.ones(2),
        )

        memory.apply_deltas(delta)
        pruned = memory.prune()

        assert pruned == 1
        assert memory.num_voxels == 1


class TestDeltaVPredictor:
    """Test DeltaV predictor."""

    def test_forward_shape(self):
        from genesis.deltav.predictor import DeltaVPredictor
        from genesis.config import DeltaVConfig

        config = DeltaVConfig(input_dim=256, max_deltas_per_frame=100)
        predictor = DeltaVPredictor(config)

        features = torch.randn(2, 16, 16, 256)  # [B, H, W, D]
        delta_v = predictor(features)

        assert delta_v.coords.shape[1] == 4  # batch, x, y, z
        assert delta_v.features.shape[1] == 7  # PBR channels
        assert delta_v.num_deltas <= 100 * 2  # max_deltas * batch


class TestGenesis:
    """Test full Genesis model."""

    def test_init(self):
        from genesis import Genesis, GenesisConfig

        config = GenesisConfig()
        model = Genesis(config, use_stubs=True)

        assert model is not None

    def test_param_count(self):
        from genesis import Genesis, GenesisConfig

        config = GenesisConfig()
        model = Genesis(config, use_stubs=True)

        counts = model.get_param_count()

        # Check total is reasonable (should be ~2B for full config)
        assert counts['total'] > 0
        print(f"Total parameters: {counts['total']:,}")

    def test_initialize_world(self):
        from genesis import Genesis, GenesisConfig

        config = GenesisConfig()
        model = Genesis(config, use_stubs=True)

        # Initialize empty world
        memory = model.initialize_world()

        assert memory is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
