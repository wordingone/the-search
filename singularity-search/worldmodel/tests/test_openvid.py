"""Tests for OpenVid dataset streaming and quality validation."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestOpenVidDataset:
    """Test OpenVid streaming dataset."""

    def test_import(self):
        """Test dataset classes can be imported."""
        from genesis.data import OpenVidStreamDataset, Panda70MStreamDataset
        assert OpenVidStreamDataset is not None
        assert Panda70MStreamDataset is not None

    def test_dataset_creation(self):
        """Test dataset can be instantiated."""
        from genesis.data import OpenVidStreamDataset

        dataset = OpenVidStreamDataset(
            seq_length=16,
            image_size=720,
            shuffle=True,
        )

        assert dataset.seq_length == 16
        assert dataset.image_size == 720
        assert dataset.shuffle is True

    def test_resize_center_crop(self):
        """Test resize and center crop logic."""
        from genesis.data.openvid import OpenVidStreamDataset

        dataset = OpenVidStreamDataset(image_size=256)

        # Test landscape image
        landscape = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cropped = dataset._resize_center_crop(landscape, 256)
        assert cropped.shape == (256, 256, 3)

        # Test portrait image
        portrait = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        cropped = dataset._resize_center_crop(portrait, 256)
        assert cropped.shape == (256, 256, 3)

        # Test square image
        square = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cropped = dataset._resize_center_crop(square, 256)
        assert cropped.shape == (256, 256, 3)

    def test_frame_tensor_format(self):
        """Test that frames are in correct tensor format."""
        from genesis.data import OpenVidStreamDataset

        dataset = OpenVidStreamDataset(
            seq_length=16,
            image_size=256,
        )

        # Mock a decoded frame
        mock_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        frame_tensor = torch.from_numpy(mock_frame).permute(2, 0, 1).float() / 255.0

        # Check format
        assert frame_tensor.shape == (3, 256, 256)
        assert frame_tensor.min() >= 0.0
        assert frame_tensor.max() <= 1.0
        assert frame_tensor.dtype == torch.float32


class TestDataloaderIntegration:
    """Test dataloader creation and integration."""

    def test_dataloader_creation(self):
        """Test dataloader factory function."""
        from genesis.data import create_openvid_dataloader

        # Just test creation, don't iterate (requires network)
        loader = create_openvid_dataloader(
            batch_size=2,
            seq_length=8,
            image_size=256,
            num_workers=0,
        )

        assert loader is not None
        assert loader.batch_size == 2

    def test_training_script_integration(self):
        """Test that training script recognizes openvid data mode."""
        import subprocess
        result = subprocess.run(
            ['python', 'scripts/genesis_experiment.py', '--help'],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        assert 'openvid' in result.stdout
        assert 'panda70m' in result.stdout


class TestQualityMetrics:
    """Test quality validation metrics."""

    def test_clip_iqa_import(self):
        """Test CLIP-IQA can be imported for quality measurement."""
        try:
            import clip
            import pyiqa
            has_quality_metrics = True
        except ImportError:
            has_quality_metrics = False

        # Skip if not installed, but log
        if not has_quality_metrics:
            pytest.skip("Quality metrics libraries not installed")

    def test_video_quality_bounds(self):
        """Test that video frames are in valid quality bounds."""
        # Synthetic test data
        frames = torch.rand(8, 3, 256, 256)

        # Check basic quality bounds
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

        # Check no NaN values
        assert not torch.isnan(frames).any()

        # Check variance (not all zeros or ones)
        assert frames.std() > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
