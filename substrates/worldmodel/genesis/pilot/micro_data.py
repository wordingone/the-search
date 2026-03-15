"""Micro Data: Bouncing square with occlusion for 16x16 @ 8 frames.

Generates minimal dataset to test whether 4D field helps occlusion recovery.

Data characteristics:
- Resolution: 16x16 grayscale
- Frames: 8 per sequence
- Object: Single 4x4 white square
- Occluder: Fixed vertical gray bar (x=7-8)
- Motion: Square moves horizontally, passes behind bar
- Occlusion frames: ~3-4 (depending on starting position)

The test: Can the model predict the square's position after it reappears?
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def generate_sequence(
    size: int = 16,
    frames: int = 8,
    square_size: int = 4,
    occluder_x: Tuple[int, int] = (7, 9),
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single bouncing square sequence with occlusion.

    The square moves horizontally and passes behind a fixed vertical occluder.
    This tests whether the model can maintain object state during occlusion.

    Args:
        size: Image resolution (square)
        frames: Number of frames
        square_size: Size of the moving square
        occluder_x: X range of the occluder bar (start, end)
        seed: Random seed for reproducibility

    Returns:
        sequence: [frames, size, size] grayscale images (float32, 0-1)
        occlusion_mask: [frames] boolean mask, True when square is occluded
    """
    if seed is not None:
        np.random.seed(seed)

    seq = np.zeros((frames, size, size), dtype=np.float32)
    occlusion_mask = np.zeros(frames, dtype=bool)

    # Square starting position: left side, random y
    # Start at x=0-3 so it will definitely cross the occluder
    x = np.random.randint(0, 4)
    y = np.random.randint(2, size - square_size - 2)

    # Velocity: always move right (could add variation later)
    vx = 1

    for t in range(frames):
        frame = np.zeros((size, size), dtype=np.float32)

        # Draw occluder (fixed vertical bar)
        frame[2:size - 2, occluder_x[0]:occluder_x[1]] = 0.3

        # Check if square overlaps with occluder
        square_left = x
        square_right = x + square_size
        is_occluded = (
            square_right > occluder_x[0] and square_left < occluder_x[1]
        )
        occlusion_mask[t] = is_occluded

        # Draw square (only visible part)
        if not is_occluded:
            # Square fully visible
            x_start = max(0, x)
            x_end = min(size, x + square_size)
            y_start = max(0, y)
            y_end = min(size, y + square_size)
            frame[y_start:y_end, x_start:x_end] = 1.0
        else:
            # Square partially visible (draw parts outside occluder)
            y_start = max(0, y)
            y_end = min(size, y + square_size)

            # Left visible part
            if square_left < occluder_x[0]:
                x_start = max(0, x)
                x_end = occluder_x[0]
                frame[y_start:y_end, x_start:x_end] = 1.0

            # Right visible part
            if square_right > occluder_x[1]:
                x_start = occluder_x[1]
                x_end = min(size, x + square_size)
                frame[y_start:y_end, x_start:x_end] = 1.0

        seq[t] = frame
        x += vx

        # Clamp position (though square will exit right side)
        x = min(x, size - 1)

    return seq, occlusion_mask


class MicroDataset(Dataset):
    """Dataset of bouncing squares with occlusion."""

    def __init__(
        self,
        num_sequences: int = 500,
        size: int = 16,
        frames: int = 8,
        seed: int = 42,
    ):
        """Initialize dataset.

        Args:
            num_sequences: Number of sequences to generate
            size: Image resolution
            frames: Frames per sequence
            seed: Random seed for reproducibility
        """
        self.num_sequences = num_sequences
        self.size = size
        self.frames = frames
        self.seed = seed

        # Pre-generate all sequences
        self.sequences = []
        self.occlusion_masks = []

        for i in range(num_sequences):
            seq, mask = generate_sequence(
                size=size,
                frames=frames,
                seed=seed + i,
            )
            self.sequences.append(seq)
            self.occlusion_masks.append(mask)

        self.sequences = np.stack(self.sequences)  # [N, T, H, W]
        self.occlusion_masks = np.stack(self.occlusion_masks)  # [N, T]

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its occlusion mask.

        Returns:
            frames: [T, 1, H, W] tensor
            occlusion_mask: [T] boolean tensor
        """
        seq = torch.from_numpy(self.sequences[idx])  # [T, H, W]
        seq = seq.unsqueeze(1)  # [T, 1, H, W]
        mask = torch.from_numpy(self.occlusion_masks[idx])  # [T]
        return seq, mask


def get_recovery_frames(occlusion_mask: torch.Tensor) -> torch.Tensor:
    """Find recovery frames (first visible frames after occlusion).

    These are the critical frames for testing occlusion recovery.
    The 4D model should excel specifically on these frames.

    Args:
        occlusion_mask: [T] or [B, T] boolean mask

    Returns:
        recovery_mask: [T] or [B, T] boolean mask, True for recovery frames
    """
    if occlusion_mask.dim() == 1:
        occlusion_mask = occlusion_mask.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, T = occlusion_mask.shape
    recovery_mask = torch.zeros_like(occlusion_mask)

    for b in range(B):
        in_occlusion = False
        for t in range(T):
            if occlusion_mask[b, t]:
                in_occlusion = True
            elif in_occlusion:
                # First visible frame after occlusion
                recovery_mask[b, t] = True
                # Mark next frame too (recovery window)
                if t + 1 < T:
                    recovery_mask[b, t + 1] = True
                in_occlusion = False

    if squeeze:
        recovery_mask = recovery_mask.squeeze(0)

    return recovery_mask


def save_dataset(dataset: MicroDataset, path: Path) -> None:
    """Save dataset to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / "sequences.npy", dataset.sequences)
    np.save(path / "occlusion_masks.npy", dataset.occlusion_masks)

    # Save metadata
    with open(path / "metadata.txt", "w") as f:
        f.write(f"num_sequences: {dataset.num_sequences}\n")
        f.write(f"size: {dataset.size}\n")
        f.write(f"frames: {dataset.frames}\n")
        f.write(f"seed: {dataset.seed}\n")

    print(f"Saved {dataset.num_sequences} sequences to {path}")


def load_dataset(path: Path) -> MicroDataset:
    """Load dataset from disk."""
    path = Path(path)

    sequences = np.load(path / "sequences.npy")
    occlusion_masks = np.load(path / "occlusion_masks.npy")

    # Create dataset without regenerating
    dataset = MicroDataset.__new__(MicroDataset)
    dataset.sequences = sequences
    dataset.occlusion_masks = occlusion_masks
    dataset.num_sequences = len(sequences)
    dataset.size = sequences.shape[2]
    dataset.frames = sequences.shape[1]
    dataset.seed = 42  # Default

    return dataset


def visualize_sequence(seq: np.ndarray, mask: np.ndarray, save_path: Path | None = None):
    """Visualize a sequence with occlusion frames marked."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    T = seq.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(T * 2, 2))

    for t in range(T):
        axes[t].imshow(seq[t], cmap="gray", vmin=0, vmax=1)
        title = f"t={t}"
        if mask[t]:
            title += " (occ)"
            axes[t].set_title(title, color="red")
        else:
            axes[t].set_title(title)
        axes[t].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate micro pilot dataset")
    parser.add_argument("--generate", type=int, default=500, help="Number of sequences")
    parser.add_argument("--output", type=str, default="pilot_data/micro/", help="Output path")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Generate dataset
    dataset = MicroDataset(
        num_sequences=args.generate,
        seed=args.seed,
    )

    # Save to disk
    save_dataset(dataset, Path(args.output))

    # Print statistics
    total_occluded = dataset.occlusion_masks.sum()
    total_frames = dataset.num_sequences * dataset.frames
    print(f"\nDataset statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Occluded frames: {total_occluded} ({100*total_occluded/total_frames:.1f}%)")

    # Identify recovery frames
    recovery_count = 0
    for i in range(len(dataset)):
        _, mask = dataset[i]
        recovery = get_recovery_frames(mask)
        recovery_count += recovery.sum().item()
    print(f"  Recovery frames: {recovery_count} ({100*recovery_count/total_frames:.1f}%)")

    # Visualize sample
    if args.visualize:
        seq, mask = dataset[0]
        visualize_sequence(seq.squeeze(1).numpy(), mask.numpy())
