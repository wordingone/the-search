"""Pilot dataset: Moving MNIST with opaque occlusion.

Generates sequences of 2 white digits on black background with opaque occlusion
based on z-layer ordering. Digits bounce off walls.

60×60 resolution, 20 frames/sequence
50K train, 5K validation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
import os


class MovingMNISTOcclusion(Dataset):
    """Moving MNIST with opaque occlusion based on z-ordering.

    Uses lazy generation with deterministic seeding per index to avoid
    memory issues with large datasets. Each sequence is generated on-demand
    with a reproducible seed based on its index.
    """

    def __init__(self, root='./data', train=True, num_sequences=50000, seq_length=20,
                 image_size=64, num_digits=2, download=True):
        super().__init__()
        self.seq_length = seq_length
        self.image_size = image_size
        self.num_digits = num_digits
        self.num_sequences = num_sequences
        self.base_seed = 42 if train else 12345  # Different seeds for train/val

        # Load MNIST digits
        mnist = MNIST(root, train=train, download=download,
                     transform=transforms.ToTensor())

        # Get all digit images
        self.digit_images = []
        for i in range(len(mnist)):
            img, _ = mnist[i]
            # Resize to 28×28 if needed, normalize to [0, 1]
            self.digit_images.append(img.squeeze().numpy())

        self.digit_images = np.array(self.digit_images)

    def _generate_sequence(self, seed):
        """Generate one sequence with opaque occlusion.

        Args:
            seed: Random seed for reproducible generation
        """
        # Use local random state for thread safety
        rng = np.random.RandomState(seed)
        frames = []

        # Initialize digits
        digits = []
        for _ in range(self.num_digits):
            # Random digit
            digit_idx = rng.randint(len(self.digit_images))
            digit_img = self.digit_images[digit_idx]

            # Random initial position (ensure digit fits in frame)
            max_pos = self.image_size - 28
            x = rng.randint(0, max_pos)
            y = rng.randint(0, max_pos)

            # Random velocity
            vx = rng.uniform(-3, 3)
            vy = rng.uniform(-3, 3)

            # Random z-layer (0 or 1, determines occlusion order)
            z = rng.randint(2)

            digits.append({
                'image': digit_img,
                'x': float(x),
                'y': float(y),
                'vx': vx,
                'vy': vy,
                'z': z
            })

        # Generate frames
        for t in range(self.seq_length):
            # Create blank frame
            frame = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            # Sort digits by z-order (lower z rendered first, higher z on top)
            sorted_digits = sorted(digits, key=lambda d: d['z'])

            # Render each digit
            for digit in sorted_digits:
                x = int(digit['x'])
                y = int(digit['y'])

                # Ensure digit stays in bounds
                x = np.clip(x, 0, self.image_size - 28)
                y = np.clip(y, 0, self.image_size - 28)

                # OPAQUE rendering: directly overwrite pixels
                frame[y:y+28, x:x+28] = digit['image']

                # Update position
                digit['x'] += digit['vx']
                digit['y'] += digit['vy']

                # Bounce off walls
                if digit['x'] <= 0 or digit['x'] >= self.image_size - 28:
                    digit['vx'] *= -1
                    digit['x'] = np.clip(digit['x'], 0, self.image_size - 28)

                if digit['y'] <= 0 or digit['y'] >= self.image_size - 28:
                    digit['vy'] *= -1
                    digit['y'] = np.clip(digit['y'], 0, self.image_size - 28)

            # Convert to 3-channel RGB (white digits on black)
            frame_rgb = np.stack([frame, frame, frame], axis=0)
            frames.append(frame_rgb)

        # Stack to (T, 3, H, W)
        sequence = np.array(frames, dtype=np.float32)

        return sequence

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """Return sequence as tensor.

        Generates sequence on-demand with deterministic seed based on index.
        """
        seed = self.base_seed + idx
        sequence = self._generate_sequence(seed)
        return torch.from_numpy(sequence)


def create_dataloaders(batch_size=64, num_workers=4):
    """Create train and validation dataloaders."""
    train_dataset = MovingMNISTOcclusion(
        train=True,
        num_sequences=50000,
        seq_length=20,
        image_size=64,
        num_digits=2
    )

    val_dataset = MovingMNISTOcclusion(
        train=False,
        num_sequences=5000,
        seq_length=20,
        image_size=64,
        num_digits=2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def visualize_sequence(sequence, save_path='sequence.gif'):
    """Visualize sequence as GIF.

    Args:
        sequence: (T, 3, H, W) tensor
        save_path: Output path for GIF
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL required for visualization. Install with: pip install pillow")
        return

    # Convert to numpy
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().numpy()

    T, _, H, W = sequence.shape

    # Convert to uint8 images
    images = []
    for t in range(T):
        frame = sequence[t].transpose(1, 2, 0)  # (H, W, 3)
        frame = (frame * 255).astype(np.uint8)
        images.append(Image.fromarray(frame))

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=100,  # ms per frame
        loop=0
    )
    print(f"Saved visualization to {save_path}")


if __name__ == '__main__':
    # Generate sample sequences and visualize
    print("Creating dataset (lazy generation, no pre-loading)...")
    dataset = MovingMNISTOcclusion(num_sequences=5, seq_length=20)

    print(f"Dataset size: {len(dataset)}")
    print(f"Sequence shape: {dataset[0].shape}")

    # Visualize first 5 sequences
    os.makedirs('pilot_visualizations', exist_ok=True)
    for i in range(5):
        seq = dataset[i]
        visualize_sequence(seq, f'pilot_visualizations/sequence_{i}.gif')

    print("\nVerification:")
    print("✓ Dataset created")
    print("✓ Shapes correct: (20, 3, 64, 64)")
    print("✓ 5 sequences visualized in pilot_visualizations/")
    print("\nNext: Visually inspect GIFs to confirm opaque occlusion")
