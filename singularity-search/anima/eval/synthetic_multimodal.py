"""
Synthetic Multi-Modal Benchmark for ANIMA
==========================================

Tests ANIMA across all 7 modalities using synthetic data:
1. Text/Symbolic - Sequence prediction, pattern completion
2. Vision/Spatial - Patch classification, spatial reasoning
3. Video/Temporal - Temporal pattern recognition
4. Audio/Acoustic - Frequency pattern detection
5. 3D/Spatial - Point cloud classification
6. Sensor/Medical - Time series anomaly detection
7. Scientific - Molecular property prediction

No data download required - all synthetic.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

class SyntheticTextData:
    """Text/Symbolic: Token sequence patterns."""

    def __init__(self, vocab_size: int = 100, seq_len: int = 64, num_classes: int = 10):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_classes = num_classes

    def generate_batch(self, batch_size: int, task: str = "next_token") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate text-like sequence data."""
        if task == "next_token":
            # Pattern: repeating sequences - predict which pattern class
            # Each class has a distinct repeating pattern
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.zeros(batch_size, self.seq_len, 1)

            for i in range(batch_size):
                c = labels[i].item()
                # Different patterns per class
                pattern_len = 4 + c % 5  # Pattern length varies by class
                base = torch.arange(pattern_len).float() / pattern_len + c * 0.1
                pattern = base.repeat(self.seq_len // pattern_len + 1)[:self.seq_len]
                x[i, :, 0] = pattern + torch.randn(self.seq_len) * 0.05

            return x, labels

        elif task == "pattern_complete":
            # A B A B A ? -> class based on A,B values
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.zeros(batch_size, self.seq_len, 1)

            for i in range(batch_size):
                c = labels[i].item()
                a_val = c * 0.1
                b_val = c * 0.1 + 0.05
                pattern = torch.tensor([a_val, b_val]).repeat(self.seq_len // 2 + 1)[:self.seq_len]
                x[i, :, 0] = pattern + torch.randn(self.seq_len) * 0.02

            return x, labels


class SyntheticVisionData:
    """Vision/Spatial: Patch-based image classification."""

    def __init__(self, num_patches: int = 16, patch_dim: int = 8, num_classes: int = 10):
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.num_classes = num_classes

    def generate_batch(self, batch_size: int, task: str = "classify") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate vision-like patch data."""
        if task == "classify":
            # Each class has a distinct spatial pattern
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.randn(batch_size, self.num_patches, self.patch_dim) * 0.1

            # Add class-specific patterns
            for i in range(batch_size):
                c = labels[i].item()
                # Pattern based on class: different patches activated
                active_patches = [(c * 2 + j) % self.num_patches for j in range(3)]
                for p in active_patches:
                    x[i, p, :] += torch.randn(self.patch_dim) * 0.5 + c * 0.1

            return x, labels

        elif task == "spatial_relation":
            # Detect if pattern is in top-half or bottom-half
            labels = torch.randint(0, 2, (batch_size,))
            x = torch.randn(batch_size, self.num_patches, self.patch_dim) * 0.1

            half = self.num_patches // 2
            for i in range(batch_size):
                if labels[i] == 0:
                    x[i, :half, :] += 0.5  # Top half bright
                else:
                    x[i, half:, :] += 0.5  # Bottom half bright

            return x, labels


class SyntheticVideoData:
    """Video/Temporal: Temporal pattern recognition."""

    def __init__(self, num_frames: int = 32, frame_dim: int = 8, num_classes: int = 5):
        self.num_frames = num_frames
        self.frame_dim = frame_dim
        self.num_classes = num_classes

    def generate_batch(self, batch_size: int, task: str = "action") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate video-like temporal data."""
        if task == "action":
            # Different temporal dynamics per class
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.zeros(batch_size, self.num_frames, self.frame_dim)

            t = torch.linspace(0, 4 * np.pi, self.num_frames)
            for i in range(batch_size):
                c = labels[i].item()
                freq = 0.5 + c * 0.3  # Different frequency per class
                phase = c * np.pi / self.num_classes

                # Temporal pattern
                pattern = torch.sin(freq * t + phase)
                x[i, :, 0] = pattern
                x[i, :, 1] = torch.cos(freq * t + phase)
                # Add noise
                x[i] += torch.randn_like(x[i]) * 0.1

            return x, labels

        elif task == "direction":
            # Detect motion direction (increasing vs decreasing)
            labels = torch.randint(0, 2, (batch_size,))
            x = torch.zeros(batch_size, self.num_frames, self.frame_dim)

            for i in range(batch_size):
                if labels[i] == 0:
                    x[i, :, 0] = torch.linspace(0, 1, self.num_frames)
                else:
                    x[i, :, 0] = torch.linspace(1, 0, self.num_frames)
                x[i] += torch.randn_like(x[i]) * 0.1

            return x, labels


class SyntheticAudioData:
    """Audio/Acoustic: Frequency pattern detection."""

    def __init__(self, num_frames: int = 100, freq_bins: int = 16, num_classes: int = 5):
        self.num_frames = num_frames
        self.freq_bins = freq_bins
        self.num_classes = num_classes

    def generate_batch(self, batch_size: int, task: str = "phoneme") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate audio-like spectrogram data."""
        if task == "phoneme":
            # Different frequency profiles per class
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.randn(batch_size, self.num_frames, self.freq_bins) * 0.1

            for i in range(batch_size):
                c = labels[i].item()
                # Activate different frequency bands per class
                low = (c * 3) % self.freq_bins
                high = min(low + 4, self.freq_bins)
                x[i, :, low:high] += 0.5 + torch.randn(self.num_frames, high - low) * 0.2

            return x, labels

        elif task == "speaker":
            # Speaker identification via pitch contour
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.randn(batch_size, self.num_frames, self.freq_bins) * 0.1

            for i in range(batch_size):
                c = labels[i].item()
                base_freq = 2 + c  # Different base frequency bin
                x[i, :, base_freq] += 0.8
                x[i, :, base_freq + 1] += 0.4

            return x, labels


class SyntheticPointCloudData:
    """3D/Spatial: Point cloud classification."""

    def __init__(self, num_points: int = 128, point_dim: int = 8, num_classes: int = 5):
        self.num_points = num_points
        self.point_dim = point_dim  # Output dim (will pad 3D coords)
        self.num_classes = num_classes

    def generate_batch(self, batch_size: int, task: str = "shape") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 3D point cloud data."""
        if task == "shape":
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.zeros(batch_size, self.num_points, self.point_dim)

            for i in range(batch_size):
                c = labels[i].item()
                if c == 0:  # Sphere
                    theta = torch.rand(self.num_points) * 2 * np.pi
                    phi = torch.rand(self.num_points) * np.pi
                    x[i, :, 0] = torch.sin(phi) * torch.cos(theta)
                    x[i, :, 1] = torch.sin(phi) * torch.sin(theta)
                    x[i, :, 2] = torch.cos(phi)
                elif c == 1:  # Cube
                    x[i, :, :3] = torch.rand(self.num_points, 3) * 2 - 1
                elif c == 2:  # Cylinder
                    theta = torch.rand(self.num_points) * 2 * np.pi
                    x[i, :, 0] = torch.cos(theta)
                    x[i, :, 1] = torch.sin(theta)
                    x[i, :, 2] = torch.rand(self.num_points) * 2 - 1
                elif c == 3:  # Cone
                    h = torch.rand(self.num_points)
                    theta = torch.rand(self.num_points) * 2 * np.pi
                    x[i, :, 0] = (1 - h) * torch.cos(theta)
                    x[i, :, 1] = (1 - h) * torch.sin(theta)
                    x[i, :, 2] = h
                else:  # Torus
                    theta = torch.rand(self.num_points) * 2 * np.pi
                    phi = torch.rand(self.num_points) * 2 * np.pi
                    R, r = 0.7, 0.3
                    x[i, :, 0] = (R + r * torch.cos(phi)) * torch.cos(theta)
                    x[i, :, 1] = (R + r * torch.cos(phi)) * torch.sin(theta)
                    x[i, :, 2] = r * torch.sin(phi)

                # Add distance from origin as feature
                x[i, :, 3] = torch.sqrt((x[i, :, :3] ** 2).sum(dim=-1))

                # Add noise to remaining dims
                if self.point_dim > 4:
                    x[i, :, 4:] = torch.randn(self.num_points, self.point_dim - 4) * 0.1

            return x, labels


class SyntheticSensorData:
    """Sensor/Medical: Time series anomaly detection."""

    def __init__(self, seq_len: int = 200, num_channels: int = 4, num_classes: int = 4):
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.num_classes = num_classes

    def generate_batch(self, batch_size: int, task: str = "anomaly") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sensor-like time series data."""
        if task == "anomaly":
            # Normal vs different anomaly types
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.zeros(batch_size, self.seq_len, self.num_channels)

            t = torch.linspace(0, 10 * np.pi, self.seq_len)
            for i in range(batch_size):
                c = labels[i].item()

                # Base signal (normal heartbeat-like)
                x[i, :, 0] = torch.sin(t) + torch.sin(2 * t) * 0.5
                x[i, :, 1] = torch.cos(t) * 0.8

                if c == 1:  # Tachycardia (fast)
                    x[i, :, 0] = torch.sin(2 * t) + torch.sin(4 * t) * 0.5
                elif c == 2:  # Bradycardia (slow)
                    x[i, :, 0] = torch.sin(0.5 * t) + torch.sin(t) * 0.5
                elif c == 3:  # Arrhythmia (irregular)
                    irregular = torch.sin(t) * (1 + 0.5 * torch.sin(0.3 * t))
                    x[i, :, 0] = irregular

                # Add noise
                x[i] += torch.randn_like(x[i]) * 0.1

            return x, labels


class SyntheticMolecularData:
    """Scientific: Molecular property prediction."""

    def __init__(self, max_atoms: int = 32, atom_dim: int = 8, num_classes: int = 5):
        self.max_atoms = max_atoms
        self.atom_dim = atom_dim
        self.num_classes = num_classes

    def generate_batch(self, batch_size: int, task: str = "property") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate molecular-like data."""
        if task == "property":
            # Different molecular structures -> different properties
            labels = torch.randint(0, self.num_classes, (batch_size,))
            x = torch.zeros(batch_size, self.max_atoms, self.atom_dim)

            for i in range(batch_size):
                c = labels[i].item()
                num_atoms = 10 + c * 5  # Different sizes

                # Atom types (one-hot style)
                atom_types = torch.randint(0, 4, (num_atoms,))
                x[i, :num_atoms, 0] = atom_types.float() / 4

                # 3D positions (chain-like for simplicity)
                for j in range(num_atoms):
                    x[i, j, 1] = j * 0.1 + torch.randn(1).item() * 0.02
                    x[i, j, 2] = np.sin(j * 0.5 + c) * 0.2
                    x[i, j, 3] = np.cos(j * 0.5 + c) * 0.2

                # Add some structural features
                x[i, :num_atoms, 4] = torch.linspace(0, 1, num_atoms)  # Position encoding

            return x, labels


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

@dataclass
class ModalityResult:
    """Results for a single modality."""
    name: str
    task: str
    accuracy: float
    loss: float
    samples_per_sec: float
    num_samples: int


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    model_name: str
    model_params: int
    modalities: List[ModalityResult]
    overall_accuracy: float
    total_time: float


def run_synthetic_benchmark(
    model: nn.Module,
    sensory_dim: int = 8,
    output_dim: int = 10,
    samples_per_modality: int = 1000,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> BenchmarkResults:
    """
    Run comprehensive synthetic benchmark across all modalities.

    Args:
        model: ANIMA model to evaluate
        sensory_dim: Input dimension (must match model)
        output_dim: Output dimension (must match model)
        samples_per_modality: Samples to test per modality
        batch_size: Batch size for evaluation
        device: Torch device

    Returns:
        BenchmarkResults with per-modality and aggregate scores
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    model_name = model.__class__.__name__

    # Define modalities and their generators
    modalities = [
        ("Text/Symbolic", "next_token", SyntheticTextData(vocab_size=100, seq_len=64, num_classes=output_dim)),
        ("Vision/Spatial", "classify", SyntheticVisionData(num_patches=64, patch_dim=sensory_dim, num_classes=output_dim)),
        ("Video/Temporal", "action", SyntheticVideoData(num_frames=64, frame_dim=sensory_dim, num_classes=min(5, output_dim))),
        ("Audio/Acoustic", "phoneme", SyntheticAudioData(num_frames=64, freq_bins=sensory_dim, num_classes=min(5, output_dim))),
        ("3D/Spatial", "shape", SyntheticPointCloudData(num_points=64, point_dim=sensory_dim, num_classes=min(5, output_dim))),
        ("Sensor/Medical", "anomaly", SyntheticSensorData(seq_len=64, num_channels=sensory_dim, num_classes=min(4, output_dim))),
        ("Scientific", "property", SyntheticMolecularData(max_atoms=64, atom_dim=sensory_dim, num_classes=min(5, output_dim))),
    ]

    results = []
    total_start = time.time()

    criterion = nn.CrossEntropyLoss()

    for name, task, generator in modalities:
        correct = 0
        total_loss = 0.0
        total_samples = 0

        start_time = time.time()

        with torch.no_grad():
            num_batches = samples_per_modality // batch_size
            for _ in range(num_batches):
                x, y = generator.generate_batch(batch_size, task)

                # Ensure correct dimensions
                if x.dim() == 2:
                    x = x.unsqueeze(-1).expand(-1, -1, sensory_dim)
                elif x.shape[-1] != sensory_dim:
                    # Project to sensory_dim
                    x = x[..., :sensory_dim] if x.shape[-1] > sensory_dim else \
                        torch.cat([x, torch.zeros(*x.shape[:-1], sensory_dim - x.shape[-1])], dim=-1)

                x = x.to(device)
                y = y.to(device)

                # Forward pass
                out = model(x)

                # Pool over sequence for classification
                out_pooled = out.mean(dim=1)  # [batch, output_dim]

                # Ensure output matches num_classes
                num_classes = y.max().item() + 1
                if out_pooled.shape[-1] < num_classes:
                    out_pooled = torch.cat([out_pooled, torch.zeros(out_pooled.shape[0], num_classes - out_pooled.shape[-1], device=device)], dim=-1)
                out_pooled = out_pooled[:, :num_classes]

                # Compute loss and accuracy
                loss = criterion(out_pooled, y)
                preds = out_pooled.argmax(dim=-1)

                correct += (preds == y).sum().item()
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        elapsed = time.time() - start_time
        accuracy = correct / total_samples
        avg_loss = total_loss / total_samples
        samples_per_sec = total_samples / elapsed

        results.append(ModalityResult(
            name=name,
            task=task,
            accuracy=accuracy,
            loss=avg_loss,
            samples_per_sec=samples_per_sec,
            num_samples=total_samples,
        ))

    total_time = time.time() - total_start
    overall_accuracy = np.mean([r.accuracy for r in results])

    return BenchmarkResults(
        model_name=model_name,
        model_params=num_params,
        modalities=results,
        overall_accuracy=overall_accuracy,
        total_time=total_time,
    )


def print_benchmark_results(results: BenchmarkResults):
    """Print formatted benchmark results."""
    print()
    print("=" * 70)
    print(f"SYNTHETIC MULTI-MODAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {results.model_name}")
    print(f"Parameters: {results.model_params:,}")
    print(f"Total Time: {results.total_time:.1f}s")
    print("-" * 70)
    print(f"{'Modality':<20} {'Task':<12} {'Accuracy':>10} {'Loss':>10} {'Samp/s':>10}")
    print("-" * 70)

    for r in results.modalities:
        print(f"{r.name:<20} {r.task:<12} {r.accuracy:>10.2%} {r.loss:>10.4f} {r.samples_per_sec:>10.0f}")

    print("-" * 70)
    print(f"{'OVERALL':<20} {'':<12} {results.overall_accuracy:>10.2%}")
    print("=" * 70)
    print()


def compare_models(
    models: Dict[str, nn.Module],
    sensory_dim: int = 8,
    output_dim: int = 10,
    samples_per_modality: int = 500,
) -> Dict[str, BenchmarkResults]:
    """Compare multiple models on synthetic benchmark."""
    all_results = {}

    for name, model in models.items():
        print(f"Benchmarking {name}...")
        results = run_synthetic_benchmark(
            model,
            sensory_dim=sensory_dim,
            output_dim=output_dim,
            samples_per_modality=samples_per_modality,
        )
        all_results[name] = results

    # Print comparison table
    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    modality_names = [r.name for r in list(all_results.values())[0].modalities]
    print(f"{'Model':<20} {'Params':>10}", end="")
    for m in modality_names:
        print(f" {m[:8]:>8}", end="")
    print(f" {'OVERALL':>10}")
    print("-" * 80)

    # Results per model
    for name, results in all_results.items():
        print(f"{name:<20} {results.model_params:>10,}", end="")
        for r in results.modalities:
            print(f" {r.accuracy:>8.1%}", end="")
        print(f" {results.overall_accuracy:>10.1%}")

    print("=" * 80)

    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from anima.core import Anima

    # Fixed input/output dims for fair comparison
    SENSORY_DIM = 8
    OUTPUT_DIM = 10

    # Test at different parameter counts (vary only d_model and bottleneck)
    configs = [
        ("Tiny", 16, 8),       # ~2K params
        ("Small", 32, 16),     # ~5K params
        ("Medium", 64, 32),    # ~17K params
        ("Large", 128, 64),    # ~66K params
        ("XL", 256, 128),      # ~260K params
    ]

    models = {}
    for name, d_model, bottleneck in configs:
        model = Anima(
            sensory_dim=SENSORY_DIM,
            d_model=d_model,
            bottleneck_dim=bottleneck,
            output_dim=OUTPUT_DIM,
        )
        params = sum(p.numel() for p in model.parameters())
        models[f"{name} ({params//1000}K)"] = model

    # Run comparison
    results = compare_models(
        models,
        sensory_dim=SENSORY_DIM,
        output_dim=OUTPUT_DIM,
        samples_per_modality=500,
    )
