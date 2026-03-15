"""
Shared configuration for quantization experiments.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class ExperimentConfig:
    """Configuration for quantization experiments."""

    # Quantization settings
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False

    # Model dimensions (kept small for fast iteration)
    dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4

    # 3D field settings
    field_size: int = 16  # 16^3 = 4096 voxels
    window_size: int = 4  # 4^3 = 64 tokens per window

    # Sparsity settings
    default_sparsity: float = 0.05  # 5% occupancy

    # Training settings
    batch_size: int = 4
    num_steps: int = 100
    learning_rate: float = 1e-4

    # FSQ settings (matches genesis/tokenizer/fsq.py)
    fsq_levels: List[int] = field(default_factory=lambda: [8, 6, 5, 5, 5])

    # Device
    device: str = "cuda"

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"


@dataclass
class MetricThresholds:
    """Pass/fail thresholds for experiments."""

    # Sub-problem 1: Structured
    structured_loss_ratio: float = 0.5  # Windowed Q8 loss < 0.5x Full Q8 loss

    # Sub-problem 2: Sparse
    sparse_error_threshold: float = 0.01  # <1% error at 5% sparsity

    # Sub-problem 3: Discrete (FSQ)
    fsq_codebook_utilization: float = 0.90  # >90% codebook used

    # Sub-problem 4: Bounded
    bounded_loss_ratio: float = 2.0  # Q8/FP32 loss ratio < 2x

    # Sub-problem 5: Local
    error_growth_rate: float = 1.5  # <1.5x per layer

    # Integration
    full_q8_loss_ratio: float = 3.0  # Q8 loss < 3x FP32 baseline


# Default configs
DEFAULT_CONFIG = ExperimentConfig()
DEFAULT_THRESHOLDS = MetricThresholds()
