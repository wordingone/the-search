"""
Quantization Hypothesis Testing Framework

Tests the hypothesis:
"Fully quantized training for Genesis is plausible only in structured,
sparse, or discrete latent spaces with bounded activations and local propagation"

Sub-problems:
1. Structured: 3D voxel grid with spatial locality
2. Sparse: <5% voxel occupancy
3. Discrete: FSQ-style quantized latents
4. Bounded: Activations in known range
5. Local: Windowed attention (W=4)
"""

from .config import ExperimentConfig
from .metrics import (
    gradient_snr,
    quantization_error,
    activation_range,
    error_propagation_rate,
)
from .utils import (
    fake_quant,
    fake_quant_tensor,
    QuantizedSTE,
    measure_memory_usage,
)

__all__ = [
    "ExperimentConfig",
    "gradient_snr",
    "quantization_error",
    "activation_range",
    "error_propagation_rate",
    "fake_quant",
    "fake_quant_tensor",
    "QuantizedSTE",
    "measure_memory_usage",
]
