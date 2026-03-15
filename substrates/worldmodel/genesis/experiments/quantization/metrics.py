"""
Quantization error metrics for hypothesis testing.
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import numpy as np


def gradient_snr(grad_fp32: Tensor, grad_q8: Tensor) -> float:
    """
    Signal-to-noise ratio of quantized gradients.

    Higher SNR = better gradient preservation.
    SNR > 20 dB is generally good for training.

    Args:
        grad_fp32: Reference FP32 gradients
        grad_q8: Quantized gradients

    Returns:
        SNR in decibels
    """
    signal = grad_fp32.pow(2).mean()
    noise = (grad_fp32 - grad_q8).pow(2).mean()
    snr = 10 * torch.log10(signal / (noise + 1e-10))
    return snr.item()


def quantization_error(
    x_fp32: Tensor,
    x_quant: Tensor,
    metric: str = "relative"
) -> float:
    """
    Measure quantization error between FP32 and quantized tensors.

    Args:
        x_fp32: Original FP32 tensor
        x_quant: Quantized tensor
        metric: "relative" (NRMSE), "absolute" (RMSE), or "max"

    Returns:
        Error value
    """
    diff = x_fp32 - x_quant

    if metric == "absolute":
        return diff.pow(2).mean().sqrt().item()
    elif metric == "max":
        return diff.abs().max().item()
    elif metric == "relative":
        # Normalized RMSE (relative to signal magnitude)
        rmse = diff.pow(2).mean().sqrt()
        signal_mag = x_fp32.pow(2).mean().sqrt()
        return (rmse / (signal_mag + 1e-10)).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def activation_range(x: Tensor) -> Tuple[float, float]:
    """
    Measure activation range for boundedness analysis.

    Args:
        x: Activation tensor

    Returns:
        (min, max) tuple
    """
    return x.min().item(), x.max().item()


def activation_stats(x: Tensor) -> Dict[str, float]:
    """
    Comprehensive activation statistics.

    Args:
        x: Activation tensor

    Returns:
        Dictionary with min, max, mean, std, 1%ile, 99%ile
    """
    x_flat = x.flatten().float()
    return {
        "min": x_flat.min().item(),
        "max": x_flat.max().item(),
        "mean": x_flat.mean().item(),
        "std": x_flat.std().item(),
        "percentile_1": torch.quantile(x_flat, 0.01).item(),
        "percentile_99": torch.quantile(x_flat, 0.99).item(),
        "abs_max": x_flat.abs().max().item(),
    }


def error_propagation_rate(
    errors_by_layer: List[float]
) -> Tuple[float, str]:
    """
    Analyze error propagation across layers.

    Args:
        errors_by_layer: List of error magnitudes at each layer

    Returns:
        (growth_rate, pattern) where pattern is "linear", "exponential", or "stable"
    """
    if len(errors_by_layer) < 2:
        return 1.0, "stable"

    errors = np.array(errors_by_layer)
    layers = np.arange(len(errors))

    # Fit exponential: error = a * exp(b * layer)
    # Take log: log(error) = log(a) + b * layer
    log_errors = np.log(errors + 1e-10)
    exp_fit = np.polyfit(layers, log_errors, 1)
    exp_rate = np.exp(exp_fit[0])  # Growth rate per layer

    # Fit linear: error = a + b * layer
    lin_fit = np.polyfit(layers, errors, 1)
    lin_rate = 1 + lin_fit[0] / (errors[0] + 1e-10)

    # Determine pattern based on which fits better
    exp_pred = np.exp(exp_fit[0] * layers + exp_fit[1])
    lin_pred = lin_fit[0] * layers + lin_fit[1]

    exp_mse = np.mean((errors - exp_pred) ** 2)
    lin_mse = np.mean((errors - lin_pred) ** 2)

    if exp_rate < 1.1:
        return exp_rate, "stable"
    elif exp_mse < lin_mse and exp_rate > 1.5:
        return exp_rate, "exponential"
    else:
        return lin_rate, "linear"


def scatter_add_error(
    values: Tensor,
    indices: Tensor,
    output_size: int,
    dtype_test: torch.dtype = torch.float16,
) -> float:
    """
    Measure accumulation error in scatter_add operation.

    This tests whether sparsity reduces quantization error in
    the core scatter_add operation used for field updates.

    Args:
        values: Values to accumulate (N,) or (N, C)
        indices: Target indices (N,)
        output_size: Size of output tensor
        dtype_test: Dtype to test (float16, int8, etc.)

    Returns:
        Relative error vs FP32 reference
    """
    device = values.device

    # FP32 reference
    if values.dim() == 1:
        ref = torch.zeros(output_size, dtype=torch.float32, device=device)
        ref.scatter_add_(0, indices, values.float())

        # Test dtype
        test = torch.zeros(output_size, dtype=dtype_test, device=device)
        values_test = values.to(dtype_test)
        test.scatter_add_(0, indices, values_test)
        test = test.float()
    else:
        C = values.shape[1]
        ref = torch.zeros(output_size, C, dtype=torch.float32, device=device)
        idx_expanded = indices.unsqueeze(-1).expand(-1, C)
        ref.scatter_add_(0, idx_expanded, values.float())

        test = torch.zeros(output_size, C, dtype=dtype_test, device=device)
        values_test = values.to(dtype_test)
        test.scatter_add_(0, idx_expanded, values_test)
        test = test.float()

    return quantization_error(ref, test, metric="relative")


def codebook_utilization(indices: Tensor, num_codes: int) -> float:
    """
    Measure codebook utilization for FSQ/VQ.

    Args:
        indices: Token indices from quantization
        num_codes: Total number of codes in codebook

    Returns:
        Fraction of codes used (0 to 1)
    """
    unique_codes = torch.unique(indices.flatten())
    return len(unique_codes) / num_codes


def loss_ratio(loss_test: float, loss_baseline: float) -> float:
    """
    Compute loss ratio for pass/fail determination.

    Args:
        loss_test: Loss from test configuration
        loss_baseline: Loss from baseline (FP32)

    Returns:
        Ratio (test / baseline)
    """
    return loss_test / (loss_baseline + 1e-10)


class MetricsTracker:
    """
    Track metrics across training steps for analysis.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def log(self, name: str, value: float):
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])

    def mean(self, name: str) -> float:
        """Get mean of a metric."""
        values = self.get(name)
        return np.mean(values) if values else 0.0

    def std(self, name: str) -> float:
        """Get std of a metric."""
        values = self.get(name)
        return np.std(values) if values else 0.0

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        return {
            name: {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
            for name, values in self.metrics.items()
            if values
        }
