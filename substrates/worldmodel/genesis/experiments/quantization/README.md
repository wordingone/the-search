# Quantization Hypothesis Testing Framework

## The Hypothesis

> "Fully quantized training for Genesis is plausible **only in structured, sparse, or discrete latent spaces with bounded activations and local propagation**"

## Decomposition into Testable Sub-Problems

| Condition | Definition | Experiment | Pass Criterion |
|-----------|------------|------------|----------------|
| **Structured** | 3D voxel grid with spatial locality | `exp1_structured/` | Windowed Q8 < 0.5x Full Q8 loss |
| **Sparse** | <5% voxel occupancy | `exp2_sparse/` | Q8 error < 1% at 5% sparsity |
| **Discrete** | FSQ-style quantized latents | `exp3_discrete/` | Codebook utilization > 90% |
| **Bounded** | Activations in known range | `exp4_bounded/` | Bounded Q8 < 2x FP32 loss |
| **Local** | Windowed attention (W=4) | `exp5_local/` | Error growth < 1.5x per layer |

## Quick Start

```bash
# Run all experiments
cd B:\M\ArtificialArchitecture\worldmodel
python -m pytest genesis/experiments/quantization/ -v

# Run specific experiment
python genesis/experiments/quantization/exp1_structured/test_windowed_vs_full.py

# Run integration test (after sub-problems pass)
python genesis/experiments/quantization/final_integration/test_full_quantized.py
```

## Directory Structure

```
genesis/experiments/quantization/
├── __init__.py              # Package exports
├── config.py                # Shared configuration
├── metrics.py               # Quantization error metrics
├── utils.py                 # Fake quantization utilities
├── README.md                # This file
│
├── exp1_structured/         # Sub-problem 1: Windowed vs full attention
│   ├── test_windowed_vs_full.py
│   └── results/
│
├── exp2_sparse/             # Sub-problem 2: Sparsity levels
│   ├── test_sparsity_levels.py
│   └── results/
│
├── exp3_discrete/           # Sub-problem 3: FSQ gradient flow
│   ├── test_fsq_gradient_flow.py
│   └── results/
│
├── exp4_bounded/            # Sub-problem 4: Bounded activations
│   ├── test_activation_bounds.py
│   └── results/
│
├── exp5_local/              # Sub-problem 5: Error propagation
│   ├── test_error_propagation.py
│   └── results/
│
└── final_integration/       # Combined test
    ├── test_full_quantized.py
    └── results/
```

## Key Metrics

### Gradient SNR (Signal-to-Noise Ratio)
```python
def gradient_snr(grad_fp32, grad_q8):
    signal = grad_fp32.pow(2).mean()
    noise = (grad_fp32 - grad_q8).pow(2).mean()
    return 10 * torch.log10(signal / (noise + 1e-10))
```
- SNR > 20 dB: Good gradient preservation
- SNR < 10 dB: Significant noise, may affect training

### Error Propagation Rate
- **Linear**: Error grows as `1.0 → 1.2 → 1.4 → 1.6` (good)
- **Exponential**: Error grows as `1.0 → 2.0 → 4.0 → 8.0` (bad)

### Codebook Utilization
- > 90%: Healthy discrete space
- < 50%: Codebook collapse, quantization too aggressive

## Expected Outcomes

### If Hypothesis Confirmed
- All 5 conditions are necessary for Q8 training
- Removing any condition causes divergence
- Genesis architecture is quantization-friendly by design
- Path to INT8 inference is clear

### If Hypothesis Partially Confirmed
- Some conditions are necessary, others helpful
- Document which are necessary vs sufficient
- Refine hypothesis for future work

### If Hypothesis Rejected
- Q8 training fails even with all conditions
- Identify which operations are irreducible precision requirements
- Document minimum precision requirements per component

## Reference Files (Read-Only)

| Pattern | Source |
|---------|--------|
| STE implementation | `genesis/tokenizer/fsq.py:55-85` |
| Windowed attention | `genesis/pilot/windowed_attn.py:168-220` |
| Scatter_add accumulation | `genesis/pilot/field_model.py:554-614` |
| Softmax stability | `genesis/dynamics/attention.py:177-184` |

## Quarantine Notice

This experiment framework is **completely isolated** from main Genesis code:
- No imports from main Genesis modules (self-contained implementations)
- No modifications to existing files
- Results stored in `results/` subdirectories
- Can be deleted without affecting Genesis

## The Breath

> "If explicit 3D state is claimed efficient, the computation over that state should exploit its structure."

The original insight applied to efficiency. This experiment applies the same principle to precision:

> "If structured, sparse, discrete, bounded, local computation is claimed quantization-friendly, each condition must be tested independently."

We don't claim. We measure.
