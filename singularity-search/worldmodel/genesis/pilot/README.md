# Phase 0: Validation Pilot

## Hypothesis

Maintaining explicit 3D state updated via temporal stereo is more data-efficient than inferring state from pixel streams for learning object permanence under occlusion.

## Models

### Field Model (`field_model.py`)
- **Temporal stereo encoder:** Computes cost volume from frame pairs to estimate depth
- **3D field:** 16³ × 32ch persistent spatial state
- **Propagator:** 6-layer transformer with full attention (4096 tokens)
- **Persistence gate:** Learned per-channel blending of propagated vs. persistent state
- **Decoder:** Renders field slice to frame
- **Parameters:** ~22M

### Baseline Model (`baseline_model.py`)
- **Frame encoder:** Concatenated frame pair → features
- **Transformer:** 6-layer temporal reasoning (same architecture as field propagator)
- **Decoder:** Features → predicted frame
- **NO field, NO temporal stereo**
- **Parameters:** ~22M (matched within 0.9%)

## Dataset (`data.py`)

Moving MNIST with opaque occlusion:
- 2 white digits on black background
- 64×64 resolution, 20 frames/sequence
- Z-layer assignment determines occlusion order
- Front digit fully occludes back digit (opaque)
- 50K train, 5K validation

## Training (`train.py`)

```bash
# Train field model
python train.py --model field --epochs 50

# Train baseline model
python train.py --model baseline --epochs 50

# Compare results
python train.py --compare
```

### Metrics

1. **MSE** - Mean squared error (standard)
2. **SSIM** - Structural similarity (standard)
3. **Occlusion Recovery Score** - MSE on frames where occluded objects reappear (PRIMARY METRIC)

### Decision Gate

**PASS (proceed to Phase 1):**
- Field model >10% better on occlusion recovery
- SSIM comparable (within 5%)

**FAIL (hypothesis invalid):**
- Baseline wins or improvement <10%
- DO NOT proceed to Phase 1

## Quick Start

```bash
# 1. Verify implementation
python ../../tests/pilot/test_pilot.py

# 2. Generate visualizations
python data.py

# 3. Run full workflow (Windows)
..\..\run_phase0.bat

# 4. Run full workflow (Linux/Mac)
bash ../../run_phase0.sh
```

## Implementation Details

### Temporal Stereo Encoding

```python
# Frame pair → cost volume
cost_volume = correlate_features_across_disparities(feat_prev, feat_curr)

# Depth estimation
prob = softmax(refine(cost_volume))
depth = soft_argmin(prob)
confidence = 1 - normalized_entropy(prob)

# Splat features into 3D field at estimated depth
field[x, y, depth] += confidence * features
```

### Field Propagation

```python
# Transformer propagates field state
propagated = transformer(field)

# Learnable persistence gate
gate = sigmoid(persistence_params)
field_next = gate * propagated + (1 - gate) * field_prev
```

### Key Differences from Baseline

| Aspect | Field Model | Baseline |
|--------|------------|----------|
| **State** | Explicit 3D field persists | No persistent state |
| **Depth** | Estimated via temporal stereo | Not computed |
| **Occlusion** | Field tracks occluded objects | Purely correlation-based |
| **Prediction** | Render from field state | Direct frame prediction |

## Expected Results

If hypothesis is correct:
- Field model should have **lower occlusion recovery MSE** (better at predicting reappearing objects)
- SSIM should be comparable or better
- Training curves should show faster convergence

If hypothesis is wrong:
- Baseline wins or no significant difference
- Field overhead doesn't justify complexity
- Back to drawing board

## Files

```
genesis/pilot/
├── README.md           # This file
├── field_model.py      # Temporal stereo + field model
├── baseline_model.py   # Frame-to-frame baseline
├── data.py            # Moving MNIST with occlusion
└── train.py           # Training + comparison

tests/pilot/
└── test_pilot.py      # Verification tests

pilot_logs/            # Created during training
├── field_training.csv
├── baseline_training.csv
└── comparison.png

pilot_checkpoints/     # Created during training
├── field_best.pth
├── baseline_best.pth
├── field_epoch_*.pth
└── baseline_epoch_*.pth
```

## Hardware Requirements

- **GPU:** RTX 4090 or equivalent (12GB+ VRAM sufficient with windowed attention)
- **Training time:** ~1-2 days per model (50 epochs, batch_size=12)
- **Total time:** ~2-4 days for full comparison

**Note:** Original estimates of "3-5 hours" were incorrect due to full O(N²) attention.
The current implementation uses windowed 3D attention (O(N × W³)) which enables
batch_size=12 with 9.9 GB peak memory, achieving 53.8× speedup over the original.

## Next Steps

After decision gate passes:
→ **Phase 1:** Unified Genesis implementation with real-world data

After decision gate fails:
→ Analyze failure modes, revise hypothesis, iterate
