# Genesis Implementation Summary

## Status: Phase 0 Complete ✓

**Date:** 2026-02-03
**Phase:** Validation Pilot
**Next:** Train models → Decision gate → Phase 1 (if pass)

---

## What Was Implemented

### Phase 0: Validation Pilot

Complete pilot implementation to test core hypothesis before scaling.

**Hypothesis:**
> Maintaining explicit 3D state via temporal stereo is more data-efficient than pixel-stream inference for learning object permanence.

**Test:**
- Field model (temporal stereo + 3D field) vs. Baseline (frame-to-frame)
- Matched parameters (~22M each)
- Evaluated on Moving MNIST with opaque occlusion
- Primary metric: Occlusion recovery score

### Files Created

```
genesis/pilot/
├── __init__.py
├── field_model.py        # 21.7M params - Temporal stereo + field (WindowedFieldPropagator)
├── baseline_model.py     # 21.9M params - Frame-to-frame control
├── data.py              # Moving MNIST with occlusion
├── train.py             # Training + automated comparison + certainty-weighted loss
├── windowed_attn.py     # O(N×W³) windowed 3D attention (89× memory reduction)
└── README.md            # Pilot documentation

tests/pilot/
└── test_pilot.py        # Verification tests (ALL PASS)

Documentation:
├── PHASE0_STATUS.md      # Detailed status + next steps
├── CHANGELOG.md         # Updated with Phase 0 evolution entry
├── IMPLEMENTATION_SUMMARY.md  # This file
├── CLAUDE.md            # Updated with Architectural Constraints
├── run_phase0.bat       # Windows workflow script
└── run_phase0.sh        # Linux/Mac workflow script
```

### Verification Status

**All gates passed:**

| Check | Status |
|-------|--------|
| Field model params in [20M, 30M] | ✓ PASS (21.7M) |
| Baseline params in [20M, 30M] | ✓ PASS (21.9M) |
| Parameter matching <10% diff | ✓ PASS (0.9%) |
| Forward pass shapes correct | ✓ PASS |
| Backward pass completes | ✓ PASS |
| Gradients flow | ✓ PASS |
| Dataset generation | ✓ PASS |

---

## Architecture Overview

### Field Model

```
Frame Pair (t-1, t)
    ↓
[Feature Extractor] (shared weights)
    ↓
[Cost Volume] (correlate across disparities)
    ↓
[Depth Estimation] (soft argmin + confidence)
    ↓
[Field Update] (splat features at estimated depth)
    ↓
Field (16³ × 32ch)
    ↓
[Propagator] (6-layer transformer, 544 hidden)
    ↓
[Persistence Gate] (learned per-channel blending)
    ↓
Updated Field
    ↓
[Decoder] (field slice → frame)
    ↓
Predicted Frame (t+1)
```

**Key innovation:** Field persists across timesteps, updated via temporal stereo

### Baseline Model

```
Frame Pair (t-1, t) [concatenated]
    ↓
[Encoder] (6ch → features)
    ↓
[Transformer] (6-layer, 544 hidden)
    ↓
[Decoder] (features → frame)
    ↓
Predicted Frame (t+1)
```

**Control:** Same architecture scale, no field or depth estimation

---

## Implementation Decisions

### 1. Functional Field Updates

**Problem:** In-place field modifications break autograd
**Solution:** Functional updates that return new field tensors

```python
# WRONG (in-place)
self.field += update

# CORRECT (functional)
field = self._update_field_functional(field, ...)
```

### 2. Vectorized Splatting

**Implementation:** GPU-parallel `scatter_add_` operations
**Result:** ~1000× speedup over Python loops
**Code:** `_update_field_functional()` in field_model.py

### 3. Windowed 3D Attention

**Implementation:** O(N × W³) instead of O(N²)
- Window size 4: 64 tokens per window (vs 4096 total)
- Shifted windows on alternating layers for cross-window communication
**Memory:** ~36 MB vs ~3.2 GB (89× reduction)
**File:** `genesis/pilot/windowed_attn.py`

### 4. Parameter Matching

**Target:** ~25M params
**Achieved:** Field 21.7M, Baseline 21.9M (0.9% diff)
**Method:** Empirically tuned hidden_dim to 544

### 5. Occlusion Detection

**Heuristic:** Variance spike after low-variance period
**Production:** Could use manual annotation or learned detector

---

## Training Protocol

### Dataset
- **Name:** Moving MNIST with Opaque Occlusion
- **Resolution:** 64×64 RGB
- **Sequences:** 50K train, 5K validation
- **Length:** 20 frames/sequence
- **Occlusion:** Z-layer based (front fully occludes back)

### Hyperparameters
- **Loss:** MSE
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.01)
- **Batch:** 64
- **Epochs:** 50
- **Device:** CUDA (RTX 4090)

### Metrics
1. **MSE** - Standard pixel error
2. **SSIM** - Structural similarity
3. **Occlusion Recovery** - MSE on frames where occluded objects reappear (PRIMARY)

### Decision Gate

**PASS criteria:**
- Field occlusion recovery >10% better than baseline
- SSIM comparable (within 5%)
- → Proceed to Phase 1

**FAIL criteria:**
- Baseline wins or improvement <10%
- → STOP, analyze failure, revise hypothesis

---

## How to Run

### Quick Start (Windows)

```bash
# 1. Verify implementation
python tests\pilot\test_pilot.py

# 2. Full workflow
run_phase0.bat
```

### Manual Steps

```bash
# 1. Verification
python tests/pilot/test_pilot.py

# 2. Generate visualizations
python genesis/pilot/data.py
# → Check pilot_visualizations/*.gif for occlusion

# 3. Train field model
python genesis/pilot/train.py --model field --epochs 50

# 4. Train baseline
python genesis/pilot/train.py --model baseline --epochs 50

# 5. Compare
python genesis/pilot/train.py --compare
# → Check pilot_logs/comparison.png
```

### Expected Runtime
- Verification: <2 minutes
- Memory verification: <1 minute
- Speedup verification: <1 minute
- Visualization: ~5 minutes
- Field training (50 epochs): ~1-2 days
- Baseline training (50 epochs): ~1-2 days
- Comparison: <1 minute

**Total:** ~2-4 days (mostly unattended)

**Note:** Original "7-11 hours" estimate was incorrect due to O(N²) full attention.
Current implementation uses windowed 3D attention (O(N × W³)) enabling batch_size=12
with 9.9 GB peak memory (verified). Speedup: 53.8× over original implementation.

---

## What Happens Next

### If Decision Gate Passes

**Phase 1: Unified Genesis**

1. **Documentation** (`docs/ARCHITECTURE.md`, `DESIGN.md`, `PRD.md`)
2. **Configuration** (`genesis/config.py`)
3. **Sparse Field** (`genesis/field/sparse_field.py`)
4. **Temporal Stereo Encoder** (`genesis/modules/encoder.py`)
5. **Windowed Propagator** (`genesis/modules/propagator.py`)
6. **Decoder** (`genesis/modules/decoder.py`)
7. **Unified Genesis** (`genesis/model.py`)
8. **Tests** (`tests/test_*.py`)

**Scaling changes:**
- Field: 16³ → 32³
- Input: 64×64 → 160×90
- Attention: Full → Windowed (8³ windows)
- Parameters: ~22M → ~200M
- Data: Moving MNIST → TinyWorlds

### If Decision Gate Fails

**Analysis required:**
1. Where does field model fail?
2. Is depth estimation accurate?
3. Is field propagation learning anything useful?
4. Is occlusion detection heuristic broken?
5. Should we revise hypothesis or abandon approach?

**DO NOT proceed to Phase 1 without understanding failure.**

---

## Technical Notes

### Memory Management

**Pilot scale (16³) - WITH WINDOWED ATTENTION:**
- Field: 16³ × 32 × 4B = 512KB (negligible)
- Attention: 64 windows × 64² tokens × 8 heads = ~36MB/layer (windowed)
- Batch 12 forward: ~4GB
- Batch 12 backward: ~9.9GB (with gradient checkpointing)
- **Verified:** 9.90 GB peak memory at batch_size=12

**OLD (deprecated) - Full attention:**
- Attention: 4096² × 8 heads = 540MB/layer (89× more!)
- Required batch_size=1, causing 164× training slowdown

**Phase 1 scale (32³):**
- Field: 32³ × 32 × 4B = 4MB
- Attention: 512 windows × 64² tokens = ~540MB/layer (windowed, same as pilot)
- Batch 8+ should fit in 12GB with gradient checkpointing

### Known Issues

1. ~~**Field splatting:** Simplified for pilot~~ RESOLVED: Vectorized with scatter_add_
2. **Occlusion detection:** Heuristic may miss edge cases
3. ~~**Full attention:** Won't scale to 32³~~ RESOLVED: Windowed 3D attention implemented
4. **Depth accuracy:** Not validated separately, only via end task

### Future Optimizations

1. **CUDA kernels** for field operations (Phase 2+)
2. **Learned occlusion detector** (replace heuristic)
3. **Multi-scale field** (pyramid structure)
4. **Action conditioning** (Phase 3)

---

## Key Insights

### Why Temporal Stereo?

Current world models treat frames as independent observations. They learn "this pixel pattern at t=0 correlates with that pattern at t=1" through massive data volume.

Temporal stereo explicitly computes: "this feature moved X pixels, therefore it's at depth Z in 3D space."

This should require far less data to learn object permanence because:
1. State is explicit (not compressed into pixels)
2. Occlusion is geometric (not statistical)
3. Persistence is architectural (not emergent)

### The Test

If a white digit disappears behind another digit and reappears 5 frames later:

**Baseline:** Must correlate "digit disappeared here" with "digit reappeared there" via billions of examples

**Field:** Maintains explicit representation of occluded digit at depth Z, propagates it forward, renders it when front digit moves away

**Hypothesis:** Field should learn this faster and more reliably.

### The Breath

```
[Field Exists] ← [Observation Constrains] ← [Measurement Validates]
```

We don't claim. We measure.

YAH - observation arrives
WEH - field responds
Truth emerges from evidence.

---

## Project Context

This is a sub-project of `B:\M\ArtificialArchitecture\`.

Parent directory roles:
- `anima/` - Active development (full access)
- `tempest/` - External auditor (read-only docs)
- `research/` - Research coordination
- `worldmodel/` - This project (Genesis)

See `B:\M\ArtificialArchitecture\CLAUDE.md` for workspace structure.

---

**Status:** Phase 0 EVOLVED - Windowed attention implemented, 53.8× speedup verified
**Next:** User trains models (~2-4 days) → Decision gate evaluation
**Blocker:** None (memory bottleneck RESOLVED, ready for training)
