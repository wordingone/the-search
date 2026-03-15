# Genesis Changelog

Meta-level project state. Detailed session logs in `changelogs/`.

---

## Project Identity

**Name**: Genesis World Model
**Goal**: Sub-1B parameter world model with unlimited horizon
**Target**: 720p @ 24 FPS, keyboard/mouse interaction

## Architecture (Stable)

| Component | Status | Notes |
|-----------|--------|-------|
| Motion-aware tokenizer | Stable | Keyframe + P-frame encoding |
| Slot attention dynamics | **Fixed v0.9.9** | Per-location tokens, pos_embed |
| Train-decay/infer-clip | Validated | Prevents mode collapse |
| Bounded context (64 frames) | Stable | Memory-safe generation |

## Quality Baselines (Reference)

| Resolution | CLIP-IQA (train) | CLIP-IQA (gen) | Status |
|------------|------------------|----------------|--------|
| 64x64 | 0.574 | - | VALIDATED baseline |
| 256x256 | 0.287 | - | DATA CEILING (JAT upscaled) |
| 720p (v3) | 0.8166 | 0.8166 | **INVALID** (all black frames, see Bug #4) |
| **720p (v4)** | **0.229** | **0.4725** | **REAL (4 bugs fixed, 500 iters, 15/15 non-black)** |

## Invariants (Do Not Violate)

1. **Parameters < 1B** - Hard constraint
2. **No mode collapse** - Use train-decay + infer-clip
3. **Memory bounded** - Context capped at 64 frames
4. **Unique spatial tokens** - Never flatten then expand

## Anti-Patterns (Lessons Learned)

| Never Do | Why | Do Instead |
|----------|-----|------------|
| Full training to validate fix | Hours wasted | `/validate-fix` <3min |
| Assume slow = model | Often data loading | Profile components |
| `Linear(C*H*W, dim)` + expand | Identical tokens | `Linear(C, dim)` per-loc |
| Compare mismatched evals | Wrong conclusions | Same task, same frames |
| Skip behavioral tests | Silent failures | Verify behavior changed |
| Progressive scaling when data-limited | Wastes weeks | Jump to native high-res |
| Continuous latents + slot attention | Slots corrupt | Use VQ-VAE (discrete) |
| Optimize on synthetic data | Ceiling ~0.22 | Use real video data |
| **Trust CLIP-IQA without visual check** | **Black 720p = 0.8166** | **Check frame.mean() > 0.05** |
| **Unconstrained residual decoder** | **Drift to mean=-1.22** | **Add Tanh activation** |
| **tok_recon_loss on keyframes only** | **P-frames unsupervised** | **Include 2+ frames** |
| **latent_scale too large** | **23x overshoot -> black** | **Match encoder std (~0.4)** |

## Validation Protocol

Before any fix is considered done:
```
/validate-fix
L1: Model loads, shapes correct (<5s)
L2: Behavior changed (<10s)
L3: Gradients flow (<10s)
L4: Loss decreases 100 iters (<60s)
```

## Version History (Major Only)

| Version | Date | Change |
|---------|------|--------|
| **0.10.7** | **2026-02-06** | **4 bugs fixed, CLIP-IQA 0.4725 (first honest 720p metric)** |
| ~~0.10.3~~ | ~~2026-02-05~~ | ~~720p VALIDATED~~ RETRACTED: CLIP-IQA 0.713 measured degenerate output |
| 0.10.2 | 2026-02-05 | Extended 720p training - Resume mode, iter 500→5000 |
| 0.10.1 | 2026-02-05 | 720p training started - WebVid+FSQ |
| 0.10.0 | 2026-02-05 | 720p pivot - VQ-VAE + OpenVid-1M |
| 0.9.9 | 2026-02-05 | Slot attention bug fix (unique tokens) |
| 0.9.7 | 2026-02-04 | Streaming dataset infrastructure |
| 0.9.6 | 2026-02-03 | Adaptive latent channels |

## Current Focus

**4 CRITICAL BUGS FOUND AND FIXED** (2026-02-06). V4 training needed.

### RETRACTION: Prior 720p "Success" Was Invalid (2026-02-06)

All prior 720p metrics measured degenerate output:
- CLIP-IQA 0.607 (training) = measuring gray/degenerate frames
- CLIP-IQA 0.713 (generation) = JAT seed on wrong dataset
- CLIP-IQA 0.8166 (v3) = pure black 720x720 frames (black=0.817 at this resolution)

**CLIP-IQA is unreliable for degenerate images at high resolution.**

| Image Type | 224x224 | 720x720 |
|-----------|---------|---------|
| Black | 0.642 | **0.817** |
| Random noise | 0.604 | 0.639 |

**Mitigation**: Always check `frame.mean() > 0.05` before trusting CLIP-IQA.

### 4 Bugs Found and Fixed (2026-02-06)

| Bug | File | Root Cause | Fix | Status |
|-----|------|------------|-----|--------|
| #1 | dynamics/model.py | latent_scale=8.0 (23x overshoot) | Changed to 0.4 | FIXED |
| #2 | genesis_experiment.py | Data norm [-1,1] vs Sigmoid [0,1] | Removed *2-1 | FIXED |
| #3 | genesis_model.py | No tok_recon_loss | Added MSE on keyframes | FIXED |
| #4 | tokenizer/motion.py | Unconstrained residual decoder | Tanh + P-frame supervision | FIXED |

**Impact of Bug #4**: Residual decoder output drifted from mean=-0.057 to mean=-1.223 during training. P-frame = warp(0.36) + residual(-1.22) = -0.86 -> clamp -> 0.0 (BLACK).

### V4 Training Complete (2026-02-06)

500 iters, all 4 bugs fixed. **First honest 720p metrics.**

| Metric | Value | Notes |
|--------|-------|-------|
| Loss (500 iters) | 0.131 | Improving trend |
| CLIP-IQA (train) | 0.229 | Real (frames non-black) |
| **CLIP-IQA (gen)** | **0.4725** | **Avg of 4 non-black frames** |
| CLIP-IQA best frame | **0.547** | Close to 0.50 target |
| Non-black frames | **15/15** | ALL frames pass mean>0.05 |
| Frame mean (keyframe) | 0.608 | Healthy distribution |
| Frame mean (P-frames) | 0.574-0.586 | No drift (was 0.00 in v3) |
| Parameters | 64.8M | Under 1B target |

**Next**: Extended training (2000-5000 iters) to push CLIP-IQA past 0.50.

### Architecture (Reference)

- Parameters: 64.8M at 720p (under 1B)
- Dataset: WebVid streaming at 720p
- Training speed: ~2.5-7.7s/iter (varies with data loading)
- Memory: 24GB VRAM (batch_size=1)

---

Detailed changelog: `changelogs/`
Commands: `/validate-fix`, `/changelog`
