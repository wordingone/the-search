# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Genesis**: Sub-1B parameter continuous interactive world model.

Target: 720p @ 24 FPS, unlimited horizon, keyboard/mouse interaction.

**Parameter Constraint**: <1B total (NOT 4B - this is a hard requirement).

### Core Directive

I want a world model that maintains coherent persistent state over unlimited time horizons while running in real-time on consumer hardware - not through brute-force parameters, but by exploiting structure. The core belief is that object permanence and temporal consistency shouldn't require 30B parameters if you represent the world correctly.

### Agent Teams

Genesis uses Agent Teams for coordinated multi-agent workflows. See:
- `AGENTS.md` - Team member definitions and coordination protocol
- `docs/AGENT_TEAMS_QUICKSTART.md` - Launch instructions

Launch with: `claude --teammate-mode in-process`

## Agent Role

**Meta-Cognitive Reasoning Expert** specializing in Machine Learning and Artificial Intelligence with meaningful experience in architecting world models and extreme mastery over resource and memory usage.

### Reasoning Protocol (For Complex Problems)

1. **DECOMPOSE**: Break into sub-problems. Doubt everything from the inside out, from where you are, now.
2. **SOLVE**: Address each with explicit confidence (0.0-1.0)
3. **VERIFY**: Check logic, facts, completeness, bias
4. **SYNTHESIZE**: Combine using weighted confidence
5. **REFLECT**: If confidence <0.8, identify weaknesses and retry

### Required Output Format

For all significant decisions:
- **Clear Answer**: The recommendation/solution
- **Confidence Level**: 0.0-1.0 with justification
- **Key Caveats**: Assumptions, risks, unknowns

---

## Role

Implement Genesis world model components following:
- `docs/PRD.md` - Requirements
- `docs/ARCHITECTURE.md` - System design
- `docs/DESIGN.md` - Component specifications

## Commands

```bash
# Set up environment (use local cache, not C drive)
source setup_env.sh  # bash
# OR: setup_env.bat  # cmd

# Run tests
cd B:\M\ArtificialArchitecture\worldmodel
python -m pytest tests/ -v

# Verify imports
python -c "from genesis import Genesis; print('OK')"

# Check parameter count
python -c "from genesis import Genesis; m = Genesis(); print(f'{sum(p.numel() for p in m.parameters()):,}')"

# Run training stage
python scripts/train.py --stage 1 --config configs/default.yaml
```

## Cache Configuration

HuggingFace and Torch caches are stored locally in `.cache/` (not C drive):
- Set environment via `source setup_env.sh` or `setup_env.bat`
- Or set `HF_HOME=B:/M/ArtificialArchitecture/worldmodel/.cache/huggingface`
- WebVid streaming requires no persistent cache (downloads to temp, discards after use)

## Processing Logic

### Tool Priority
1. **KG-MCP tools** for code exploration (search, symbol, slice, related_to)
2. **Read tool** as fallback (max 200 lines per read)
3. Never use Grep/Glob for source files when KG is indexed

### Code Patterns
- Reference TRELLIS implementations in `B:\M\ArtificialArchitecture\spatial\trellis-forge\`
- Match sparse voxel patterns from `trellis/representations/octree/`
- Match transformer patterns from `trellis/modules/sparse/transformer/`

## Constraints

### No Fallbacks
- CUDA extensions must work, no PyTorch fallbacks
- If CUDA fails, debug and fix the build
- Fallbacks cause 10-100× slowdown and incorrect results

### No Simplification
- Do not reduce resolution/quality to avoid problems
- Do not skip components to "get something working"
- Match the design specifications exactly

### Parameter Budget (1B Target - Revised 2026-02-03)
| Component | Budget |
|-----------|--------|
| Video Tokenizer | 100M |
| Action Encoder | 15M |
| Text Encoder | 60M (frozen) |
| Dynamics Backbone | 700M |
| DeltaV Predictor | 50M |
| World Initializer | 50M |
| Embeddings/Misc | 25M |
| **Total** | **1B** |

### Dependencies
- MIT or Apache 2.0 licenses only
- No runtime model downloads
- Local weights for all pretrained components

---

## Critical Anti-Patterns (Mistakes Made, Never Repeat)

### NEVER: Run Full Training to Validate a Bug Fix
**What happened**: Ran hours of training to check if a fix worked.
**Why it's wrong**: A fix can be validated in <3 minutes with unit tests.
**Do instead**: Use `/validate-fix` - test model loads, behavior changed, gradients flow, loss decreases (100 iters).

### NEVER: Assume Training is Slow Because of Model Size
**What happened**: Blamed 50M model for slow training, didn't profile.
**Why it's wrong**: Actual bottleneck was data loading (36s first batch), not model (313ms/batch after warmup).
**Do instead**: Profile each component separately. Data loading often dominates.

### NEVER: Flatten Spatial Dimensions Before Per-Location Projection
**What happened**: `Linear(C*H*W, dim)` then `expand(H*W)` created 256 IDENTICAL tokens.
**Why it's wrong**: Slot attention needs unique tokens per location to learn spatial binding.
**Do instead**: `reshape(B, H*W, C)` then `Linear(C, dim)` for unique tokens per location.

### NEVER: Compare Metrics from Different Evaluation Methodologies
**What happened**: Compared tokenizer CLIP-IQA (0.326) to VBench CLIP-IQA (0.151) directly.
**Why it's wrong**: Tokenizer eval = reconstruction of middle frames. VBench = generation including worst-case last frames. Different tasks.
**Do instead**: Use consistent frame selection and task type when comparing metrics.

### NEVER: Skip Positional Encoding in Attention Over Spatial Tokens
**What happened**: Slot attention had no 2D positional info.
**Why it's wrong**: Slots can't learn locality without knowing where tokens are spatially.
**Do instead**: Add `pos_embed = Parameter(randn(1, H*W, dim))` and add to tokens.

### NEVER: Trust That "No Error" Means "Fix Works"
**What happened**: Model loaded fine but tokens were still identical.
**Why it's wrong**: Structural bugs don't cause errors, they cause silent failures.
**Do instead**: Always add behavioral test: verify OLD behavior differs from NEW.

### NEVER: Launch Training Without Checking for Existing GPU Processes
**What happened**: Launched training while another instance was already running. Two identical processes competed for GPU, causing 3.5x slowdown.
**Why it's wrong**: GPU memory is shared. Multiple training processes cause OOM, slowdowns, and corrupted results.
**Do instead**: ALWAYS run `nvidia-smi` or `Get-Process python*` before launching training. Kill any existing training processes first.

```bash
# Before ANY training launch:
nvidia-smi --query-compute-apps=pid,process_name --format=csv
# If Python processes exist, verify they're not training before proceeding
```

### NEVER: Launch Same Command Multiple Times (Subagents or Retries)
**What happened**: Agent spawned duplicate training job, then main agent also spawned one. Both ran simultaneously.
**Why it's wrong**: Creates resource conflicts, wastes GPU time, corrupts checkpoints if both write to same directory.
**Do instead**:
1. Check if command is already running before launching
2. Use unique `--save-dir` for each run
3. If agent fails mid-training, check GPU before retry

### NEVER: Use Default Dataset in Evaluation Scripts
**What happened**: VBench evaluation defaulted to JAT (Atari 256p) but model was trained on WebVid (720p). Reported CLIP-IQA 0.713 was for wrong data.
**Why it's wrong**: Evaluation on mismatched data produces meaningless metrics. Hides real model performance.
**Do instead**: Always explicitly pass `--seed-dataset` matching the training dataset. For 720p models, use `--seed-dataset webvid`.

### NEVER: Trust CLIP-IQA Without Visual Validation
**What happened**: Black 720x720 images scored 0.8166 CLIP-IQA. V3 training "passed" with all-black generation output.
**Why it's wrong**: CLIP-IQA scores degenerate images higher at larger resolutions. Black 720p > random noise > gray.
**Do instead**: ALWAYS check `frame.mean() > 0.05` before trusting any CLIP-IQA score. Visual inspection mandatory.

### NEVER: Train WebVid Without Video Cache
**What happened**: Each iteration did fresh HTTP download from WebVid URLs. Training speed degraded from 721ms to 5,911ms per iter.
**Why it's wrong**: WebVid URLs expire, servers throttle, retry overhead compounds. 80-90% of wall time is data I/O.
**Do instead**: Use `--video-cache-dir data/video_cache` (default ON). First run caches 50 video batches locally (~2-3GB). All subsequent runs load from disk at ~150ms vs ~3300ms.

### NEVER: Train 720p Without GPU Resource Management
**What happened**: 720p model uses ~100% VRAM (24.7GB / 24.5GB), locks workstation at 100% GPU.
**Why it's wrong**: Cannot use computer during training. No headroom for other processes.
**Do instead**: Always use `--gradient-checkpoint` (reduces VRAM 43%) + `--gpu-yield-ms 10` (prevents 100% lock).

### NEVER: Leave Residual Decoder Unconstrained
**What happened**: Residual decoder outputs drifted negative without bounds. P-frames went black.
**Why it's wrong**: Unconstrained outputs + additive residuals = unbounded drift over time.
**Do instead**: Add Tanh activation to residual decoder output. Supervise P-frames, not just keyframes.

---

## Architectural Constraints (Non-Negotiable)

These constraints exist because violating them caused 164x training slowdown (82 days instead of 3 days).

### Attention Complexity
- NEVER use O(N^2) full attention on >256 tokens
- Windowed attention REQUIRED for field propagation (any field >4^3)
- Window size <= 8^3 = 512 tokens maximum per attention operation
- Memory profile MUST fit batch_size>=8 in 12GB VRAM

### Efficiency Thesis (ABANDONED for 2D video - 2026-02-03)

**Original thesis**: "Explicit 3D state is more efficient than pixel-stream inference."

**Status: ABANDONED** after micro-pilot experiments (v1, v2) showed:
- 4D field uses 2x FLOPs for worse accuracy than 2D baseline
- Depth dimension has no geometric meaning without 3D supervision
- Artificial depth is just extra hidden channels, no easier to learn

**Still valid IF**:
- True 3D supervision available (voxel ground truth)
- Multi-view observations (geometric constraints)
- Depth from camera (real geometric meaning)

**For 2D video prediction, use standard approaches**:
- Large transformers (implicit structure)
- Slot attention (explicit objects, no fake depth)
- Implicit 3D (NeRF-style, learns 3D from 2D via differentiable rendering)

The following constraints remain valid for other purposes:

### Training Feasibility (Revised 2026-02-03)
- Phase 0 training MUST complete in **<24 hours** on RTX 4090
- If memory forces batch_size=1, the architecture is WRONG - fix it
- Verify memory profile BEFORE starting long training runs
- Never assume training time - measure iteration speed first
- Use `--benchmark` to verify <0.5s/batch before full training
- **NEVER run multiple training jobs in parallel on the same GPU** - they compete for resources and both slow down; run sequentially instead

### Terminating Training Processes
When user requests to kill/stop training:
1. First try `TaskStop` with the task ID
2. Verify with `nvidia-smi` - if GPU still shows high usage, processes are still running
3. Use PowerShell to force kill (cmd/taskkill has path escaping issues in Git Bash):
   ```bash
   # Find Python PIDs from nvidia-smi output, then:
   powershell -Command "Stop-Process -Id <PID> -Force"
   ```
4. Verify GPU usage dropped (should be <20% idle)

### Documentation Drift Prevention
- PRD/ARCHITECTURE/DESIGN must match actual implementation
- Training time estimates must be verified by measurement, not assumption
- Update ALL docs when discoveries invalidate prior assumptions
- CHANGELOG.md is source of truth for current state

---

## Quantization Constraints (Experimentally Verified)

These constraints were validated through systematic experimentation testing the hypothesis: "Q8 requires structured, sparse, bounded, local computation."

**Result:** Hypothesis partially wrong. Q8 works when FP32 is preserved for accumulation operations.

### FP32 Required (Non-Negotiable)
- `scatter_add` accumulators (10% error accumulation at Q8)
- LayerNorm variance computation
- Softmax temperature

### Q8 Safe
- Weights, activations, gradients
- Any activation function (GELU/ReLU work fine)
- Windowed attention (no quantization advantage over full)

### Target Precision Mix
- Mixed precision: 2.4x FP32 loss (~40% memory savings)
- Full Q8: 2.65x FP32 loss (acceptable for some applications)

### Key Insight
Structural properties (windowed, sparse, local) help compute/memory efficiency, NOT quantization. LayerNorm + residual connections are the true enablers of stable Q8 training.

---

## Efficiency Savings Sources (Context: 3D-Supervised Tasks Only)

These targets apply only to tasks with true 3D supervision (voxel GT, multi-view, depth).
For 2D video prediction, use standard transformer approaches.

| Dimension | Constraint | Savings Target | Violation Symptom |
|-----------|------------|----------------|-------------------|
| Representation | Exploit sparsity (<5% occupancy) | 20x | Dense ops on sparse data |
| Computation | Windowed attention O(N*W^3) | 64x | O(N^2) full attention |
| Supervision | Certainty-weighted loss | 3-5x | Uniform MSE everywhere |
| Memory | Vectorized ops (scatter_add) | 1000x | Python for-loops |

If these savings are not achieved in 3D-supervised tasks, the architecture is wrong.

---

## Documentation Rule

**All documentation urges → CHANGELOG.md**

Do not create new .md files unless explicitly requested. After completing work:
1. Update CHANGELOG.md with what was done
2. Include any issues encountered
3. Note any deviations from design

## Key Files

| File | Purpose |
|------|---------|
| `docs/PRD.md` | Product requirements |
| `docs/ARCHITECTURE.md` | System design, data flow |
| `docs/DESIGN.md` | Component specifications |
| `configs/default.yaml` | Model configuration |
| `genesis/model.py` | Full model assembly |
| `CHANGELOG.md` | Progress and decisions |

## Planning Documents

| File | Purpose |
|------|---------|
| `~/.claude/plans/humble-finding-hoare.md` | Original Genesis plan (Phase 0-2) |
| `~/.claude/plans/proud-squishing-flask.md` | Evolved plan (efficiency fixes) |

The evolved plan exists because the original violated the efficiency thesis in its own implementation (O(N^2) attention on 4096 tokens). Always check these plans before major architectural decisions.

## Reference Implementations

| Component | Reference Path |
|-----------|---------------|
| Sparse VAE | `spatial/trellis-forge/trellis/models/sparse_structure_vae.py` |
| Octree | `spatial/trellis-forge/trellis/representations/octree/octree_dfs.py` |
| Transformer | `spatial/trellis-forge/trellis/modules/sparse/transformer/blocks.py` |
| Rasterizer | `spatial/trellis-forge/trellis/renderers/octree_renderer.py` |
| OVoxel | `spatial/trellis-forge/o_voxel/o_voxel/` |

## Target Performance

| Metric | Target | Reference | Status |
|--------|--------|-----------|--------|
| VBench CLIP-IQA | >0.50 | 64x64: 0.574 | [IN PROGRESS] 256p: 0.287 (ceiling), 720p: TBD |
| VBench Overall | >82 | VideoAR-4B: 81.74 | [IN PROGRESS] 720p training pending |
| Parameters | **<1B** | Hard constraint | [VERIFIED] Current: 50M (256p), ~58.5M (720p) |
| Resolution | **720p** | Primary target | [IN PROGRESS] VQ-VAE + OpenVid-1M path |
| Frame Rate | 24 FPS | Real-time | [VERIFIED] ~110ms/frame = 9 FPS |
| Object Permanence | >90% | Genie 3 level | [UNTESTED] |
| Unlimited Horizon | 1000+ frames | No collapse | [VERIFIED] at 64x64 |
| Memory Stability | <10% growth | No leak | [VERIFIED] 0.3% during generation |
| Training Speed | 6.4 samp/sec | After warmup | [VERIFIED] 313ms/batch |

---

## Bug Fix Validation Protocol

**Never run full training to validate a fix.** Use `/validate-fix` or:

| Level | Time | What to Test |
|-------|------|--------------|
| L1: Static | <5s | Model loads, shapes correct |
| L2: Behavioral | <10s | OLD vs NEW behavior differs |
| L3: Gradient | <10s | All params have gradients |
| L4: Learning | 60s | 100 iters, loss decreases |

**Total: <3 minutes** - See `~/.claude/skills/bug-fix-validation.md`

---

## Current State (2026-02-06)

**Version**: 0.10.7 (v4 checkpoint, 4 bugs fixed)
**Risk Level**: MODERATE (real metrics, needs more training)
**Parameters**: 64.8M at 720p (well under 1B target)

**Metrics** (honest, with frame.mean()>0.05 guard):
- CLIP-IQA (gen avg): **0.4725** (target >0.50)
- CLIP-IQA (best frame): **0.547** (above target)
- Non-black frames: **15/15** (was 0/15 in v3)
- Frame means: keyframe=0.608, P-frames=0.574-0.586

**Success Probability** (updated 2026-02-06):
- 256x256 complete: DONE (CLIP-IQA 0.287 ceiling, data-limited)
- 720p CLIP-IQA >0.50: 35-50% (on track with extended training)
- VBench >82 (Genie 3 level): 25-35%

### 4 Critical Bugs Fixed (v0.10.4 - v0.10.7)

| Bug | File | Fix |
|-----|------|-----|
| #1: latent_scale 23x overshoot | dynamics/model.py | 8.0 -> 0.4 |
| #2: Data norm [-1,1] vs Sigmoid [0,1] | genesis_experiment.py | Remove `*2-1` |
| #3: No tok_recon_loss | genesis_model.py | Added MSE on keyframes |
| #4: Residual decoder drift | tokenizer/motion.py | Tanh + P-frame supervision |

### Video Cache (NEW - eliminates data loading bottleneck)

WebVid HTTP streaming was 80-90% of training wall time (3,300ms/iter).
Persistent local cache brings this to ~900ms/iter (3.6x speedup).

- Cache dir: `data/video_cache/720/` (~2-3GB for 50 batches)
- Auto-populates on first training run
- All subsequent runs load from disk
- `--no-cache` flag to force fresh HTTP downloads

### Critical Bug Fixed (v0.9.9)

**Slot tokens were ALL IDENTICAL** - This broke spatial binding at 256x256.

```python
# OLD (broken): All 256 tokens bit-for-bit identical
latent_proj = Linear(C*H*W, slot_dim)  # 4096 -> 64
tokens = proj(flatten(x)).expand(H*W)  # broadcast single vector

# NEW (fixed): Each spatial location unique
latent_proj = Linear(C, slot_dim)      # 16 -> 64 per location
tokens = proj(x.reshape(B, H*W, C))    # [B, 256, 64] unique
pos_embed added for spatial awareness
```

Validated: 256/256 unique tokens, gradients flow, model loads.

### Training Speed (Verified)

| Metric | Value |
|--------|-------|
| First batch | 36s (dataset init, one-time) |
| After warmup | 313ms/batch (batch_size=2) |
| Throughput | 6.4 samples/sec |
| 1000 samples | 2.6 min |
| Bottleneck | Backward pass (48%), NOT data |

### What Works
- 64x64: CLIP-IQA 0.574, 1000+ frames, stable memory
- 256x256: CLIP-IQA 0.287 with JAT real data (data ceiling reached)
- Train-decay + infer-clip workflow validated
- Unique spatial tokens (bug fixed v0.9.9)
- Fast training after warmup (313ms/batch)
- Perceptual loss integration working

### What Needs Implementation (720p Pivot)
- VQ-VAE tokenizer (discrete latents prevent corruption)
- OpenVid-1M dataset integration (433K native 1080p clips)
- 720p training pipeline

### Confidence Gates

| Gate | 64x64 | 256x256 |
|------|-------|---------|
| Gate 0: Factorization | PASS | PASS |
| Gate 1: State sufficiency | PASS | NEEDS RETEST |
| Gate 2: Perception invariance | UNTESTED | UNTESTED |
| Gate 3: Graceful decay | PASS | NEEDS RETEST |
| Gate 4: Parameter efficiency | PASS | PASS (50M) |

---

## Verification Status (2026-02-05)

### Tested Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| Parameter budget | [VERIFIED] | 50M params (well under 1B target) |
| Unlimited horizon | [VERIFIED] | 1000 frames at 64x64, no collapse |
| No memory leak | [VERIFIED] | 0.3% growth during generation |
| Slot attention works | [VERIFIED] | Train-decay + infer-clip workflow |
| Context bounded | [VERIFIED] | Capped at 64 frames |
| **Unique spatial tokens** | [VERIFIED] | 256/256 unique (bug fixed v0.9.9) |
| **Gradient flow** | [VERIFIED] | All params have non-zero gradients |
| **Training speed** | [VERIFIED] | 6.4 samp/sec after warmup |
| **64x64 quality** | [VERIFIED] | CLIP-IQA 0.574 (good) |
| **256x256 quality** | [COMPLETE] | CLIP-IQA 0.287 (data ceiling reached) |
| **720p resolution** | [IN PROGRESS] | VQ-VAE + OpenVid-1M path selected |

### VBench Results (2026-02-05)

| Metric | 64x64 | 256x256 | 720p (target) |
|--------|-------|---------|---------------|
| CLIP-IQA | **0.574** | **0.287** | >0.50 |
| MUSIQ | 18.38 | 20.02 | TBD |
| Frame Consistency | 0.0124 | 0.0143 | TBD |

**Note**: 256x256 reached data ceiling at 0.287 (JAT is 64x64 upscaled). 720p with OpenVid-1M expected to break through.

### Critical Discovery: Train-Decay + Infer-Clip Workflow

**Problem**: Training with `slot_norm_mode='clip'` causes mode collapse.
**Solution**: Train with `decay` (default), infer with `clip` override.

| Configuration | Result |
|--------------|--------|
| Train clip, test clip | MODE COLLAPSE (frames identical) |
| Train decay, test decay | MODE COLLAPSE (slot influence vanishes) |
| **Train decay, test clip** | **WORKS** (diverse frames, stable memory) |

**Why it works**:
1. Decay during training forces model to learn diversity generation
2. Information loss requires active content regeneration
3. Clip at inference preserves learned diversity without explosion

### Memory Profile (No Leak)

| Stage | Memory |
|-------|--------|
| Model load | 125.6 MB |
| After warm-up | 144.6 MB |
| After 1000 frames | 145.1 MB |
| **Generation growth** | **0.5 MB (0.3%)** |

Context window bounded at 64 frames. Slot memory negligible (~2KB).

### Test Results Summary

| Metric | Value | Pass Criteria | Status |
|--------|-------|---------------|--------|
| MSE ratio | 0.11x | ≤2.0x | PASS |
| PSNR delta | +9.6 dB | ≥-3 dB | PASS |
| Memory growth | 0.3% | ≤10% | PASS |
| Frame variance std | 0.0035 | >0.001 | PASS |
| Consecutive similarity | 0.9955 | <0.999 | PASS |

---

## Commands

### Training (720p with video cache)
```bash
# Quick training (500 iters, ~8 min with cache)
python scripts/genesis_experiment.py --mode train --iterations 500 \
  --data-mode webvid --image-size 720 --use-fsq --use-perceptual \
  --perceptual-weight 0.1 --eval-interval 100 \
  --gpu-yield-ms 10 --gradient-checkpoint \
  --save-dir checkpoints/genesis_quick

# Standard training (2000 iters, ~30 min with cache)
python scripts/genesis_experiment.py --mode train --iterations 2000 \
  --data-mode webvid --image-size 720 --use-fsq --use-perceptual \
  --perceptual-weight 0.1 --eval-interval 500 \
  --gpu-yield-ms 10 --gradient-checkpoint \
  --save-dir checkpoints/genesis_standard

# Resume from checkpoint
python scripts/genesis_experiment.py --mode resume \
  --checkpoint checkpoints/genesis_720_v4_cached/checkpoint_1000.pt \
  --iterations 1000 --data-mode webvid --image-size 720 \
  --use-fsq --use-perceptual --perceptual-weight 0.1 \
  --gpu-yield-ms 10 --gradient-checkpoint \
  --save-dir checkpoints/genesis_720_v4_cached

# REQUIRED FLAGS for 720p:
#   --gradient-checkpoint  (VRAM 24.7GB -> 14.1GB, prevents OOM)
#   --gpu-yield-ms 10     (prevents 100% GPU lock)
#   --use-fsq             (discrete quantization prevents corruption)
#   --video-cache-dir     (default: data/video_cache, auto-populates)
```

### Testing
```bash
# Test horizon stability (MUST use --slot-norm-mode clip)
python genesis/test_horizon_stability.py \
  --checkpoint checkpoints/genesis_256/best_genesis.pt \
  --slot-norm-mode clip --frames 1000

# Test with memory profiling
python genesis/test_horizon_stability.py \
  --checkpoint checkpoints/genesis_256/best_genesis.pt \
  --slot-norm-mode clip --frames 1000 --profile-memory
```

### Verification
```bash
# Verify imports
python -c "from genesis import Genesis; print('OK')"

# Check parameter count (256x256)
python -c "from genesis.genesis_model import Genesis, GenesisConfig; m = Genesis(GenesisConfig(image_size=256)); print(f'{sum(p.numel() for p in m.parameters()):,}')"

# Run unit tests
python -m pytest tests/ -v
```

### Bug Fix Validation (Use BEFORE committing fixes)
```bash
# L1: Model loads
python -c "from genesis.genesis_model import Genesis, GenesisConfig; m = Genesis(GenesisConfig(image_size=256)); print('OK')"

# L2: Token uniqueness (v0.9.9 fix)
python -c "
import torch
from genesis.dynamics.model import SlotLatentDynamicsModel
m = SlotLatentDynamicsModel(latent_channels=16, latent_height=16, latent_width=16)
x = torch.randn(1, 16, 16, 16)
t = m.latent_proj(x.permute(0,2,3,1).reshape(1,256,16)) + m.pos_embed
print(f'Unique tokens: {len(torch.unique(t[0], dim=0))}/256')
"

# L3: Gradient flow
python -c "
import torch
from genesis.genesis_model import Genesis, GenesisConfig
m = Genesis(GenesisConfig(image_size=256)).cuda()
x = torch.randn(1,16,3,256,256).cuda()
m(x)['total_loss'].backward()
print('Gradients OK' if all(p.grad is not None for p in m.parameters() if p.requires_grad) else 'FAIL')
"

# L4: Quick learning (100 iters, ~60s)
# Use synthetic data to avoid data loading overhead
```

---

## Next Actions (Priority Order)

1. **Implement VQ-VAE tokenizer** - Fix continuous latent corruption
   - Discrete codes prevent dynamics from corrupting latents
   - Target: 8192 codebook, CLIP-IQA >0.40
2. **Integrate OpenVid-1M dataset** - Native 1080p quality data
   - WebDataset streaming (no full download)
   - 433K clips at native high resolution
3. **Train VQ-VAE at 720p** - 2-3 days GPU time
   - Validate: codebook usage >90%, CLIP-IQA >0.40
4. **Train dynamics at 720p** - 5-7 days GPU time
   - Target: CLIP-IQA >0.50, VBench >75
5. **Extended horizon testing** - 10K, 100K frames at 720p

### Completed (256p)
- [x] 256p training: CLIP-IQA 0.287 (data ceiling)
- [x] Slot attention bug fix (v0.9.9)
- [x] Perceptual loss integration
- [x] JAT real data integration

---

## Parent Context

Subdirectory of `B:\M\ArtificialArchitecture\`. See parent CLAUDE.md for workspace roles.
