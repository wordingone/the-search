# Genesis: Product Requirements Document

## Vision

Genesis is a continuous, interactive world model that learns dynamics from video unsupervised, maintains persistent 3D memory, and supports unlimited horizon inference with real-time user interaction.

## Problem Statement

Current world models face fundamental limitations:
1. **Context window bounds**: Genie 1 limited to 16 frames, quality degrades beyond
2. **Full-frame prediction**: Computationally expensive, redundant for mostly-static scenes
3. **No persistent memory**: Each frame generated independently, no accumulated state
4. **Parameter inefficiency**: Diffusion models require 10-30B params for quality

Genesis solves these through **Delta-Voxel Prediction**: predicting sparse 3D updates to persistent memory instead of full frames.

## Target Performance

| Metric | Target | Benchmark |
|--------|--------|-----------|
| VBench Overall | >82 | VideoAR-4B: 81.74 |
| VBench Semantic | >77 | VideoAR-4B: 77.15 (SOTA) |
| VBench CLIP-IQA | >0.50 | 64x64 baseline: 0.574 |
| Parameters | **<1B** | Hard constraint (NOT 4B) |
| Resolution | 720p | 1280×720 (via 256x256 → 512x512 → 720p) |
| Frame Rate | 24 FPS | Real-time interaction |
| Horizon | Unlimited | Persistent memory |
| Object Permanence | >90% | Genie 3 benchmark |

## Input Modalities

### World Initialization
| Input | Purpose | Processing |
|-------|---------|------------|
| Text | Scene description | T5-small → cross-attention → initial voxels |
| Image | Single-view bootstrap | CLIP → depth estimation → voxel lifting |
| Video | Multi-frame initialization | 3D VAE → latent history → populated memory |

### Continuous Control
| Input | Purpose | Processing |
|-------|---------|------------|
| WASD | Camera/character movement | Discrete action encoding |
| Mouse | Look direction | Continuous delta encoding |
| Space/Shift | Jump/crouch | Binary action flags |

## Functional Requirements

### FR-1: Video Understanding
- Encode arbitrary video to latent tokens
- Learn temporal dynamics without supervision
- Compress 16× spatial, 4× temporal

### FR-2: Action Learning
- Extract latent actions from frame transitions (unsupervised)
- Map keyboard/mouse to learned action space
- Support 8+ discrete action categories

### FR-3: Dynamics Prediction
- Autoregressive next-frame prediction
- Condition on action and history
- 16-frame context window with persistent memory

### FR-4: Delta-Voxel Output
- Predict sparse voxel changes (add/modify/remove)
- <5000 voxel updates per frame typical
- PBR material attributes (RGB, metallic, roughness, opacity)

### FR-5: Persistent Memory
- OVoxel sparse voxel storage
- SVO encoding for efficient queries
- Support 10M+ active voxels

### FR-6: Real-time Rendering
- CUDA octree rasterization
- 24+ FPS at 720p on RTX 4090
- Front-to-back alpha blending

## Non-Functional Requirements

### NFR-1: Performance
- Inference latency <42ms per frame (24 FPS)
- VRAM usage <20GB on RTX 4090
- Memory growth <1MB per minute of simulation

### NFR-2: Quality
- Visual fidelity comparable to Genie 3
- Temporal consistency over minutes
- No visible artifacts from delta accumulation

### NFR-3: Scalability
- Train on WebVid-10M, Panda-70M scale
- Distributed training support
- Checkpoint compatibility across versions

## Constraints

### Hardware
- Primary target: Windows + CUDA (RTX 4090, 24GB)
- Must not require >24GB VRAM for inference
- CUDA extensions required (no CPU fallbacks)

### Dependencies
- MIT or Apache 2.0 licenses only
- No runtime model downloads
- Local weights for all components

### Architecture
- **<1B total parameters** (hard constraint)
- Autoregressive (not diffusion) for latency
- Sparse operations for efficiency
- Resolution-adaptive latent channels (maintain bits/pixel across resolutions)

## Success Criteria

### Phase 1: Foundation
- [ ] Project structure complete
- [ ] Documentation (PRD, Architecture, Design)
- [ ] Configuration system

### Phase 2: Components
- [ ] FSQ quantizer working
- [ ] 3D RoPE implemented
- [ ] OVoxel memory operational

### Phase 3: Integration
- [ ] Full model assembles
- [ ] Forward pass executes
- [ ] Parameter count verified <4B

### Phase 4: Training
- [ ] Stage 1 tokenizer converges
- [ ] Stage 2 dynamics learns
- [ ] Stage 3 DeltaV produces valid outputs
- [ ] VBench evaluation >82

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Delta accumulation drift | Quality degrades over time | Periodic memory consolidation, stability losses |
| 2D→3D lifting ambiguity | Incorrect depth | Multi-view supervision, depth priors |
| Real-time budget exceeded | <24 FPS | Profile early, optimize critical path |
| OVoxel memory explosion | OOM | Pruning strategy, LOD system |

## Timeline

| Phase | Deliverable | Duration |
|-------|-------------|----------|
| 1 | Foundation + Docs | 1 session |
| 2 | Core Components | 2-3 sessions |
| 3 | Integration | 2-3 sessions |
| 4 | Training Pipeline | 3-4 sessions |
| 5 | Evaluation | 1-2 sessions |

## References

- Genie 1: arXiv 2402.15391
- Genie 2/3: DeepMind blog posts
- VideoAR: arXiv 2601.05966
- TinyWorlds: github.com/AlmondGod/tinyworlds
- TRELLIS: github.com/microsoft/TRELLIS
