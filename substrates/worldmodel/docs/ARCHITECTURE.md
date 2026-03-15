# Genesis: System Architecture

## Overview

Genesis implements a two-phase pipeline:
1. **World Initialization**: Text/image/video → initial OVoxel state
2. **Continuous Simulation**: Actions → delta predictions → memory updates → rendered frames

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GENESIS ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     WORLD INITIALIZATION                             │    │
│  │                                                                      │    │
│  │   Text ──→ T5-small ──┐                                             │    │
│  │                       ├──→ Cross-Attention ──→ WorldInitializer ──┐ │    │
│  │  Image ──→ CLIP ──────┘          ↑                                │ │    │
│  │                                  │                                │ │    │
│  │  Video ──→ VideoTokenizer ───────┘                                │ │    │
│  │                                                                   │ │    │
│  └───────────────────────────────────────────────────────────────────│─┘    │
│                                                                      │      │
│                                                                      ▼      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      OVOXEL MEMORY                                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │   Coords    │  │  Features   │  │    SVO      │                  │    │
│  │  │  [N, 4]     │  │  [N, 7]     │  │  Octree     │                  │    │
│  │  │ batch,x,y,z │  │ RGB,M,R,O,S │  │  Index      │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↑                      │                        │
│                              │                      ▼                        │
│  ┌───────────────────────────│─────────────────────────────────────────┐    │
│  │                    CONTINUOUS SIMULATION                             │    │
│  │                              │                                       │    │
│  │  User Input ──→ ActionEncoder ──┐                                   │    │
│  │  (WASD/mouse)       [B, A]      │                                   │    │
│  │                                 ▼                                   │    │
│  │  LatentHistory ──→ DynamicsBackbone ──→ DeltaVPredictor ────────────┤    │
│  │  [B, T, H, W, C]    (1.8B AR Xfmr)      [B, K, 11]                  │    │
│  │        ↑                                 coords + feats + op        │    │
│  │        │                                                            │    │
│  │        └──────────── VideoTokenizer.encode(rendered_frame) ←────────┤    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                      │      │
│                                                                      ▼      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      CUDA RASTERIZER                                 │    │
│  │                                                                      │    │
│  │   OVoxel Memory + Camera ──→ TileBasedRenderer ──→ Frame [3,H,W]    │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Precision Flow

Data precision through the pipeline (based on quantization validation - see CHANGELOG.md [0.3.2]):

```
Input (FP32) → VideoTokenizer (Q8/FSQ) → Dynamics (Q8 weights, FP32 LayerNorm/Softmax)
           ↓
     DeltaVPredictor (Q8) → OVoxelMemory (FP32 scatter_add) → Rasterizer (Q8)
           ↓
      Output (FP32)
```

**Critical FP32 paths:** LayerNorm variance, Softmax, scatter_add accumulators.

---

## Component Interfaces

### VideoTokenizer

```python
class VideoTokenizer:
    def encode(self, video: Tensor) -> Tensor:
        """
        Args:
            video: [B, T, 3, H, W] - Raw RGB frames
        Returns:
            latent: [B, T//4, H//16, W//16, C] - Compressed latents
        """

    def decode(self, latent: Tensor) -> Tensor:
        """
        Args:
            latent: [B, T', H', W', C] - Latent representation
        Returns:
            video: [B, T'*4, 3, H'*16, W'*16] - Reconstructed frames
        """

    def quantize(self, latent: Tensor) -> Tuple[Tensor, Tensor]:
        """FSQ quantization for discrete tokens."""
```

### ActionEncoder

```python
class ActionEncoder:
    def encode(self, keyboard: Tensor, mouse: Tensor) -> Tensor:
        """
        Args:
            keyboard: [B, 6] - WASD + Space + Shift binary
            mouse: [B, 2] - dx, dy continuous
        Returns:
            action: [B, A] - Latent action embedding
        """
```

### LatentActionModel

```python
class LatentActionModel:
    def infer_action(self, z_t: Tensor, z_t1: Tensor) -> Tensor:
        """
        Unsupervised action inference from frame transition.

        Args:
            z_t: [B, H', W', C] - Current latent
            z_t1: [B, H', W', C] - Next latent
        Returns:
            action: [B, A] - Inferred latent action
        """
```

### DynamicsBackbone

```python
class DynamicsBackbone:
    def forward(
        self,
        latent_history: Tensor,
        action_history: Tensor,
    ) -> Tensor:
        """
        Autoregressive next-latent prediction.

        Args:
            latent_history: [B, T, H', W', C] - Past latent frames
            action_history: [B, T, A] - Past actions
        Returns:
            next_latent: [B, H', W', D] - Predicted features for DeltaV
        """
```

### DeltaVPredictor

```python
class DeltaVPredictor:
    def forward(self, dynamics_out: Tensor) -> DeltaV:
        """
        Predict sparse voxel updates.

        Args:
            dynamics_out: [B, H', W', D] - Dynamics backbone output
        Returns:
            DeltaV containing:
                coords: [K, 4] - (batch, x, y, z) voxel coordinates
                features: [K, 7] - PBR attributes
                op_type: [K] - 0=remove, 1=modify, 2=add
        """
```

### OVoxelMemory

```python
class OVoxelMemory:
    def apply_deltas(self, delta_v: DeltaV) -> None:
        """Apply sparse updates to persistent memory."""

    def query_frustum(self, camera: Camera) -> Tuple[Tensor, Tensor]:
        """Return visible voxels for rendering."""

    def prune(self) -> None:
        """Remove zero-opacity voxels."""

    def serialize(self) -> bytes:
        """SVO-encoded checkpoint."""
```

### CUDARasterizer

```python
class CUDARasterizer:
    def render(
        self,
        coords: Tensor,
        features: Tensor,
        camera: Camera,
    ) -> Tensor:
        """
        Render voxels to image.

        Args:
            coords: [N, 4] - Voxel coordinates
            features: [N, 7] - PBR features
            camera: Camera pose and intrinsics
        Returns:
            frame: [3, H, W] - RGB image
        """
```

## Data Flow

### Initialization Flow

```
1. User provides: text OR image OR video
2. Conditioning encoder processes input:
   - Text: T5-small → [B, L, 512] embeddings
   - Image: CLIP → [B, 257, 768] tokens
   - Video: VideoTokenizer → [B, T', H', W', C] latents
3. WorldInitializer generates initial OVoxel state:
   - Cross-attention fuses conditioning
   - 3D deconvolution outputs dense voxel grid
   - Sparsification keeps top-K by confidence
4. OVoxelMemory populated with initial state
```

### Simulation Loop

```
for each frame:
    1. Get user input (keyboard, mouse)
    2. ActionEncoder → latent action [B, A]
    3. DynamicsBackbone(history, actions) → features [B, H', W', D]
    4. DeltaVPredictor(features) → sparse deltas [K, 11]
    5. OVoxelMemory.apply_deltas(deltas)
    6. CUDARasterizer.render(memory, camera) → frame [3, H, W]
    7. VideoTokenizer.encode(frame) → add to history
    8. Display frame
```

## Memory Management

### OVoxel Lifecycle

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   CREATED    │ ──→ │    ACTIVE    │ ──→ │   PRUNED     │
│  (op=add)    │     │  (op=modify) │     │  (op=remove) │
└──────────────┘     └──────────────┘     └──────────────┘
      │                    │                     │
      │                    │                     │
      └──────── Memory ────┴───── Lazy Delete ───┘
```

### Memory Budget

| Component | Budget | Notes |
|-----------|--------|-------|
| OVoxel coords | 16 MB / 1M voxels | int32 [N, 4] |
| OVoxel features | 14 MB / 1M voxels | fp16 [N, 7] |
| SVO index | ~1 MB / 1M voxels | Compressed octree |
| Latent history | 64 MB | 16 frames × [1, 16, 16, 8] |
| Model weights | ~4.1 GB | 2.03B params @ fp16 |
| **Total** | ~4.5 GB base + 30 MB/M voxels |

### Pruning Strategy

```python
# Every 100 frames
if frame_count % 100 == 0:
    # Remove voxels with zero opacity
    mask = memory.features[:, 5] > 0.01  # opacity channel
    memory.coords = memory.coords[mask]
    memory.features = memory.features[mask]
    memory.rebuild_svo()
```

## Training Pipeline

### Stage 1: Video Tokenizer (2 weeks)

```
Input: WebVid-10M video clips
Output: Trained encoder/decoder with FSQ

Loss = L_recon + β*L_perceptual + γ*L_fsq
     = ||x - decode(encode(x))||²
     + ||VGG(x) - VGG(x̂)||²
     + commitment_loss
```

### Stage 2: Dynamics + LAM (3 weeks)

```
Input: Video clips with frozen tokenizer
Output: Trained dynamics backbone + latent action model

Loss = L_next_frame + α*L_action_variance + β*L_consistency
     = ||ẑ_{t+1} - z_{t+1}||²
     + -log(Var(actions))
     + ||ẑ_{t+1} - ẑ_t||² * smoothness
```

### Stage 3: DeltaV with 3D Supervision (2 weeks)

```
Input: Objaverse + DL3DV multi-view data
Output: Trained 2D→3D lifting

Loss = L_voxel + α*L_sparsity + β*L_render
     = ||voxels_pred - voxels_gt||²
     + ||num_deltas||₀
     + ||render(memory + delta) - target||²
```

### Stage 4: End-to-End (1 week)

```
Input: Full pipeline with all components
Output: Fine-tuned complete model

Loss = L_render + α*L_consistency + β*L_stability
     = LPIPS(rendered, target)
     + temporal_smoothness
     + delta_magnitude_regularization
```

## Scaling Configurations

### Small (Debug)

```yaml
tokenizer:
  channels: [32, 64, 128, 256]
dynamics:
  layers: 12
  hidden: 768
  heads: 12
total_params: ~500M
```

### Default (Development)

```yaml
tokenizer:
  channels: [64, 128, 256, 512]
dynamics:
  layers: 24
  hidden: 1536
  heads: 24
total_params: ~1.2B
```

### Full (Production)

```yaml
tokenizer:
  channels: [64, 128, 256, 512]
dynamics:
  layers: 32
  hidden: 2048
  heads: 32
total_params: ~2.03B  # Verified actual count
```

**Note on frozen components:**
- T5-small (60M) and CLIP (86M) are frozen during training
- These are NOT counted toward the 4B parameter budget
- Only trainable parameters count: ~2.03B

## Error Handling

### Delta Accumulation Drift

Problem: Small errors accumulate over thousands of frames.

Mitigation:
1. **Stability loss**: Penalize large delta magnitudes
2. **Periodic consolidation**: Re-encode rendered frame every N frames
3. **Confidence gating**: Only apply high-confidence deltas

### Memory Explosion

Problem: Voxel count grows unbounded.

Mitigation:
1. **Active pruning**: Remove low-opacity voxels
2. **LOD system**: Merge distant voxels
3. **Hard cap**: Maximum 50M voxels, oldest pruned first

### Temporal Inconsistency

Problem: Objects flicker or teleport.

Mitigation:
1. **Temporal smoothing**: EMA on delta features
2. **Operation constraints**: Modify preferred over add/remove
3. **History conditioning**: Longer context window

---

## Efficiency Architecture (v0.6.0)

Genesis focuses on factual efficiency gains that are guaranteed and measurable.

### Guaranteed Savings (Build on These)

| Source | Savings | Implementation | Status |
|--------|---------|----------------|--------|
| Latent Tokenization | 12-32x | 8x spatial + 4x temporal VAE compression | Verified |
| Windowed Attention | 64x | O(N*W) vs O(N^2), window<=512 tokens | Verified |
| Mixed Precision | 2x | BF16 weights/activations, FP32 accumulators | Verified |

### NOT Guaranteed (Avoid Overcommitting)

| Source | Claimed | Reality | Notes |
|--------|---------|---------|-------|
| Sparse Compute | 20x | 2-7x | Gather/scatter overhead |
| Weighted Loss | 5x epochs | Task-dependent | Overhead can negate |
| Explicit 4D Field | ??? | ABANDONED | 2x overhead, worse accuracy |

### Why 4D Field Was Abandoned

Micro-pilot experiments (v0.5.1, v0.5.2) showed:
- 4D field uses **1.92x FLOPs** for **11% worse** accuracy vs baseline
- Depth dimension has no geometric meaning without 3D supervision
- Artificial depth is just extra hidden channels, no easier to learn

**Valid IF** (not applicable to 2D video):
- True 3D supervision available (voxel ground truth)
- Multi-view observations (geometric constraints)
- Depth from camera (real geometric meaning)

### Architectural Constraints (Non-Negotiable)

| Constraint | Rationale |
|------------|-----------|
| Window size <= 512 tokens | Memory/compute for batch_size>=8 |
| FP32 for accumulators | 10% error accumulation in Q8 scatter_add |
| FP32 for LayerNorm variance | Numerical stability |
| No O(N^2) attention on >256 tokens | 164x training slowdown |

### Module Organization

```
genesis/
├── tokenizer/     # Video VAE (latent compression)
├── dynamics/      # Transformer backbone (windowed attention)
├── action/        # Action encoding
├── deltav/        # Sparse voxel prediction
├── memory/        # OVoxel persistent storage
├── render/        # CUDA rasterizer
├── model.py       # Full model assembly
└── config.py      # Configuration
```
