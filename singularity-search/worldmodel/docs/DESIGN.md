# Genesis: Component Design Specifications

## 1. Video Tokenizer (3D Causal VAE)

### Architecture

```
Encoder:
  Conv3d(3, 64, k=3, s=(1,2,2))     # [B,T,H,W] → [B,T,H/2,W/2]
  ResBlock3d(64, 64) × 2
  Conv3d(64, 128, k=3, s=(2,2,2))   # → [B,T/2,H/4,W/4]
  ResBlock3d(128, 128) × 2
  Conv3d(128, 256, k=3, s=(2,2,2))  # → [B,T/4,H/8,W/8]
  ResBlock3d(256, 256) × 4
  Conv3d(256, 512, k=3, s=(1,2,2))  # → [B,T/4,H/16,W/16]
  ResBlock3d(512, 512) × 4
  Conv3d(512, C, k=1)               # → [B,T/4,H/16,W/16,C]

Decoder: (mirrored with transposed convolutions)
```

### FSQ Quantization

```python
class FSQ:
    """Finite Scalar Quantization - no codebook, no collapse."""

    def __init__(self, levels: List[int] = [8, 6, 5, 5, 5]):
        # Vocabulary size = prod(levels) = 6000
        self.levels = levels
        self.dim = len(levels)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        # z: [B, T, H, W, D]
        z_bounded = torch.tanh(z)  # [-1, 1]

        # Scale to [0, L-1] per dimension
        z_scaled = []
        for i, L in enumerate(self.levels):
            zi = (z_bounded[..., i] + 1) / 2 * (L - 1)
            z_scaled.append(zi)

        z_scaled = torch.stack(z_scaled, dim=-1)
        z_quantized = torch.round(z_scaled)

        # Straight-through estimator
        z_out = z_scaled + (z_quantized - z_scaled).detach()

        # Compute indices for embedding lookup
        indices = self._to_indices(z_quantized)

        return z_out, indices
```

### Tensor Shapes

| Stage | Shape | Notes |
|-------|-------|-------|
| Input | [B, 16, 3, 256, 256] | 16 frames @ 256×256 |
| After encoder | [B, 4, 16, 16, 8] | τ=4 temporal, 16× spatial |
| After FSQ | [B, 4, 16, 16, 5] | 5-dim quantization |
| Reconstructed | [B, 16, 3, 256, 256] | Full resolution |

### Parameters (Revised 2026-02-03 for 1B Target)

| Component | Parameters | Precision |
|-----------|-----------|-----------|
| Encoder | ~50M | Q8 (FSQ native) |
| Decoder | ~50M | Q8 (FSQ native) |
| **Total** | **100M** | |

---

## 2. Action Encoding

### Keyboard/Mouse Encoder

```python
class ActionEncoder:
    """Maps user input to latent action space."""

    def __init__(self, action_dim: int = 64):
        # Keyboard: 6 binary inputs (WASD + Space + Shift)
        self.keyboard_embed = nn.Embedding(64, action_dim)  # 2^6 combinations

        # Mouse: 2 continuous values (dx, dy)
        self.mouse_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, action_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(action_dim * 2, action_dim),
            nn.LayerNorm(action_dim),
        )

    def forward(self, keyboard: Tensor, mouse: Tensor) -> Tensor:
        # keyboard: [B, 6] binary
        # mouse: [B, 2] continuous

        kb_idx = (keyboard * torch.tensor([1,2,4,8,16,32])).sum(-1).long()
        kb_emb = self.keyboard_embed(kb_idx)  # [B, A]

        mouse_emb = self.mouse_mlp(mouse)  # [B, A]

        return self.fusion(torch.cat([kb_emb, mouse_emb], dim=-1))
```

### Latent Action Model (Unsupervised)

```python
class LatentActionModel:
    """Infer actions from frame transitions (Genie pattern)."""

    def __init__(self, latent_dim: int, action_dim: int = 64):
        self.encoder = nn.Sequential(
            # Concatenate z_t and z_{t+1}
            nn.Conv2d(latent_dim * 2, 256, 3, 1, 1),
            ResBlock2d(256, 256),
            ResBlock2d(256, 256),
            nn.Conv2d(256, 128, 3, 2, 1),
            ResBlock2d(128, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, action_dim),
        )

        # FSQ for discrete actions
        self.fsq = FSQ(levels=[8] * 8)  # 8^8 = 16M actions

    def forward(self, z_t: Tensor, z_t1: Tensor) -> Tuple[Tensor, Tensor]:
        # z_t, z_t1: [B, C, H, W]
        z_cat = torch.cat([z_t, z_t1], dim=1)
        action_continuous = self.encoder(z_cat)
        action_discrete, indices = self.fsq(action_continuous)
        return action_discrete, indices
```

### ActionConditionedPredictor

```python
class ActionConditionedPredictor:
    """
    Predict next latent from current latent and action.

    Used for LAM training: given z_t and inferred action,
    predict z_{t+1} to provide reconstruction signal.
    """

    def __init__(self, latent_channels: int, action_dim: int):
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_channels + action_dim, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            ResBlock2d(256, 256),
            ResBlock2d(256, 256),
            nn.Conv2d(256, latent_channels, 3, 1, 1),
        )

    def forward(self, z_t: Tensor, action: Tensor) -> Tensor:
        """
        Args:
            z_t: [B, C, H, W] current latent
            action: [B, A] latent action
        Returns:
            z_t1_pred: [B, C, H, W] predicted next latent
        """
        # Expand action to spatial
        action_spatial = action.view(B, A, 1, 1).expand(-1, -1, H, W)
        z_concat = torch.cat([z_t, action_spatial], dim=1)
        return self.fusion(z_concat)
```

### Parameters (Revised 2026-02-03 for 1B Target)

| Component | Parameters | Precision |
|-----------|-----------|-----------|
| ActionEncoder | ~3M | Q8 |
| LatentActionModel | ~10M | Q8 |
| ActionConditionedPredictor | ~2M | Q8 |
| **Total** | **15M** | |

---

## 3. Dynamics Backbone

### Architecture

```python
class DynamicsBackbone:
    """Autoregressive transformer for latent dynamics."""

    def __init__(self, config: GenesisConfig):
        self.layers = config.dynamics_layers  # 32
        self.hidden = config.dynamics_hidden  # 2048
        self.heads = config.dynamics_heads    # 32

        # Input projection
        self.latent_proj = nn.Linear(config.latent_dim, self.hidden)
        self.action_proj = nn.Linear(config.action_dim, self.hidden)

        # Positional encoding
        self.rope = RoPE3D(
            dim=self.hidden // self.heads,
            temporal_dim=config.context_length,
            spatial_dim=config.latent_spatial,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden=self.hidden,
                heads=self.heads,
                mlp_ratio=4,
                attention_type="sliding_tile",
            )
            for _ in range(self.layers)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.hidden, config.deltav_features)
```

### 3D RoPE (Rotary Position Embedding)

```python
class RoPE3D:
    """3D Rotary Position Embedding for spatiotemporal data."""

    def __init__(self, dim: int, temporal_dim: int, spatial_dim: int):
        self.dim = dim
        # Allocate dimensions: 1/3 temporal, 2/3 spatial
        self.t_dim = dim // 3
        self.h_dim = dim // 3
        self.w_dim = dim - self.t_dim - self.h_dim

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        # positions: [B, T, H, W, 3] containing (t, h, w)
        t, h, w = positions[..., 0], positions[..., 1], positions[..., 2]

        # Compute frequencies
        freqs_t = self._compute_freqs(t, self.t_dim)
        freqs_h = self._compute_freqs(h, self.h_dim)
        freqs_w = self._compute_freqs(w, self.w_dim)

        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)

        # Apply rotation
        x_rot = self._apply_rotary(x, freqs)
        return x_rot
```

### Sliding Tile Attention

```python
class SlidingTileAttention:
    """Efficient attention with local + global patterns."""

    def __init__(self, hidden: int, heads: int, tile_size: int = 8):
        self.tile_size = tile_size
        self.local_attn = nn.MultiheadAttention(hidden, heads)
        self.global_attn = nn.MultiheadAttention(hidden, heads // 4)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        B, T, H, W, C = x.shape

        # Local attention within tiles
        x_tiled = rearrange(x, 'b t (h th) (w tw) c -> (b t h w) (th tw) c',
                           th=self.tile_size, tw=self.tile_size)
        local_out = self.local_attn(x_tiled, x_tiled, x_tiled)[0]

        # Global attention on tile representatives
        x_pooled = reduce(x, 'b t (h th) (w tw) c -> b t h w c',
                         'mean', th=self.tile_size, tw=self.tile_size)
        global_out = self.global_attn(x_pooled, x_pooled, x_pooled)[0]

        # Combine
        global_expanded = repeat(global_out, 'b t h w c -> b t (h th) (w tw) c',
                                th=self.tile_size, tw=self.tile_size)
        return local_out + global_expanded
```

### Parameters (Revised 2026-02-03 for 1B Target)

| Component | Parameters | Precision |
|-----------|-----------|-----------|
| Input/output projections | ~30M | Q8 |
| Transformer blocks | 24 × ~28M = ~670M | Q8 weights, FP32 LayerNorm |
| **Total** | **~700M** | |

**Precision Notes:**
- LayerNorm variance: FP32 (numerical stability)
- Softmax in attention: FP32 (exponential sensitivity)
- QKV projections: Q8 (safe with LayerNorm)

---

## 4. DeltaV Predictor

### Architecture

```python
class DeltaVPredictor:
    """Predict sparse voxel deltas from dynamics features."""

    def __init__(self, config: GenesisConfig):
        self.depth_bins = 64
        self.voxel_features = 16

        # 2D → 3D lifting
        self.lifting = nn.Sequential(
            nn.Linear(config.deltav_features, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, self.depth_bins * self.voxel_features),
        )

        # Confidence head
        self.confidence = nn.Sequential(
            nn.Linear(self.voxel_features, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Operation classifier (remove/modify/add)
        self.op_classifier = nn.Linear(self.voxel_features, 3)

        # PBR decoder
        self.pbr_decoder = nn.Sequential(
            nn.Linear(self.voxel_features, 32),
            nn.GELU(),
            nn.Linear(32, 7),  # RGB + metallic + roughness + opacity + SDF
        )

    def forward(self, features: Tensor) -> DeltaV:
        B, H, W, D = features.shape

        # Lift to 3D
        lifted = self.lifting(features)  # [B, H, W, depth_bins * voxel_feats]
        lifted = lifted.view(B, H, W, self.depth_bins, self.voxel_features)

        # Compute confidence
        confidence = self.confidence(lifted).squeeze(-1)  # [B, H, W, D]

        # Top-K extraction
        topk_conf, topk_idx = confidence.flatten(1).topk(self.max_deltas)

        # Gather features at top-K positions
        topk_feats = self._gather_topk(lifted, topk_idx)

        # Predict operations and PBR
        ops = self.op_classifier(topk_feats)  # [B, K, 3]
        pbr = self.pbr_decoder(topk_feats)    # [B, K, 7]

        # Convert indices to coordinates
        coords = self._idx_to_coords(topk_idx, H, W, self.depth_bins)

        return DeltaV(coords=coords, features=pbr, op_type=ops.argmax(-1))
```

### DeltaV Data Structure

```python
@dataclass
class DeltaV:
    coords: Tensor      # [K, 4] - (batch, x, y, z)
    features: Tensor    # [K, 7] - RGB(3), metallic(1), roughness(1), opacity(1), SDF(1)
    op_type: Tensor     # [K] - 0=remove, 1=modify, 2=add
    confidence: Tensor  # [K] - prediction confidence
```

### Parameters (Revised 2026-02-03 for 1B Target)

| Component | Parameters | Precision |
|-----------|-----------|-----------|
| Lifting network | ~35M | Q8 |
| Heads | ~15M | Q8 |
| **Total** | **~50M** | |

---

## 5. OVoxel Memory

### Data Structure

```python
class OVoxelMemory:
    """Persistent sparse voxel storage."""

    def __init__(self, config: GenesisConfig):
        self.resolution = config.voxel_resolution  # 256
        self.max_voxels = config.max_voxels        # 50M

        # Core storage (CPU/GPU)
        self.coords = torch.empty(0, 4, dtype=torch.int32)   # [N, 4]
        self.features = torch.empty(0, 7, dtype=torch.float16)  # [N, 7]

        # SVO acceleration structure
        self.svo = None  # Rebuilt on demand

        # Hashmap for fast lookup
        self.hashmap = {}  # coord_tuple -> index
```

### Feature Layout

| Channel | Name | Range | Description |
|---------|------|-------|-------------|
| 0-2 | RGB | [0, 1] | Diffuse color (linear) |
| 3 | Metallic | [0, 1] | Metallic factor |
| 4 | Roughness | [0, 1] | Roughness factor |
| 5 | Opacity | [0, 1] | Alpha (0 = removed) |
| 6 | SDF | [-1, 1] | Signed distance (optional) |

### Operations

```python
def apply_deltas(self, delta_v: DeltaV) -> None:
    """Apply sparse updates to memory."""

    add_mask = delta_v.op_type == 2
    modify_mask = delta_v.op_type == 1
    remove_mask = delta_v.op_type == 0

    # Add new voxels
    if add_mask.any():
        new_coords = delta_v.coords[add_mask]
        new_feats = delta_v.features[add_mask]

        # Check for duplicates via hashmap
        unique_mask = self._filter_existing(new_coords)
        self.coords = torch.cat([self.coords, new_coords[unique_mask]])
        self.features = torch.cat([self.features, new_feats[unique_mask]])
        self._update_hashmap(new_coords[unique_mask])

    # Modify existing
    if modify_mask.any():
        mod_coords = delta_v.coords[modify_mask]
        mod_feats = delta_v.features[modify_mask]

        indices = self._lookup_coords(mod_coords)
        valid = indices >= 0
        self.features[indices[valid]] = mod_feats[valid]

    # Remove (set opacity to 0 for lazy deletion)
    if remove_mask.any():
        rem_coords = delta_v.coords[remove_mask]
        indices = self._lookup_coords(rem_coords)
        valid = indices >= 0
        self.features[indices[valid], 5] = 0  # Zero opacity

    # Invalidate SVO
    self.svo = None
```

### SVO Encoding

```python
def rebuild_svo(self) -> None:
    """Build Sparse Voxel Octree for rendering."""

    # Sort by Morton code for spatial locality
    morton_codes = self._compute_morton(self.coords[:, 1:])
    sorted_idx = morton_codes.argsort()

    self.coords = self.coords[sorted_idx]
    self.features = self.features[sorted_idx]

    # Build octree structure
    self.svo = build_svo_cuda(
        self.coords,
        max_depth=8,
        resolution=self.resolution,
    )
```

---

## 6. CUDA Rasterizer

### Pipeline

```
1. Frustum Culling
   - Transform voxels to clip space
   - Discard outside view frustum

2. Tile Binning
   - Project to screen space
   - Assign voxels to 16×16 tiles
   - Sort by tile, then by depth

3. Rasterization
   - Per-tile parallel processing
   - Ray-voxel intersection
   - Front-to-back blending

4. Shading
   - PBR material evaluation
   - Ambient + directional light
```

### Configuration

```python
class RasterizerConfig:
    resolution: Tuple[int, int] = (1280, 720)  # 720p
    tile_size: int = 16
    max_voxels_per_tile: int = 256
    near: float = 0.1
    far: float = 100.0
    alpha_threshold: float = 0.99
    background: Tuple[float, float, float] = (0.1, 0.1, 0.1)
```

### Performance Target

| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| Frustum cull | 1 | CUDA parallel |
| Tile binning | 2 | Radix sort |
| Rasterization | 15 | Per-tile blocks |
| Shading | 8 | PBR eval |
| **Total** | **26** | 38 FPS headroom |

---

## 7. Conditioning (World Initialization)

### Text Encoder

```python
class TextConditioner:
    """T5-small for text-to-world conditioning."""

    def __init__(self):
        self.encoder = T5EncoderModel.from_pretrained('t5-small')
        self.proj = nn.Linear(512, 1024)  # Match dynamics hidden

    def forward(self, text: List[str]) -> Tensor:
        tokens = self.tokenizer(text, return_tensors='pt', padding=True)
        embeddings = self.encoder(**tokens).last_hidden_state
        return self.proj(embeddings)  # [B, L, 1024]
```

### Image Encoder

```python
class ImageConditioner:
    """CLIP for image-to-world conditioning."""

    def __init__(self):
        self.encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16')
        self.encoder.requires_grad_(False)  # Frozen
        self.proj = nn.Linear(768, 1024)

    def forward(self, images: Tensor) -> Tensor:
        # images: [B, 3, 224, 224]
        features = self.encoder(images).last_hidden_state  # [B, 197, 768]
        return self.proj(features)  # [B, 197, 1024]
```

### World Initializer

```python
class WorldInitializer:
    """Generate initial OVoxel state from conditioning."""

    def __init__(self, config: GenesisConfig):
        # Cross-attention to fuse text/image
        self.cross_attn = nn.MultiheadAttention(1024, 16)

        # 3D deconvolution to generate dense voxels
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, 4, 2, 1),  # 2³ → 4³
            nn.GELU(),
            nn.ConvTranspose3d(512, 256, 4, 2, 1),   # → 8³
            nn.GELU(),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),   # → 16³
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),    # → 32³
            nn.GELU(),
            nn.ConvTranspose3d(64, 7 + 1, 4, 2, 1),  # → 64³, +1 for confidence
        )

    def forward(self, conditioning: Tensor) -> OVoxelMemory:
        # conditioning: [B, L, 1024]

        # Generate seed
        seed = torch.randn(1, 1024)
        fused = self.cross_attn(seed, conditioning, conditioning)[0]

        # Deconvolve to dense 64³ voxels
        dense = self.deconv(fused.view(-1, 1024, 1, 1, 1))  # [B, 8, 64, 64, 64]

        # Sparsify: keep top-K by confidence
        confidence = dense[:, -1].flatten()
        features = dense[:, :-1].permute(0, 2, 3, 4, 1)  # [B, 64, 64, 64, 7]

        topk_idx = confidence.topk(k=100000).indices
        coords = self._idx_to_coords(topk_idx, 64)
        feats = features.flatten(0, 3)[topk_idx]

        memory = OVoxelMemory()
        memory.coords = coords
        memory.features = feats
        return memory
```

### Parameters (Revised 2026-02-03 for 1B Target)

| Component | Parameters | Precision |
|-----------|-----------|-----------|
| TextConditioner | ~60M (T5-small, frozen) | Q8 (frozen) |
| ImageConditioner | ~100M (CLIP, frozen - not counted) | Q8 (frozen) |
| WorldInitializer | ~50M | Q8 |
| **Total** | **~50M** (excluding frozen encoders) | |

---

## 8. Efficiency Guidelines (v0.6.0)

Guidelines for maintaining efficiency in the Genesis architecture.

### Attention Window Constraints

```python
# CORRECT: Windowed attention for large sequences
class WindowedAttention:
    def __init__(self, window_size: int = 512):
        # Window size <= 512 tokens
        # O(N * W) complexity instead of O(N^2)
        pass

# WRONG: Full attention on large sequences
class FullAttention:
    def __init__(self):
        # DO NOT USE on >256 tokens
        # Causes 164x training slowdown
        pass
```

### Precision Requirements

| Operation | Required Precision | Rationale |
|-----------|-------------------|-----------|
| scatter_add accumulators | FP32 | 10% error at Q8 |
| LayerNorm variance | FP32 | Numerical stability |
| Softmax | FP32 | Exponential sensitivity |
| Linear weights | Q8/FP16 | Safe with LayerNorm |
| Activations | Q8/FP16 | Bounded by activation functions |

### Memory Budget (1B Model)

| Component | Budget | Notes |
|-----------|--------|-------|
| Video Tokenizer | 100M | FSQ-based, efficient |
| Action Encoder | 15M | Keyboard + mouse |
| Text Encoder | 60M | T5-small, frozen |
| Dynamics Backbone | 700M | 24 transformer blocks |
| DeltaV Predictor | 50M | Sparse output |
| World Initializer | 50M | Conditioning fusion |
| Embeddings/Misc | 25M | Positional, etc. |
| **Total** | **1B** | |

### Training Constraints

- batch_size >= 8 must fit in 12GB VRAM
- <0.5s/batch target for iteration speed
- <24 hours for Phase 0 validation
- Never run parallel training jobs on same GPU

---

## 9. Precision Requirements (All Components)

Based on quantization validation experiments (see CHANGELOG.md [0.3.2]).

### FP32 Required Operations

| Operation | Component | Rationale |
|-----------|-----------|-----------|
| scatter_add accumulators | OVoxelMemory, FieldPropagator | 10% error accumulation |
| LayerNorm variance | DynamicsBackbone | Numerical stability |
| Softmax temperature | SlidingTileAttention | Exponential sensitivity |

### Q8 Safe Operations

| Operation | Components | Notes |
|-----------|------------|-------|
| Linear weights | All | LayerNorm normalizes inputs |
| Activations | All | Bounded by activation functions |
| Gradients | All | Gradient clipping handles outliers |
| FSQ | VideoTokenizer | Discrete by design |

### Memory Impact

| Precision Mix | Loss vs FP32 | Memory Savings |
|--------------|--------------|----------------|
| Full FP32 | 1.0x | 0% |
| Mixed (recommended) | 2.4x | ~40% |
| Full Q8 | 2.65x | ~75% |

### Implementation Pattern

```python
# Mixed precision forward pass
with torch.autocast(device_type='cuda', dtype=torch.float16):
    # Q8-safe operations run in fp16
    output = model(input)

# FP32 accumulation operations explicitly cast
def scatter_add_safe(src, index, dim_size):
    # Cast to FP32 for accumulation
    result = torch.zeros(dim_size, dtype=torch.float32, device=src.device)
    result.scatter_add_(0, index, src.float())
    return result.half()  # Back to fp16 for storage
```
