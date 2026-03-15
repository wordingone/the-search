"""Genesis model configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml
from pathlib import Path


@dataclass
class FSQConfig:
    """Finite Scalar Quantization configuration."""
    levels: List[int] = field(default_factory=lambda: [8, 6, 5, 5, 5])
    dim: int = 5

    @property
    def vocab_size(self) -> int:
        """Vocabulary size = product of levels."""
        result = 1
        for L in self.levels:
            result *= L
        return result


@dataclass
class TokenizerConfig:
    """3D Causal VAE configuration."""
    input_channels: int = 3
    latent_channels: int = 5  # Must match FSQ dim
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 4, 4])
    decoder_depths: List[int] = field(default_factory=lambda: [4, 4, 2, 2])
    temporal_downsample: int = 4
    spatial_downsample: int = 16
    fsq: FSQConfig = field(default_factory=FSQConfig)


@dataclass
class ActionEncoderConfig:
    """Keyboard/mouse encoder configuration."""
    keyboard_keys: int = 6
    mouse_dims: int = 2
    action_dim: int = 64


@dataclass
class LAMConfig:
    """Latent Action Model configuration."""
    hidden_channels: List[int] = field(default_factory=lambda: [256, 128])
    fsq_levels: List[int] = field(default_factory=lambda: [8] * 8)
    variance_loss_weight: float = 0.1


@dataclass
class ActionConfig:
    """Complete action system configuration."""
    encoder: ActionEncoderConfig = field(default_factory=ActionEncoderConfig)
    lam: LAMConfig = field(default_factory=LAMConfig)


@dataclass
class RoPEConfig:
    """3D Rotary Position Embedding configuration."""
    type: str = "3d"
    base: int = 10000
    temporal_fraction: float = 0.333


@dataclass
class AttentionConfig:
    """Sliding tile attention configuration."""
    type: str = "sliding_tile"
    tile_size: int = 8
    global_heads: int = 8


@dataclass
class DynamicsConfig:
    """Autoregressive transformer configuration."""
    layers: int = 32
    hidden_dim: int = 2048
    num_heads: int = 32
    head_dim: int = 64
    mlp_ratio: int = 4
    dropout: float = 0.0
    context_length: int = 16
    rope: RoPEConfig = field(default_factory=RoPEConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)


@dataclass
class DeltaVConfig:
    """Delta-Voxel predictor configuration."""
    input_dim: int = 2048
    depth_bins: int = 64
    voxel_features: int = 16
    max_deltas_per_frame: int = 4096
    confidence_threshold: float = 0.5


@dataclass
class SVOConfig:
    """Sparse Voxel Octree configuration."""
    max_depth: int = 8
    serialize: str = "z_order"


@dataclass
class MemoryConfig:
    """OVoxel memory configuration."""
    resolution: int = 256
    max_voxels: int = 50_000_000
    dtype: str = "float16"
    feature_channels: int = 7
    prune_interval: int = 100
    opacity_threshold: float = 0.01
    svo: SVOConfig = field(default_factory=SVOConfig)


@dataclass
class RenderConfig:
    """CUDA rasterizer configuration."""
    resolution: Tuple[int, int] = (1280, 720)
    tile_size: int = 16
    max_voxels_per_tile: int = 256
    near: float = 0.1
    far: float = 100.0
    alpha_threshold: float = 0.99
    background: Tuple[float, float, float] = (0.1, 0.1, 0.1)


@dataclass
class TextConditioningConfig:
    """Text encoder configuration."""
    model: str = "t5-small"
    hidden_dim: int = 512
    proj_dim: int = 1024


@dataclass
class ImageConditioningConfig:
    """Image encoder configuration."""
    model: str = "clip-vit-base-patch16"
    hidden_dim: int = 768
    proj_dim: int = 1024
    freeze: bool = True


@dataclass
class InitializerConfig:
    """World initializer configuration."""
    deconv_channels: List[int] = field(default_factory=lambda: [1024, 512, 256, 128, 64])
    output_resolution: int = 64
    top_k_voxels: int = 100_000


@dataclass
class ConditioningConfig:
    """Complete conditioning configuration."""
    text: TextConditioningConfig = field(default_factory=TextConditioningConfig)
    image: ImageConditioningConfig = field(default_factory=ImageConditioningConfig)
    initializer: InitializerConfig = field(default_factory=InitializerConfig)


@dataclass
class GenesisConfig:
    """Complete Genesis model configuration."""
    name: str = "genesis"
    version: str = "0.1.0"

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    deltav: DeltaVConfig = field(default_factory=DeltaVConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "GenesisConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data.get("model", data))

    @classmethod
    def _from_dict(cls, data: dict) -> "GenesisConfig":
        """Recursively construct config from dict."""
        # This is a simplified version - full implementation would
        # recursively instantiate nested dataclasses
        return cls(
            name=data.get("name", "genesis"),
            version=data.get("version", "0.1.0"),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump({"model": data}, f, default_flow_style=False)
