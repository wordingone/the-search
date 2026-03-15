"""
ANIMA Training Configuration
============================

YAML-based configuration system for training runs.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    variant: str = "anima_zero"  # anima_zero, anima_one, transformer
    sensory_dim: int = 8
    output_dim: int = 4
    target_params: int = 25000
    # Variant-specific
    d_model: int = 32
    d_bottleneck: int = 16
    chunk_size: int = 4
    spectral_radius: float = 0.99


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    epochs: int = 100
    max_steps: Optional[int] = None  # If set, overrides epochs
    lr: float = 0.001
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    scheduler: str = "cosine"  # cosine, linear, constant
    # Checkpointing
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset: str = "synthetic"  # synthetic, pile, laion, webvid, etc.
    data_path: Optional[str] = None
    streaming: bool = True
    prefetch: int = 4
    num_workers: int = 4
    max_seq_len: int = 512
    # Splits
    train_samples: Optional[int] = None  # None = use all
    val_samples: int = 1000
    test_samples: int = 1000


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    benchmarks: List[str] = field(default_factory=lambda: ["reasoning", "physics"])
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    save_predictions: bool = False


@dataclass
class Config:
    """Complete training configuration."""
    # Experiment metadata
    name: str = "anima_training"
    seed: int = 42
    device: str = "cuda"  # cuda, cpu, auto
    output_dir: str = "outputs"

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Distributed
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Parse nested configs
    config = Config(
        name=data.get('name', 'anima_training'),
        seed=data.get('seed', 42),
        device=data.get('device', 'cuda'),
        output_dir=data.get('output_dir', 'outputs'),
        distributed=data.get('distributed', False),
        world_size=data.get('world_size', 1),
        local_rank=data.get('local_rank', 0),
    )

    if 'model' in data:
        config.model = ModelConfig(**data['model'])
    if 'training' in data:
        config.training = TrainingConfig(**data['training'])
    if 'data' in data:
        config.data = DataConfig(**data['data'])
    if 'eval' in data:
        config.eval = EvalConfig(**data['eval'])

    return config


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    data = {
        'name': config.name,
        'seed': config.seed,
        'device': config.device,
        'output_dir': config.output_dir,
        'distributed': config.distributed,
        'world_size': config.world_size,
        'local_rank': config.local_rank,
        'model': asdict(config.model),
        'training': asdict(config.training),
        'data': asdict(config.data),
        'eval': asdict(config.eval),
    }

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Config, overrides: Dict[str, Any]) -> Config:
    """Merge override dict into base config."""
    import copy
    config = copy.deepcopy(base)

    for key, value in overrides.items():
        if '.' in key:
            # Nested key like "training.lr"
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)

    return config
