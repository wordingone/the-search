"""
ANIMA Data Loading
==================

Streaming data loaders for large-scale training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, List, Callable
from dataclasses import dataclass
import random
import json


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset: str = "synthetic"
    data_path: Optional[str] = None
    batch_size: int = 64
    seq_len: int = 32
    sensory_dim: int = 8
    output_dim: int = 4
    num_workers: int = 4
    prefetch: int = 2
    streaming: bool = True
    shuffle: bool = True
    max_samples: Optional[int] = None


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing ANIMA training.

    Generates tasks similar to the benchmark suite:
    - Sequence prediction
    - Pattern recognition
    - Physics simulation
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 32,
        sensory_dim: int = 8,
        output_dim: int = 4,
        task: str = 'mixed',
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.sensory_dim = sensory_dim
        self.output_dim = output_dim
        self.task = task
        self.rng = np.random.RandomState(seed)

        # Pre-generate data
        self._generate_data()

    def _generate_data(self):
        """Generate synthetic data."""
        self.inputs = []
        self.targets = []

        for i in range(self.num_samples):
            if self.task == 'mixed':
                task = self.rng.choice(['sequence', 'pattern', 'physics'])
            else:
                task = self.task

            inp, tgt = self._generate_sample(task)
            self.inputs.append(inp)
            self.targets.append(tgt)

        self.inputs = np.stack(self.inputs)
        self.targets = np.stack(self.targets)

    def _generate_sample(self, task: str) -> tuple:
        """Generate single sample."""
        if task == 'sequence':
            return self._gen_sequence()
        elif task == 'pattern':
            return self._gen_pattern()
        elif task == 'physics':
            return self._gen_physics()
        else:
            return self._gen_sequence()

    def _gen_sequence(self) -> tuple:
        """Arithmetic sequence prediction."""
        start = self.rng.uniform(0, 1)
        step = self.rng.uniform(0.01, 0.1)

        # Generate sequence
        seq = np.array([start + i * step for i in range(self.seq_len + 1)])
        seq = seq % 1.0  # Keep in [0, 1]

        # Pad to sensory_dim
        inp = np.zeros((self.seq_len, self.sensory_dim))
        inp[:, 0] = seq[:-1]

        # Target: predict next value
        tgt = np.zeros((self.seq_len, self.output_dim))
        tgt[:, 0] = seq[1:]

        return inp.astype(np.float32), tgt.astype(np.float32)

    def _gen_pattern(self) -> tuple:
        """Repeating pattern recognition."""
        pattern_len = self.rng.randint(2, 6)
        pattern = self.rng.uniform(0, 1, size=pattern_len)

        # Repeat pattern
        repeats = (self.seq_len + 1) // pattern_len + 1
        full_seq = np.tile(pattern, repeats)[:self.seq_len + 1]

        # Pad to sensory_dim
        inp = np.zeros((self.seq_len, self.sensory_dim))
        inp[:, 0] = full_seq[:-1]

        # Target: predict next in pattern
        tgt = np.zeros((self.seq_len, self.output_dim))
        tgt[:, 0] = full_seq[1:]

        return inp.astype(np.float32), tgt.astype(np.float32)

    def _gen_physics(self) -> tuple:
        """Simple physics simulation (projectile motion)."""
        v0 = self.rng.uniform(0.5, 2.0)
        angle = self.rng.uniform(0.3, 0.7)  # radians
        g = 0.1

        # Trajectory
        t = np.linspace(0, 2, self.seq_len + 1)
        x = v0 * np.cos(angle) * t
        y = v0 * np.sin(angle) * t - 0.5 * g * t ** 2
        y = np.maximum(y, 0)  # Ground at y=0

        # Normalize
        x = x / x.max() if x.max() > 0 else x
        y = y / max(y.max(), 0.01)

        # Input: current position
        inp = np.zeros((self.seq_len, self.sensory_dim))
        inp[:, 0] = x[:-1]
        inp[:, 1] = y[:-1]

        # Target: next position
        tgt = np.zeros((self.seq_len, self.output_dim))
        tgt[:, 0] = x[1:]
        tgt[:, 1] = y[1:]

        return inp.astype(np.float32), tgt.astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input': torch.from_numpy(self.inputs[idx]),
            'target': torch.from_numpy(self.targets[idx]),
        }


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for large-scale data.

    Loads data on-the-fly without keeping everything in memory.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 32,
        sensory_dim: int = 8,
        output_dim: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.sensory_dim = sensory_dim
        self.output_dim = output_dim
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples

        # Find all data files
        self.files = list(self.data_path.glob("*.json")) + \
                     list(self.data_path.glob("*.jsonl")) + \
                     list(self.data_path.glob("*.npy"))

        if not self.files:
            raise ValueError(f"No data files found in {data_path}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()

        # Handle multi-worker
        if worker_info is not None:
            # Split files among workers
            per_worker = len(self.files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
            files = self.files[start:end]
        else:
            files = self.files

        # Shuffle files
        if self.shuffle:
            rng = random.Random(self.seed)
            files = files.copy()
            rng.shuffle(files)

        count = 0
        for file_path in files:
            for sample in self._load_file(file_path):
                yield sample
                count += 1
                if self.max_samples and count >= self.max_samples:
                    return

    def _load_file(self, path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Load and yield samples from a file."""
        if path.suffix == '.npy':
            data = np.load(path)
            for i in range(len(data)):
                yield self._process_sample(data[i])
        elif path.suffix in ['.json', '.jsonl']:
            with open(path, 'r') as f:
                if path.suffix == '.jsonl':
                    for line in f:
                        sample = json.loads(line)
                        yield self._process_sample(sample)
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for sample in data:
                            yield self._process_sample(sample)
                    else:
                        yield self._process_sample(data)

    def _process_sample(self, sample: Any) -> Dict[str, torch.Tensor]:
        """Process raw sample into model input."""
        if isinstance(sample, np.ndarray):
            # Assume [seq_len, features] format
            inp = torch.from_numpy(sample[:self.seq_len]).float()
            tgt = torch.from_numpy(sample[1:self.seq_len + 1]).float()
        elif isinstance(sample, dict):
            inp = torch.tensor(sample.get('input', sample.get('obs', [])), dtype=torch.float32)
            tgt = torch.tensor(sample.get('target', sample.get('label', [])), dtype=torch.float32)
        else:
            raise ValueError(f"Unknown sample type: {type(sample)}")

        # Pad/truncate to expected dimensions
        if inp.shape[-1] < self.sensory_dim:
            inp = torch.nn.functional.pad(inp, (0, self.sensory_dim - inp.shape[-1]))
        if tgt.shape[-1] < self.output_dim:
            tgt = torch.nn.functional.pad(tgt, (0, self.output_dim - tgt.shape[-1]))

        return {'input': inp[:, :self.sensory_dim], 'target': tgt[:, :self.output_dim]}


class StreamingDataLoader:
    """
    Wrapper for streaming data loading with prefetch.
    """

    def __init__(
        self,
        config: DataConfig,
        split: str = 'train',
    ):
        self.config = config
        self.split = split

        # Create dataset
        if config.dataset == 'synthetic':
            self.dataset = SyntheticDataset(
                num_samples=config.max_samples or 10000,
                seq_len=config.seq_len,
                sensory_dim=config.sensory_dim,
                output_dim=config.output_dim,
                seed=42 if split == 'train' else 123,
            )
        elif config.data_path:
            self.dataset = StreamingDataset(
                data_path=config.data_path,
                seq_len=config.seq_len,
                sensory_dim=config.sensory_dim,
                output_dim=config.output_dim,
                shuffle=(split == 'train'),
                max_samples=config.max_samples,
            )
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=(split == 'train' and not config.streaming),
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch if config.num_workers > 0 else None,
            pin_memory=True,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        if hasattr(self.dataset, '__len__'):
            return len(self.dataset) // self.config.batch_size
        return None  # Unknown for streaming


def get_dataloaders(
    config: DataConfig,
    val_ratio: float = 0.1,
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        config: Data configuration
        val_ratio: Fraction of data for validation

    Returns:
        (train_loader, val_loader)
    """
    train_config = DataConfig(**vars(config))
    val_config = DataConfig(**vars(config))
    val_config.shuffle = False

    if config.max_samples:
        train_samples = int(config.max_samples * (1 - val_ratio))
        val_samples = config.max_samples - train_samples
        train_config.max_samples = train_samples
        val_config.max_samples = val_samples

    train_loader = StreamingDataLoader(train_config, split='train')
    val_loader = StreamingDataLoader(val_config, split='val')

    return train_loader, val_loader
