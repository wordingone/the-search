"""
ANIMA Training Infrastructure
=============================

Unified training framework for all ANIMA variants on real-world datasets.
"""

from .trainer import Trainer, TrainerState
from .data_loader import StreamingDataLoader, DataConfig
from .config import load_config, save_config
from .losses import TaskLoss, CrossEntropyLoss, MSELoss, ContrastiveLoss
from .schedulers import get_scheduler
from .checkpointing import save_checkpoint, load_checkpoint

__all__ = [
    'Trainer',
    'TrainerConfig',
    'StreamingDataLoader',
    'DataConfig',
    'load_config',
    'save_config',
    'TaskLoss',
    'CrossEntropyLoss',
    'MSELoss',
    'ContrastiveLoss',
    'get_scheduler',
    'save_checkpoint',
    'load_checkpoint',
]
