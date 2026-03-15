"""
ANIMA Learning Rate Schedulers
==============================

Learning rate scheduling strategies.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class CosineAnnealingWarmup(_LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class LinearWarmupDecay(_LRScheduler):
    """Linear warmup followed by linear decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            decay_factor = 1.0 - progress
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]


class ConstantWithWarmup(_LRScheduler):
    """Constant learning rate with linear warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs


class PolynomialDecay(_LRScheduler):
    """Polynomial decay with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            decay_factor = (1.0 - progress) ** self.power
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    warmup_steps: int = 100,
    total_steps: int = 10000,
    min_lr: float = 0.0,
    **kwargs,
) -> _LRScheduler:
    """Factory function for learning rate schedulers."""

    schedulers = {
        'cosine': lambda: CosineAnnealingWarmup(
            optimizer, warmup_steps, total_steps, min_lr
        ),
        'linear': lambda: LinearWarmupDecay(
            optimizer, warmup_steps, total_steps, min_lr
        ),
        'constant': lambda: ConstantWithWarmup(optimizer, warmup_steps),
        'polynomial': lambda: PolynomialDecay(
            optimizer, warmup_steps, total_steps,
            power=kwargs.get('power', 1.0), min_lr=min_lr
        ),
    }

    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")

    return schedulers[name]()
