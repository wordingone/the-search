"""
ANIMA Checkpointing
===================

Save and load model checkpoints for training resumption.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Any] = None,
    **extra,
) -> str:
    """
    Save training checkpoint.

    Args:
        path: Directory or file path for checkpoint
        model: Model to save
        optimizer: Optimizer state (optional)
        scheduler: LR scheduler state (optional)
        epoch: Current epoch
        step: Current global step
        metrics: Current metrics (optional)
        config: Training config (optional)
        **extra: Additional data to save

    Returns:
        Full path to saved checkpoint
    """
    path = Path(path)

    # If directory, create filename
    if path.is_dir() or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        path = path / filename
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dict
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    if config is not None:
        # Handle dataclass configs
        if hasattr(config, '__dataclass_fields__'):
            from dataclasses import asdict
            checkpoint['config'] = asdict(config)
        else:
            checkpoint['config'] = config

    # Add extra data
    checkpoint.update(extra)

    # Save
    torch.save(checkpoint, path)

    # Save metadata as JSON for easy inspection
    meta_path = path.with_suffix('.json')
    meta = {k: v for k, v in checkpoint.items()
            if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return str(path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda',
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint to
        strict: Whether to strictly enforce state dict keys

    Returns:
        Checkpoint metadata (epoch, step, metrics, etc.)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)

    # Load model
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Return metadata
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', None),
        'timestamp': checkpoint.get('timestamp', None),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last: int = 5,
    keep_best: bool = True,
) -> int:
    """
    Remove old checkpoints, keeping only recent and best.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
        keep_best: Whether to also keep 'best_checkpoint.pt'

    Returns:
        Number of checkpoints deleted
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return 0

    # Find all checkpoints except 'best'
    checkpoints = [p for p in checkpoint_dir.glob("checkpoint_*.pt")
                   if 'best' not in p.name]

    if len(checkpoints) <= keep_last:
        return 0

    # Sort by modification time (oldest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime)

    # Delete old ones
    to_delete = checkpoints[:-keep_last]
    deleted = 0

    for ckpt in to_delete:
        ckpt.unlink()
        # Also delete metadata JSON if exists
        meta = ckpt.with_suffix('.json')
        if meta.exists():
            meta.unlink()
        deleted += 1

    return deleted


class CheckpointManager:
    """Manages checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last: int = 5,
        keep_best: bool = True,
        metric_name: str = 'loss',
        metric_mode: str = 'min',  # 'min' or 'max'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.best_metric = float('inf') if metric_mode == 'min' else float('-inf')

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Any = None,
    ) -> str:
        """Save checkpoint and manage old ones."""
        # Save regular checkpoint
        path = save_checkpoint(
            self.checkpoint_dir,
            model, optimizer, scheduler,
            epoch, step, metrics, config
        )

        # Check if this is the best
        if self.keep_best and self.metric_name in metrics:
            current = metrics[self.metric_name]
            is_best = (
                (self.metric_mode == 'min' and current < self.best_metric) or
                (self.metric_mode == 'max' and current > self.best_metric)
            )
            if is_best:
                self.best_metric = current
                best_path = self.checkpoint_dir / 'best_checkpoint.pt'
                save_checkpoint(
                    best_path,
                    model, optimizer, scheduler,
                    epoch, step, metrics, config,
                    is_best=True
                )

        # Cleanup old checkpoints
        cleanup_old_checkpoints(self.checkpoint_dir, self.keep_last, self.keep_best)

        return path

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
    ) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint if exists."""
        latest = find_latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            return None
        return load_checkpoint(latest, model, optimizer, scheduler, device)

    def load_best(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
    ) -> Optional[Dict[str, Any]]:
        """Load best checkpoint if exists."""
        best_path = self.checkpoint_dir / 'best_checkpoint.pt'
        if not best_path.exists():
            return None
        return load_checkpoint(best_path, model, optimizer, scheduler, device)
