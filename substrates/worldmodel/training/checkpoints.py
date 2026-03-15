"""Checkpoint management for Genesis training."""

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import shutil
import os


class CheckpointManager:
    """
    Manage training checkpoints with automatic cleanup.

    Features:
    - Save/load model, optimizer, scheduler, and training state
    - Keep only last N checkpoints
    - Best model tracking
    - Distributed training support
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        keep_best_n: int = 3,
        metric_name: str = "loss",
        metric_mode: str = "min",
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            keep_best_n: Number of best checkpoints to keep
            metric_name: Name of metric for best model selection
            metric_mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        # Track checkpoints
        self.history_file = self.checkpoint_dir / "checkpoint_history.json"
        self.history = self._load_history()

    def _load_history(self) -> Dict[str, Any]:
        """Load checkpoint history from disk."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {
            "last_checkpoints": [],
            "best_checkpoints": [],
            "latest": None,
        }

    def _save_history(self) -> None:
        """Save checkpoint history to disk."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            extra: Extra data to save
            is_best: Mark as best checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Build checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state": self._get_state_dict(model),
        }

        if optimizer is not None:
            checkpoint["optimizer_state"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state"] = scheduler.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if extra is not None:
            checkpoint.update(extra)

        # Save checkpoint
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        # Update history
        self._add_to_history(filepath, metrics, is_best)
        self._cleanup()

        return filepath

    def _get_state_dict(self, model: nn.Module) -> Dict[str, Any]:
        """Get model state dict, handling DDP wrapper."""
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    def _add_to_history(
        self,
        filepath: Path,
        metrics: Optional[Dict[str, float]],
        is_best: bool,
    ) -> None:
        """Add checkpoint to history."""
        entry = {
            "path": str(filepath),
            "metrics": metrics,
        }

        # Add to last checkpoints
        self.history["last_checkpoints"].append(entry)
        self.history["latest"] = str(filepath)

        # Check if best
        if is_best or self._is_best(metrics):
            self.history["best_checkpoints"].append(entry)
            self._sort_best()

        self._save_history()

    def _is_best(self, metrics: Optional[Dict[str, float]]) -> bool:
        """Check if current metrics are best."""
        if metrics is None or self.metric_name not in metrics:
            return False

        current = metrics[self.metric_name]
        best_checkpoints = self.history["best_checkpoints"]

        if not best_checkpoints:
            return True

        # Get best metric value
        best_metrics = [c["metrics"] for c in best_checkpoints if c["metrics"]]
        if not best_metrics:
            return True

        best_values = [m[self.metric_name] for m in best_metrics if self.metric_name in m]
        if not best_values:
            return True

        if self.metric_mode == "min":
            return current < min(best_values)
        else:
            return current > max(best_values)

    def _sort_best(self) -> None:
        """Sort best checkpoints by metric."""
        def get_metric(entry):
            if entry["metrics"] is None:
                return float("inf") if self.metric_mode == "min" else float("-inf")
            return entry["metrics"].get(self.metric_name, float("inf") if self.metric_mode == "min" else float("-inf"))

        reverse = self.metric_mode == "max"
        self.history["best_checkpoints"].sort(key=get_metric, reverse=reverse)

    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping best and last N."""
        # Files to keep
        keep_paths = set()

        # Keep last N
        last_n = self.history["last_checkpoints"][-self.keep_last_n:]
        for entry in last_n:
            keep_paths.add(entry["path"])
        self.history["last_checkpoints"] = last_n

        # Keep best N
        best_n = self.history["best_checkpoints"][:self.keep_best_n]
        for entry in best_n:
            keep_paths.add(entry["path"])
        self.history["best_checkpoints"] = best_n

        # Delete old checkpoints
        for filepath in self.checkpoint_dir.glob("checkpoint_*.pt"):
            if str(filepath) not in keep_paths:
                try:
                    os.remove(filepath)
                except OSError:
                    pass

        self._save_history()

    def load(
        self,
        path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            path: Path to checkpoint (uses latest if None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load onto
            strict: Strict state dict loading

        Returns:
            Checkpoint dict with epoch, step, metrics
        """
        if path is None:
            path = self.history.get("latest")
            if path is None:
                raise ValueError("No checkpoint found")

        checkpoint = torch.load(path, map_location=device)

        if model is not None:
            self._load_model_state(model, checkpoint["model_state"], strict)

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint

    def _load_model_state(
        self,
        model: nn.Module,
        state_dict: Dict[str, Any],
        strict: bool,
    ) -> None:
        """Load model state dict, handling DDP wrapper."""
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)

    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        return self.history.get("latest")

    def get_best(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if self.history["best_checkpoints"]:
            return self.history["best_checkpoints"][0]["path"]
        return None

    def list_checkpoints(self) -> List[str]:
        """List all tracked checkpoints."""
        paths = []
        for entry in self.history["last_checkpoints"]:
            paths.append(entry["path"])
        return paths

    def cleanup(self, keep_last_n: Optional[int] = None) -> int:
        """
        Clean up old checkpoints.

        Args:
            keep_last_n: Override for number to keep

        Returns:
            Number of checkpoints deleted
        """
        old_keep = self.keep_last_n
        if keep_last_n is not None:
            self.keep_last_n = keep_last_n

        before = len(list(self.checkpoint_dir.glob("checkpoint_*.pt")))
        self._cleanup()
        after = len(list(self.checkpoint_dir.glob("checkpoint_*.pt")))

        self.keep_last_n = old_keep
        return before - after
