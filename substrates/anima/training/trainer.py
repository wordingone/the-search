"""
ANIMA Trainer
=============

Unified training loop for all ANIMA variants.
"""

import os
import sys
import time
import json
import logging
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field
from tqdm import tqdm

# Local imports
from .config import Config, load_config, save_config
from .losses import TaskLoss, get_loss
from .schedulers import get_scheduler
from .checkpointing import CheckpointManager, load_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Tracks training state."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    early_stop_counter: int = 0
    train_loss_history: list = field(default_factory=list)
    val_loss_history: list = field(default_factory=list)


class Trainer:
    """
    Unified trainer for ANIMA architectures.

    Supports:
    - Multiple ANIMA variants (Zero, One, V2-V5)
    - Configurable loss functions
    - Learning rate scheduling with warmup
    - Checkpointing and resumption
    - Early stopping
    - Distributed training (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_dataloader: Iterator,
        val_dataloader: Optional[Iterator] = None,
        loss_fn: Optional[TaskLoss] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")

        self.model.to(self.device)

        # Setup loss
        self.loss_fn = loss_fn or get_loss('regression')

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Calculate total steps
        if config.training.max_steps:
            self.total_steps = config.training.max_steps
        else:
            # Estimate from epochs
            self.total_steps = config.training.epochs * len(train_dataloader)

        # Setup scheduler
        self.scheduler = get_scheduler(
            config.training.scheduler,
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            total_steps=self.total_steps,
        )

        # Setup checkpointing
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(
            self.output_dir / 'checkpoints',
            keep_last=5,
            keep_best=True,
            metric_name='val_loss',
            metric_mode='min',
        )

        # Save config
        save_config(config, self.output_dir / 'config.yaml')

        # Initialize state
        self.state = TrainerState()

        # Try to resume from checkpoint
        self._try_resume()

    def _try_resume(self):
        """Try to resume from latest checkpoint."""
        meta = self.checkpoint_manager.load_latest(
            self.model, self.optimizer, self.scheduler, str(self.device)
        )
        if meta:
            self.state.epoch = meta['epoch']
            self.state.global_step = meta['step']
            if 'metrics' in meta and meta['metrics']:
                self.state.best_metric = meta['metrics'].get('val_loss', float('inf'))
            logger.info(f"Resumed from epoch {self.state.epoch}, step {self.state.global_step}")

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Final metrics dictionary
        """
        logger.info(f"Starting training: {self.config.name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()
        start_epoch = self.state.epoch

        try:
            for epoch in range(start_epoch, self.config.training.epochs):
                self.state.epoch = epoch
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs}")
                logger.info(f"{'='*60}")

                # Train epoch
                train_metrics = self._train_epoch()
                self.state.train_loss_history.append(train_metrics['loss'])

                # Validate
                if self.val_dataloader is not None:
                    val_metrics = self._validate()
                    self.state.val_loss_history.append(val_metrics['loss'])
                    metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                else:
                    metrics = train_metrics

                # Log
                self._log_metrics(metrics, epoch)

                # Checkpoint
                self.checkpoint_manager.save(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.state.global_step, metrics, self.config
                )

                # Early stopping
                if self._check_early_stopping(metrics.get('val_loss', metrics['loss'])):
                    logger.info("Early stopping triggered")
                    break

                # Check max steps
                if self.config.training.max_steps and self.state.global_step >= self.config.training.max_steps:
                    logger.info("Max steps reached")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        # Final metrics
        total_time = time.time() - start_time
        final_metrics = {
            'epochs_completed': self.state.epoch + 1,
            'total_steps': self.state.global_step,
            'total_time_seconds': total_time,
            'final_train_loss': self.state.train_loss_history[-1] if self.state.train_loss_history else None,
            'final_val_loss': self.state.val_loss_history[-1] if self.state.val_loss_history else None,
            'best_val_loss': self.state.best_metric,
        }

        # Save final metrics
        with open(self.output_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"\nTraining complete!")
        logger.info(f"Total time: {total_time / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.state.best_metric:.4f}")

        return final_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc="Training", leave=False)

        for batch in pbar:
            # Move to device
            batch = self._to_device(batch)

            # Forward pass
            loss, metrics = self._training_step(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            # Track
            total_loss += loss.item()
            num_batches += 1
            self.state.global_step += 1

            # Update progress
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Logging
            if self.state.global_step % self.config.training.log_every == 0:
                self._log_step(loss.item(), metrics)

            # Eval during training
            if (self.val_dataloader is not None and
                self.state.global_step % self.config.training.eval_every == 0):
                val_metrics = self._validate()
                logger.info(f"Step {self.state.global_step} | Val Loss: {val_metrics['loss']:.4f}")
                self.model.train()  # Back to training mode

            # Check max steps
            if self.config.training.max_steps and self.state.global_step >= self.config.training.max_steps:
                break

        return {'loss': total_loss / max(num_batches, 1)}

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step."""
        # ANIMA models expect observation sequence
        if hasattr(self.model, 'reset'):
            # ANIMA variant
            self.model.reset(batch['input'].shape[0], self.device)

            outputs = []
            for t in range(batch['input'].shape[1]):
                result = self.model.step(batch['input'][:, t])
                outputs.append(result['action'])
            outputs = torch.stack(outputs, dim=1)
        else:
            # Standard model (transformer)
            outputs = self.model(batch['input'])

        # Compute loss
        loss = self.loss_fn(outputs, batch['target'])

        if isinstance(loss, dict):
            metrics = {k: v.item() for k, v in loss.items() if k != 'total'}
            loss = loss['total']
        else:
            metrics = {}

        return loss, metrics

    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
                batch = self._to_device(batch)
                loss, _ = self._training_step(batch)
                total_loss += loss.item()
                num_batches += 1

        return {'loss': total_loss / max(num_batches, 1)}

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _log_step(self, loss: float, metrics: Dict[str, float]):
        """Log training step."""
        lr = self.scheduler.get_last_lr()[0]
        msg = f"Step {self.state.global_step} | Loss: {loss:.4f} | LR: {lr:.2e}"
        if metrics:
            msg += " | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(msg)

    def _log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log epoch metrics."""
        logger.info(f"Epoch {epoch + 1} Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    def _check_early_stopping(self, current_metric: float) -> bool:
        """Check if should early stop."""
        if current_metric < self.state.best_metric - self.config.training.early_stopping_min_delta:
            self.state.best_metric = current_metric
            self.state.early_stop_counter = 0
        else:
            self.state.early_stop_counter += 1

        return self.state.early_stop_counter >= self.config.training.early_stopping_patience


def create_model(config: Config) -> nn.Module:
    """Create model from config."""
    # Import ANIMA variants
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    variant = config.model.variant.lower()

    if variant == 'anima_zero':
        from anima.core.anima_zero import ANIMA, ANIMAConfig
        model_config = ANIMAConfig(
            sensory_dim=config.model.sensory_dim,
            output_dim=config.model.output_dim,
            world_dim=config.model.d_model,
            internal_dim=config.model.d_model,
            action_dim=config.model.d_model,
            spectral_radius=config.model.spectral_radius,
        )
        return ANIMA(model_config)

    elif variant == 'anima_one':
        from anima.core.anima_one import ANIMA1, ANIMA1Config
        model_config = ANIMA1Config(
            sensory_dim=config.model.sensory_dim,
            output_dim=config.model.output_dim,
            d_model=config.model.d_model,
            d_bottleneck=config.model.d_bottleneck,
            d_state=config.model.d_model,
            chunk_size=config.model.chunk_size,
            spectral_radius=config.model.spectral_radius,
        )
        return ANIMA1(model_config)

    elif variant == 'transformer':
        from anima.tests.fair_benchmark import TransformerMatched
        return TransformerMatched(
            sensory_dim=config.model.sensory_dim,
            output_dim=config.model.output_dim,
            target_params=config.model.target_params,
        )

    else:
        raise ValueError(f"Unknown model variant: {variant}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ANIMA Trainer")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--override', type=str, nargs='*', help="Override config values (key=value)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.override:
        from .config import merge_configs
        overrides = {}
        for ov in args.override:
            key, value = ov.split('=')
            # Try to parse value
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            overrides[key] = value
        config = merge_configs(config, overrides)

    # Create model
    model = create_model(config)
    logger.info(f"Created model: {config.model.variant}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy dataloaders for testing
    # In production, these would come from anima/data/
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = torch.randn(1000, 10, config.model.sensory_dim)
    dummy_targets = torch.randn(1000, 10, config.model.output_dim)
    dummy_dataset = TensorDataset(dummy_data, dummy_targets)

    def collate_fn(batch):
        inputs, targets = zip(*batch)
        return {
            'input': torch.stack(inputs),
            'target': torch.stack(targets),
        }

    train_loader = DataLoader(
        dummy_dataset, batch_size=config.training.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dummy_dataset, batch_size=config.training.batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader)

    # Train
    metrics = trainer.train()
    print(f"\nFinal metrics: {metrics}")


if __name__ == '__main__':
    main()
