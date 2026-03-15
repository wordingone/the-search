"""
ANIMA Loss Functions
====================

Task-specific loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TaskLoss(nn.Module):
    """Base class for task-specific losses."""

    def __init__(self, name: str = "base"):
        super().__init__()
        self.name = name

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class CrossEntropyLoss(TaskLoss):
    """Cross-entropy loss for classification tasks."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__("cross_entropy")
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: [batch, ..., num_classes] logits
            target: [batch, ...] class indices
            mask: [batch, ...] optional mask (1 = valid, 0 = ignore)
        """
        # Flatten for loss computation
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1)

        loss = F.cross_entropy(
            pred_flat, target_flat,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )

        if mask is not None:
            mask_flat = mask.reshape(-1)
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class MSELoss(TaskLoss):
    """Mean squared error for regression tasks."""

    def __init__(self):
        super().__init__("mse")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.mse_loss(pred, target, reduction='none')

        if mask is not None:
            loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * pred.shape[-1] + 1e-8)
        else:
            loss = loss.mean()

        return loss


class ContrastiveLoss(TaskLoss):
    """Contrastive loss for retrieval/matching tasks."""

    def __init__(self, temperature: float = 0.07):
        super().__init__("contrastive")
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.

        Args:
            anchor: [batch, dim] anchor embeddings
            positive: [batch, dim] positive embeddings
            negative: [batch, num_neg, dim] optional negative embeddings
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)

        # Positive similarity
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature

        if negative is not None:
            negative = F.normalize(negative, dim=-1)
            # [batch, num_neg]
            neg_sim = torch.bmm(negative, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature
            # [batch, 1 + num_neg]
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # In-batch negatives
            # [batch, batch]
            sim_matrix = torch.mm(anchor, positive.T) / self.temperature
            labels = torch.arange(anchor.shape[0], device=anchor.device)
            loss = F.cross_entropy(sim_matrix, labels)

        return loss


class PredictionLoss(TaskLoss):
    """Combined prediction loss for ANIMA (next observation + action)."""

    def __init__(self, obs_weight: float = 1.0, action_weight: float = 1.0):
        super().__init__("prediction")
        self.obs_weight = obs_weight
        self.action_weight = action_weight

    def forward(self, pred_obs: torch.Tensor, target_obs: torch.Tensor,
                pred_action: torch.Tensor, target_action: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Combined loss for observation prediction and action matching.
        """
        obs_loss = F.mse_loss(pred_obs, target_obs, reduction='none')
        action_loss = F.mse_loss(pred_action, target_action, reduction='none')

        if mask is not None:
            obs_loss = (obs_loss * mask.unsqueeze(-1)).sum() / (mask.sum() * pred_obs.shape[-1] + 1e-8)
            action_loss = (action_loss * mask.unsqueeze(-1)).sum() / (mask.sum() * pred_action.shape[-1] + 1e-8)
        else:
            obs_loss = obs_loss.mean()
            action_loss = action_loss.mean()

        total = self.obs_weight * obs_loss + self.action_weight * action_loss

        return {
            'total': total,
            'obs_loss': obs_loss,
            'action_loss': action_loss,
        }


class LanguageModelingLoss(TaskLoss):
    """Autoregressive language modeling loss."""

    def __init__(self, vocab_size: int, label_smoothing: float = 0.0):
        super().__init__("lm")
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [batch, seq, vocab_size]
            targets: [batch, seq] token indices
            mask: [batch, seq] attention mask
        """
        # Shift for autoregressive
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            label_smoothing=self.label_smoothing,
            reduction='none'
        )

        if mask is not None:
            shift_mask = mask[:, 1:].contiguous().view(-1)
            loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        # Perplexity
        perplexity = torch.exp(loss)

        return {
            'loss': loss,
            'perplexity': perplexity,
        }


def get_loss(task_type: str, **kwargs) -> TaskLoss:
    """Factory function for loss functions."""
    losses = {
        'classification': CrossEntropyLoss,
        'regression': MSELoss,
        'contrastive': ContrastiveLoss,
        'prediction': PredictionLoss,
        'lm': LanguageModelingLoss,
    }

    if task_type not in losses:
        raise ValueError(f"Unknown loss type: {task_type}. Available: {list(losses.keys())}")

    return losses[task_type](**kwargs)
