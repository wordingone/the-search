"""
ANIMA Evaluation Metrics
========================

Standardized metrics for model evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.2,
    mode: str = 'regression',
) -> float:
    """
    Compute accuracy for predictions.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        threshold: Tolerance for regression accuracy
        mode: 'regression' (within threshold) or 'classification' (argmax)

    Returns:
        Accuracy as float in [0, 1]
    """
    # Ensure same shape
    if predictions.shape != targets.shape:
        # Take first dim of predictions if mismatch
        if predictions.dim() > targets.dim():
            predictions = predictions[..., 0]
        elif targets.dim() > predictions.dim():
            targets = targets[..., 0]
        # Truncate to smaller size
        min_size = min(predictions.shape[-1], targets.shape[-1])
        predictions = predictions[..., :min_size]
        targets = targets[..., :min_size]

    if mode == 'classification':
        if predictions.dim() > 1 and predictions.shape[-1] > 1:
            pred_classes = predictions.argmax(dim=-1)
        else:
            pred_classes = (predictions.squeeze() > 0.5).long()
        target_classes = targets.long().squeeze() if targets.dim() > 0 else targets.long()

        return (pred_classes == target_classes).float().mean().item()

    else:  # regression
        diff = torch.abs(predictions - targets)
        # Use scalar threshold comparison
        tol = threshold * torch.abs(targets).clamp(min=0.1).mean().item()
        tol = max(tol, threshold)
        correct = (diff < tol).float()
        return correct.mean().item()


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean squared error."""
    return torch.nn.functional.mse_loss(predictions, targets).item()


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error."""
    return torch.nn.functional.l1_loss(predictions, targets).item()


def perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity for language modeling.

    Args:
        logits: [batch, seq, vocab] logits
        targets: [batch, seq] target token indices
        ignore_index: Index to ignore in loss computation

    Returns:
        Perplexity (exp of cross-entropy loss)
    """
    # Flatten
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)

    # Cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )

    return torch.exp(loss).item()


def retrieval_recall(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute retrieval recall at K.

    Args:
        query_embeddings: [num_queries, dim]
        key_embeddings: [num_keys, dim]
        k: Top-k to consider

    Returns:
        Dict with R@1, R@5, R@K
    """
    # Normalize
    query_norm = torch.nn.functional.normalize(query_embeddings, dim=-1)
    key_norm = torch.nn.functional.normalize(key_embeddings, dim=-1)

    # Similarity matrix
    sim = torch.mm(query_norm, key_norm.T)  # [num_queries, num_keys]

    # Get rankings
    _, indices = sim.topk(k, dim=-1)

    # Compute recalls (assuming diagonal is correct)
    num_queries = query_embeddings.shape[0]
    correct_indices = torch.arange(num_queries, device=query_embeddings.device)

    recall_at_1 = (indices[:, 0] == correct_indices).float().mean().item()
    recall_at_5 = (indices[:, :5] == correct_indices.unsqueeze(1)).any(dim=1).float().mean().item()
    recall_at_k = (indices == correct_indices.unsqueeze(1)).any(dim=1).float().mean().item()

    return {
        'R@1': recall_at_1,
        'R@5': recall_at_5,
        f'R@{k}': recall_at_k,
    }


def cosine_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
) -> float:
    """Average cosine similarity between two sets of embeddings."""
    norm1 = torch.nn.functional.normalize(embeddings1, dim=-1)
    norm2 = torch.nn.functional.normalize(embeddings2, dim=-1)
    return (norm1 * norm2).sum(dim=-1).mean().item()


def direction_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.7,
) -> float:
    """
    Accuracy for directional predictions (e.g., goal navigation).

    Checks if predicted direction aligns with target direction.
    """
    # Normalize to unit vectors
    pred_norm = torch.nn.functional.normalize(predictions, dim=-1)
    target_norm = torch.nn.functional.normalize(targets, dim=-1)

    # Cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=-1)

    # Accuracy is fraction above threshold
    return (cos_sim > threshold).float().mean().item()


def task_specific_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
) -> float:
    """
    Compute task-specific accuracy.

    Args:
        predictions: Model predictions
        targets: Ground truth
        task_type: One of 'sequence', 'pattern', 'conditional', 'analogy',
                   'projectile', 'collision', 'goal', 'momentum'

    Returns:
        Accuracy for the specific task type
    """
    if task_type in ['sequence', 'pattern', 'conditional', 'analogy', 'projectile', 'momentum']:
        # Regression tasks with 20% tolerance
        return accuracy(predictions, targets, threshold=0.2, mode='regression')

    elif task_type == 'collision':
        # Binary classification
        return accuracy(predictions, targets, mode='classification')

    elif task_type == 'goal':
        # Directional accuracy
        return direction_accuracy(predictions, targets, threshold=0.7)

    else:
        # Default to regression
        return accuracy(predictions, targets, threshold=0.2, mode='regression')


class MetricTracker:
    """Tracks metrics across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._values = defaultdict(list)

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self._values[k].append(v)

    def compute(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self._values.items()}

    def __str__(self) -> str:
        metrics = self.compute()
        return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
