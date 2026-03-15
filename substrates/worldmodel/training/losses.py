"""Loss functions for Genesis training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np


class ChamferLoss(nn.Module):
    """
    Bidirectional Chamfer distance for sparse voxel coordinates.

    Measures the average nearest-neighbor distance between predicted
    and ground truth point sets.
    """

    def __init__(self, symmetric: bool = True):
        super().__init__()
        self.symmetric = symmetric

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute Chamfer distance.

        Args:
            pred: [N, 3] predicted coordinates
            target: [M, 3] target coordinates

        Returns:
            Chamfer distance (scalar)
        """
        if pred.shape[0] == 0 or target.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred = pred.float()
        target = target.float()

        # Compute pairwise distances [N, M]
        dist = torch.cdist(pred, target, p=2)

        # Forward: min distance from pred to target
        forward_dist = dist.min(dim=1).values.mean()

        if self.symmetric:
            # Backward: min distance from target to pred
            backward_dist = dist.min(dim=0).values.mean()
            return (forward_dist + backward_dist) / 2
        else:
            return forward_dist


class HungarianMatcher(nn.Module):
    """
    Optimal bipartite matching using Hungarian algorithm.

    Used for assigning predicted deltas to ground truth for loss computation.
    """

    def __init__(
        self,
        cost_coord: float = 1.0,
        cost_feature: float = 1.0,
        cost_class: float = 1.0,
    ):
        super().__init__()
        self.cost_coord = cost_coord
        self.cost_feature = cost_feature
        self.cost_class = cost_class

    @torch.no_grad()
    def forward(
        self,
        pred_coords: Tensor,
        pred_features: Tensor,
        pred_classes: Tensor,
        target_coords: Tensor,
        target_features: Tensor,
        target_classes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute optimal assignment between predictions and targets.

        Returns:
            pred_idx: [K] indices into predictions
            target_idx: [K] corresponding indices into targets
        """
        if pred_coords.shape[0] == 0 or target_coords.shape[0] == 0:
            device = pred_coords.device
            return torch.empty(0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        # Compute cost matrix
        # Coordinate cost
        coord_cost = torch.cdist(
            pred_coords.float(), target_coords.float(), p=2
        )

        # Feature cost
        feature_cost = torch.cdist(
            pred_features.float(), target_features.float(), p=1
        )

        # Class cost (cross-entropy based)
        pred_probs = F.softmax(pred_classes, dim=-1) if pred_classes.dim() > 1 else F.one_hot(pred_classes, 3).float()
        target_onehot = F.one_hot(target_classes.long(), 3).float()
        class_cost = -torch.matmul(pred_probs, target_onehot.T)

        # Combined cost
        C = (
            self.cost_coord * coord_cost +
            self.cost_feature * feature_cost +
            self.cost_class * class_cost
        )

        # Hungarian algorithm (on CPU for scipy)
        C_np = C.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(C_np)

        return (
            torch.tensor(row_ind, dtype=torch.long, device=pred_coords.device),
            torch.tensor(col_ind, dtype=torch.long, device=pred_coords.device),
        )


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss using pretrained VGG.

    Uses the LPIPS library for more accurate perceptual similarity.
    Falls back to VGG perceptual loss if LPIPS not available.
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        self._lpips = None
        self._net = net
        self._initialized = False

    def _init_lpips(self, device):
        if self._initialized:
            return

        try:
            import lpips
            self._lpips = lpips.LPIPS(net=self._net, verbose=False).to(device)
            self._lpips.requires_grad_(False)
        except ImportError:
            # Fallback to VGG perceptual loss
            self._lpips = None

        self._initialized = True

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute LPIPS distance.

        Args:
            pred: [B, 3, H, W] predicted images (normalized to [-1, 1])
            target: [B, 3, H, W] target images (normalized to [-1, 1])

        Returns:
            LPIPS distance (scalar)
        """
        self._init_lpips(pred.device)

        # Convert from [0, 1] to [-1, 1] if needed
        if pred.min() >= 0:
            pred = pred * 2 - 1
            target = target * 2 - 1

        if self._lpips is not None:
            return self._lpips(pred, target).mean()
        else:
            # Fallback: simple L1 in feature space
            return F.l1_loss(pred, target)


class MultiViewLoss(nn.Module):
    """
    Cross-view consistency loss for 3D reconstruction.

    Ensures voxel features are consistent across multiple camera views.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features_view1: Tensor,
        features_view2: Tensor,
        correspondences: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute multi-view consistency loss.

        Args:
            features_view1: [N, C] features from view 1
            features_view2: [M, C] features from view 2
            correspondences: [K, 2] optional correspondences (idx1, idx2)

        Returns:
            Consistency loss (scalar)
        """
        if correspondences is not None:
            # Use known correspondences
            f1 = features_view1[correspondences[:, 0]]
            f2 = features_view2[correspondences[:, 1]]
            return F.mse_loss(f1, f2)
        else:
            # Use nearest neighbor matching
            if features_view1.shape[0] == 0 or features_view2.shape[0] == 0:
                return torch.tensor(0.0, device=features_view1.device)

            dist = torch.cdist(features_view1, features_view2, p=2)
            nn_dist = dist.min(dim=1).values

            # Contrastive loss: pull correspondences together
            return F.relu(nn_dist - self.margin).mean()


class ReconstructionLoss(nn.Module):
    """Video reconstruction loss."""

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.loss_type == "l2":
            return F.mse_loss(pred, target)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss.

    Compares features at intermediate VGG layers.
    """

    def __init__(self, layers: list = None):
        super().__init__()
        self.layers = layers or [4, 9, 16, 23]  # VGG16 ReLU layers
        self._vgg = None
        self._vgg_loaded = False

    def _load_vgg(self, device):
        if self._vgg_loaded:
            return

        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            vgg.requires_grad_(False)
            vgg.eval()
            self._vgg = vgg.to(device)
            self._vgg_loaded = True
        except ImportError:
            # Fallback: no perceptual loss
            self._vgg = None
            self._vgg_loaded = True

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        self._load_vgg(pred.device)

        if self._vgg is None:
            return torch.tensor(0.0, device=pred.device)

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        loss = 0.0
        x_pred = pred_norm
        x_target = target_norm

        for i, layer in enumerate(self._vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.layers:
                loss = loss + F.l1_loss(x_pred, x_target)

        return loss / len(self.layers)


class FSQCommitmentLoss(nn.Module):
    """FSQ commitment loss for stable quantization."""

    def forward(self, z_continuous: Tensor, z_quantized: Tensor) -> Tensor:
        return F.mse_loss(z_continuous, z_quantized.detach())


class ActionVarianceLoss(nn.Module):
    """
    Encourage diverse latent actions.

    Prevents collapse where all transitions map to same action.
    """

    def __init__(self, target_variance: float = 0.1):
        super().__init__()
        self.target_variance = target_variance

    def forward(self, actions: Tensor) -> Tensor:
        # actions: [B, T, A] or [B, A]
        if actions.dim() == 3:
            actions = actions.flatten(0, 1)  # [B*T, A]

        var = actions.var(dim=0).mean()
        return (var - self.target_variance).pow(2)


class TemporalConsistencyLoss(nn.Module):
    """Encourage smooth temporal transitions."""

    def forward(self, predictions: Tensor) -> Tensor:
        # predictions: [B, T, ...]
        if predictions.shape[1] < 2:
            return torch.tensor(0.0, device=predictions.device)

        diff = predictions[:, 1:] - predictions[:, :-1]
        return diff.pow(2).mean()


class SparsityLoss(nn.Module):
    """Encourage sparse DeltaV predictions."""

    def __init__(self, target_sparsity: float = 0.1):
        super().__init__()
        self.target_sparsity = target_sparsity

    def forward(self, num_deltas: int, max_deltas: int) -> Tensor:
        sparsity = num_deltas / max_deltas
        return (sparsity - self.target_sparsity).pow(2)


class GenesisCriterion(nn.Module):
    """
    Combined loss function for Genesis training.

    Supports different loss configurations for each training stage.
    """

    def __init__(
        self,
        stage: int = 1,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        fsq_weight: float = 0.25,
        action_var_weight: float = 0.1,
        temporal_weight: float = 0.05,
        sparsity_weight: float = 0.01,
    ):
        super().__init__()
        self.stage = stage

        self.weights = {
            'recon': recon_weight,
            'perceptual': perceptual_weight,
            'fsq': fsq_weight,
            'action_var': action_var_weight,
            'temporal': temporal_weight,
            'sparsity': sparsity_weight,
        }

        self.recon_loss = ReconstructionLoss("l1")
        self.perceptual_loss = PerceptualLoss()
        self.fsq_loss = FSQCommitmentLoss()
        self.action_var_loss = ActionVarianceLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.sparsity_loss = SparsityLoss()

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute losses.

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            dict of individual losses and total
        """
        losses = {}

        # Stage 1: Tokenizer training
        if self.stage >= 1:
            if 'reconstruction' in outputs and 'video' in targets:
                # Flatten batch and time for reconstruction loss
                pred = outputs['reconstruction']
                target = targets['video']

                # Handle shape differences
                if pred.shape != target.shape:
                    # Resize if needed
                    B, T = target.shape[:2]
                    pred = F.interpolate(
                        pred.flatten(0, 1),
                        size=target.shape[-2:],
                        mode='bilinear',
                    ).view(B, -1, *target.shape[2:])

                losses['recon'] = self.recon_loss(pred, target) * self.weights['recon']

                # Perceptual loss (on frames)
                if self.weights['perceptual'] > 0:
                    pred_frames = pred.flatten(0, 1)
                    target_frames = target.flatten(0, 1)
                    losses['perceptual'] = self.perceptual_loss(
                        pred_frames, target_frames
                    ) * self.weights['perceptual']

            if 'latent' in outputs:
                latent = outputs['latent']
                # FSQ commitment (simplified - would need pre-quantized values)
                losses['fsq'] = torch.tensor(0.0, device=latent.device)

        # Stage 2: Dynamics training
        if self.stage >= 2:
            if 'actions' in outputs:
                losses['action_var'] = self.action_var_loss(
                    outputs['actions']
                ) * self.weights['action_var']

            if 'predicted_features' in outputs:
                losses['temporal'] = self.temporal_loss(
                    outputs['predicted_features']
                ) * self.weights['temporal']

        # Stage 3: DeltaV training
        if self.stage >= 3:
            if 'delta_v' in outputs:
                delta_v = outputs['delta_v']
                losses['sparsity'] = self.sparsity_loss(
                    delta_v.num_deltas,
                    4096,  # max_deltas
                ) * self.weights['sparsity']

        # Total loss
        losses['total'] = sum(losses.values())

        return losses
