"""DeltaV predictor: predict sparse voxel updates from dynamics features."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple, Optional, Tuple
from dataclasses import dataclass

from genesis.config import DeltaVConfig
from genesis.deltav.lifting import FeatureLifting, CoordinateLifting


@dataclass
class DeltaV:
    """
    Sparse voxel delta prediction.

    Represents changes to apply to OVoxel memory:
    - coords: Where to apply changes
    - features: What values to set
    - op_type: What operation (remove/modify/add)
    - confidence: How confident the prediction is
    """
    coords: Tensor      # [K, 4] - (batch, x, y, z)
    features: Tensor    # [K, 7] - PBR: RGB(3), metallic(1), roughness(1), opacity(1), SDF(1)
    op_type: Tensor     # [K] - 0=remove, 1=modify, 2=add
    confidence: Tensor  # [K] - prediction confidence

    # Internal: stored for loss computation (avoid re-prediction bug)
    _op_logits: Optional[Tensor] = None  # [K, 3] - logits before argmax
    _voxel_features: Optional[Tensor] = None  # [K, voxel_features] - intermediate features

    def to(self, device: torch.device) -> "DeltaV":
        """Move to device."""
        result = DeltaV(
            coords=self.coords.to(device),
            features=self.features.to(device),
            op_type=self.op_type.to(device),
            confidence=self.confidence.to(device),
        )
        if self._op_logits is not None:
            result._op_logits = self._op_logits.to(device)
        if self._voxel_features is not None:
            result._voxel_features = self._voxel_features.to(device)
        return result

    def filter_by_confidence(self, threshold: float) -> "DeltaV":
        """Keep only predictions above confidence threshold."""
        mask = self.confidence >= threshold
        result = DeltaV(
            coords=self.coords[mask],
            features=self.features[mask],
            op_type=self.op_type[mask],
            confidence=self.confidence[mask],
        )
        if self._op_logits is not None:
            result._op_logits = self._op_logits[mask]
        if self._voxel_features is not None:
            result._voxel_features = self._voxel_features[mask]
        return result

    @property
    def num_deltas(self) -> int:
        """Number of delta predictions."""
        return self.coords.shape[0]


class DeltaVPredictor(nn.Module):
    """
    Predict sparse voxel deltas from dynamics features.

    Core innovation of Genesis: instead of predicting full frames,
    predict sparse updates to persistent 3D memory.

    Pipeline:
    1. Lift 2D features to 3D (depth prediction)
    2. Predict per-voxel confidence
    3. Extract top-K confident positions
    4. Predict operation type and PBR features
    """

    def __init__(self, config: DeltaVConfig):
        super().__init__()
        self.config = config

        # Feature lifting (2D -> 3D)
        self.feature_lifting = FeatureLifting(
            input_dim=config.input_dim,
            output_dim=config.voxel_features,
            depth_bins=config.depth_bins,
            hidden_dim=1024,
        )

        # Coordinate lifting
        self.coord_lifting = CoordinateLifting(voxel_resolution=256)

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.voxel_features, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Operation classifier (remove=0, modify=1, add=2)
        self.op_classifier = nn.Sequential(
            nn.Linear(config.voxel_features, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # PBR decoder
        # Output: RGB(3) + metallic(1) + roughness(1) + opacity(1) + SDF(1) = 7
        self.pbr_decoder = nn.Sequential(
            nn.Linear(config.voxel_features, 64),
            nn.GELU(),
            nn.Linear(64, 7),
        )

        # Activation functions for PBR values
        self.rgb_act = nn.Sigmoid()  # RGB in [0, 1]
        self.factor_act = nn.Sigmoid()  # Metallic, roughness, opacity in [0, 1]
        self.sdf_act = nn.Tanh()  # SDF in [-1, 1]

    def forward(
        self,
        features: Tensor,
        max_deltas: Optional[int] = None,
    ) -> DeltaV:
        """
        Predict sparse voxel deltas.

        Args:
            features: [B, H, W, D] dynamics backbone output
            max_deltas: Override for maximum deltas per frame

        Returns:
            DeltaV with sparse predictions
        """
        B, H, W, D = features.shape
        max_k = max_deltas or self.config.max_deltas_per_frame
        device = features.device

        # Lift to 3D: [B, H, W, depth_bins, voxel_features]
        voxel_features, depth_probs = self.feature_lifting(features)
        depth_bins = voxel_features.shape[3]

        # Predict confidence for each voxel position
        confidence = self.confidence_head(voxel_features).squeeze(-1)  # [B, H, W, D]
        confidence = confidence * depth_probs  # Weight by depth probability

        # Flatten and get top-K
        conf_flat = confidence.view(B, -1)  # [B, H*W*depth_bins]

        # Handle empty tensor case
        if conf_flat.numel() == 0:
            return DeltaV(
                coords=torch.empty(0, 4, dtype=torch.int32, device=device),
                features=torch.empty(0, 7, device=device),
                op_type=torch.empty(0, dtype=torch.long, device=device),
                confidence=torch.empty(0, device=device),
            )

        topk_conf, topk_idx = conf_flat.topk(min(max_k, conf_flat.shape[1]), dim=1)

        # Gather features at top-K positions
        voxel_flat = voxel_features.view(B, -1, self.config.voxel_features)
        topk_features = torch.gather(
            voxel_flat,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, self.config.voxel_features),
        )  # [B, K, voxel_features]

        # Convert flat indices to (h, w, d) coordinates
        hw_size = H * W
        d_idx = topk_idx % depth_bins
        hw_idx = topk_idx // depth_bins
        h_idx = hw_idx // W
        w_idx = hw_idx % W

        # Convert to voxel coordinates
        coords_list = []
        for b in range(B):
            voxel_coords = self.coord_lifting(
                h_idx[b], w_idx[b], d_idx[b],
                H, W, depth_bins,
            )
            batch_idx = torch.full((voxel_coords.shape[0], 1), b, device=device)
            coords_list.append(torch.cat([batch_idx, voxel_coords], dim=-1))

        coords = torch.cat(coords_list, dim=0)  # [B*K, 4]

        # Flatten batch dimension for features
        topk_features_flat = topk_features.view(-1, self.config.voxel_features)
        topk_conf_flat = topk_conf.view(-1)

        # Predict operation types (store logits for loss computation)
        op_logits = self.op_classifier(topk_features_flat)
        op_type = op_logits.argmax(dim=-1)  # [B*K]

        # Predict PBR features
        pbr_raw = self.pbr_decoder(topk_features_flat)  # [B*K, 7]

        # Apply activations
        rgb = self.rgb_act(pbr_raw[:, :3])
        metallic = self.factor_act(pbr_raw[:, 3:4])
        roughness = self.factor_act(pbr_raw[:, 4:5])
        opacity = self.factor_act(pbr_raw[:, 5:6])
        sdf = self.sdf_act(pbr_raw[:, 6:7])

        features_out = torch.cat([rgb, metallic, roughness, opacity, sdf], dim=-1)

        delta_v = DeltaV(
            coords=coords.int(),
            features=features_out,
            op_type=op_type,
            confidence=topk_conf_flat,
        )

        # Store intermediate values for loss computation (avoid re-prediction bug)
        delta_v._op_logits = op_logits
        delta_v._voxel_features = topk_features_flat

        return delta_v

    def compute_loss(
        self,
        pred: DeltaV,
        target_coords: Tensor,
        target_features: Tensor,
        target_ops: Tensor,
    ) -> dict:
        """
        Compute training losses.

        Args:
            pred: Predicted DeltaV
            target_coords: [N, 4] ground truth coordinates
            target_features: [N, 7] ground truth PBR
            target_ops: [N] ground truth operations

        Returns:
            dict of losses
        """
        # This is a simplified loss - full implementation would use
        # Hungarian matching or other assignment strategies

        losses = {}

        # Handle empty predictions
        if pred.num_deltas == 0:
            device = target_coords.device if target_coords.numel() > 0 else torch.device('cpu')
            losses["coord"] = torch.tensor(0.0, device=device, requires_grad=True)
            losses["feature"] = torch.tensor(0.0, device=device, requires_grad=True)
            losses["op"] = torch.tensor(0.0, device=device, requires_grad=True)
            losses["sparsity"] = torch.tensor(0.0, device=device)
            return losses

        # Coordinate loss (chamfer distance)
        pred_coords_float = pred.coords[:, 1:].float()  # Exclude batch dim
        target_coords_float = target_coords[:, 1:].float()

        # Simplified: MSE on matched pairs (assumes same ordering)
        min_len = min(pred_coords_float.shape[0], target_coords_float.shape[0])
        if min_len > 0:
            losses["coord"] = nn.functional.mse_loss(
                pred_coords_float[:min_len],
                target_coords_float[:min_len],
            )

        # Feature loss
        if min_len > 0:
            losses["feature"] = nn.functional.l1_loss(
                pred.features[:min_len],
                target_features[:min_len],
            )

        # Operation classification loss - use stored logits to avoid re-prediction bug
        if min_len > 0:
            if pred._op_logits is not None:
                # Use stored logits (correct: consistent with prediction)
                losses["op"] = nn.functional.cross_entropy(
                    pred._op_logits[:min_len],
                    target_ops[:min_len],
                )
            elif pred._voxel_features is not None:
                # Fallback: use stored intermediate features
                op_logits = self.op_classifier(pred._voxel_features[:min_len])
                losses["op"] = nn.functional.cross_entropy(
                    op_logits,
                    target_ops[:min_len],
                )
            else:
                # Last resort: skip op loss if no intermediate features available
                losses["op"] = torch.tensor(0.0, device=pred.coords.device, requires_grad=True)

        # Sparsity regularization (encourage fewer deltas)
        losses["sparsity"] = torch.tensor(
            pred.num_deltas / self.config.max_deltas_per_frame,
            device=pred.coords.device
        )

        return losses
