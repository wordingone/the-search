"""Loss functions for video tokenizer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from einops import rearrange


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.

    Computes L1 distance in VGG feature space for sharper reconstructions.
    """

    def __init__(
        self,
        layers: List[str] = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'],
        weights: List[float] = [1.0, 1.0, 1.0, 1.0],
        normalize_input: bool = True,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.weights = weights

        # Load pretrained VGG19
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()

        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False

        # Layer indices for VGG19
        layer_indices = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_4': 18,
            'relu4_4': 27,
            'relu5_4': 36,
        }

        # Build feature extractors
        self.slices = nn.ModuleList()
        prev_idx = 0
        for layer in layers:
            idx = layer_indices[layer]
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:idx]))
            prev_idx = idx

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize to ImageNet stats."""
        return (x - self.mean) / self.std

    def extract_features(self, x: Tensor) -> List[Tensor]:
        """Extract multi-scale VGG features."""
        if self.normalize_input:
            x = self.normalize(x)

        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: [B, C, H, W] or [B, T, C, H, W] predicted images/video
            target: [B, C, H, W] or [B, T, C, H, W] target images/video

        Returns:
            perceptual loss scalar
        """
        # Handle video input by flattening temporal dimension
        is_video = pred.dim() == 5
        if is_video:
            B, T = pred.shape[:2]
            pred = rearrange(pred, 'b t c h w -> (b t) c h w')
            target = rearrange(target, 'b t c h w -> (b t) c h w')

        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)

        loss = 0.0
        for pf, tf, w in zip(pred_features, target_features, self.weights):
            loss += w * F.l1_loss(pf, tf)

        return loss


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS).

    More robust than VGG perceptual loss for measuring image similarity.
    """

    def __init__(self, net: str = 'vgg'):
        super().__init__()
        try:
            import lpips
            self.lpips = lpips.LPIPS(net=net)
            for param in self.lpips.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: lpips not installed, falling back to VGG perceptual loss")
            self.lpips = None
            self.vgg_loss = VGGPerceptualLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute LPIPS loss."""
        if self.lpips is None:
            return self.vgg_loss(pred, target)

        # Handle video input
        is_video = pred.dim() == 5
        if is_video:
            pred = rearrange(pred, 'b t c h w -> (b t) c h w')
            target = rearrange(target, 'b t c h w -> (b t) c h w')

        # LPIPS expects inputs in [-1, 1]
        pred = pred * 2 - 1
        target = target * 2 - 1

        return self.lpips(pred, target).mean()


class TokenizerLoss(nn.Module):
    """
    Combined loss for video tokenizer training.

    Components:
    - Reconstruction loss (L1 + MSE)
    - Perceptual loss (VGG features)
    - GAN loss (adversarial)
    - Commitment loss (for FSQ stability)
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        gan_weight: float = 0.1,
        commitment_weight: float = 0.25,
        use_lpips: bool = False,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        self.commitment_weight = commitment_weight

        # Perceptual loss
        if use_lpips:
            self.perceptual = LPIPSLoss()
        else:
            self.perceptual = VGGPerceptualLoss()

    def reconstruction_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Combined L1 + MSE reconstruction loss."""
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return l1 + mse

    def commitment_loss(self, latent: Tensor, codes: Tensor) -> Tensor:
        """
        Commitment loss for FSQ.

        Encourages encoder output to stay close to quantized values.
        """
        return F.mse_loss(latent, codes.detach())

    def forward(
        self,
        recon: Tensor,
        target: Tensor,
        latent: Tensor,
        codes: Tensor,
        gan_loss: Optional[Tensor] = None,
    ) -> dict:
        """
        Compute combined tokenizer loss.

        Args:
            recon: reconstructed video
            target: original video
            latent: continuous latent before quantization
            codes: quantized codes
            gan_loss: optional pre-computed GAN loss

        Returns:
            dict with individual losses and total
        """
        losses = {}

        # Reconstruction
        losses['recon'] = self.reconstruction_loss(recon, target)

        # Perceptual
        losses['perceptual'] = self.perceptual(recon, target)

        # Commitment
        losses['commitment'] = self.commitment_loss(latent, codes)

        # GAN (if provided)
        if gan_loss is not None:
            losses['gan'] = gan_loss
        else:
            losses['gan'] = torch.tensor(0.0, device=recon.device)

        # Total
        losses['total'] = (
            self.recon_weight * losses['recon'] +
            self.perceptual_weight * losses['perceptual'] +
            self.commitment_weight * losses['commitment'] +
            self.gan_weight * losses['gan']
        )

        return losses


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for video reconstruction.

    Penalizes temporal flickering by measuring frame-to-frame differences.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: [B, T, C, H, W] predicted video
            target: [B, T, C, H, W] target video

        Returns:
            temporal consistency loss
        """
        # Compute temporal differences
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]

        # L1 loss on temporal differences
        return self.weight * F.l1_loss(pred_diff, target_diff)
