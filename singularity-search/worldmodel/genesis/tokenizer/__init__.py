"""Video tokenizer with 3D Causal VAE and FSQ quantization."""

from genesis.tokenizer.fsq import FSQ, FSQEmbedding
from genesis.tokenizer.encoder import VideoEncoder
from genesis.tokenizer.decoder import VideoDecoder, VideoTokenizer
from genesis.tokenizer.discriminator import (
    PatchDiscriminator3d,
    MultiScaleDiscriminator,
    hinge_loss_d,
    hinge_loss_g,
)
from genesis.tokenizer.losses import (
    VGGPerceptualLoss,
    TokenizerLoss,
    TemporalConsistencyLoss,
)
from genesis.tokenizer.motion import (
    MotionAwareTokenizer,
    MotionAwareLoss,
    FlowEstimator,
    warp_frame,
)

__all__ = [
    # Core components
    "FSQ",
    "FSQEmbedding",
    "VideoEncoder",
    "VideoDecoder",
    "VideoTokenizer",
    # Motion-aware tokenizer
    "MotionAwareTokenizer",
    "MotionAwareLoss",
    "FlowEstimator",
    "warp_frame",
    # Discriminator
    "PatchDiscriminator3d",
    "MultiScaleDiscriminator",
    "hinge_loss_d",
    "hinge_loss_g",
    # Losses
    "VGGPerceptualLoss",
    "TokenizerLoss",
    "TemporalConsistencyLoss",
]
