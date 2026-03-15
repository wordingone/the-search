"""
Genesis: World Model for Video Prediction.

Target: 720p @ 24 FPS, unlimited horizon, keyboard/mouse interaction.
Architecture: Latent tokenization + windowed attention transformer.
"""

# Genesis model (baseline architecture that passed validation)
from genesis.model import Genesis
from genesis.config import GenesisConfig

__version__ = "0.6.0"
__all__ = [
    "Genesis",
    "GenesisConfig",
]
