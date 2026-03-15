"""
Anima Variants Module
=====================

Surgical variations of base Anima for behavioral analysis.

W/I/T Architecture (implements S/M/D from Formal Theory):
    W (World) - implements Type S (State) - environment encoding
    I (Internal) - implements Type M (Memory) - learning/adaptation
    T (Time) - implements Type D (Decision) - temporal context

V1 Variants (branch from Anima V1):
- Mortal: Energy-constrained (dies from depletion)
- Metamorphic: Transforms on energy crisis instead of dying
- Collective: Shared resource pool with cooperation bonus
- Neuroplastic: Dynamic N (grows/prunes its own variable count)

V2 Hybrids (branch from AnimaV2 trunk - synthesized baseline):
- Adaptive: Neuroplastic growth + Metamorphic transform-shrink
- Resonant: Collective cooperation + Internal multi-voice harmony
- Phoenix: Metamorphic transformation + Staged energy cycles
- Pressured: Mortal urgency + Collective stability (learning rate, not survival)

Each variant changes exactly ONE behavioral axis.
"""

# V1 Variants
from .mortal import AnimaMortal, AnimaMortalConfig
from .metamorphic import AnimaMetamorphic, AnimaMetamorphicConfig
from .collective import AnimaCollective, AnimaCollectiveConfig, AnimaSwarm
from .neuroplastic import AnimaNeuroplastic, AnimaNeuroplasticConfig

# V2 Hybrids
from .v2_hybrids import (
    AnimaAdaptive, AnimaAdaptiveConfig,
    AnimaResonant, AnimaResonantConfig,
    AnimaPhoenix, AnimaPhoenixConfig,
    AnimaPressured, AnimaPressuredConfig,
)

__all__ = [
    # V1 Variants
    'AnimaMortal', 'AnimaMortalConfig',
    'AnimaMetamorphic', 'AnimaMetamorphicConfig',
    'AnimaCollective', 'AnimaCollectiveConfig', 'AnimaSwarm',
    'AnimaNeuroplastic', 'AnimaNeuroplasticConfig',
    # V2 Hybrids
    'AnimaAdaptive', 'AnimaAdaptiveConfig',
    'AnimaResonant', 'AnimaResonantConfig',
    'AnimaPhoenix', 'AnimaPhoenixConfig',
    'AnimaPressured', 'AnimaPressuredConfig',
]
