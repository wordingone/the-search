"""
Anima Archive
=============

Historical variants preserved for reference.

V1 (base.py): Original baseline - 38k params
V2 (v2_core.py): Synthesized baseline - 98k params
V3 (v3_collective.py): Collective-Democratic - 327k params
V4 (v4_minimal.py): Minimal Efficient - 20k params
V4-Telos (v4_telos.py): Goal as Embodied Preference - 21k params

Benchmark Results (2026-01-11):
  V4-MinimalEfficient: 65.5% overall, 20k params, 15% goal
  V4-Telos: 41.0% overall, 21k params, 20% goal (+5pp)
  V1-Base: 56.6% overall, 38k params
  V2-Core: 61.0% overall, 98k params
  V3-Collective: 61.0% overall, 327k params

Key Finding: V4-Telos improved goal-seeking but hurt all other tasks.
This led to V5 design: temporal horizon structure to enhance ALL tasks.
"""

from .v1_base import Anima as AnimaV1, AnimaConfig as AnimaV1Config
from .v2_core import AnimaV2, AnimaV2Config
from .v3_collective import AnimaV3, AnimaV3Config
from .v4_minimal import AnimaV4, AnimaV4Config
from .v4_telos import AnimaTelos, AnimaTelosConfig

__all__ = [
    'AnimaV1', 'AnimaV1Config',
    'AnimaV2', 'AnimaV2Config',
    'AnimaV3', 'AnimaV3Config',
    'AnimaV4', 'AnimaV4Config',
    'AnimaTelos', 'AnimaTelosConfig',
]
