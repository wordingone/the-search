"""
Anima Core Module - Generation 2
================================

The second generation of Anima architectures, designed from first-principles
causal analysis of benchmark performance across 14 synthetic tasks.

NEW ARCHITECTURES (Generation 2):
    - AnimaHierarchical: Fast/slow temporal horizons with adaptive mixing
    - AnimaInduction: Explicit induction circuit for in-context learning
    - AnimaModular: Separate logic vs memory circuits with soft routing

DESIGN PRINCIPLES:
    1. Task-Adaptive Gating: Learn WHEN to couple vs independent gates
    2. Hierarchical State: Fast state for immediate, slow state for long-term
    3. Induction-Preserving: Explicit copy-shift patterns as inductive bias
    4. Superposition-Specialization Balance: Modular circuits for each regime

CAUSAL INSIGHTS (from comprehensive benchmark):
    - Gate independence enables accumulation (+50pp on analogy)
    - Gate coupling enables threshold decisions (+42pp on conditional)
    - Structured state enables multi-scale memory (+30pp on delay)
    - State-dependent discretization enables intentionality (+20pp on momentum)
    - Induction score correlates r=0.71 with overall performance
    - Superposition correlates r=-0.68 with logic performance

ARCHITECTURE COMPARISON:
    | Model            | State Structure          | Gate Type    | Specialization |
    |------------------|--------------------------|--------------|----------------|
    | AnimaHierarchical| Fast [d] + Slow [d,s]    | Adaptive     | Temporal       |
    | AnimaInduction   | SSM [d,s] + History      | Independent  | ICL            |
    | AnimaModular     | Logic [d] + Memory [d,s] | Both         | Task-type      |

LEGACY ARCHITECTURES:
    All previous variants (Anima, AnimaOptimized, AnimaISSM, AnimaATR,
    AnimaEvolved, AnimaEvolvedV2/V3/V4, AnimaRouter) are archived in:
    archive/legacy/anima_core_v1/

USAGE:
    from anima.core import AnimaHierarchical, AnimaInduction, AnimaModular

    # Temporal tasks (delay, sequence, projectile)
    model = AnimaHierarchical(sensory_dim=8, d_model=32, output_dim=4)

    # In-context learning (pattern, analogy, associative)
    model = AnimaInduction(sensory_dim=8, d_model=32, output_dim=4)

    # Mixed tasks (both logic AND memory)
    model = AnimaModular(sensory_dim=8, d_model=32, output_dim=4)
"""

from .anima_hierarchical import AnimaHierarchical
from .anima_induction import AnimaInduction
from .anima_modular import AnimaModular

__all__ = [
    # Generation 2 architectures
    'AnimaHierarchical',
    'AnimaInduction',
    'AnimaModular',
]


# Factory functions for common configurations
def create_hierarchical(
    sensory_dim: int = 8,
    d_model: int = 32,
    bottleneck_dim: int = 16,
    output_dim: int = 4,
    d_state: int = 16,
    d_meta: int = 8,
):
    """Create AnimaHierarchical optimized for temporal tasks."""
    return AnimaHierarchical(
        sensory_dim=sensory_dim,
        d_model=d_model,
        bottleneck_dim=bottleneck_dim,
        output_dim=output_dim,
        d_state=d_state,
        d_meta=d_meta,
    )


def create_induction(
    sensory_dim: int = 8,
    d_model: int = 32,
    bottleneck_dim: int = 16,
    output_dim: int = 4,
    d_state: int = 16,
    lookback: int = 4,
):
    """Create AnimaInduction optimized for in-context learning."""
    return AnimaInduction(
        sensory_dim=sensory_dim,
        d_model=d_model,
        bottleneck_dim=bottleneck_dim,
        output_dim=output_dim,
        d_state=d_state,
        lookback=lookback,
    )


def create_modular(
    sensory_dim: int = 8,
    d_model: int = 32,
    bottleneck_dim: int = 16,
    output_dim: int = 4,
    d_state: int = 16,
):
    """Create AnimaModular optimized for mixed logic/memory tasks."""
    return AnimaModular(
        sensory_dim=sensory_dim,
        d_model=d_model,
        bottleneck_dim=bottleneck_dim,
        output_dim=output_dim,
        d_state=d_state,
    )
