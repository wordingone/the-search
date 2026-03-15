"""
Anima Metamorphic Variant
=========================

SURGICAL DIFFERENCE: Energy crisis triggers transformation, not death.

Base Anima: Energy is optional constraint
Mortal:     Energy depletion → death
Metamorphic: Energy depletion → transform → continue

This tests whether transformation ability affects:
- Long-term survival
- Memory accumulation patterns
- State evolution over multiple "lives"

From Echo findings:
- Metamorphosis Echo had 11 transformations over 1500 steps
- Each transform: compress memory → reset energy → time jump
- Survived entire run with increasing irreversible memory norm
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from ..core.base import Anima, AnimaConfig


class AnimaMetamorphicConfig(AnimaConfig):
    """Config for metamorphic Anima."""

    def __init__(self, **kwargs):
        # Enable energy
        kwargs.setdefault('use_energy', True)
        kwargs.setdefault('E_init', 1.0)
        kwargs.setdefault('E_decay', 0.003)  # Moderate decay

        # Transformation parameters
        self.transform_threshold = kwargs.pop('transform_threshold', 0.05)
        self.rebirth_energy = kwargs.pop('rebirth_energy', 0.3)
        self.time_jump = kwargs.pop('time_jump', 10.0)  # Phase cycles to add

        # Intake (same as mortal for comparison)
        self.intake_max = kwargs.pop('intake_max', 0.002)
        self.intake_sensitivity = kwargs.pop('intake_sensitivity', 2.0)

        super().__init__(**kwargs)


class AnimaMetamorphic(Anima):
    """
    Metamorphic Anima - transforms instead of dying.

    SINGLE CHANGE from base Anima:
    - Energy reaching threshold triggers metamorphose() not die()

    Transformation process:
    1. Compress internal memory → new internal state seed
    2. Fade world memory (partial reset)
    3. Restore partial energy
    4. Jump time (mark discontinuity)
    """

    def __init__(self, config: Optional[AnimaMetamorphicConfig] = None):
        if config is None:
            config = AnimaMetamorphicConfig()
        super().__init__(config)
        self.config: AnimaMetamorphicConfig = config

        # Track transformations
        self.transformation_count = 0
        self.transformation_history: List[Dict] = []

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with transformation check."""
        if not self.alive:
            return {'alive': False, 'cause': self.death_cause}

        # Base step
        result = super().step(observation)

        if not result['alive']:
            return result

        # Compute energy intake
        prediction_error = result.get('prediction_error', 0.5)
        intake = self.config.intake_max * np.exp(-prediction_error * self.config.intake_sensitivity)

        # Update energy
        self.E = self.E - self.config.E_decay + intake

        # === SURGICAL DIFFERENCE: Transform instead of die ===
        if self.E <= self.config.transform_threshold:
            self.metamorphose()

        result['energy'] = self.E
        result['transformations'] = self.transformation_count

        return result

    def metamorphose(self):
        """
        Transform instead of dying.

        Process:
        1. Compress internal memory into new internal state
        2. Fade world memory
        3. Restore partial energy
        4. Record transformation
        """
        self.transformation_count += 1

        # Record state before transform
        self.transformation_history.append({
            'step': self.step_count,
            'cycle': self.cycle_count,
            'phase': self.phase,
            'energy_before': self.E,
            'world_norm': self.world_state.norm().item(),
            'internal_norm': self.internal_state.norm().item(),
            'internal_memory_norm': self.internal_memory.norm().item(),
        })

        # Compress internal memory into new internal state seed
        memory_mean = self.internal_memory.mean(dim=1)  # (1, D)
        memory_norm = memory_mean / (memory_mean.norm() + 1e-6)

        # Mix with current internal state (preserve some continuity)
        self.internal_state = 0.3 * self.internal_state + 0.7 * memory_norm

        # Fade world memory (partial reset - keep some learning)
        self.world_memory = self.world_memory * 0.5

        # Restore partial energy
        self.E = self.config.rebirth_energy

        # Time jump (mark discontinuity in experience)
        self.phase += self.config.time_jump
        while self.phase >= 2 * np.pi:
            self.phase -= 2 * np.pi
            self.cycle_count += 1

    def get_final_report(self) -> Dict[str, Any]:
        """Extended report with transformation data."""
        report = super().get_final_report()
        report['variant'] = 'Metamorphic'
        report['transformation_count'] = self.transformation_count
        report['transformation_history'] = self.transformation_history
        return report
