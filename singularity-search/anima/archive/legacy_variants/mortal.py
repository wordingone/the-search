"""
Anima Mortal Variant
====================

SURGICAL DIFFERENCE: Energy depletion causes death.

Base Anima: Energy is optional constraint (doesn't kill)
Mortal:     Energy depletion → death (like Echo Original)

This tests whether mortality pressure affects:
- Learning speed (urgency)
- Memory consolidation patterns
- State stability

From Echo findings:
- Original Echo died at step 542 from energy depletion
- Energy creates "survival pressure" that may drive adaptation
"""

import torch
from typing import Dict, Any, Optional
from ..core.base import Anima, AnimaConfig


class AnimaMortalConfig(AnimaConfig):
    """Config for mortal Anima."""

    def __init__(self, **kwargs):
        # Mortality parameters
        kwargs.setdefault('use_energy', True)
        kwargs.setdefault('E_init', 1.0)
        kwargs.setdefault('E_decay', 0.002)  # Faster decay than base

        # Death threshold
        self.death_threshold = kwargs.pop('death_threshold', 0.01)

        # Energy intake (reward for good prediction)
        self.intake_max = kwargs.pop('intake_max', 0.003)
        self.intake_sensitivity = kwargs.pop('intake_sensitivity', 2.0)

        super().__init__(**kwargs)


class AnimaMortal(Anima):
    """
    Mortal Anima - energy depletion causes death.

    SINGLE CHANGE from base Anima:
    - Energy reaching threshold triggers die()

    Everything else identical to base Anima.
    """

    def __init__(self, config: Optional[AnimaMortalConfig] = None):
        if config is None:
            config = AnimaMortalConfig()
        super().__init__(config)
        self.config: AnimaMortalConfig = config

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with mortality check."""
        if not self.alive:
            return {'alive': False, 'cause': self.death_cause}

        # Base step
        result = super().step(observation)

        if not result['alive']:
            return result

        # === SURGICAL DIFFERENCE: Mortality ===
        # Compute energy intake from prediction quality
        prediction_error = result.get('prediction_error', 0.5)
        intake = self.config.intake_max * torch.exp(
            torch.tensor(-prediction_error * self.config.intake_sensitivity)
        ).item()

        # Update energy
        self.E = self.E - self.config.E_decay + intake

        # Check death
        if self.E <= self.config.death_threshold:
            self.die("Energy depletion (mortality)")
            result['alive'] = False
            result['cause'] = self.death_cause

        result['energy'] = self.E
        result['intake'] = intake

        return result

    def get_final_report(self) -> Dict[str, Any]:
        """Extended report with mortality data."""
        report = super().get_final_report()
        report['variant'] = 'Mortal'
        report['death_threshold'] = self.config.death_threshold
        return report
