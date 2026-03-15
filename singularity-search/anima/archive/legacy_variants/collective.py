"""
Anima Collective Variant
========================

SURGICAL DIFFERENCE: Multiple entities share resource pool.

Base Anima: Singular entity, independent resources
Collective: Swarm shares energy pool, cooperation bonus

This tests whether cooperation affects:
- Collective survival
- Individual vs group optimization
- Emergent coordination

From Echo findings:
- Collective Echo: 85% cooperation rate, all 5 survived
- Shared pool grew from 3.0 to 51.0 over 1500 steps
- Individual contribution varied (0.81-0.87)
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from ..core.base import Anima, AnimaConfig


class AnimaCollectiveConfig(AnimaConfig):
    """Config for collective Anima member."""

    def __init__(self, **kwargs):
        # Swarm parameters (extracted before super)
        self.swarm_size = kwargs.pop('swarm_size', 5)
        self.pool_init = kwargs.pop('pool_init', 3.0)
        self.pool_drain = kwargs.pop('pool_drain', 0.01)
        self.cooperation_threshold = kwargs.pop('cooperation_threshold', 0.3)
        self.cooperation_bonus = kwargs.pop('cooperation_bonus', 1.5)

        # Individual members slightly weaker
        kwargs.setdefault('use_energy', True)
        kwargs.setdefault('E_init', 0.5)
        kwargs.setdefault('E_decay', 0.001)

        super().__init__(**kwargs)


class AnimaCollective(Anima):
    """
    Collective Anima member - part of a swarm.

    SINGLE CHANGE from base Anima:
    - Energy managed at swarm level, not individual
    - Tracks contribution to collective (inverse of prediction error)
    """

    def __init__(self, config: Optional[AnimaCollectiveConfig] = None, member_id: int = 0):
        if config is None:
            config = AnimaCollectiveConfig()
        super().__init__(config)
        self.config: AnimaCollectiveConfig = config

        self.member_id = member_id
        self.last_prediction_error = 0.5
        self.contribution = 0.0

    def step_without_energy(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Step without individual energy management.
        Energy is handled at swarm level.
        """
        if not self.alive:
            return {'alive': False, 'cause': self.death_cause}

        # Base step (but ignore energy update)
        original_E = self.E
        result = super().step(observation)
        self.E = original_E  # Restore - swarm handles energy

        if not result['alive']:
            return result

        # Track contribution (inverse of error = better prediction = more contribution)
        self.last_prediction_error = result.get('prediction_error', 0.5)
        self.contribution = 1.0 / (1.0 + self.last_prediction_error)

        result['member_id'] = self.member_id
        result['contribution'] = self.contribution

        return result

    def get_final_report(self) -> Dict[str, Any]:
        """Extended report with collective data."""
        report = super().get_final_report()
        report['variant'] = 'Collective'
        report['member_id'] = self.member_id
        report['contribution'] = self.contribution
        return report


class AnimaSwarm:
    """
    Swarm of collective Anima entities.

    Manages:
    - Shared energy pool (E_pool)
    - Collective prediction quality (collective_error)
    - Proportional energy distribution
    - Cooperation bonus when collective does well
    """

    def __init__(self, config: AnimaCollectiveConfig):
        self.config = config

        # Shared energy pool
        self.E_pool = config.pool_init

        # Initialize members
        self.members: List[AnimaCollective] = []
        for i in range(config.swarm_size):
            member = AnimaCollective(config, member_id=i)
            self.members.append(member)

        self.step_count = 0
        self.alive = True

        # Metrics
        self.pool_history: List[float] = []
        self.cooperation_history: List[bool] = []

    def step(self, observation: torch.Tensor) -> Dict[str, Any]:
        """Step all members with shared resource management."""
        self.step_count += 1

        # Step all living members
        living = [m for m in self.members if m.alive]
        if not living:
            self.alive = False
            return {'alive': False, 'cause': 'Swarm extinction'}

        errors = []
        contributions = []

        for member in living:
            result = member.step_without_energy(observation)
            if result['alive']:
                errors.append(member.last_prediction_error)
                contributions.append(member.contribution)

        # Recount living after step
        living = [m for m in self.members if m.alive]
        if not living:
            self.alive = False
            return {'alive': False, 'cause': 'Swarm extinction'}

        # Collective metrics
        collective_error = np.mean(errors) if errors else 1.0
        cooperation_active = collective_error < self.config.cooperation_threshold

        # === SURGICAL DIFFERENCE: Shared pool with cooperation ===
        # Pool intake (better collective prediction = more intake)
        base_intake = 0.01 * len(living) * np.exp(-collective_error)
        if cooperation_active:
            base_intake *= self.config.cooperation_bonus

        # Pool drain (each member costs)
        drain = self.config.pool_drain * len(living)

        # Update pool
        self.E_pool = max(0.0, self.E_pool + base_intake - drain)

        # Distribute energy to members proportionally
        total_contribution = sum(contributions) + 1e-6
        for member in living:
            share = (member.contribution / total_contribution) * 0.1 * self.E_pool
            member.E = min(1.0, member.E + share)

            # Check individual death from pool depletion
            if member.E <= 0.01 and self.E_pool <= 0.1:
                member.die("Pool depletion")

        # Record metrics
        self.pool_history.append(self.E_pool)
        self.cooperation_history.append(cooperation_active)

        return {
            'alive': self.alive,
            'step': self.step_count,
            'E_pool': self.E_pool,
            'population': len([m for m in self.members if m.alive]),
            'collective_error': collective_error,
            'cooperation_active': cooperation_active,
        }

    def get_report(self) -> Dict[str, Any]:
        """Get swarm-level report."""
        living = [m for m in self.members if m.alive]
        coop_rate = sum(self.cooperation_history) / len(self.cooperation_history) if self.cooperation_history else 0

        return {
            'variant': 'Collective Swarm',
            'alive': self.alive,
            'total_steps': self.step_count,
            'final_population': len(living),
            'max_population': self.config.swarm_size,
            'final_E_pool': self.E_pool,
            'cooperation_rate': coop_rate,
            'member_states': [{
                'id': m.member_id,
                'alive': m.alive,
                'energy': m.E,
                'contribution': m.contribution,
                'cycles': m.cycle_count,
            } for m in self.members],
        }
