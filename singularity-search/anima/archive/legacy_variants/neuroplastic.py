"""
Anima Neuroplastic Variant
==========================

SURGICAL DIFFERENCE: Can grow/prune its own variable count (N).

Base Anima: Fixed architecture (N constant)
Neuroplastic: Dynamic N based on task demands

This tests whether self-modifying architecture affects:
- Adaptive capacity over time
- Efficiency (pruning unused capacity)
- Capability growth (adding capacity when needed)

Inspired by:
- MoE (Mixture of Experts) - but learned, not designed
- Neurogenesis - brain grows new neurons
- Synaptic pruning - brain removes unused connections
- Evolution - ancestors bake in what works

Key Innovation: Anima learns WHEN and HOW to modify its own N.

Variable Types (τ):
    S (State) - W modules
    M (Memory) - I modules
    D (Decision) - Output heads

Growth triggers:
- Prediction error consistently high → need more capacity
- Novel patterns detected → allocate specialist module

Prune triggers:
- Module contribution consistently low → remove it
- Redundancy detected → merge modules
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from ..core.base import Anima, AnimaConfig


@dataclass
class ModuleStats:
    """Statistics for a dynamic module."""
    id: int
    module_type: str  # 'world', 'internal', 'output'
    created_step: int
    total_activations: int = 0
    avg_contribution: float = 0.0
    last_active_step: int = 0


class AnimaNeuroplasticConfig(AnimaConfig):
    """Config for neuroplastic Anima."""

    def __init__(self, **kwargs):
        # Growth/prune parameters
        self.min_world_modules = kwargs.pop('min_world_modules', 1)
        self.max_world_modules = kwargs.pop('max_world_modules', 8)
        self.min_internal_modules = kwargs.pop('min_internal_modules', 1)
        self.max_internal_modules = kwargs.pop('max_internal_modules', 8)

        # Growth triggers
        self.growth_error_threshold = kwargs.pop('growth_error_threshold', 0.6)
        self.growth_error_window = kwargs.pop('growth_error_window', 50)
        self.growth_cooldown = kwargs.pop('growth_cooldown', 100)

        # Prune triggers
        self.prune_contribution_threshold = kwargs.pop('prune_contribution_threshold', 0.05)
        self.prune_inactive_steps = kwargs.pop('prune_inactive_steps', 200)
        self.prune_cooldown = kwargs.pop('prune_cooldown', 100)

        # Module dimensions (smaller for efficiency)
        self.module_dim = kwargs.pop('module_dim', 16)

        # Ensure base config uses aggregated dimensions
        kwargs.setdefault('world_dim', 32)  # Will be overridden by module count
        kwargs.setdefault('internal_dim', 32)
        kwargs.setdefault('use_energy', False)

        super().__init__(**kwargs)


class DynamicModule(nn.Module):
    """A single dynamically-created module."""

    def __init__(self, input_dim: int, output_dim: int, module_id: int):
        super().__init__()
        self.module_id = module_id

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
        )

        # Gating (learned importance)
        self.gate = nn.Parameter(torch.ones(1))

        # Track contribution
        self.contribution_history: List[float] = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Forward pass with contribution tracking."""
        output = self.net(x) * torch.sigmoid(self.gate)
        contribution = output.abs().mean().item()
        self.contribution_history.append(contribution)
        if len(self.contribution_history) > 100:
            self.contribution_history.pop(0)
        return output, contribution

    def avg_contribution(self) -> float:
        if not self.contribution_history:
            return 1.0  # New modules get benefit of doubt
        return np.mean(self.contribution_history)


class AnimaNeuroplastic(Anima):
    """
    Neuroplastic Anima - can grow and prune its own architecture.

    SINGLE CHANGE from base Anima:
    - Variable count N changes over time based on task demands

    Architecture:
    - World state = aggregation of world modules
    - Internal state = aggregation of internal modules
    - Modules can be added (growth) or removed (pruning)
    """

    def __init__(self, config: Optional[AnimaNeuroplasticConfig] = None):
        if config is None:
            config = AnimaNeuroplasticConfig()

        # Initialize base with minimal architecture
        super().__init__(config)
        self.config: AnimaNeuroplasticConfig = config

        # Replace fixed modules with dynamic module lists
        self.world_modules: nn.ModuleList = nn.ModuleList()
        self.internal_modules: nn.ModuleList = nn.ModuleList()

        # Module statistics
        self.module_stats: List[ModuleStats] = []
        self.next_module_id = 0

        # Initialize with minimum modules
        for _ in range(config.min_world_modules):
            self._add_world_module()
        for _ in range(config.min_internal_modules):
            self._add_internal_module()

        # Aggregation layers (combine module outputs)
        self._rebuild_aggregators()

        # Growth/prune tracking
        self.last_growth_step = -config.growth_cooldown
        self.last_prune_step = -config.prune_cooldown
        self.recent_errors: List[float] = []

        # History
        self.architecture_history: List[Dict] = []

    def _add_world_module(self) -> DynamicModule:
        """Add a new world module."""
        module = DynamicModule(
            self.config.sensory_dim,
            self.config.module_dim,
            self.next_module_id
        )
        self.world_modules.append(module)

        self.module_stats.append(ModuleStats(
            id=self.next_module_id,
            module_type='world',
            created_step=self.step_count,
        ))
        self.next_module_id += 1

        return module

    def _add_internal_module(self) -> DynamicModule:
        """Add a new internal module."""
        # Internal modules take world output as input
        input_dim = len(self.world_modules) * self.config.module_dim if self.world_modules else self.config.sensory_dim
        input_dim = max(input_dim, self.config.module_dim)  # Ensure minimum

        module = DynamicModule(
            input_dim,
            self.config.module_dim,
            self.next_module_id
        )
        self.internal_modules.append(module)

        self.module_stats.append(ModuleStats(
            id=self.next_module_id,
            module_type='internal',
            created_step=self.step_count,
        ))
        self.next_module_id += 1

        return module

    def _rebuild_aggregators(self):
        """Rebuild aggregation layers after architecture change."""
        world_out_dim = len(self.world_modules) * self.config.module_dim
        internal_out_dim = len(self.internal_modules) * self.config.module_dim

        # World aggregator
        self.world_aggregator = nn.Linear(
            max(world_out_dim, 1),
            self.config.world_dim
        )

        # Internal aggregator
        self.internal_aggregator = nn.Linear(
            max(internal_out_dim, 1),
            self.config.internal_dim
        )

        # Update internal module input sizes if needed
        for module in self.internal_modules:
            if module.net[0].in_features != world_out_dim:
                # Reinitialize with correct size
                module.net = nn.Sequential(
                    nn.Linear(max(world_out_dim, self.config.module_dim), self.config.module_dim),
                    nn.Tanh(),
                )

    def _remove_module(self, module_type: str, idx: int):
        """Remove a module by index."""
        if module_type == 'world':
            if len(self.world_modules) > self.config.min_world_modules:
                module = self.world_modules[idx]
                del self.world_modules[idx]
                # Remove stats
                self.module_stats = [s for s in self.module_stats if s.id != module.module_id]
        elif module_type == 'internal':
            if len(self.internal_modules) > self.config.min_internal_modules:
                module = self.internal_modules[idx]
                del self.internal_modules[idx]
                self.module_stats = [s for s in self.module_stats if s.id != module.module_id]

        self._rebuild_aggregators()

    def _check_growth(self) -> bool:
        """Check if we should grow architecture."""
        if self.step_count - self.last_growth_step < self.config.growth_cooldown:
            return False

        if len(self.recent_errors) < self.config.growth_error_window:
            return False

        avg_error = np.mean(self.recent_errors[-self.config.growth_error_window:])

        if avg_error > self.config.growth_error_threshold:
            # High error → need more capacity
            can_grow_world = len(self.world_modules) < self.config.max_world_modules
            can_grow_internal = len(self.internal_modules) < self.config.max_internal_modules

            if can_grow_world or can_grow_internal:
                return True

        return False

    def _check_prune(self) -> Optional[Tuple[str, int]]:
        """Check if we should prune a module. Returns (type, idx) or None."""
        if self.step_count - self.last_prune_step < self.config.prune_cooldown:
            return None

        # Check world modules
        for idx, module in enumerate(self.world_modules):
            if len(self.world_modules) <= self.config.min_world_modules:
                break
            if module.avg_contribution() < self.config.prune_contribution_threshold:
                return ('world', idx)

        # Check internal modules
        for idx, module in enumerate(self.internal_modules):
            if len(self.internal_modules) <= self.config.min_internal_modules:
                break
            if module.avg_contribution() < self.config.prune_contribution_threshold:
                return ('internal', idx)

        return None

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with dynamic architecture modification."""
        if not self.alive:
            return {'alive': False, 'cause': self.death_cause}

        self.step_count += 1

        if observation is not None:
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)

        # === FORWARD THROUGH DYNAMIC MODULES ===

        # World modules (parallel)
        world_outputs = []
        world_contributions = []
        for module in self.world_modules:
            out, contrib = module(observation if observation is not None else torch.zeros(1, self.config.sensory_dim))
            world_outputs.append(out)
            world_contributions.append(contrib)

        # Aggregate world outputs
        if world_outputs:
            world_combined = torch.cat(world_outputs, dim=-1)
            world_state = self.world_aggregator(world_combined)
        else:
            world_state = torch.zeros(1, self.config.world_dim)

        # Internal modules (take world output)
        internal_outputs = []
        internal_contributions = []
        for module in self.internal_modules:
            # Adjust input size if needed
            inp = world_combined if world_outputs else torch.zeros(1, self.config.module_dim)
            if inp.shape[-1] != module.net[0].in_features:
                # Pad or truncate
                target_size = module.net[0].in_features
                if inp.shape[-1] < target_size:
                    inp = torch.cat([inp, torch.zeros(1, target_size - inp.shape[-1])], dim=-1)
                else:
                    inp = inp[:, :target_size]

            out, contrib = module(inp)
            internal_outputs.append(out)
            internal_contributions.append(contrib)

        # Aggregate internal outputs
        if internal_outputs:
            internal_combined = torch.cat(internal_outputs, dim=-1)
            internal_state = self.internal_aggregator(internal_combined)
        else:
            internal_state = torch.zeros(1, self.config.internal_dim)

        # Update base states
        self.world_state = world_state
        self.internal_state = internal_state

        # Compute prediction error (simplified)
        if observation is not None:
            prediction = self.world_model.predict(self.world_state, self.world_memory)
            prediction_error = (observation - prediction).abs().mean().item()
        else:
            prediction_error = 0.5

        self.recent_errors.append(prediction_error)
        if len(self.recent_errors) > 200:
            self.recent_errors.pop(0)

        # Generate action from internal state
        action = self.internal_model.generate_action(self.internal_state)

        # Advance time
        activity_rate = action.abs().mean().item()
        self.phase += self.time_model.compute_phase_advance(activity_rate)
        if self.phase >= 2 * np.pi:
            self.phase -= 2 * np.pi
            self.cycle_count += 1
            self.consolidate_to_memory()

        # === SURGICAL DIFFERENCE: Dynamic architecture ===

        # Check growth
        if self._check_growth():
            # Decide what to grow based on module contributions
            world_contrib = np.mean(world_contributions) if world_contributions else 0
            internal_contrib = np.mean(internal_contributions) if internal_contributions else 0

            if world_contrib < internal_contrib and len(self.world_modules) < self.config.max_world_modules:
                self._add_world_module()
                growth_type = 'world'
            elif len(self.internal_modules) < self.config.max_internal_modules:
                self._add_internal_module()
                growth_type = 'internal'
            else:
                self._add_world_module()
                growth_type = 'world'

            self._rebuild_aggregators()
            self.last_growth_step = self.step_count

            self.architecture_history.append({
                'step': self.step_count,
                'action': 'grow',
                'type': growth_type,
                'world_count': len(self.world_modules),
                'internal_count': len(self.internal_modules),
                'trigger': 'high_error',
            })

        # Check prune
        prune_target = self._check_prune()
        if prune_target:
            module_type, idx = prune_target
            self._remove_module(module_type, idx)
            self.last_prune_step = self.step_count

            self.architecture_history.append({
                'step': self.step_count,
                'action': 'prune',
                'type': module_type,
                'world_count': len(self.world_modules),
                'internal_count': len(self.internal_modules),
                'trigger': 'low_contribution',
            })

        # Check coherence
        if world_state.norm().item() > self.config.world_coherence_max or \
           internal_state.norm().item() > self.config.internal_coherence_max:
            self.die("Coherence failure")
            return {'alive': False, 'cause': self.death_cause}

        return {
            'alive': self.alive,
            'step': self.step_count,
            'world_state_norm': world_state.norm().item(),
            'internal_state_norm': internal_state.norm().item(),
            'phase': self.phase,
            'cycle': self.cycle_count,
            'prediction_error': prediction_error,
            'action': action.detach(),
            'activity_rate': activity_rate,
            # Neuroplastic-specific
            'world_module_count': len(self.world_modules),
            'internal_module_count': len(self.internal_modules),
            'total_N': self._get_total_N(),
        }

    def _get_total_N(self) -> int:
        """Get total variable count N."""
        return sum(p.numel() for p in self.parameters())

    def get_final_report(self) -> Dict[str, Any]:
        """Extended report with architecture evolution data."""
        report = super().get_final_report()
        report['variant'] = 'Neuroplastic'
        report['final_world_modules'] = len(self.world_modules)
        report['final_internal_modules'] = len(self.internal_modules)
        report['final_N'] = self._get_total_N()
        report['architecture_history'] = self.architecture_history
        report['growth_events'] = len([h for h in self.architecture_history if h['action'] == 'grow'])
        report['prune_events'] = len([h for h in self.architecture_history if h['action'] == 'prune'])
        return report
