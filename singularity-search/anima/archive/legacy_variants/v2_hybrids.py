"""
A N I M A  V 2  H Y B R I D  V A R I A N T S
=============================================

Four hybrid variants synthesized from benchmark findings.
All share the AnimaV2 trunk (urgency + compression + coherence + stability).

Synthesis Matrix:
┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│                 │ Neuroplastic │ Metamorphic  │ Collective   │ Mortal       │
│                 │ (44% proj)   │ (100% coll)  │ (59% mom)    │ (11% seq)    │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ ADAPTIVE        │     ██████   │     ██████   │              │              │
│ grow+transform  │   dynamic N  │  compress    │              │              │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ RESONANT        │              │              │     ██████   │     ████     │
│ internal swarm  │              │              │  multi-voice │   urgency    │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ PHOENIX         │              │     ██████   │              │     ██████   │
│ staged growth   │              │  transform   │              │   energy     │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ PRESSURED       │     ████     │              │     ████     │     ██████   │
│ learning rate   │   capacity   │              │  cooperation │   pressure   │
└─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

Each variant branches from AnimaV2 trunk with ONE primary modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import copy

from ..core.anima_v2 import AnimaV2, AnimaV2Config


# ============================================================================
# VARIANT 1: ADAPTIVE (Neuroplastic + Metamorphic)
# ============================================================================

@dataclass
class AnimaAdaptiveConfig(AnimaV2Config):
    """Config for Adaptive variant - grows capacity, transforms to shrink."""

    # Growth parameters (from Neuroplastic)
    min_capacity_modules: int = 2
    max_capacity_modules: int = 6
    growth_error_threshold: float = 0.65
    growth_window: int = 40

    # Transform-shrink parameters (from Metamorphic - replaces pruning)
    shrink_contribution_threshold: float = 0.08
    shrink_inactive_steps: int = 150

    # Cooldowns
    growth_cooldown: int = 80
    shrink_cooldown: int = 80

    # Module dimension
    capacity_module_dim: int = 12


class CapacityModule(nn.Module):
    """A growable/transformable capacity module."""

    def __init__(self, input_dim: int, output_dim: int, module_id: int):
        super().__init__()
        self.module_id = module_id
        self.created_step = 0

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
        )

        # Soft gate (instead of hard pruning)
        self.gate = nn.Parameter(torch.ones(1))

        # Contribution tracking
        self.contributions: List[float] = []
        self.last_active = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        gated_out = self.net(x) * torch.sigmoid(self.gate)
        contribution = gated_out.abs().mean().item()

        self.contributions.append(contribution)
        if len(self.contributions) > 80:
            self.contributions.pop(0)

        if contribution > 0.01:
            self.last_active = 0
        else:
            self.last_active += 1

        return gated_out, contribution

    def avg_contribution(self) -> float:
        if not self.contributions:
            return 1.0
        return np.mean(self.contributions)

    def compress_into(self, other: 'CapacityModule', ratio: float = 0.5):
        """Transform this module's knowledge into another (Metamorphic-inspired)."""
        with torch.no_grad():
            # Blend weights
            for (name1, p1), (name2, p2) in zip(self.net.named_parameters(),
                                                  other.net.named_parameters()):
                if p1.shape == p2.shape:
                    p2.data = (1 - ratio) * p2.data + ratio * p1.data


class AnimaAdaptive(AnimaV2):
    """
    ADAPTIVE = Neuroplastic (growth) + Metamorphic (transform-shrink)

    Key insight: Neuroplastic's pruning disrupted learned patterns (3% reasoning).
    Metamorphic's transformation preserved them (12% analogy).

    Solution: Grow capacity when needed, but TRANSFORM modules instead of
    pruning them. Low-contribution modules compress their knowledge into
    high-contribution ones before being removed.

    Expected strengths:
    - Projectile prediction (from Neuroplastic's dynamic capacity)
    - Pattern preservation (from Metamorphic's transformation)
    """

    def __init__(self, config: Optional[AnimaAdaptiveConfig] = None):
        if config is None:
            config = AnimaAdaptiveConfig()

        super().__init__(config)
        self.config: AnimaAdaptiveConfig = config

        # Initialize tracking BEFORE adding modules
        self.next_module_id = 0
        self.last_growth_step = -config.growth_cooldown
        self.last_shrink_step = -config.shrink_cooldown
        self.architecture_history: List[Dict] = []

        # Capacity modules (replaces fixed architecture)
        self.capacity_modules: nn.ModuleList = nn.ModuleList()

        # Initialize with minimum modules
        for i in range(config.min_capacity_modules):
            self._add_capacity_module()

        # Aggregator
        self._rebuild_aggregator()

    def _add_capacity_module(self) -> CapacityModule:
        """Add a new capacity module."""
        module = CapacityModule(
            self.config.sensory_dim + self.config.internal_dim,
            self.config.capacity_module_dim,
            self.next_module_id
        )
        module.created_step = self.step_count
        self.capacity_modules.append(module)
        self.next_module_id += 1
        return module

    def _rebuild_aggregator(self):
        """Rebuild aggregator after architecture change."""
        total_dim = len(self.capacity_modules) * self.config.capacity_module_dim
        self.capacity_aggregator = nn.Linear(
            max(total_dim, 1),
            self.config.internal_dim
        )

    def _transform_shrink(self, weak_idx: int, strong_idx: int):
        """Transform weak module into strong one (Metamorphic-inspired)."""
        weak = self.capacity_modules[weak_idx]
        strong = self.capacity_modules[strong_idx]

        # Compress weak's knowledge into strong
        weak.compress_into(strong, ratio=0.3)

        # Record history
        self.architecture_history.append({
            'step': self.step_count,
            'action': 'transform_shrink',
            'from_module': weak.module_id,
            'to_module': strong.module_id,
            'module_count': len(self.capacity_modules) - 1,
        })

        # Remove weak module
        del self.capacity_modules[weak_idx]
        self._rebuild_aggregator()

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with adaptive capacity management."""
        # Base step first
        result = super().step(observation)

        if not result['alive']:
            return result

        # Process through capacity modules
        if observation is not None:
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)

            # Combine observation with internal state for capacity input
            cap_input = torch.cat([observation, self.internal_state], dim=-1)

            outputs = []
            contributions = []

            for module in self.capacity_modules:
                out, contrib = module(cap_input)
                outputs.append(out)
                contributions.append(contrib)

            if outputs:
                combined = torch.cat(outputs, dim=-1)
                capacity_signal = self.capacity_aggregator(combined)

                # Add capacity signal to internal state
                self.internal_state = self.internal_state + 0.2 * capacity_signal

        # === GROWTH CHECK (from Neuroplastic) ===
        if self.step_count - self.last_growth_step >= self.config.growth_cooldown:
            if len(self.prediction_errors) >= self.config.growth_window:
                recent_error = np.mean(self.prediction_errors[-self.config.growth_window:])

                if recent_error > self.config.growth_error_threshold:
                    if len(self.capacity_modules) < self.config.max_capacity_modules:
                        self._add_capacity_module()
                        self._rebuild_aggregator()
                        self.last_growth_step = self.step_count

                        self.architecture_history.append({
                            'step': self.step_count,
                            'action': 'grow',
                            'module_count': len(self.capacity_modules),
                            'trigger': f'error={recent_error:.3f}',
                        })

        # === TRANSFORM-SHRINK CHECK (Metamorphic-inspired) ===
        if self.step_count - self.last_shrink_step >= self.config.shrink_cooldown:
            if len(self.capacity_modules) > self.config.min_capacity_modules:
                # Find weakest and strongest modules
                contribs = [(i, m.avg_contribution()) for i, m in enumerate(self.capacity_modules)]
                contribs.sort(key=lambda x: x[1])

                weak_idx, weak_contrib = contribs[0]
                strong_idx, strong_contrib = contribs[-1]

                if weak_contrib < self.config.shrink_contribution_threshold:
                    self._transform_shrink(weak_idx, strong_idx)
                    self.last_shrink_step = self.step_count

        # Add capacity info to result
        result['capacity_modules'] = len(self.capacity_modules)
        result['total_params'] = sum(p.numel() for p in self.parameters())

        return result

    def get_final_report(self) -> Dict[str, Any]:
        report = super().get_final_report()
        report['variant'] = 'Adaptive'
        report['final_capacity_modules'] = len(self.capacity_modules)
        report['architecture_history'] = self.architecture_history
        report['growth_events'] = len([h for h in self.architecture_history if h['action'] == 'grow'])
        report['transform_events'] = len([h for h in self.architecture_history if h['action'] == 'transform_shrink'])
        return report


# ============================================================================
# VARIANT 2: RESONANT (Collective + Internal Harmony)
# ============================================================================

@dataclass
class AnimaResonantConfig(AnimaV2Config):
    """Config for Resonant variant - internal swarm of sub-states."""

    # Internal swarm parameters
    n_voices: int = 4  # Number of internal "voices"
    resonance_threshold: float = 0.6  # Agreement threshold for consensus
    resonance_bonus: float = 1.8  # Bonus when voices agree

    # Voice dimensions
    voice_dim: int = 16


class InternalVoice(nn.Module):
    """A single internal voice that processes and votes."""

    def __init__(self, input_dim: int, voice_dim: int, voice_id: int):
        super().__init__()
        self.voice_id = voice_id

        self.processor = nn.Sequential(
            nn.Linear(input_dim, voice_dim),
            nn.Tanh(),
        )

        self.vote_head = nn.Sequential(
            nn.Linear(voice_dim, voice_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input and generate vote."""
        processed = self.processor(x)
        vote = self.vote_head(processed)
        return processed, vote


class AnimaResonant(AnimaV2):
    """
    RESONANT = Collective (cooperation) + Internal Harmony

    Key insight: Collective's cooperation created balanced performance
    (9.5% reasoning, 58.9% momentum). The cooperation happened between
    EXTERNAL entities. What if we create cooperation INSIDE a single entity?

    Solution: Multiple internal "voices" that each process the input
    and vote on actions. When voices agree (resonance), learning is boosted.
    When they disagree, the system explores more.

    Expected strengths:
    - Pattern recognition (multiple perspectives)
    - Momentum tracking (collective velocity estimation)
    """

    def __init__(self, config: Optional[AnimaResonantConfig] = None):
        if config is None:
            config = AnimaResonantConfig()

        super().__init__(config)
        self.config: AnimaResonantConfig = config

        # Internal voices (swarm within)
        self.voices: nn.ModuleList = nn.ModuleList([
            InternalVoice(
                config.sensory_dim + config.internal_dim,
                config.voice_dim,
                i
            )
            for i in range(config.n_voices)
        ])

        # Consensus aggregator
        self.consensus_net = nn.Sequential(
            nn.Linear(config.n_voices * config.voice_dim, config.internal_dim),
            nn.Tanh(),
        )

        # Resonance history
        self.resonance_history: List[float] = []

    def _compute_resonance(self, votes: List[torch.Tensor]) -> float:
        """Compute agreement level between voices."""
        if len(votes) < 2:
            return 1.0

        # Pairwise cosine similarities
        similarities = []
        for i in range(len(votes)):
            for j in range(i + 1, len(votes)):
                sim = F.cosine_similarity(
                    votes[i].view(1, -1),
                    votes[j].view(1, -1)
                ).item()
                similarities.append(sim)

        return np.mean(similarities)

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with internal resonance."""
        # Base step
        result = super().step(observation)

        if not result['alive']:
            return result

        # Process through voices
        if observation is not None:
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)

            # Voice input
            voice_input = torch.cat([observation, self.internal_state], dim=-1)

            processed_states = []
            votes = []

            for voice in self.voices:
                processed, vote = voice(voice_input)
                processed_states.append(processed)
                votes.append(vote)

            # Compute resonance (agreement)
            resonance = self._compute_resonance(votes)
            self.resonance_history.append(resonance)
            if len(self.resonance_history) > 100:
                self.resonance_history.pop(0)

            # Consensus from all voices
            all_processed = torch.cat(processed_states, dim=-1)
            consensus = self.consensus_net(all_processed)

            # Apply resonance effect
            if resonance > self.config.resonance_threshold:
                # High agreement → strong consensus signal, boost learning
                consensus_strength = 0.3 * self.config.resonance_bonus
            else:
                # Low agreement → weak consensus, explore more
                consensus_strength = 0.1

            self.internal_state = self.internal_state + consensus_strength * consensus

            # Resonance boosts coherence module
            if resonance > self.config.resonance_threshold:
                self.coherence_module.coherence_history.append(resonance)

        result['resonance'] = resonance if 'resonance' in dir() else 0.5
        result['n_voices'] = len(self.voices)

        return result

    def get_final_report(self) -> Dict[str, Any]:
        report = super().get_final_report()
        report['variant'] = 'Resonant'
        report['n_voices'] = len(self.voices)
        report['avg_resonance'] = np.mean(self.resonance_history) if self.resonance_history else 0
        report['high_resonance_ratio'] = (
            sum(1 for r in self.resonance_history if r > self.config.resonance_threshold) /
            len(self.resonance_history)
        ) if self.resonance_history else 0
        return report


# ============================================================================
# VARIANT 3: PHOENIX (Metamorphic + Staged Energy Cycles)
# ============================================================================

@dataclass
class AnimaPhoenixConfig(AnimaV2Config):
    """Config for Phoenix variant - staged transformations."""

    # Life stages
    n_stages: int = 3  # egg → juvenile → adult
    stage_cycle_thresholds: List[int] = field(default_factory=lambda: [5, 15, 30])

    # Stage-specific parameters (W/I/T balance shifts)
    stage_world_weights: List[float] = field(default_factory=lambda: [0.6, 0.4, 0.3])
    stage_internal_weights: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.5])
    stage_time_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2])

    # Transformation energy
    transformation_cost: float = 0.15
    energy_recovery_rate: float = 0.005


class AnimaPhoenix(AnimaV2):
    """
    PHOENIX = Metamorphic (transformation) + Staged Growth

    Key insight: Metamorphic's crisis-triggered transformation preserved
    structure (100% collision, 12% analogy). But transformation only
    happened at crisis. What if transformation is PLANNED?

    Solution: Phoenix has life stages (egg → juvenile → adult). Each
    stage has different W/I/T balance. Transformation happens at stage
    transitions, compressing and restructuring knowledge appropriately.

    Expected strengths:
    - Long-term adaptation (staged development)
    - Collision avoidance (transformation preserves reactive patterns)
    """

    def __init__(self, config: Optional[AnimaPhoenixConfig] = None):
        if config is None:
            config = AnimaPhoenixConfig()

        super().__init__(config)
        self.config: AnimaPhoenixConfig = config

        # Stage tracking
        self.current_stage = 0
        self.stage_names = ['Egg', 'Juvenile', 'Adult']
        self.transformation_history: List[Dict] = []

        # Energy for transformation
        self.phoenix_energy = 1.0

        # Stage-specific layers
        self.stage_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.internal_dim, config.internal_dim),
                nn.Tanh(),
            )
            for _ in range(config.n_stages)
        ])

    def _get_stage_weights(self) -> Tuple[float, float, float]:
        """Get W/I/T weights for current stage."""
        return (
            self.config.stage_world_weights[self.current_stage],
            self.config.stage_internal_weights[self.current_stage],
            self.config.stage_time_weights[self.current_stage],
        )

    def _transform_to_next_stage(self):
        """Transform to next life stage (Metamorphic-inspired)."""
        if self.current_stage >= self.config.n_stages - 1:
            return  # Already at final stage

        old_stage = self.current_stage

        # Compress current knowledge
        with torch.no_grad():
            # Compress internal state through current stage modulator
            compressed = self.stage_modulators[self.current_stage](self.internal_state)

            # Compress memory
            memory_mean = self.internal_memory.mean(dim=1, keepdim=True)
            self.internal_memory = 0.7 * self.internal_memory + 0.3 * memory_mean

        # Advance stage
        self.current_stage += 1

        # Initialize new stage with compressed knowledge
        with torch.no_grad():
            new_modulator = self.stage_modulators[self.current_stage]
            self.internal_state = new_modulator(compressed)

        # Energy cost
        self.phoenix_energy -= self.config.transformation_cost

        # Record
        self.transformation_history.append({
            'step': self.step_count,
            'cycle': self.cycle_count,
            'from_stage': self.stage_names[old_stage],
            'to_stage': self.stage_names[self.current_stage],
            'energy_after': self.phoenix_energy,
        })

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with staged development."""
        # Check for stage transition
        if self.current_stage < self.config.n_stages - 1:
            threshold = self.config.stage_cycle_thresholds[self.current_stage]
            if self.cycle_count >= threshold:
                self._transform_to_next_stage()

        # Get stage-specific weights
        w_weight, i_weight, t_weight = self._get_stage_weights()

        # Base step
        result = super().step(observation)

        if not result['alive']:
            return result

        # Apply stage-specific modulation
        stage_modulated = self.stage_modulators[self.current_stage](self.internal_state)
        self.internal_state = (1 - i_weight) * self.internal_state + i_weight * stage_modulated

        # Energy recovery
        self.phoenix_energy = min(1.0, self.phoenix_energy + self.config.energy_recovery_rate)

        # Add stage info to result
        result['stage'] = self.stage_names[self.current_stage]
        result['stage_idx'] = self.current_stage
        result['phoenix_energy'] = self.phoenix_energy
        result['w_weight'] = w_weight
        result['i_weight'] = i_weight

        return result

    def get_final_report(self) -> Dict[str, Any]:
        report = super().get_final_report()
        report['variant'] = 'Phoenix'
        report['final_stage'] = self.stage_names[self.current_stage]
        report['transformations'] = len(self.transformation_history)
        report['transformation_history'] = self.transformation_history
        return report


# ============================================================================
# VARIANT 4: PRESSURED (Mortal Learning Rate + Collective Stability)
# ============================================================================

@dataclass
class AnimaPressuredConfig(AnimaV2Config):
    """Config for Pressured variant - energy modulates learning, not survival."""

    # Pressure energy (NOT survival energy)
    pressure_init: float = 0.8
    pressure_decay: float = 0.001
    pressure_recovery_rate: float = 0.002

    # Learning modulation
    min_learning_rate: float = 0.2
    max_learning_rate: float = 0.9
    pressure_learning_scale: float = 2.0  # How much pressure affects learning

    # Collective stability (prevents Mortal's physics failure)
    stability_cooperation: float = 0.3
    stability_threshold: float = 0.4


class AnimaPressured(AnimaV2):
    """
    PRESSURED = Mortal (energy pressure) + Collective (stability)

    Key insight: Mortal's energy pressure improved reasoning (11% sequence,
    11% conditional) but destroyed physics (0% collision). The pressure
    created URGENCY but also PANIC.

    Solution: Keep the pressure for urgency but use it to modulate LEARNING
    RATE, not survival. Add Collective's stability mechanism to prevent
    panic-driven physics failures.

    Expected strengths:
    - Sequential reasoning (from pressure-driven focus)
    - Conditional logic (from heightened attention)
    - Stable physics (from collective stability mechanism)
    """

    def __init__(self, config: Optional[AnimaPressuredConfig] = None):
        if config is None:
            config = AnimaPressuredConfig()

        super().__init__(config)
        self.config: AnimaPressuredConfig = config

        # Pressure state (like energy but doesn't kill)
        self.pressure = config.pressure_init

        # Stability accumulator (Collective-inspired)
        self.stability_accumulator = 0.5

        # Learning rate history
        self.learning_rate_history: List[float] = []

        # Pressure-aware update layer
        self.pressure_modulator = nn.Sequential(
            nn.Linear(config.internal_dim + 1, config.internal_dim),
            nn.Tanh(),
        )

    def _compute_learning_rate(self) -> float:
        """Compute learning rate from pressure (higher pressure = faster learning)."""
        # Invert pressure: low pressure (like low energy) = high learning rate
        inverted = 1.0 - self.pressure

        # Scale to learning rate range
        lr = self.config.min_learning_rate + inverted * (
            self.config.max_learning_rate - self.config.min_learning_rate
        ) * self.config.pressure_learning_scale

        return np.clip(lr, self.config.min_learning_rate, self.config.max_learning_rate)

    def _update_stability(self, action_magnitude: float):
        """Update stability accumulator (Collective-inspired)."""
        # Stability increases when actions are moderate
        if action_magnitude < 0.5:
            self.stability_accumulator += self.config.stability_cooperation * 0.1
        else:
            self.stability_accumulator -= self.config.stability_cooperation * 0.05

        self.stability_accumulator = np.clip(self.stability_accumulator, 0.0, 1.0)

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with pressure-modulated learning."""
        # Base step
        result = super().step(observation)

        if not result['alive']:
            return result

        # Update pressure (decay over time)
        self.pressure -= self.config.pressure_decay

        # Recovery from good predictions
        if result['prediction_error'] < 0.3:
            self.pressure += self.config.pressure_recovery_rate * 2
        else:
            self.pressure += self.config.pressure_recovery_rate * 0.5

        self.pressure = np.clip(self.pressure, 0.1, 1.0)

        # Compute pressure-modulated learning rate
        learning_rate = self._compute_learning_rate()
        self.learning_rate_history.append(learning_rate)
        if len(self.learning_rate_history) > 100:
            self.learning_rate_history.pop(0)

        # Apply pressure modulation to internal state
        pressure_tensor = torch.tensor([[self.pressure]], dtype=torch.float32)
        pressure_input = torch.cat([self.internal_state, pressure_tensor], dim=-1)
        pressure_signal = self.pressure_modulator(pressure_input)

        self.internal_state = (1 - learning_rate) * self.internal_state + learning_rate * pressure_signal

        # Update stability (Collective-inspired)
        action_mag = result['action'].abs().mean().item()
        self._update_stability(action_mag)

        # Stability dampens extreme actions (prevents Mortal's panic)
        if self.stability_accumulator > self.config.stability_threshold:
            # High stability = moderate actions
            result['action'] = result['action'] * (0.5 + 0.5 * self.stability_accumulator)

        result['pressure'] = self.pressure
        result['learning_rate'] = learning_rate
        result['stability'] = self.stability_accumulator

        return result

    def get_final_report(self) -> Dict[str, Any]:
        report = super().get_final_report()
        report['variant'] = 'Pressured'
        report['final_pressure'] = self.pressure
        report['avg_learning_rate'] = np.mean(self.learning_rate_history) if self.learning_rate_history else 0
        report['final_stability'] = self.stability_accumulator
        return report


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'AnimaAdaptive', 'AnimaAdaptiveConfig',
    'AnimaResonant', 'AnimaResonantConfig',
    'AnimaPhoenix', 'AnimaPhoenixConfig',
    'AnimaPressured', 'AnimaPressuredConfig',
]
