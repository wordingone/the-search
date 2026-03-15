"""
A N I M A  V 3  -  C O L L E C T I V E - D E M O C R A T I C
==============================================================

V3 introduces democratic mode selection where agents vote on operating modes.
Based on empirical findings from V1/V2 benchmarking:

Mathematical Foundation: S' = (V, τ, F, φ, Ω, Γ) where:
  V = Variables (N degrees of freedom)
  τ = Type assignment {W, I, T} → {State, Memory, Decision}
  F = Evolution function with cross-type coupling
  φ = Output map (V_T → Actions)
  Ω = Collective operator (multi-entity cooperation)
  Γ = Mode selector (variant selection function)

Key Empirical Insights Incorporated:
1. Collective has lowest reasoning variance (generalization)
2. V2-Core has best projectile prediction (urgency mechanism)
3. Mortal creates reasoning focus but physics panic
4. Neuroplastic achieves 100% collision but disrupts learning
5. Mechanism Synergy: Urgency + Compression + Cooperation enhance each other
6. Mechanism Interference: Dynamic N interferes with all other mechanisms

Democratic Mode Selection:
- URGENT:    High error → fast learning, prediction focus
- TRANSFORM: Low energy → compress and restructure
- COOPERATE: Multiple signals → consensus decision
- EXPAND:    Novel pattern → grow capacity (carefully!)
- STABLE:    Normal operation → standard W/I/T evolution

Performance Targets (based on data synthesis):
- Reasoning: >9.5% avg (beat V1-Metamorphic)
- Physics: >52% avg (beat V1-Collective)
- Overall: >30% (beat all existing variants)

Anti-Patterns (from data):
- EXPAND cooldown: 100 steps minimum
- TRANSFORM blocked during collision warning
- Energy affects learning rate, never survival
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .anima_v2 import AnimaV2Config, UrgencyModule, CoherenceModule, CompressionModule


class Mode(Enum):
    """Operating modes for democratic selection."""
    URGENT = "urgent"       # High error → fast learning
    TRANSFORM = "transform" # Energy crisis → compress/restructure
    COOPERATE = "cooperate" # Multiple signals → consensus
    EXPAND = "expand"       # Novel pattern → grow capacity
    STABLE = "stable"       # Normal operation


@dataclass
class AnimaV3Config(AnimaV2Config):
    """Configuration for Anima V3 - Collective-Democratic."""

    # Collective parameters
    n_agents: int = 3                    # Number of agents in collective
    contribution_decay: float = 0.1      # How fast contribution weights decay

    # Democratic mode selection
    mode_blend_strength: float = 0.7     # How much mode affects behavior (0=ignore, 1=full)
    vote_temperature: float = 1.0        # Softmax temperature for voting

    # Mode-specific parameters
    urgent_lr_boost: float = 2.0         # Learning rate multiplier in URGENT mode
    transform_ratio: float = 0.7         # Compression ratio in TRANSFORM mode
    cooperate_weight: float = 0.5        # How much to blend consensus in COOPERATE mode
    expand_cooldown: int = 100           # Steps between EXPAND operations
    expand_threshold: float = 0.8        # Sustained error threshold for EXPAND

    # Anti-patterns
    collision_warning_threshold: float = 0.3  # Block TRANSFORM when collision imminent
    max_expansion_per_session: int = 3        # Limit total expansions


class ModeVoter(nn.Module):
    """
    Each agent votes on operating mode based on local state.

    Inputs:
        - prediction_error: recent error magnitude
        - energy_level: current energy (if using energy)
        - coherence: W-I alignment
        - novelty: how novel is current observation
        - collision_warning: imminent collision detected

    Outputs:
        - vote: softmax over modes [URGENT, TRANSFORM, COOPERATE, EXPAND, STABLE]
    """

    def __init__(self, config: AnimaV3Config):
        super().__init__()
        self.config = config

        # Input: [error, energy, coherence, novelty, collision_warning, urgency_history]
        self.voter = nn.Sequential(
            nn.Linear(6, 16),
            nn.Tanh(),
            nn.Linear(16, 5),  # 5 modes
        )

        # Mode bias (learned specialization)
        self.mode_bias = nn.Parameter(torch.zeros(5))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute vote distribution over modes.

        features: (batch, 6) - [error, energy, coherence, novelty, collision_warning, urgency_history]
        returns: (batch, 5) - softmax over modes
        """
        logits = self.voter(features) + self.mode_bias
        return F.softmax(logits / self.config.vote_temperature, dim=-1)


class CollectiveAgent(nn.Module):
    """
    Individual agent within the collective.

    Each agent:
    - Has its own W/I/T states
    - Votes on operating mode
    - Contributes to collective consensus
    - Can specialize through mode_bias learning
    """

    def __init__(self, config: AnimaV3Config, agent_id: int):
        super().__init__()
        self.config = config
        self.agent_id = agent_id

        # Core components (shared architecture, independent weights)
        # Sensory encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.sensory_dim, config.world_dim),
            nn.Tanh(),
        )

        # World model
        self.world_predictor = nn.Sequential(
            nn.Linear(config.world_dim * 2, config.world_dim),
            nn.Tanh(),
            nn.Linear(config.world_dim, config.sensory_dim),
        )
        self.world_memory_attn = nn.MultiheadAttention(
            config.world_dim, num_heads=4, batch_first=True
        )
        self.world_update = nn.GRUCell(config.world_dim, config.world_dim)

        # Internal model
        self.error_processor = nn.Sequential(
            nn.Linear(config.sensory_dim, config.internal_dim),
            nn.Tanh(),
        )
        self.internal_update = nn.GRUCell(config.internal_dim, config.internal_dim)
        self.internal_memory_attn = nn.MultiheadAttention(
            config.internal_dim, num_heads=4, batch_first=True
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(config.internal_dim + config.time_dim, config.internal_dim),
            nn.Tanh(),
            nn.Linear(config.internal_dim, config.action_dim),
            nn.Tanh(),
        )

        # Time model
        self.phase_embedding = nn.Linear(2, config.time_dim)
        self.cycle_embedding = nn.Embedding(100, config.time_dim)

        # Mode voter
        self.mode_voter = ModeVoter(config)

        # Urgency module
        self.urgency_module = UrgencyModule(config)

        # Compression module
        self.compression_module = CompressionModule(config)

        # States
        self.world_state = torch.zeros(1, config.world_dim)
        self.internal_state = torch.zeros(1, config.internal_dim)
        self.world_memory = torch.zeros(1, config.n_world_slots, config.world_dim)
        self.internal_memory = torch.zeros(1, config.n_internal_slots, config.internal_dim)

        # Time
        self.phase = 0.0
        self.cycle_count = 0

        # Tracking
        self.prediction_errors: List[float] = []
        self.novelty_scores: List[float] = []
        self.contribution_score = 1.0 / config.n_agents  # Start equal
        self.last_mode = Mode.STABLE

    def encode_observation(self, observation: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode observation to world space."""
        if observation is None:
            return torch.zeros(1, self.config.world_dim)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        return self.encoder(observation)

    def predict(self) -> torch.Tensor:
        """Generate prediction from current state."""
        attended, _ = self.world_memory_attn(
            self.world_state.unsqueeze(1),
            self.world_memory, self.world_memory
        )
        attended = attended.squeeze(1)
        combined = torch.cat([self.world_state, attended], dim=-1)
        return self.world_predictor(combined)

    def get_time_embedding(self) -> torch.Tensor:
        """Get temporal embedding."""
        phase_features = torch.tensor(
            [[np.sin(self.phase), np.cos(self.phase)]],
            dtype=torch.float32
        )
        cycle_idx = torch.tensor([min(self.cycle_count, 99)])
        return self.phase_embedding(phase_features) + self.cycle_embedding(cycle_idx)

    def vote_on_mode(self, error: float, energy: float, coherence: float,
                     novelty: float, collision_warning: float) -> torch.Tensor:
        """Vote on operating mode."""
        urgency_history = np.mean(self.prediction_errors[-10:]) if self.prediction_errors else 0.5

        features = torch.tensor([[
            error, energy, coherence, novelty, collision_warning, urgency_history
        ]], dtype=torch.float32)

        return self.mode_voter(features)

    def step_with_mode(self, observation: Optional[torch.Tensor],
                       mode: Mode, mode_strength: float,
                       consensus_world: Optional[torch.Tensor] = None,
                       consensus_internal: Optional[torch.Tensor] = None
                       ) -> Dict[str, Any]:
        """
        Step agent with mode-specific behavior.

        Returns prediction error, action, and state info.
        """
        # Encode observation
        obs_encoding = self.encode_observation(observation)

        # Predict
        prediction = self.predict()

        # Compute error
        if observation is not None:
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            prediction_error = observation - prediction
            error_magnitude = prediction_error.abs().mean().item()
        else:
            prediction_error = torch.zeros(1, self.config.sensory_dim)
            error_magnitude = 0.5

        self.prediction_errors.append(error_magnitude)
        if len(self.prediction_errors) > 100:
            self.prediction_errors.pop(0)

        # Mode-specific learning rate
        if mode == Mode.URGENT:
            lr_multiplier = self.config.urgent_lr_boost
        else:
            lr_multiplier = 1.0

        # Urgency signal
        urgency = self.urgency_module(error_magnitude)

        # Update world state
        world_input = obs_encoding
        base_world = self.world_update(world_input, self.world_state)

        # Mode-specific world update
        if mode == Mode.TRANSFORM and mode_strength > 0.5:
            # Compress world state
            base_world = self._compress_state(base_world, self.config.transform_ratio)
        elif mode == Mode.COOPERATE and consensus_world is not None:
            # Blend with consensus
            base_world = (1 - self.config.cooperate_weight * mode_strength) * base_world + \
                        self.config.cooperate_weight * mode_strength * consensus_world

        # Apply update with learning rate
        update_strength = 0.3 * lr_multiplier * (1 + 0.3 * urgency)
        self.world_state = (1 - update_strength) * self.world_state + update_strength * base_world

        # Update internal state
        error_encoding = self.error_processor(prediction_error)
        attended, _ = self.internal_memory_attn(
            self.internal_state.unsqueeze(1),
            self.internal_memory, self.internal_memory
        )
        attended = attended.squeeze(1)
        internal_input = error_encoding + 0.3 * attended

        base_internal = self.internal_update(internal_input, self.internal_state)

        # Mode-specific internal update
        if mode == Mode.TRANSFORM and mode_strength > 0.5:
            base_internal = self._compress_state(base_internal, self.config.transform_ratio)
        elif mode == Mode.COOPERATE and consensus_internal is not None:
            base_internal = (1 - self.config.cooperate_weight * mode_strength) * base_internal + \
                           self.config.cooperate_weight * mode_strength * consensus_internal

        # Apply update
        self.internal_state = (1 - update_strength) * self.internal_state + update_strength * base_internal

        # Generate action
        time_embedding = self.get_time_embedding()
        action = self.action_head(torch.cat([self.internal_state, time_embedding], dim=-1))

        # Advance phase
        activity_rate = action.abs().mean().item()
        phase_advance = self.config.base_frequency * (1 + 0.2 * activity_rate)
        if mode == Mode.URGENT:
            phase_advance *= 1.3  # Faster time in urgent mode
        self.phase += phase_advance

        # Cycle completion
        if self.phase >= 2 * np.pi:
            self.phase -= 2 * np.pi
            self.cycle_count += 1
            self._consolidate_memory()

        # Track mode
        self.last_mode = mode

        return {
            'action': action.detach(),
            'prediction_error': error_magnitude,
            'urgency': urgency,
            'world_state': self.world_state.detach().clone(),
            'internal_state': self.internal_state.detach().clone(),
            'phase': self.phase,
            'cycle': self.cycle_count,
        }

    def _compress_state(self, state: torch.Tensor, ratio: float) -> torch.Tensor:
        """Compress state by reducing magnitude of small components."""
        threshold = state.abs().quantile(1 - ratio)
        mask = state.abs() > threshold
        return state * mask.float() + state * 0.1 * (~mask).float()

    def _consolidate_memory(self):
        """Consolidate to memory."""
        # World memory
        slot = self.cycle_count % self.config.n_world_slots
        self.world_memory[0, slot, :] = self.world_state.detach().squeeze()

        # Internal memory (compressed)
        slot = self.cycle_count % self.config.n_internal_slots
        self.internal_memory = self.compression_module.compress(
            self.internal_memory, self.internal_state
        )
        self.internal_memory[0, slot, :] = self.internal_state.detach().squeeze()

    def expand_capacity(self) -> bool:
        """
        Expand capacity (EXPAND mode).
        Returns True if expansion happened.

        Note: This is very conservative due to empirical finding that
        Neuroplastic's dynamic N interferes with learning.
        """
        # Add a small amount to existing dimensions via slight noise
        # Instead of actually growing N (which disrupts), we expand effective capacity
        # by initializing new memory slots
        old_slot = np.random.randint(0, self.config.n_internal_slots)
        self.internal_memory[0, old_slot, :] *= 0.9  # Decay old
        return True


class CollectiveOperator(nn.Module):
    """
    The Ω operator: combines multiple agent states into collective behavior.

    Ω: S^n → S (combines n entities into emergent behavior)

    Ω(s₁, s₂, ..., sₙ) = {
        W_collective = Σᵢ αᵢ · Wᵢ  where αᵢ = contribution(sᵢ)
        I_collective = ⊕ᵢ Iᵢ       where ⊕ = weighted memory union
        T_collective = sync(T₁...Tₙ) where sync = phase alignment
    }
    """

    def __init__(self, config: AnimaV3Config):
        super().__init__()
        self.config = config

        # Contribution weighting network
        self.contribution_net = nn.Sequential(
            nn.Linear(2, 8),  # [prediction_error, coherence]
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def compute_contributions(self, agents: List[CollectiveAgent]) -> torch.Tensor:
        """Compute contribution weight for each agent."""
        contributions = []
        for agent in agents:
            avg_error = np.mean(agent.prediction_errors[-20:]) if agent.prediction_errors else 0.5
            features = torch.tensor([[avg_error, agent.contribution_score]], dtype=torch.float32)
            contribution = self.contribution_net(features).item()
            contributions.append(contribution)

        # Normalize to sum to 1
        total = sum(contributions)
        if total > 0:
            contributions = [c / total for c in contributions]
        else:
            contributions = [1.0 / len(agents)] * len(agents)

        return torch.tensor(contributions)

    def combine_world_states(self, agents: List[CollectiveAgent],
                             contributions: torch.Tensor) -> torch.Tensor:
        """Combine world states weighted by contribution."""
        world_states = torch.stack([agent.world_state for agent in agents], dim=0)
        weights = contributions.view(-1, 1, 1)
        return (world_states * weights).sum(dim=0)

    def combine_internal_states(self, agents: List[CollectiveAgent],
                                contributions: torch.Tensor) -> torch.Tensor:
        """Combine internal states weighted by contribution."""
        internal_states = torch.stack([agent.internal_state for agent in agents], dim=0)
        weights = contributions.view(-1, 1, 1)
        return (internal_states * weights).sum(dim=0)

    def sync_phases(self, agents: List[CollectiveAgent]) -> float:
        """Synchronize phases across agents (average)."""
        phases = [agent.phase for agent in agents]
        return np.mean(phases)


class AnimaV3(nn.Module):
    """
    Anima V3 - Collective-Democratic

    A collective of agents that democratically select operating modes.
    Implements S' = (V, τ, F, φ, Ω, Γ) where:
      Ω = Collective operator
      Γ = Mode selector (democratic vote)

    Key features:
    - Multiple agents vote on operating mode
    - Consensus-based mode selection
    - Contribution-weighted state combination
    - Anti-patterns enforced (EXPAND cooldown, TRANSFORM block during collision)
    """

    def __init__(self, config: Optional[AnimaV3Config] = None):
        super().__init__()

        if config is None:
            config = AnimaV3Config()
        self.config = config

        # Create agents
        self.agents = nn.ModuleList([
            CollectiveAgent(config, i) for i in range(config.n_agents)
        ])

        # Collective operator
        self.collective_operator = CollectiveOperator(config)

        # Coherence module (shared)
        self.coherence_module = CoherenceModule(config)

        # Shared resource pool (energy)
        self.E_pool = 3.0
        self.E_regen_rate = 0.01

        # Tracking
        self.step_count = 0
        self.alive = True
        self.death_cause = None

        # Mode tracking
        self.current_mode = Mode.STABLE
        self.mode_history: List[Mode] = []
        self.last_expand_step = -config.expand_cooldown
        self.total_expansions = 0

        # Performance tracking
        self.collective_errors: List[float] = []
        self.collective_coherence: List[float] = []

    def detect_collision_warning(self, observation: Optional[torch.Tensor]) -> float:
        """
        Detect if collision is imminent.
        Returns 0-1 where 1 = collision imminent.

        Uses prediction variance across agents as proxy for uncertainty.
        """
        if observation is None:
            return 0.0

        predictions = []
        for agent in self.agents:
            pred = agent.predict()
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        variance = predictions.var(dim=0).mean().item()

        # High variance = high uncertainty = potential collision
        return min(1.0, variance * 5)  # Scale to [0, 1]

    def democratic_vote(self, error: float, coherence: float,
                        novelty: float, collision_warning: float) -> Tuple[Mode, torch.Tensor]:
        """
        Agents vote on operating mode.

        Returns:
            mode: the selected mode
            mode_weights: soft weights over all modes for blending
        """
        # Compute individual votes
        votes = []
        for agent in self.agents:
            vote = agent.vote_on_mode(
                error=error,
                energy=self.E_pool / 3.0,  # Normalized energy
                coherence=coherence,
                novelty=novelty,
                collision_warning=collision_warning
            )
            votes.append(vote * agent.contribution_score)

        # Combine votes (weighted by contribution)
        total_contribution = sum(agent.contribution_score for agent in self.agents)
        combined_vote = torch.stack(votes, dim=0).sum(dim=0) / total_contribution

        # Mode selection with constraints
        mode_idx = combined_vote.argmax().item()
        modes = [Mode.URGENT, Mode.TRANSFORM, Mode.COOPERATE, Mode.EXPAND, Mode.STABLE]
        selected_mode = modes[mode_idx]

        # Anti-pattern: Block TRANSFORM during collision warning
        if selected_mode == Mode.TRANSFORM and collision_warning > self.config.collision_warning_threshold:
            selected_mode = Mode.STABLE
            combined_vote[0, 1] = 0  # Zero out TRANSFORM vote
            combined_vote = combined_vote / combined_vote.sum()  # Renormalize

        # Anti-pattern: EXPAND cooldown
        if selected_mode == Mode.EXPAND:
            if self.step_count - self.last_expand_step < self.config.expand_cooldown:
                selected_mode = Mode.STABLE
                combined_vote[0, 3] = 0  # Zero out EXPAND vote
                combined_vote = combined_vote / combined_vote.sum()
            elif self.total_expansions >= self.config.max_expansion_per_session:
                selected_mode = Mode.STABLE
                combined_vote[0, 3] = 0
                combined_vote = combined_vote / combined_vote.sum()

        return selected_mode, combined_vote.squeeze(0)

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Step the collective with democratic mode selection.

        Protocol:
        1. Each agent processes observation independently
        2. Each agent votes on mode based on local state
        3. Collective computes consensus mode
        4. Mode-specific modulation applied to all agents
        5. Agent states combined via Ω
        6. Shared pool updated based on collective performance
        """
        if not self.alive:
            return {'alive': False, 'cause': self.death_cause}

        self.step_count += 1

        # === PHASE 1: INDIVIDUAL PROCESSING ===
        # Each agent makes prediction, we get collective error
        errors = []
        for agent in self.agents:
            pred = agent.predict()
            if observation is not None:
                obs = observation.unsqueeze(0) if observation.dim() == 1 else observation
                err = (obs - pred).abs().mean().item()
            else:
                err = 0.5
            errors.append(err)

        collective_error = np.mean(errors)
        self.collective_errors.append(collective_error)
        if len(self.collective_errors) > 100:
            self.collective_errors.pop(0)

        # === PHASE 2: COMPUTE COLLECTIVE STATE ===
        contributions = self.collective_operator.compute_contributions(self.agents)
        consensus_world = self.collective_operator.combine_world_states(self.agents, contributions)
        consensus_internal = self.collective_operator.combine_internal_states(self.agents, contributions)

        # Coherence between collective states
        coherence, _, _ = self.coherence_module(consensus_world, consensus_internal)
        self.collective_coherence.append(coherence)
        if len(self.collective_coherence) > 100:
            self.collective_coherence.pop(0)

        # Novelty estimation
        if len(self.collective_errors) > 10:
            recent_mean = np.mean(self.collective_errors[-10:])
            historical_mean = np.mean(self.collective_errors)
            novelty = abs(collective_error - recent_mean) / (historical_mean + 1e-6)
        else:
            novelty = 0.5

        # Collision warning
        collision_warning = self.detect_collision_warning(observation)

        # === PHASE 3: DEMOCRATIC MODE SELECTION ===
        selected_mode, mode_weights = self.democratic_vote(
            error=collective_error,
            coherence=coherence,
            novelty=min(1.0, novelty),
            collision_warning=collision_warning
        )

        self.current_mode = selected_mode
        self.mode_history.append(selected_mode)
        if len(self.mode_history) > 100:
            self.mode_history.pop(0)

        # === PHASE 4: AGENT STEPS WITH MODE ===
        actions = []
        for agent in self.agents:
            result = agent.step_with_mode(
                observation=observation,
                mode=selected_mode,
                mode_strength=self.config.mode_blend_strength,
                consensus_world=consensus_world,
                consensus_internal=consensus_internal
            )
            actions.append(result['action'])

            # Update contribution score based on error
            agent_error = result['prediction_error']
            if agent_error < collective_error:
                agent.contribution_score = min(1.0, agent.contribution_score + 0.01)
            else:
                agent.contribution_score = max(0.1, agent.contribution_score - 0.005)

        # === PHASE 5: HANDLE EXPAND MODE ===
        if selected_mode == Mode.EXPAND:
            # Pick the agent with highest contribution to expand
            best_agent = max(self.agents, key=lambda a: a.contribution_score)
            if best_agent.expand_capacity():
                self.last_expand_step = self.step_count
                self.total_expansions += 1

        # === PHASE 6: ENERGY POOL UPDATE ===
        # Energy cost based on mode
        mode_costs = {
            Mode.STABLE: 0.01,
            Mode.URGENT: 0.02,
            Mode.TRANSFORM: 0.015,
            Mode.COOPERATE: 0.005,  # Cooperation is efficient
            Mode.EXPAND: 0.03,
        }
        self.E_pool -= mode_costs.get(selected_mode, 0.01)
        self.E_pool += self.E_regen_rate * (1 - collective_error)  # Regen on good predictions
        self.E_pool = max(0.1, min(3.0, self.E_pool))  # Clamp

        # === PHASE 7: COMBINE ACTIONS ===
        # Weighted action combination
        actions_tensor = torch.stack(actions, dim=0)
        weights = contributions.view(-1, 1, 1)
        collective_action = (actions_tensor * weights).sum(dim=0)

        # === PHASE 8: COHERENCE CHECK ===
        world_norm = consensus_world.norm().item()
        internal_norm = consensus_internal.norm().item()

        if world_norm > self.config.world_coherence_max or \
           internal_norm > self.config.internal_coherence_max:
            self.alive = False
            self.death_cause = "Collective coherence failure"
            return {'alive': False, 'cause': self.death_cause}

        return {
            'alive': self.alive,
            'step': self.step_count,
            'action': collective_action.detach(),
            'mode': selected_mode.value,
            'mode_weights': mode_weights.detach(),
            'collective_error': collective_error,
            'coherence': coherence,
            'E_pool': self.E_pool,
            'contributions': contributions.tolist(),
            'collision_warning': collision_warning,
            'world_norm': world_norm,
            'internal_norm': internal_norm,
        }

    def get_final_report(self) -> Dict[str, Any]:
        """Get comprehensive report of collective performance."""
        mode_counts = {mode.value: 0 for mode in Mode}
        for mode in self.mode_history:
            mode_counts[mode.value] += 1

        return {
            'variant': 'AnimaV3-CollectiveDemocratic',
            'alive': self.alive,
            'death_cause': self.death_cause,
            'total_steps': self.step_count,
            'total_cycles': max(agent.cycle_count for agent in self.agents),
            'n_agents': len(self.agents),
            'final_E_pool': self.E_pool,
            'total_expansions': self.total_expansions,
            'mode_distribution': mode_counts,
            'avg_collective_error': np.mean(self.collective_errors) if self.collective_errors else 0,
            'avg_coherence': np.mean(self.collective_coherence) if self.collective_coherence else 0,
            'agent_contributions': [agent.contribution_score for agent in self.agents],
            'agent_cycles': [agent.cycle_count for agent in self.agents],
        }


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

AnimaCollectiveDemocratic = AnimaV3
AnimaCollectiveDemocraticConfig = AnimaV3Config
