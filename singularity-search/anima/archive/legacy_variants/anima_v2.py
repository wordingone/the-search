"""
A N I M A  V 2  -  S Y N T H E S I Z E D  B A S E L I N E
=========================================================

New baseline incorporating empirical findings from variant benchmarking.

Key Findings Incorporated:
1. URGENCY (from Mortal): Energy pressure improves reasoning 9.5% vs 5.8%
   → Add urgency signal WITHOUT death penalty

2. COMPRESSION (from Metamorphic): Transformation preserves structure
   → Add soft compression at consolidation, not just at crisis

3. COOPERATION (from Collective): W-I coordination improves balance
   → Add cross-state coherence mechanism

4. STABILITY (anti-Neuroplastic): Dynamic architecture disrupts reasoning
   → Keep fixed architecture, add gating instead of growth

Architecture: W/I/T with Enhanced Coupling

    ┌─────────────────────────────────────────────────┐
    │                    TRUNK                         │
    │  ┌─────────┐   coherence   ┌─────────┐         │
    │  │    W    │◄────────────►│    I    │         │
    │  │ (State) │   coupling    │ (Memory)│         │
    │  └────┬────┘               └────┬────┘         │
    │       │         urgency         │              │
    │       └─────────┬───────────────┘              │
    │                 ▼                               │
    │            ┌─────────┐                         │
    │            │    T    │                         │
    │            │ (Time)  │                         │
    │            └─────────┘                         │
    │                 │                               │
    │      compression gate (soft)                   │
    │                 ▼                               │
    │            [MEMORY]                            │
    └─────────────────────────────────────────────────┘

Empirical Targets:
- Reasoning: >9.5% (beat Mortal/Collective)
- Physics: >53% (beat Metamorphic/Neuroplastic)
- Overall: >30% (beat all variants)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class AnimaV2Config:
    """Configuration for Anima V2 - synthesized baseline."""

    # Core dimensions (same as V1 for comparison)
    world_dim: int = 64
    internal_dim: int = 64
    time_dim: int = 16
    sensory_dim: int = 16
    action_dim: int = 4

    # Memory slots
    n_world_slots: int = 8
    n_internal_slots: int = 6

    # Time model
    base_frequency: float = 0.15

    # NEW: Urgency parameters (from Mortal findings)
    urgency_baseline: float = 0.3      # Base urgency level
    urgency_error_scale: float = 2.0   # How much error increases urgency
    urgency_decay: float = 0.95        # Urgency decay per step

    # NEW: Compression parameters (from Metamorphic findings)
    compression_ratio: float = 0.7     # How much to compress at consolidation
    compression_threshold: float = 0.5 # Memory similarity threshold for compression

    # NEW: Coherence parameters (from Collective findings)
    coherence_weight: float = 0.2      # W-I coupling strength
    coherence_bonus: float = 1.5       # Bonus when W-I aligned

    # Stability (from anti-Neuroplastic findings)
    stability_gate: float = 0.9        # Gate for state updates (prevents disruption)

    # Coherence bounds (death condition)
    world_coherence_max: float = 15.0
    internal_coherence_max: float = 15.0


class UrgencyModule(nn.Module):
    """
    Urgency signal generator (inspired by Mortal's energy pressure).

    Key insight: Mortal's energy pressure improved reasoning (9.5% vs 5.8%)
    but caused physics failure (0% collision). We want the urgency benefit
    without the survival penalty.

    Urgency = f(prediction_error, time_pressure)
    - High error → high urgency → faster learning
    - But urgency doesn't kill, just modulates
    """

    def __init__(self, config: AnimaV2Config):
        super().__init__()
        self.config = config

        # Urgency state
        self.urgency = config.urgency_baseline

        # Learnable urgency modulation
        self.urgency_net = nn.Sequential(
            nn.Linear(2, 8),  # [error, current_urgency]
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, prediction_error: float) -> float:
        """Compute urgency from prediction error."""
        # Input: [error, current_urgency]
        x = torch.tensor([[prediction_error, self.urgency]], dtype=torch.float32)

        # Learned urgency adjustment
        adjustment = self.urgency_net(x).item()

        # Update urgency: decay + error-driven increase
        self.urgency = (
            self.config.urgency_decay * self.urgency +
            (1 - self.config.urgency_decay) * (
                self.config.urgency_baseline +
                self.config.urgency_error_scale * prediction_error * adjustment
            )
        )

        # Clamp to [0, 1]
        self.urgency = np.clip(self.urgency, 0.0, 1.0)

        return self.urgency

    def reset(self):
        self.urgency = self.config.urgency_baseline


class CoherenceModule(nn.Module):
    """
    W-I coherence coupling (inspired by Collective's cooperation).

    Key insight: Collective achieved balanced performance (9.5% reasoning,
    40.7% physics) through cooperation. We implement internal cooperation
    between W and I states.

    Coherence = cosine_similarity(W, project(I))
    High coherence → bonus to learning
    """

    def __init__(self, config: AnimaV2Config):
        super().__init__()
        self.config = config

        # Project I to W space for comparison
        self.i_to_w = nn.Linear(config.internal_dim, config.world_dim)

        # Project W to I space for comparison
        self.w_to_i = nn.Linear(config.world_dim, config.internal_dim)

        # Coherence history
        self.coherence_history: List[float] = []

    def forward(self, world_state: torch.Tensor, internal_state: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Compute coherence and generate coupling signals.

        Returns:
            coherence: float in [-1, 1]
            w_signal: signal to add to W update
            i_signal: signal to add to I update
        """
        # Project states
        i_projected = self.i_to_w(internal_state)
        w_projected = self.w_to_i(world_state)

        # Compute coherence (cosine similarity)
        coherence = F.cosine_similarity(
            world_state.view(1, -1),
            i_projected.view(1, -1)
        ).item()

        # Track history
        self.coherence_history.append(coherence)
        if len(self.coherence_history) > 100:
            self.coherence_history.pop(0)

        # Coupling signals (weighted by coherence)
        coupling_strength = self.config.coherence_weight * (1 + coherence) / 2

        w_signal = coupling_strength * w_projected.mean() * (i_projected - world_state)
        i_signal = coupling_strength * coherence * (w_projected - internal_state)

        return coherence, w_signal, i_signal

    def get_bonus(self) -> float:
        """Get coherence bonus for learning."""
        if not self.coherence_history:
            return 1.0
        avg_coherence = np.mean(self.coherence_history[-20:])
        if avg_coherence > 0.5:
            return self.config.coherence_bonus
        return 1.0


class CompressionModule(nn.Module):
    """
    Soft compression for memory consolidation (inspired by Metamorphic).

    Key insight: Metamorphic's transformation preserved structural knowledge
    (12% analogy, 100% collision). We apply soft compression at every
    consolidation, not just at crisis.

    Compression: memory = compress(memory, current_state, threshold)
    - Similar memories get merged
    - Dissimilar memories preserved
    """

    def __init__(self, config: AnimaV2Config):
        super().__init__()
        self.config = config

        # Compression autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(config.internal_dim, config.internal_dim // 2),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.internal_dim // 2, config.internal_dim),
            nn.Tanh(),
        )

    def compress(self, memory: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """
        Compress memory while preserving important information.

        memory: (1, n_slots, dim)
        current_state: (1, dim)
        """
        n_slots = memory.shape[1]
        dim = memory.shape[2]

        # Compute similarity of each slot to current state
        similarities = F.cosine_similarity(
            memory.squeeze(0),  # (n_slots, dim)
            current_state.expand(n_slots, -1),  # (n_slots, dim)
            dim=1
        )  # (n_slots,)

        # Compress similar slots (high similarity = more compression)
        compressed = memory.clone()

        for i in range(n_slots):
            if similarities[i] > self.config.compression_threshold:
                # Compress through autoencoder
                slot = memory[0, i, :].unsqueeze(0)
                encoded = self.encoder(slot)
                decoded = self.decoder(encoded)

                # Blend original and compressed based on similarity
                blend = similarities[i] * self.config.compression_ratio
                compressed[0, i, :] = (1 - blend) * slot + blend * decoded

        return compressed


class WorldModelV2(nn.Module):
    """Enhanced world model with stability gating."""

    def __init__(self, config: AnimaV2Config):
        super().__init__()
        self.config = config

        # Sensory encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.sensory_dim, config.world_dim),
            nn.Tanh(),
        )

        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(config.world_dim * 2, config.world_dim),
            nn.Tanh(),
            nn.Linear(config.world_dim, config.sensory_dim),
        )

        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            config.world_dim, num_heads=4, batch_first=True
        )

        # Stability gate (prevents Neuroplastic-style disruption)
        self.stability_gate = nn.Sequential(
            nn.Linear(config.world_dim * 2, config.world_dim),
            nn.Sigmoid(),
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def predict(self, world_state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Attend to memory
        attended, _ = self.memory_attention(
            world_state.unsqueeze(1),
            memory, memory
        )
        attended = attended.squeeze(1)

        # Predict
        combined = torch.cat([world_state, attended], dim=-1)
        return self.predictor(combined)

    def update(self, current_state: torch.Tensor, new_encoding: torch.Tensor,
               urgency: float) -> torch.Tensor:
        """Update world state with stability gating."""
        # Compute stability gate
        combined = torch.cat([current_state, new_encoding], dim=-1)
        gate = self.stability_gate(combined)

        # Apply stability (high stability = slow update, prevents disruption)
        base_stability = self.config.stability_gate
        effective_stability = base_stability * (1 - urgency * 0.3)  # Urgency reduces stability

        # Gated update
        new_state = effective_stability * current_state + (1 - effective_stability) * gate * new_encoding

        return new_state


class InternalModelV2(nn.Module):
    """Enhanced internal model with urgency-modulated learning."""

    def __init__(self, config: AnimaV2Config):
        super().__init__()
        self.config = config

        # Error processor
        self.error_processor = nn.Sequential(
            nn.Linear(config.sensory_dim, config.internal_dim),
            nn.Tanh(),
        )

        # State update
        self.state_update = nn.GRUCell(config.internal_dim, config.internal_dim)

        # Action generator
        self.action_head = nn.Sequential(
            nn.Linear(config.internal_dim + config.time_dim, config.internal_dim),
            nn.Tanh(),
            nn.Linear(config.internal_dim, config.action_dim),
            nn.Tanh(),
        )

        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            config.internal_dim, num_heads=4, batch_first=True
        )

    def update(self, current_state: torch.Tensor, prediction_error: torch.Tensor,
               memory: torch.Tensor, urgency: float, coherence_signal: torch.Tensor) -> torch.Tensor:
        """Update internal state with urgency modulation."""
        # Process error
        error_encoding = self.error_processor(prediction_error)

        # Attend to memory
        attended, _ = self.memory_attention(
            current_state.unsqueeze(1),
            memory, memory
        )
        attended = attended.squeeze(1)

        # Combine with coherence signal
        combined_input = error_encoding + 0.3 * attended + coherence_signal

        # Urgency-modulated update (higher urgency = bigger update)
        base_state = self.state_update(combined_input, current_state)

        # Interpolate based on urgency
        update_strength = 0.3 + 0.5 * urgency  # Range: [0.3, 0.8]
        new_state = (1 - update_strength) * current_state + update_strength * base_state

        return new_state

    def generate_action(self, internal_state: torch.Tensor,
                        time_embedding: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([internal_state, time_embedding], dim=-1)
        return self.action_head(combined)


class TimeModelV2(nn.Module):
    """Time model with urgency-coupled phase advance."""

    def __init__(self, config: AnimaV2Config):
        super().__init__()
        self.config = config
        self.phase_embedding = nn.Linear(2, config.time_dim)
        self.cycle_embedding = nn.Embedding(100, config.time_dim)

    def get_embedding(self, phase: float, cycle: int) -> torch.Tensor:
        phase_features = torch.tensor([[np.sin(phase), np.cos(phase)]], dtype=torch.float32)
        cycle_idx = torch.tensor([min(cycle, 99)])
        return self.phase_embedding(phase_features) + self.cycle_embedding(cycle_idx)

    def compute_phase_advance(self, activity_rate: float, urgency: float) -> float:
        """Phase advance coupled with urgency."""
        base_advance = self.config.base_frequency * (1 + 0.2 * activity_rate)
        urgency_boost = 1 + 0.3 * urgency  # Higher urgency = faster time
        return base_advance * urgency_boost


class AnimaV2(nn.Module):
    """
    Anima V2 - Synthesized Baseline

    Incorporates empirical findings:
    - Urgency (from Mortal): Better reasoning without death penalty
    - Compression (from Metamorphic): Preserves structural knowledge
    - Coherence (from Collective): Balanced W-I coupling
    - Stability (anti-Neuroplastic): Prevents pattern disruption
    """

    def __init__(self, config: Optional[AnimaV2Config] = None):
        super().__init__()

        if config is None:
            config = AnimaV2Config()
        self.config = config

        # Core modules
        self.world_model = WorldModelV2(config)
        self.internal_model = InternalModelV2(config)
        self.time_model = TimeModelV2(config)

        # NEW: Synthesized modules
        self.urgency_module = UrgencyModule(config)
        self.coherence_module = CoherenceModule(config)
        self.compression_module = CompressionModule(config)

        # States
        self.world_state = torch.zeros(1, config.world_dim)
        self.internal_state = torch.zeros(1, config.internal_dim)

        # Memory
        self.world_memory = torch.zeros(1, config.n_world_slots, config.world_dim)
        self.internal_memory = torch.zeros(1, config.n_internal_slots, config.internal_dim)

        # Time
        self.phase = 0.0
        self.cycle_count = 0

        # Tracking
        self.step_count = 0
        self.alive = True
        self.death_cause = None
        self.pending_internal_states: List[torch.Tensor] = []
        self.prediction_errors: List[float] = []

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Step with synthesized enhancements."""
        if not self.alive:
            return {'alive': False, 'cause': self.death_cause}

        self.step_count += 1

        # Handle observation shape
        if observation is not None:
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)

        # === PHASE 1: ENCODE ===
        if observation is not None:
            obs_encoding = self.world_model.encode(observation)
        else:
            obs_encoding = torch.zeros(1, self.config.world_dim)

        # === PHASE 2: PREDICT ===
        prediction = self.world_model.predict(self.world_state, self.world_memory)

        if observation is not None:
            prediction_error = observation - prediction
            error_magnitude = prediction_error.abs().mean().item()
        else:
            prediction_error = torch.zeros(1, self.config.sensory_dim)
            error_magnitude = 0.5

        self.prediction_errors.append(error_magnitude)
        if len(self.prediction_errors) > 100:
            self.prediction_errors.pop(0)

        # === PHASE 3: URGENCY (from Mortal) ===
        urgency = self.urgency_module(error_magnitude)

        # === PHASE 4: COHERENCE (from Collective) ===
        coherence, w_signal, i_signal = self.coherence_module(
            self.world_state, self.internal_state
        )
        coherence_bonus = self.coherence_module.get_bonus()

        # === PHASE 5: UPDATE STATES ===
        # World state update with stability gating
        self.world_state = self.world_model.update(
            self.world_state, obs_encoding, urgency
        )
        self.world_state = self.world_state + 0.1 * w_signal

        # Internal state update with urgency modulation
        self.internal_state = self.internal_model.update(
            self.internal_state, prediction_error,
            self.internal_memory, urgency, i_signal
        )

        # Track for consolidation
        self.pending_internal_states.append(self.internal_state.detach().clone())
        if len(self.pending_internal_states) > 50:
            self.pending_internal_states.pop(0)

        # === PHASE 6: TIME ===
        time_embedding = self.time_model.get_embedding(self.phase, self.cycle_count)

        # Generate action
        action = self.internal_model.generate_action(self.internal_state, time_embedding)

        # Advance phase (urgency-coupled)
        activity_rate = action.abs().mean().item()
        self.phase += self.time_model.compute_phase_advance(activity_rate, urgency)

        # Cycle completion
        if self.phase >= 2 * np.pi:
            self.phase -= 2 * np.pi
            self.cycle_count += 1
            self._consolidate_with_compression()

        # === PHASE 7: COHERENCE CHECK ===
        if self.world_state.norm().item() > self.config.world_coherence_max or \
           self.internal_state.norm().item() > self.config.internal_coherence_max:
            self.alive = False
            self.death_cause = "Coherence failure"
            return {'alive': False, 'cause': self.death_cause}

        return {
            'alive': self.alive,
            'step': self.step_count,
            'action': action.detach(),
            'prediction_error': error_magnitude,
            'urgency': urgency,
            'coherence': coherence,
            'coherence_bonus': coherence_bonus,
            'phase': self.phase,
            'cycle': self.cycle_count,
            'world_state_norm': self.world_state.norm().item(),
            'internal_state_norm': self.internal_state.norm().item(),
        }

    def _consolidate_with_compression(self):
        """Consolidate memory with soft compression (from Metamorphic)."""
        if len(self.pending_internal_states) < 10:
            return

        # Compute consolidated state
        recent = torch.stack(self.pending_internal_states[-20:], dim=1)
        consolidated = recent.mean(dim=1)

        # Compress existing memory
        self.internal_memory = self.compression_module.compress(
            self.internal_memory, consolidated
        )

        # Add new consolidated state to oldest slot
        oldest_slot = self.cycle_count % self.config.n_internal_slots
        self.internal_memory[0, oldest_slot, :] = consolidated.squeeze()

        # Also update world memory
        self.world_memory[0, self.cycle_count % self.config.n_world_slots, :] = \
            self.world_state.detach().squeeze()

    def die(self, cause: str):
        self.alive = False
        self.death_cause = cause

    def get_final_report(self) -> Dict[str, Any]:
        return {
            'variant': 'AnimaV2',
            'alive': self.alive,
            'death_cause': self.death_cause,
            'total_steps': self.step_count,
            'total_cycles': self.cycle_count,
            'final_world_norm': self.world_state.norm().item(),
            'final_internal_norm': self.internal_state.norm().item(),
            'world_memory_norm': self.world_memory.norm().item(),
            'internal_memory_norm': self.internal_memory.norm().item(),
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0,
            'final_urgency': self.urgency_module.urgency,
            'avg_coherence': np.mean(self.coherence_module.coherence_history) if self.coherence_module.coherence_history else 0,
        }


# ============================================================================
# CONVENIENCE ALIAS
# ============================================================================

AnimaCore = AnimaV2
AnimaCoreConfig = AnimaV2Config
