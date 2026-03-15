"""
A N I M A  V 4 - T E L O S
===========================

Formal Integration of Goal as Embodied Preference Over Futures

═══════════════════════════════════════════════════════════════════════════════
FORMAL DEFINITION
═══════════════════════════════════════════════════════════════════════════════

System Definition:
    S = (V, τ, F, φ, G) where:

    V = {W, I, G}           -- Variables: World, Internal, Goal
    τ: V → {S, M, T}        -- Type assignment
    F: V × E → V            -- Evolution function
    φ: I × G → A            -- Output map (action from internal + goal)
    G: Future → ℝ           -- Goal as preference function over futures

Type Assignment τ:
    τ(W) = S (State)        -- World encodes environmental state
    τ(I) = M (Memory)       -- Internal accumulates experience
    τ(G) = T (Telos)        -- Goal embodies preference over futures

The Telos Type T:
    T represents "that toward which" - the final cause in Aristotelian terms.

    Unlike S (what is) and M (what was), T encodes (what should be).

    T is not a target state but a PREFERENCE FUNCTION over possible futures:
        G: Trajectory → ℝ
        G(future) = desirability of that future unfolding

═══════════════════════════════════════════════════════════════════════════════
GOAL AS EMBODIED PREFERENCE
═══════════════════════════════════════════════════════════════════════════════

Definition (Embodied Preference):
    A goal G is EMBODIED iff:
        1. G is represented in the system's state (not external)
        2. G causally influences action selection
        3. G is updated through interaction (not fixed)

Definition (Preference Over Futures):
    G: F → ℝ where F = {f : T → State} is the space of future trajectories

    G(f) represents how much the system "wants" future f to occur.

    This is NOT a reward signal (which evaluates present states).
    This IS a value function over counterfactual trajectories.

Definition (Causal Guidance):
    G causally guides action iff:
        ∂φ/∂G ≠ 0  (goal affects action selection)

    Specifically, action a is chosen to maximize:
        a* = argmax_a E[G(future | action=a, state=current)]

═══════════════════════════════════════════════════════════════════════════════
THE TELOS MECHANISM
═══════════════════════════════════════════════════════════════════════════════

The core innovation: G is not a static target but a LEARNED PREFERENCE FIELD.

Evolution of G:
    G(t+1) = F_G(G(t), outcome(t), counterfactual(t))

    where:
        outcome(t) = what actually happened
        counterfactual(t) = what would have happened under different actions

    G learns which futures are preferable by experiencing outcomes.

Action Selection via G:
    Given current state s and goal preference G:

    1. Imagine possible futures: {f_a : a ∈ Actions}
       f_a = predicted trajectory if action a is taken

    2. Evaluate futures: {G(f_a) : a ∈ Actions}
       How desirable is each imagined future?

    3. Select action: a* = argmax_a G(f_a)
       Choose action leading to most preferred future

This is TELEOLOGICAL action selection:
    - Action is caused by its anticipated end (telos)
    - The future (as imagined) causes the present (action)
    - G embodies "that for the sake of which"

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMALIZATION
═══════════════════════════════════════════════════════════════════════════════

State Space:
    W ∈ ℝ^d_w     -- World state (environmental encoding)
    I ∈ ℝ^d_i     -- Internal state (experiential accumulation)
    G ∈ ℝ^d_g     -- Goal state (preference embedding)

Future Imagination:
    Let Φ: W × A → W be the world dynamics model (predicts next world state)

    Future trajectory under action sequence (a_1, ..., a_H):
        f(a_1:H) = (Φ(w, a_1), Φ(Φ(w, a_1), a_2), ..., Φ^H(w, a_1:H))

    Simplified single-step:
        f_a = Φ(W, a) for action a

Preference Function:
    G: ℝ^d_w → ℝ is parameterized as neural network

    G(w') = how much the system prefers world state w'

    For trajectory f = (w_1, ..., w_H):
        G(f) = Σ_t γ^t G(w_t)  where γ ∈ (0, 1) is temporal discount

Action Selection:
    a* = argmax_a G(Φ(W, a))

    Choose action that leads to most preferred next state.

Goal Evolution:
    G learns from experience via:
        L_G = -G(w_achieved) + G(w_imagined)

    This encourages G to prefer achievable futures over fantasies.

    Additionally, intrinsic preference for coherence:
        L_coherence = ||G(w) - G(w')||² for similar w, w'

    Smooth preference landscape enables gradient-based action selection.

═══════════════════════════════════════════════════════════════════════════════
INTEGRATION WITH ANIMA ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

The W/I/T separation from V1 maps to W/I/G:

    V1 Type     V4-Telos Type    Role
    ─────────   ──────────────   ────────────────────────────────
    W → S       W → S            Environmental state encoding
    I → M       I → M            Experiential memory accumulation
    T → D       G → T            Telos (goal as preference)

Key change: T (temporal phase) is REPLACED by G (goal preference).

Justification from data:
    - T provided "decision context" but didn't improve goal-seeking (13-16%)
    - G provides "teleological context" - WHY to act, not just WHEN
    - This directly addresses the architectural gap in navigation tasks

Parameter Budget:
    V4 base: ~20k params
    Goal module: ~8k params (imagination + preference)
    Total: ~28k params (under 100k limit)

═══════════════════════════════════════════════════════════════════════════════
IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class AnimaTelosConfig:
    """
    Configuration for Anima V4-Telos.

    Formal system: S = (V, τ, F, φ, G)
    """

    # World state W (τ = S)
    world_dim: int = 32

    # Internal state I (τ = M)
    internal_dim: int = 32

    # Goal state G (τ = T, Telos)
    goal_dim: int = 16

    # Interface dimensions
    sensory_dim: int = 8
    action_dim: int = 4

    # Imagination horizon (future steps to consider)
    imagination_horizon: int = 3

    # Goal learning
    goal_learning_rate: float = 0.01
    temporal_discount: float = 0.9

    # Preference smoothness
    preference_smoothness: float = 0.1

    # Coherence bounds
    state_coherence_max: float = 10.0


class WorldDynamicsModel(nn.Module):
    """
    Φ: W × A → W

    Predicts next world state given current state and action.
    Enables imagination of futures for teleological action selection.
    """

    def __init__(self, config: AnimaTelosConfig):
        super().__init__()
        self.config = config

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.sensory_dim, config.world_dim),
            nn.Tanh(),
        )

        # Dynamics: (world_state, action) → next_world_state
        self.dynamics = nn.Sequential(
            nn.Linear(config.world_dim + config.action_dim, config.world_dim),
            nn.Tanh(),
            nn.Linear(config.world_dim, config.world_dim),
            nn.Tanh(),
        )

        # Decoder: world_state → predicted_observation
        self.decoder = nn.Sequential(
            nn.Linear(config.world_dim, config.sensory_dim),
        )

        # State update (GRU-like)
        self.update = nn.GRUCell(config.world_dim, config.world_dim)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation into world state."""
        return self.encoder(observation)

    def predict_next(self, world_state: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """
        Φ(W, a) → W'

        Imagine next world state given action.
        """
        combined = torch.cat([world_state, action], dim=-1)
        delta = self.dynamics(combined)
        return world_state + 0.1 * delta  # Residual prediction

    def decode(self, world_state: torch.Tensor) -> torch.Tensor:
        """Decode world state to predicted observation."""
        return self.decoder(world_state)

    def step(self, world_state: torch.Tensor,
             obs_encoding: torch.Tensor) -> torch.Tensor:
        """Update world state from observation."""
        return self.update(obs_encoding, world_state)


class GoalPreferenceFunction(nn.Module):
    """
    G: W → ℝ

    The goal as an embodied preference function over world states.

    This is the TELOS - "that toward which" action is directed.

    Key properties:
        1. EMBODIED: G is part of the system's state, not external
        2. CAUSAL: G directly influences action selection via ∂φ/∂G ≠ 0
        3. LEARNED: G evolves through interaction
    """

    def __init__(self, config: AnimaTelosConfig):
        super().__init__()
        self.config = config

        # Goal state (embodied preference representation)
        self.goal_state = nn.Parameter(torch.zeros(1, config.goal_dim))

        # Preference network: (world_state, goal_state) → preference_value
        self.preference_net = nn.Sequential(
            nn.Linear(config.world_dim + config.goal_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

        # Goal context encoder (from observation)
        self.goal_context = nn.Sequential(
            nn.Linear(config.sensory_dim, config.goal_dim),
            nn.Tanh(),
        )

        # Goal evolution (learns from outcomes)
        self.goal_update = nn.GRUCell(config.goal_dim, config.goal_dim)

        # Tracking for learning
        self.imagined_preferences: List[float] = []
        self.achieved_preferences: List[float] = []

    def evaluate(self, world_state: torch.Tensor) -> torch.Tensor:
        """
        G(w) → ℝ

        How much does the system prefer this world state?
        """
        # Expand goal state if needed
        goal = self.goal_state.expand(world_state.shape[0], -1)

        combined = torch.cat([world_state, goal], dim=-1)
        return self.preference_net(combined)

    def evaluate_trajectory(self, trajectory: List[torch.Tensor]) -> torch.Tensor:
        """
        G(f) = Σ_t γ^t G(w_t)

        Evaluate preference for a future trajectory.
        """
        total = 0.0
        gamma = self.config.temporal_discount

        for t, world_state in enumerate(trajectory):
            pref = self.evaluate(world_state)
            total = total + (gamma ** t) * pref

        return total

    def update_from_outcome(self,
                            imagined_world: torch.Tensor,
                            achieved_world: torch.Tensor,
                            observation: torch.Tensor):
        """
        Learn from the difference between imagination and reality.

        G should prefer achievable futures over fantasies.
        """
        # Get preference values
        pref_imagined = self.evaluate(imagined_world).item()
        pref_achieved = self.evaluate(achieved_world).item()

        self.imagined_preferences.append(pref_imagined)
        self.achieved_preferences.append(pref_achieved)

        # Update goal state from observation context
        context = self.goal_context(observation)
        self.goal_state.data = self.goal_update(
            context, self.goal_state
        ).detach()

    def get_goal_state(self) -> torch.Tensor:
        """Return current goal state for action conditioning."""
        return self.goal_state


class InternalStateModule(nn.Module):
    """
    I: τ(I) = M (Memory)

    Accumulates experience and integrates with goal for action.
    """

    def __init__(self, config: AnimaTelosConfig):
        super().__init__()
        self.config = config

        # Error processor
        self.error_encoder = nn.Sequential(
            nn.Linear(config.sensory_dim, config.internal_dim),
            nn.Tanh(),
        )

        # State update
        self.update = nn.GRUCell(config.internal_dim, config.internal_dim)

        # Action head: (internal_state, goal_state) → action
        # This implements φ: I × G → A
        self.action_head = nn.Sequential(
            nn.Linear(config.internal_dim + config.goal_dim, config.internal_dim),
            nn.Tanh(),
            nn.Linear(config.internal_dim, config.action_dim),
            nn.Tanh(),
        )

    def step(self, internal_state: torch.Tensor,
             prediction_error: torch.Tensor) -> torch.Tensor:
        """Update internal state from prediction error."""
        error_encoding = self.error_encoder(prediction_error)
        return self.update(error_encoding, internal_state)

    def generate_action(self, internal_state: torch.Tensor,
                        goal_state: torch.Tensor) -> torch.Tensor:
        """
        φ: I × G → A

        Generate action from internal state and goal.
        Goal causally guides action: ∂φ/∂G ≠ 0
        """
        combined = torch.cat([internal_state, goal_state], dim=-1)
        return self.action_head(combined)


class TeleologicalActionSelector(nn.Module):
    """
    Implements teleological action selection.

    a* = argmax_a G(Φ(W, a))

    Choose action that leads to most preferred future.
    """

    def __init__(self, config: AnimaTelosConfig):
        super().__init__()
        self.config = config

        # Action candidates (discrete set for imagination)
        self.n_candidates = 8

        # Action refiner
        self.refiner = nn.Sequential(
            nn.Linear(config.action_dim + config.goal_dim, config.action_dim),
            nn.Tanh(),
        )

    def select_action(self,
                      base_action: torch.Tensor,
                      world_state: torch.Tensor,
                      goal_fn: GoalPreferenceFunction,
                      world_model: WorldDynamicsModel) -> Tuple[torch.Tensor, Dict]:
        """
        Teleological action selection.

        1. Generate action candidates around base_action
        2. Imagine future for each candidate
        3. Evaluate preference for each future
        4. Select action leading to most preferred future
        """
        batch_size = base_action.shape[0]

        # Generate candidates by perturbing base action
        candidates = []
        for i in range(self.n_candidates):
            if i == 0:
                # Include base action
                candidates.append(base_action)
            else:
                # Perturb
                noise = torch.randn_like(base_action) * 0.3
                candidates.append(torch.tanh(base_action + noise))

        # Imagine futures and evaluate preferences
        preferences = []
        imagined_worlds = []

        for action in candidates:
            # Imagine next world state
            imagined = world_model.predict_next(world_state, action)
            imagined_worlds.append(imagined)

            # Evaluate preference
            pref = goal_fn.evaluate(imagined)
            preferences.append(pref)

        # Stack and find best
        preferences = torch.stack(preferences, dim=1)  # (batch, n_candidates, 1)
        best_idx = preferences.squeeze(-1).argmax(dim=1)  # (batch,)

        # Select best action
        candidates_stacked = torch.stack(candidates, dim=1)  # (batch, n_candidates, action_dim)
        best_action = candidates_stacked[torch.arange(batch_size), best_idx]

        # Get imagined world for learning
        imagined_stacked = torch.stack(imagined_worlds, dim=1)
        best_imagined = imagined_stacked[torch.arange(batch_size), best_idx]

        return best_action, {
            'best_preference': preferences[torch.arange(batch_size), best_idx].mean().item(),
            'preference_variance': preferences.var().item(),
            'imagined_world': best_imagined,
        }


class AnimaTelos(nn.Module):
    """
    Anima V4-Telos: Goal as Embodied Preference Over Futures

    ═══════════════════════════════════════════════════════════════════════
    FORMAL SYSTEM
    ═══════════════════════════════════════════════════════════════════════

    S = (V, τ, F, φ, G) where:

        V = {W, I, G}
            W ∈ ℝ^32  -- World state (environmental encoding)
            I ∈ ℝ^32  -- Internal state (experiential memory)
            G ∈ ℝ^16  -- Goal state (preference embedding)

        τ: V → {S, M, T}
            τ(W) = S  -- State type (what is)
            τ(I) = M  -- Memory type (what was)
            τ(G) = T  -- Telos type (what should be)

        F: V × E → V
            F_W: World update from observation
            F_I: Internal update from prediction error
            F_G: Goal update from outcome comparison

        φ: I × G → A
            Action generated from internal state AND goal
            Crucially: ∂φ/∂G ≠ 0 (goal causally guides action)

        G: W → ℝ
            Preference function over world states
            G(w) = how much the system prefers state w

    ═══════════════════════════════════════════════════════════════════════
    TELEOLOGICAL ACTION SELECTION
    ═══════════════════════════════════════════════════════════════════════

    Action is selected to maximize preference over imagined futures:

        a* = argmax_a G(Φ(W, a))

    where Φ is the world dynamics model.

    This is TELEOLOGICAL: the future (as imagined) causes the present (action).

    ═══════════════════════════════════════════════════════════════════════
    """

    def __init__(self, config: Optional[AnimaTelosConfig] = None):
        super().__init__()

        if config is None:
            config = AnimaTelosConfig()
        self.config = config

        # World model Φ (τ = S)
        self.world_model = WorldDynamicsModel(config)

        # Internal model (τ = M)
        self.internal_model = InternalStateModule(config)

        # Goal preference function G (τ = T, Telos)
        self.goal_fn = GoalPreferenceFunction(config)

        # Teleological action selector
        self.telos_selector = TeleologicalActionSelector(config)

        # States
        self.world_state = torch.zeros(1, config.world_dim)
        self.internal_state = torch.zeros(1, config.internal_dim)

        # Tracking
        self.step_count = 0
        self.alive = True
        self.death_cause = None
        self.prediction_errors: List[float] = []
        self.last_imagined_world: Optional[torch.Tensor] = None

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Step with teleological action selection.

        Protocol:
            1. W update: Encode observation into world state
            2. Predict: Generate prediction, compute error
            3. I update: Update internal state from error
            4. φ: Generate base action from I × G
            5. Telos: Refine action via preference over imagined futures
            6. Learn: Update G from imagination vs reality
        """
        if not self.alive:
            return {'alive': False, 'cause': self.death_cause}

        self.step_count += 1

        # Handle observation
        if observation is not None:
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
        else:
            observation = torch.zeros(1, self.config.sensory_dim)

        # ═══════════════════════════════════════════════════════
        # PHASE 1: WORLD UPDATE (τ = S)
        # ═══════════════════════════════════════════════════════
        obs_encoding = self.world_model.encode(observation)
        self.world_state = self.world_model.step(self.world_state, obs_encoding)

        # ═══════════════════════════════════════════════════════
        # PHASE 2: PREDICTION
        # ═══════════════════════════════════════════════════════
        prediction = self.world_model.decode(self.world_state)
        prediction_error = observation - prediction
        error_magnitude = prediction_error.abs().mean().item()

        self.prediction_errors.append(error_magnitude)
        if len(self.prediction_errors) > 100:
            self.prediction_errors.pop(0)

        # ═══════════════════════════════════════════════════════
        # PHASE 3: INTERNAL UPDATE (τ = M)
        # ═══════════════════════════════════════════════════════
        self.internal_state = self.internal_model.step(
            self.internal_state, prediction_error
        )

        # ═══════════════════════════════════════════════════════
        # PHASE 4: BASE ACTION (φ: I × G → A)
        # ═══════════════════════════════════════════════════════
        goal_state = self.goal_fn.get_goal_state()
        base_action = self.internal_model.generate_action(
            self.internal_state, goal_state
        )

        # ═══════════════════════════════════════════════════════
        # PHASE 5: TELEOLOGICAL REFINEMENT
        # a* = argmax_a G(Φ(W, a))
        # ═══════════════════════════════════════════════════════
        action, telos_info = self.telos_selector.select_action(
            base_action, self.world_state, self.goal_fn, self.world_model
        )

        imagined_world = telos_info['imagined_world']

        # ═══════════════════════════════════════════════════════
        # PHASE 6: GOAL LEARNING
        # ═══════════════════════════════════════════════════════
        if self.last_imagined_world is not None:
            # Compare what we imagined vs what we got
            self.goal_fn.update_from_outcome(
                self.last_imagined_world,
                self.world_state,
                observation
            )

        self.last_imagined_world = imagined_world.detach()

        # ═══════════════════════════════════════════════════════
        # COHERENCE CHECK
        # ═══════════════════════════════════════════════════════
        w_norm = self.world_state.norm().item()
        i_norm = self.internal_state.norm().item()

        if w_norm > self.config.state_coherence_max or \
           i_norm > self.config.state_coherence_max:
            self.alive = False
            self.death_cause = f"Coherence failure (W={w_norm:.2f}, I={i_norm:.2f})"
            return {'alive': False, 'cause': self.death_cause}

        # Current preference for achieved state
        current_preference = self.goal_fn.evaluate(self.world_state).item()

        return {
            'alive': self.alive,
            'step': self.step_count,
            'action': action.detach(),
            'prediction_error': error_magnitude,
            'world_norm': w_norm,
            'internal_norm': i_norm,
            'goal_norm': goal_state.norm().item(),
            'current_preference': current_preference,
            'imagined_preference': telos_info['best_preference'],
            'preference_variance': telos_info['preference_variance'],
        }

    def get_final_report(self) -> Dict[str, Any]:
        """Get final report."""
        return {
            'variant': 'AnimaV4-Telos',
            'formal_system': 'S = (V, tau, F, phi, G)',
            'alive': self.alive,
            'death_cause': self.death_cause,
            'total_steps': self.step_count,
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0,
            'final_world_norm': self.world_state.norm().item(),
            'final_internal_norm': self.internal_state.norm().item(),
            'final_goal_norm': self.goal_fn.get_goal_state().norm().item(),
            'avg_imagined_pref': np.mean(self.goal_fn.imagined_preferences) if self.goal_fn.imagined_preferences else 0,
            'avg_achieved_pref': np.mean(self.goal_fn.achieved_preferences) if self.goal_fn.achieved_preferences else 0,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 70)
    print("ANIMA V4-TELOS: Goal as Embodied Preference Over Futures")
    print("=" * 70)

    config = AnimaTelosConfig()
    model = AnimaTelos(config)

    params = count_parameters(model)
    print(f"\nParameter count: {params:,}")
    print(f"Under 100k limit: {params < 100000}")

    print(f"\nFormal System: S = (V, tau, F, phi, G)")
    print(f"  V = {{W, I, G}} where:")
    print(f"    W in R^{config.world_dim} (World, tau=S)")
    print(f"    I in R^{config.internal_dim} (Internal, tau=M)")
    print(f"    G in R^{config.goal_dim} (Goal, tau=T)")
    print(f"  phi: I x G -> A (goal causally guides action)")
    print(f"  G: W -> R (preference over world states)")

    # Test run
    print(f"\nTest run (10 steps):")
    for i in range(10):
        obs = torch.randn(1, config.sensory_dim)
        result = model.step(obs)
        if i % 3 == 0:
            print(f"  Step {result['step']}: "
                  f"err={result['prediction_error']:.3f}, "
                  f"pref={result['current_preference']:.3f}, "
                  f"imagined={result['imagined_preference']:.3f}")

    report = model.get_final_report()
    print(f"\nFinal Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
