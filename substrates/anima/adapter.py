"""BaseSubstrate adapter for Anima — formal intelligence emergence entity (W + I + T).

Killed: Phase 2 (~Step 480). Anima implements S = (V, tau, F, phi) with three
cleanly separated state types: W (World State, predictive model), I (Internal
State, irreversible memory), T (Temporal State, phase-based cycles).

Two distinct Anima implementations exist:
  1. anima/base.py — Full nn.Module with GRU updates, attention memory,
     phase-based temporal consolidation. ~4K parameters. Designed for
     stationary sinusoidal environments (AnimaEnvironment).
  2. anima_organism.py — W+I cellular automaton for navigation harness.
     NC=6 x D=12, no T phase. Designed for sequence discrimination.

Both are killed for the same reason:

R3 FAIL: The observation->internal mapping is unjustified (U). For base.py:
  world_dim=32, internal_dim=32, time_dim=16, sensory_dim=8, action_dim=4,
  n_world_slots=6, n_internal_slots=4, base_frequency=0.1, learning_rate=0.01,
  consolidation_strength=0.3, novelty_threshold=0.1 are ALL designer-chosen.
  The GRU gates and linear layers have parameters that learn via backprop
  (not self-directed — the training loop is external).

For anima_organism.py: w_lr=0.0003, tau=0.3, gamma=3.0, D=12, NC=6 are all U.
  The obs->signal and xs->action projections are unjustified.

This adapter wraps anima_organism.py (AnimaOrganism) as the navigation-compatible
variant. The base.py Anima entity requires a training loop and external
environment — not compatible with the process(obs)->action interface without
major scaffolding.

Dependencies: none for AnimaOrganism (pure Python). base.py Anima needs torch.
"""
import copy
import math
import random
import os
import sys
import numpy as np

from substrates.base import BaseSubstrate, Observation

_anima_dir = os.path.dirname(os.path.abspath(__file__))

# Constants matching anima_organism.py
_D = 12
_NC = 6


class AnimaAdapter(BaseSubstrate):
    """Wraps AnimaOrganism (W+I cellular automaton) into BaseSubstrate protocol.

    R3 hypothesis: FAIL. The observation -> signal projection (U), the
    xs -> action projection (U), and organism hyperparameters (w_lr, tau,
    gamma) are all unjustified.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0, alive: bool = True):
        self._n_actions_val = n_actions
        self._seed = seed
        self._alive = alive
        self._t = 0
        self._reset_internal(seed)

    def _reset_internal(self, seed: int):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "anima_organism",
            os.path.join(_anima_dir, "anima_organism.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._organism = mod.AnimaOrganism(seed=seed, alive=self._alive)
        self._xs = [[0.0] * _D for _ in range(_NC)]

    def _obs_to_signal(self, x: np.ndarray) -> list:
        """Project observation to D=12 signal. UNJUSTIFIED (U element)."""
        step = max(1, len(x) // _D)
        return [float(x[min(i * step, len(x) - 1)]) for i in range(_D)]

    def process(self, observation) -> int:
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)
        signal = self._obs_to_signal(flat)
        self._t += 1

        # Run organism step
        self._xs = self._organism.step(self._xs, signal)

        # Derive action from cell state. UNJUSTIFIED (U element).
        cell_sums = [sum(row) for row in self._xs]
        action = int(max(range(_NC), key=lambda i: cell_sums[i])) % self._n_actions_val
        return action

    def get_state(self) -> dict:
        state = {
            "xs": copy.deepcopy(self._xs),
            "t": self._t,
        }
        # AnimaOrganism internal state
        if hasattr(self._organism, 'w'):
            state["W"] = copy.deepcopy(self._organism.w)
        if hasattr(self._organism, 'mem'):
            state["mem"] = copy.deepcopy(self._organism.mem)
        if hasattr(self._organism, 'mem_slow'):
            state["mem_slow"] = copy.deepcopy(self._organism.mem_slow)
        return state

    def set_state(self, state: dict) -> None:
        self._xs = copy.deepcopy(state["xs"])
        self._t = state["t"]
        if "W" in state and hasattr(self._organism, 'w'):
            self._organism.w = copy.deepcopy(state["W"])
        if "mem" in state and hasattr(self._organism, 'mem'):
            self._organism.mem = copy.deepcopy(state["mem"])
        if "mem_slow" in state and hasattr(self._organism, 'mem_slow'):
            self._organism.mem_slow = copy.deepcopy(state["mem_slow"])

    def frozen_elements(self) -> list:
        return [
            {"name": "W_world_model", "class": "M",
             "justification": "W[i][k] learns to predict neighbor interaction. System-driven."},
            {"name": "I_accumulator", "class": "M",
             "justification": "mem accumulates prediction error. System-driven. Irreversible."},
            {"name": "xs_state", "class": "M",
             "justification": "xs (NC x D cellular state) updated every step. System-driven."},
            {"name": "phi_dynamics", "class": "I",
             "justification": "tanh(alpha*x + beta*neighbor*neighbor) dynamics. Core computation."},
            {"name": "obs_to_signal_projection", "class": "U",
             "justification": "Decimation from obs_dim to D=12. Could be PCA, random projection. System doesn't choose."},
            {"name": "action_from_cells", "class": "U",
             "justification": "argmax(sum(cell)) % n_actions. Unrelated to navigation. System doesn't choose."},
            {"name": "w_lr_0003", "class": "U",
             "justification": "w_lr=0.0003. Interior optimum from Session 19 sweep. System doesn't choose."},
            {"name": "tau_03", "class": "U",
             "justification": "tau=0.3. I accumulation rate. System doesn't choose."},
            {"name": "gamma_30", "class": "U",
             "justification": "gamma=3.0. Signal coupling strength. System doesn't choose."},
            {"name": "NC_6", "class": "U",
             "justification": "NC=6 cells. Could be 4, 8, 12. System doesn't choose."},
            {"name": "D_12", "class": "U",
             "justification": "D=12 per-cell dimension. Could be 8, 16, 32. System doesn't choose."},
        ]

    def reset(self, seed: int) -> None:
        self._t = 0
        self._reset_internal(seed)

    def on_level_transition(self) -> None:
        # Reset xs but keep learned W (organism navigates new level from current W)
        self._xs = [[0.0] * _D for _ in range(_NC)]

    @property
    def n_actions(self) -> int:
        return self._n_actions_val
