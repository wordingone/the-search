"""BaseSubstrate adapter for Living Seed — online alpha plasticity organism.

Killed: Phase 2 (~Steps 480-500). The Living Seed is a cellular automaton with
NC=6 cells of D=12 dimensions. Its core equation:

  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * x_{k+1} * x_{k-1})

When alive=True, alpha shifts online using the cross-term difference between
signal-modulated and bare dynamics: |phi_sig - phi_bare| reveals per-cell,
per-dimension signal sensitivity, and alpha moves to amplify diversity on
sensitive dimensions.

The "computation IS the self-modification" claim is partially true: phi_sig
and phi_bare are produced by the same dynamics, and the plasticity signal
(their difference) cannot be separated from the computation. But the
plasticity RULE (how alpha shifts from the signal) is frozen and designer-chosen.

R3 FAIL (7 U elements): D=12, NC=6, beta=0.5, gamma=0.9, eta learning rate,
tau=0.3, eps=0.15 are all designer-chosen. The alive/still toggle is a parameter
the system doesn't choose. The observation-to-signal mapping is unjustified.

Two files exist: the_living_seed.py (self-contained) and living_seed.py
(imports from seed.py for uncertainty resolution). This adapter wraps the
self-contained Organism class from the_living_seed.py.

Dependencies: none (pure Python, standard library only).
"""
import copy
import math
import random
import os
import sys
import numpy as np

from substrates.base import BaseSubstrate, Observation

_seed_dir = os.path.dirname(os.path.abspath(__file__))

# Module-level constants matching the_living_seed.py
_D = 12
_NC = 6


class LivingSeedAdapter(BaseSubstrate):
    """Wraps the Living Seed Organism into BaseSubstrate protocol."""

    def __init__(self, n_act=4, seed=42, alive=True, eta=0.0003):
        self._n_act = n_act
        self._seed = seed
        self._alive = alive
        self._eta = eta
        self._organism = self._make_organism(seed)
        self._xs = [[0.0] * _D for _ in range(_NC)]
        self._step_count = 0

    def _make_organism(self, seed):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "the_living_seed",
            os.path.join(_seed_dir, "the_living_seed.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Organism(seed=seed, alive=self._alive, eta=self._eta)

    def _obs_to_signal(self, obs_flat):
        """Project observation to D=12 signal. U element: decimation."""
        step = max(1, len(obs_flat) // _D)
        sig = [float(obs_flat[min(i * step, len(obs_flat) - 1)]) for i in range(_D)]
        # Normalize to reasonable range for tanh dynamics
        norm = math.sqrt(sum(s * s for s in sig) + 1e-15)
        if norm > 1e-10:
            sig = [s * 0.8 / norm for s in sig]
        return sig

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)
        signal = self._obs_to_signal(flat)

        self._xs = self._organism.step(self._xs, signal)
        self._step_count += 1

        # Derive action: sum each cell's state, take argmax mod n_actions
        cell_sums = [sum(row) for row in self._xs]
        action = max(range(_NC), key=lambda i: cell_sums[i]) % self._n_act
        return action

    def get_state(self):
        return {
            "xs": copy.deepcopy(self._xs),
            "alpha": copy.deepcopy(self._organism.alpha),
            "total_alpha_shift": self._organism.total_alpha_shift,
            "step_count": self._step_count,
        }

    def set_state(self, state):
        self._xs = copy.deepcopy(state["xs"])
        self._organism.alpha = copy.deepcopy(state["alpha"])
        self._organism.total_alpha_shift = state["total_alpha_shift"]
        self._step_count = state["step_count"]

    def frozen_elements(self):
        return [
            {"name": "xs_cell_state", "class": "M",
             "justification": "NC x D cellular state updated by phi dynamics every step."},
            {"name": "alpha_per_cell_dim", "class": "M",
             "justification": "alpha[i][k] shifts online when alive=True. The self-modification."},
            {"name": "total_alpha_shift", "class": "M",
             "justification": "Cumulative alpha change. Tracking variable."},
            {"name": "phi_dynamics", "class": "I",
             "justification": "tanh(alpha*x + beta*x_{k+1}*x_{k-1}). Core computation. Removing = no dynamics."},
            {"name": "cross_term_plasticity", "class": "I",
             "justification": "|phi_sig - phi_bare| plasticity signal. Inseparable from dynamics."},
            {"name": "D_12", "class": "U",
             "justification": "D=12 per-cell dimension. Designer-chosen. Could be 8, 16, 32."},
            {"name": "NC_6", "class": "U",
             "justification": "NC=6 cells. Designer-chosen. Could be 4, 8, 12."},
            {"name": "beta_0.5", "class": "U",
             "justification": "Product term strength. Designer-chosen."},
            {"name": "gamma_0.9", "class": "U",
             "justification": "Signal coupling strength. Designer-chosen."},
            {"name": "eta_0.0003", "class": "U",
             "justification": "Alpha plasticity rate. Designer-chosen."},
            {"name": "tau_0.3", "class": "U",
             "justification": "Attention temperature. Designer-chosen."},
            {"name": "eps_0.15", "class": "U",
             "justification": "Coupling pull strength. Designer-chosen."},
            {"name": "alpha_clamp_0.3_1.8", "class": "U",
             "justification": "Alpha clipped to [0.3, 1.8]. Designer-chosen bounds."},
            {"name": "obs_to_signal_decimation", "class": "U",
             "justification": "Decimation from obs_dim to D=12 + 0.8/norm rescaling. System doesn't choose."},
            {"name": "action_cellsum_argmax", "class": "U",
             "justification": "argmax(sum(cell)) % n_actions. No principled mapping."},
        ]

    def reset(self, seed: int):
        random.seed(seed)
        self._organism = self._make_organism(seed)
        self._xs = [[0.0] * _D for _ in range(_NC)]
        self._step_count = 0

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        # Reset xs but preserve learned alpha
        self._xs = [[0.0] * _D for _ in range(_NC)]
