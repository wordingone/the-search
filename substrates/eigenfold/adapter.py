"""BaseSubstrate adapter for EigenFold — matrix codebook with eigenform dynamics.

Killed: Phase 2, Eigenform Arc (~Steps 450-470). R3 FAIL (6 U elements).
The eigenform dynamics (Phi(M) = tanh(alpha*M + beta*M^2/k)) are interesting
mathematically — zero is unstable for alpha > 1, forcing non-trivial fixed points —
but the substrate uses cosine-like perturbation matching for classification
(winner = min perturbation delta), which is structurally similar to codebook
nearest-prototype selection. All hyperparameters (alpha, beta, lr, dt,
spawn_threshold, recovery_steps) are frozen and unjustified.

Key insight preserved: state IS transformation (matrices, not vectors).
Classification IS state change (perturbation that classifies also updates).
The eigenform concept (Phi(M) = M as attractor) is genuine, but the
surrounding machinery is all U.

Dependencies: rk.py (pure Python matrix utilities, no external deps).
"""
import copy
import math
import random
import sys
import os
import numpy as np

from substrates.base import BaseSubstrate, Observation

# EigenFold imports rk.py from its own directory
_eigenfold_dir = os.path.dirname(os.path.abspath(__file__))
_foldcore_dir = os.path.join(os.path.dirname(_eigenfold_dir), 'foldcore')


class EigenFoldAdapter(BaseSubstrate):
    """Wraps EigenFold into BaseSubstrate protocol.

    Maps flat observation to k x k matrix via reshaping/padding,
    then runs one fold step (classify + update).
    """

    def __init__(self, k=4, n_act=4, alpha=1.2, beta=0.8, lr=0.1,
                 recovery_steps=5, dt=0.03, max_norm=3.0,
                 init_steps=20, spawn_threshold=1.0):
        self._k = k
        self._n_act = n_act
        self._params = dict(
            k=k, alpha=alpha, beta=beta, lr=lr,
            recovery_steps=recovery_steps, dt=dt,
            max_norm=max_norm, init_steps=init_steps,
            spawn_threshold=spawn_threshold,
        )
        self._sub = self._make_sub()
        self._step_count = 0

    def _make_sub(self):
        # Import EigenFold with rk.py on path
        if _foldcore_dir not in sys.path:
            sys.path.insert(0, _foldcore_dir)
        if _eigenfold_dir not in sys.path:
            sys.path.insert(0, _eigenfold_dir)
        from eigenfold import EigenFold
        return EigenFold(**self._params)

    def _obs_to_matrix(self, obs_flat):
        """Project flat observation to k x k matrix. U element."""
        k = self._k
        needed = k * k
        if len(obs_flat) >= needed:
            vals = obs_flat[:needed]
        else:
            vals = np.pad(obs_flat, (0, needed - len(obs_flat)))
        # Reshape to k x k list-of-lists
        return [[float(vals[i * k + j]) for j in range(k)] for i in range(k)]

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)
        R = self._obs_to_matrix(flat)

        pred_label, delta = self._sub.step(R, label=None)
        self._step_count += 1

        # Derive action from element index or perturbation pattern
        if not self._sub.elements:
            return 0
        # Use winner index mod n_actions
        deltas = [self._sub._delta(M, R) for M, _ in self._sub.elements]
        win_idx = min(range(len(deltas)), key=lambda i: deltas[i])
        return win_idx % self._n_act

    def get_state(self):
        return {
            "elements": copy.deepcopy(self._sub.elements),
            "step_count": self._step_count,
        }

    def set_state(self, state):
        self._sub.elements = copy.deepcopy(state["elements"])
        self._step_count = state["step_count"]

    def frozen_elements(self):
        return [
            {"name": "elements_matrices", "class": "M",
             "justification": "Matrix codebook entries updated by perturbation absorption + eigenform recovery every step."},
            {"name": "eigenform_phi", "class": "I",
             "justification": "Phi(M) = tanh(alpha*M + beta*M^2/k) is the self-application map. Removing = no attractor."},
            {"name": "perturbation_classify", "class": "I",
             "justification": "min-delta winner selection is the classification. Removing = no output."},
            {"name": "alpha_1.2", "class": "U",
             "justification": "alpha=1.2. Designer-chosen to make zero unstable (needs >1). Exact value arbitrary."},
            {"name": "beta_0.8", "class": "U",
             "justification": "beta=0.8. Quadratic self-interaction strength. Designer-chosen."},
            {"name": "lr_0.1", "class": "U",
             "justification": "Winner update step size. Designer-chosen."},
            {"name": "spawn_threshold_1.0", "class": "U",
             "justification": "Spawn if min perturbation > 1.0. Designer-chosen threshold."},
            {"name": "recovery_steps_5", "class": "U",
             "justification": "5 eigenform recovery iterations. Designer-chosen count."},
            {"name": "dt_0.03", "class": "U",
             "justification": "Eigenform step size. Designer-chosen."},
            {"name": "obs_to_matrix_reshape", "class": "U",
             "justification": "Flat obs reshaped to k x k. Could be random projection. System doesn't choose."},
            {"name": "action_winidx_mod", "class": "U",
             "justification": "winner_index % n_actions. No principled mapping."},
        ]

    def reset(self, seed: int):
        random.seed(seed)
        self._sub = self._make_sub()
        self._step_count = 0

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
