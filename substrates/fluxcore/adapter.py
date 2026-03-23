"""BaseSubstrate adapter for FluxCore Compressed — unified FluxCore+RK equation.

Killed: Phase 1-2 (~Steps 200-470 across variants). The FluxCore family spans
v1 through v17 plus torch/JS/CUDA variants. All share the same core: n matrix
cells in R^(k x k) with eigenform dynamics, coupling, and perception.

The compressed form unifies FluxCore's perception with RK's generation:
  dM_i = a_i * (Phi(M_i) - M_i)                     # eigenform drive
       + (1-a_i) * lr_i * (R - M_i)                  # perception
       + (1-a_i) * sum_j(w_ij * (Psi(M_i,M_j) - M_i))  # coupling

Where Phi(M) = tanh(alpha*M + beta*M^2/k) and a_i = exp(-ef_dist^2/sigma^2).

R3 FAIL: alpha, beta, dt, sigma, tau, lr_base, k_s, noise_scale, n, k,
projection matrix P are all frozen and designer-chosen. The eigenform dynamics
are genuine (zero is unstable for alpha > 1) but every hyperparameter is U.

This adapter wraps CompressedKernel from the root substrates/fluxcore_compressed.py
as the canonical representative for ALL fluxcore variants (v2 through v17, torch,
JS, CUDA are implementation variants of the same equation).

Dependencies: rk.py (pure Python matrix utilities).
"""
import copy
import math
import random
import sys
import os
import numpy as np

from substrates.base import BaseSubstrate, Observation

_substrates_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_foldcore_dir = os.path.join(_substrates_dir, 'foldcore')


class FluxCoreAdapter(BaseSubstrate):
    """Wraps CompressedKernel into BaseSubstrate protocol.

    Canonical adapter for all FluxCore variants. Uses the compressed equation
    from fluxcore_compressed.py in the substrates root.
    """

    def __init__(self, n=8, k=4, d=256, n_act=4, seed=42):
        self._n = n
        self._k = k
        self._d = d
        self._n_act = n_act
        self._seed = seed
        self._params = dict(n=n, k=k, d=d, seed=seed)
        self._sub = self._make_sub()

    def _make_sub(self):
        # CompressedKernel imports from rk.py — ensure foldcore dir is on path
        # since fluxcore_compressed.py has a hardcoded sys.path.insert
        if _foldcore_dir not in sys.path:
            sys.path.insert(0, _foldcore_dir)
        if _substrates_dir not in sys.path:
            sys.path.insert(0, _substrates_dir)

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fluxcore_compressed",
            os.path.join(_substrates_dir, "fluxcore_compressed.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.CompressedKernel(**self._params)

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float64)[:self._d]
        if len(flat) < self._d:
            flat = np.pad(flat, (0, self._d - len(flat)))
        r = flat.tolist()

        self._sub.step(r)

        # Action: use composite matrix applied to a probe vector
        C = self._sub.composite()
        k = self._k
        # Sum rows of composite to get a k-dim vector, take argmax mod n_actions
        row_sums = [sum(C[i][j] for j in range(k)) for i in range(k)]
        action = max(range(k), key=lambda i: row_sums[i])
        return action % self._n_act

    def get_state(self):
        return {
            "cells": copy.deepcopy(self._sub.cells),
            "step_count": self._sub.step_count,
        }

    def set_state(self, state):
        self._sub.cells = copy.deepcopy(state["cells"])
        self._sub.step_count = state["step_count"]

    def frozen_elements(self):
        return [
            {"name": "cells_matrices", "class": "M",
             "justification": "n matrix cells updated by eigenform + perception + coupling every step."},
            {"name": "eigenform_phi", "class": "I",
             "justification": "Phi(M) = tanh(alpha*M + beta*M^2/k). Core self-application. Removing = no dynamics."},
            {"name": "cross_apply_psi", "class": "I",
             "justification": "Psi(Mi,Mj) = tanh(alpha*(Mi+Mj)/2 + beta*Mi*Mj/k). Coupling. Removing = isolated cells."},
            {"name": "perception_drive", "class": "I",
             "justification": "(R - M_i) input absorption. Removing = blind to observations."},
            {"name": "alpha_1.2", "class": "U",
             "justification": "Eigenform linear coefficient. >1 makes zero unstable. Exact value designer-chosen."},
            {"name": "beta_0.8", "class": "U",
             "justification": "Quadratic self-interaction strength. Designer-chosen."},
            {"name": "dt_0.03", "class": "U",
             "justification": "Euler integration step. Designer-chosen."},
            {"name": "sigma_0.3", "class": "U",
             "justification": "Autonomy bandwidth. Designer-chosen."},
            {"name": "tau_0.3", "class": "U",
             "justification": "Coupling temperature. Designer-chosen."},
            {"name": "lr_base_0.08", "class": "U",
             "justification": "Base perception learning rate. Designer-chosen."},
            {"name": "k_s_20", "class": "U",
             "justification": "Surprise scaling factor. Designer-chosen."},
            {"name": "noise_0.01", "class": "U",
             "justification": "Per-step noise magnitude. Designer-chosen."},
            {"name": "n_8_cells", "class": "U",
             "justification": "8 matrix cells. Could be 4, 16. System doesn't choose."},
            {"name": "k_4_dim", "class": "U",
             "justification": "4 x 4 matrices. Could be 3, 8. System doesn't choose."},
            {"name": "projection_P", "class": "U",
             "justification": "Frozen random projection R^d -> R^(k*k). Designer-chosen at init."},
            {"name": "action_composite_argmax", "class": "U",
             "justification": "argmax(row_sums(composite)) % n_actions. No principled mapping."},
        ]

    def reset(self, seed: int):
        random.seed(seed)
        self._params['seed'] = seed
        self._sub = self._make_sub()

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
