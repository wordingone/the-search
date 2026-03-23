"""BaseSubstrate adapter for FoldCore — many-to-few kernel with codebook + RK matrix dynamics.

Killed: Phase 2 (~Steps 455-475). R3 FAIL (8+ U elements). FoldCore is a
two-layer architecture: (1) vector codebook in R^d with spawn/merge/update and
(2) fixed matrix cells in R^(k x k) with eigenform dynamics. The codebook layer
uses cosine matching + additive normalize update on unit sphere (codebook ban
applies). The matrix layer uses RK dynamics (eigenform + coupling + perception).

The many-to-few routing (many codebook vectors -> few matrix cells) is architecturally
interesting but doesn't resolve the core U problem: spawn_thresh, merge_thresh,
lr_codebook, alpha, beta, dt, sigma, tau, projection matrix P are all frozen.

Proven on CSI benchmark (33/33 divisions), but CSI is classification, not navigation.

Two main variants available:
  - ManyToFewKernel (foldcore_manytofew.py): full two-layer architecture
  - CompressedKernel (fluxcore_compressed.py / rk.py parent): matrix-only dynamics

This adapter wraps ManyToFewKernel as the canonical FoldCore implementation.

Dependencies: rk.py (pure Python matrix utilities, no external deps).
"""
import copy
import math
import random
import sys
import os
import numpy as np

from substrates.base import BaseSubstrate, Observation

_foldcore_dir = os.path.dirname(os.path.abspath(__file__))


class FoldCoreAdapter(BaseSubstrate):
    """Wraps ManyToFewKernel into BaseSubstrate protocol."""

    def __init__(self, n_matrix=8, k=4, d=256, n_act=4, seed=42,
                 spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015):
        self._n_matrix = n_matrix
        self._k = k
        self._d = d
        self._n_act = n_act
        self._seed = seed
        self._params = dict(
            n_matrix=n_matrix, k=k, d=d, seed=seed,
            spawn_thresh=spawn_thresh, merge_thresh=merge_thresh,
            lr_codebook=lr_codebook,
        )
        self._sub = self._make_sub()

    def _make_sub(self):
        if _foldcore_dir not in sys.path:
            sys.path.insert(0, _foldcore_dir)
        from foldcore_manytofew import ManyToFewKernel
        return ManyToFewKernel(**self._params)

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

        # Action: classify via codebook then map to action space
        if self._sub.codebook:
            # Use the codebook winner's assigned matrix cell index
            sims = [sum(v[j] * r[j] for j in range(min(len(v), len(r))))
                    / (math.sqrt(sum(x*x for x in v) + 1e-15)
                       * math.sqrt(sum(x*x for x in r) + 1e-15) + 1e-15)
                    for v in self._sub.codebook]
            winner = max(range(len(sims)), key=lambda i: sims[i])
            cell_idx = self._sub.cb_assignment[winner]
            return cell_idx % self._n_act
        return 0

    def get_state(self):
        return {
            "cells": copy.deepcopy(self._sub.cells),
            "codebook": copy.deepcopy(self._sub.codebook),
            "cb_assignment": list(self._sub.cb_assignment),
            "cb_labels": list(self._sub.cb_labels),
            "step_count": self._sub.step_count,
            "total_spawned": self._sub.total_spawned,
            "total_merged": self._sub.total_merged,
        }

    def set_state(self, state):
        self._sub.cells = copy.deepcopy(state["cells"])
        self._sub.codebook = copy.deepcopy(state["codebook"])
        self._sub.cb_assignment = list(state["cb_assignment"])
        self._sub.cb_labels = list(state["cb_labels"])
        self._sub.step_count = state["step_count"]
        self._sub.total_spawned = state["total_spawned"]
        self._sub.total_merged = state["total_merged"]

    def frozen_elements(self):
        return [
            {"name": "codebook_vectors", "class": "M",
             "justification": "Unit vectors in R^d. Spawned/merged/updated on every step."},
            {"name": "matrix_cells", "class": "M",
             "justification": "k x k matrices updated by eigenform + coupling + perception drives."},
            {"name": "cb_assignment", "class": "M",
             "justification": "Codebook-to-cell routing. Set at spawn time (nearest projected cosine)."},
            {"name": "eigenform_phi", "class": "I",
             "justification": "Phi(M) = tanh(alpha*M + beta*M^2/k). Core self-application. Removing = no dynamics."},
            {"name": "cosine_codebook_match", "class": "U",
             "justification": "Codebook ban: cosine matching on unit sphere for nearest prototype."},
            {"name": "normalize_additive_update", "class": "U",
             "justification": "Codebook ban: v += lr * r then normalize. Attract on unit sphere."},
            {"name": "spawn_thresh_0.5", "class": "U",
             "justification": "Spawn if max_sim < 0.5. Designer-chosen threshold."},
            {"name": "merge_thresh_0.95", "class": "U",
             "justification": "Merge if |cos| > 0.95. Designer-chosen threshold."},
            {"name": "lr_codebook_0.015", "class": "U",
             "justification": "Codebook update rate. Designer-chosen."},
            {"name": "alpha_1.2", "class": "U",
             "justification": "Eigenform linear coefficient. Designer-chosen."},
            {"name": "beta_0.8", "class": "U",
             "justification": "Eigenform quadratic coefficient. Designer-chosen."},
            {"name": "projection_P", "class": "U",
             "justification": "Frozen random matrix R^d -> R^(k*k). Designer-chosen at init."},
            {"name": "n_matrix_8", "class": "U",
             "justification": "8 matrix cells. Could be 4, 16. System doesn't choose."},
            {"name": "k_couple_5", "class": "U",
             "justification": "5 coupling neighbors. Designer-chosen."},
            {"name": "action_cell_mod", "class": "U",
             "justification": "cell_index % n_actions. No principled action mapping."},
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
