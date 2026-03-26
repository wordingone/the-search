"""
sub1114_prosecution_v30.py — Differential encoding (zero-cost ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1114 --substrate experiments/sub1114_prosecution_v30.py

FAMILY: Differential encoding (NEW prosecution family). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: The encoding self-modifies every step based on observation change.
Blocks that JUST CHANGED get amplified in the distance metric. Zero learned
parameters — same cost as defense v30 but with change-weighted encoding.

Architecture:
- enc = avgpool4(obs) → 256D (same as defense)
- diff_enc = avgpool4(|obs - prev_obs|) → 256D (observation change)
- combined_enc = enc * (1 + diff_enc / (max(diff_enc) + ε))
  Blocks that changed → amplified. Static blocks → base only.

Action selection: same reactive switching as defense v30.
Distance: L1 on combined_enc (change-weighted), not raw enc.

KILL: ARC < v30 defense (0.33 on solvable draws).
SUCCESS: ARC > 0.33 — prosecution beats defense with zero-cost ℓ_π.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class DifferentialEncodingSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._combined_enc_0 = None
        self._prev_obs_enc = None  # raw enc of previous obs (for diff)
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _make_combined_enc(self, enc, prev_enc):
        """Change-weighted encoding: amplify blocks that just changed."""
        if prev_enc is None:
            return enc.copy()
        diff = np.abs(enc - prev_enc)
        dmax = diff.max()
        if dmax < 1e-8:
            return enc.copy()
        # Amplify changed blocks: enc * (1 + normalized_diff)
        combined = enc * (1.0 + diff / (dmax + 1e-8))
        return combined

    def _dist_to_initial(self, combined_enc):
        if self._combined_enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(combined_enc - self._combined_enc_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        # Build change-weighted encoding (ℓ_π)
        combined_enc = self._make_combined_enc(enc, self._prev_obs_enc)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._combined_enc_0 = combined_enc.copy()
            self._prev_obs_enc = enc.copy()
            self._prev_dist = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(combined_enc)

        # Explore phase: cycle through actions
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_obs_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Reactive switching (v30 logic)
        progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
        no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
            self.r3_updates += 1
            self.att_updates_total += 1
        else:
            self._consecutive_progress = 0
            if self._steps_on_action >= self._patience or no_change:
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = 3

                if self._actions_tried_this_round >= self._n_actions:
                    self._current_action = self._rng.randint(self._n_actions)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action = (self._current_action + 1) % self._n_actions

        self._prev_obs_enc = enc.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        self._enc_0 = None
        self._combined_enc_0 = None
        self._prev_obs_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "differential encoding",
    "tag": "prosecution v30 (ℓ_π zero-cost change-weighted encoding, no learned params)",
}

SUBSTRATE_CLASS = DifferentialEncodingSubstrate
