"""
sub1103_defense_v32.py — Ultra-fine encoding reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1103 --substrate experiments/sub1103_defense_v32.py

FAMILY: Reactive action switching (v21 logic, ultra-fine encoding)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v30 (avgpool4 256D) broke through on one game — the encoding
resolution hypothesis is confirmed. Does FINER encoding (avgpool2, 1024D)
solve even MORE games? If the 0% wall is partially encoding-limited, finer
resolution should monotonically increase the solvable fraction.

ONE CHANGE FROM v30: encoding = avgpool2 (1024D) instead of avgpool4 (256D).
All reactive logic IDENTICAL to v21/v30.

RISK: 1024D L1 distance may be noisier (more dimensions = more noise per
dimension). If L1 distance is dominated by noise, reactive switching fails.
BLOCK_SIZE=2 means each encoding dim covers only 2×2=4 pixels — close to
raw pixel noise.

KILL: L1 < v30 (finer encoding hurts — noise dominates).
SUCCESS: L1 > 3/5 on any draw or ARC > v30 (0.4701).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 2  # avgpool2: 64/2 = 32 blocks per dim
N_BLOCKS = 32
N_DIMS = N_BLOCKS * N_BLOCKS  # 1024
N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20


def _obs_to_enc(obs):
    """avgpool2: 64x64 → 32x32 = 1024D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class UltraFineReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
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

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(self._prev_dist - dist) < 1e-6

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
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

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
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
    "family": "reactive action switching",
    "tag": "defense v32 (ℓ₁ reactive switching + avgpool2 1024D ultra-fine encoding)",
}

SUBSTRATE_CLASS = UltraFineReactiveSubstrate
