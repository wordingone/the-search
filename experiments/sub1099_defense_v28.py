"""
sub1099_defense_v28.py — Bidirectional reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1099 --substrate experiments/sub1099_defense_v28.py

FAMILY: Reactive action switching (v21 variant — bidirectional progress)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v21 detects progress ONLY as "moved toward initial" (distance
decreased). This misses games where progress = moving to a NEW state (distance
increases). Adding a "new territory" criterion (distance exceeds previous
maximum) captures BOTH types of progress without adding learned params.

ONE CHANGE FROM v21: dual progress criterion.
- Original v21: progress = dist_to_initial DECREASED
- v28: progress = dist_to_initial DECREASED **OR** dist_to_initial > max_ever

Different from v24 (run-and-tumble): v24 had dual progress BUT ALSO added
tumble bursts + Ashby escalation + different patience. v28 is JUST the dual
criterion with v21's exact switching logic. Tests whether dual progress alone
helps (isolating the variable that v24 confounded).

Everything else IDENTICAL to v21: avgpool8 64D, round-robin switching,
patience counter, explore_steps=50.

KILL: ARC worse than v21 on comparable draw.
SUCCESS: L1% higher than v21 average (~40%) or ANY L2+.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64
N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class BidirectionalReactiveSubstrate:
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
        self._max_dist = 0.0  # NEW: track maximum distance from initial
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
            self._max_dist = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            self._max_dist = max(self._max_dist, dist)
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self.step_count % self._n_actions

        # DUAL progress criterion (the ONE change from v21):
        # 1. Moved toward initial (v21's original criterion)
        toward_initial = (self._prev_dist - dist) > 1e-4
        # 2. Exceeded maximum distance (new territory)
        new_territory = dist > self._max_dist + 1e-4

        progress = toward_initial or new_territory
        no_change = abs(self._prev_dist - dist) < 1e-6

        self._max_dist = max(self._max_dist, dist)
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
        self._max_dist = 0.0
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0


CONFIG = {
    "n_dims": N_DIMS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "reactive action switching",
    "tag": "defense v28 (ℓ₁ bidirectional progress, toward-initial OR new-territory)",
}

SUBSTRATE_CLASS = BidirectionalReactiveSubstrate
