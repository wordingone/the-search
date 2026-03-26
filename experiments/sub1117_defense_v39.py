"""
sub1117_defense_v39.py — Bidirectional progress for defense (ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1117 --substrate experiments/sub1117_defense_v39.py

FAMILY: Reactive action switching (bidirectional progress, NO learned params)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v31 (prosecution) showed 3/5 L1 with learned per-action direction.
But direction learning is ℓ_π. Defense counter: use BOTH directions simultaneously
WITHOUT learning. Try each action in "toward" mode; if no progress after patience,
try the SAME action in "away" mode before moving to next action.

This doubles the effective action space (7 actions × 2 directions = 14 virtual
actions) without any learned parameters. If the game needs "away from initial,"
the substrate finds it by brute force, not by learning.

ONE CHANGE FROM v30: after exhausting patience on an action (toward mode),
try the same action in "away" mode (check if distance INCREASED) before
switching to the next action.

KILL: ARC < v31 prosecution (which got 3/5).
SUCCESS: ARC ≥ v31 — defense matches prosecution without direction learning.
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


class BidirectionalDefenseSubstrate:
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
        self._current_direction = 1  # +1 = toward initial, -1 = away
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

        # Explore phase
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Bidirectional progress check
        if self._current_direction > 0:
            # Toward initial: distance decreased
            progress = (self._prev_dist - dist) > 1e-4
        else:
            # Away from initial: distance increased
            progress = (dist - self._prev_dist) > 1e-4

        no_change = abs(self._prev_dist - dist) < 1e-6

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0
            if self._steps_on_action >= self._patience or no_change:
                self._steps_on_action = 0
                self._patience = 3

                if self._current_direction > 0:
                    # Tried "toward" — now try "away" with same action
                    self._current_direction = -1
                else:
                    # Tried both directions — move to next action
                    self._current_direction = 1
                    self._actions_tried_this_round += 1

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
        self._current_direction = 1
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
    "tag": "defense v39 (ℓ₁ bidirectional progress — brute force both directions, no learning)",
}

SUBSTRATE_CLASS = BidirectionalDefenseSubstrate
