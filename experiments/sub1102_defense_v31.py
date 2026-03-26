"""
sub1102_defense_v31.py — Fine encoding + cross-level transfer (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1102 --substrate experiments/sub1102_defense_v31.py

FAMILY: Reactive action switching (v30 encoding + v27 cross-level transfer)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v30 (avgpool4 256D) broke through on GAME_4 (ARC=0.4701).
v27 (cross-level action memory) addresses the L2+ gap but couldn't trigger
when L1=0%. Combining v30's finer encoding (enables L1 on more games) with
v27's cross-level transfer (enables L2+ when L1 works) targets BOTH gaps
simultaneously.

TWO CHANGES FROM v21:
1. Encoding: avgpool4 (256D) instead of avgpool8 (64D) [from v30]
2. Cross-level: action success counts persist across levels [from v27]

KILL: ARC < v30 (0.4701 on lucky game) → cross-level transfer hurts.
SUCCESS: L2+ on any game, or ARC > 0.5 on any game.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4  # avgpool4: finer encoding (256D)
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


class FineEncodingCrossLevelSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        # Cross-level state: persists across levels within a game
        self._action_success = np.zeros(N_KB, dtype=np.float32)
        self._init_level_state()

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def _init_level_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._action_order = None

    def _compute_action_order(self):
        order = np.argsort(-self._action_success[:self._n_actions])
        return list(order.astype(int))

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._action_success = np.zeros(N_KB, dtype=np.float32)
        self._init_level_state()

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _next_action_in_order(self):
        if self._action_order is None or len(self._action_order) == 0:
            return (self._current_action + 1) % self._n_actions
        try:
            current_idx = self._action_order.index(self._current_action)
        except ValueError:
            current_idx = -1
        next_idx = (current_idx + 1) % len(self._action_order)
        return self._action_order[next_idx]

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
            if self._action_success.sum() > 0:
                self._current_action = int(np.argmax(
                    self._action_success[:self._n_actions]))
            else:
                self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        if self._action_order is None:
            self._action_order = self._compute_action_order()
            if self._action_success.sum() > 0:
                self._current_action = self._action_order[0]

        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(self._prev_dist - dist) < 1e-6

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
            self._action_success[self._current_action] += 1.0
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
                    self._current_action = self._next_action_in_order()

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        # Keep action success counts, reset within-level state
        self._init_level_state()


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "reactive action switching + cross-level transfer",
    "tag": "defense v31 (ℓ₁ avgpool4 256D + cross-level action memory)",
}

SUBSTRATE_CLASS = FineEncodingCrossLevelSubstrate
