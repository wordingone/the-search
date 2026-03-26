"""
sub1115_defense_v38.py — Change-magnitude reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1115 --substrate experiments/sub1115_defense_v38.py

FAMILY: Reactive action switching (DIFFERENT goal function)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v30 uses "distance to initial state decreases" as progress.
This ONLY works for games where the goal returns you toward the start.
Games requiring CONSTRUCTION (building new states) need the opposite signal.

Change-magnitude switching: "keep doing an action as long as it causes the
encoding to CHANGE. Switch when it stops producing change." This is goal-
agnostic — it works whether the solution moves toward OR away from initial.

ONE CHANGE FROM v30: progress = per-step encoding change magnitude,
NOT distance-to-initial decrease. The action keeps going as long as it
causes ANY change, regardless of direction.

KILL: ARC ≤ 0 on all games (change-magnitude is noise).
SUCCESS: ARC > 0 on any previously-0% game (direction-free progress works).
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
CHANGE_THRESH = 0.1  # minimum L1 change to count as "something happened"


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class ChangeMagnitudeReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
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

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._prev_enc is None:
            self._prev_enc = enc.copy()
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        # Explore phase: cycle through actions
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            return action

        # Progress = did the action cause ANY encoding change?
        change = float(np.sum(np.abs(enc - self._prev_enc)))
        progress = change > CHANGE_THRESH
        no_change = change < 1e-6

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
        return self._current_action

    def on_level_transition(self):
        self._prev_enc = None
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
    "change_thresh": CHANGE_THRESH,
    "family": "reactive action switching",
    "tag": "defense v38 (ℓ₁ change-magnitude progress — direction-free goal function)",
}

SUBSTRATE_CLASS = ChangeMagnitudeReactiveSubstrate
