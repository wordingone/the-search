"""
sub1136_defense_v49.py — Change-maximizing reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1136 --substrate experiments/sub1136_defense_v49.py

FAMILY: Reactive action switching (NEW goal: maximize observation change)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: ALL 32 prior defense experiments used distance-to-initial as
the goal (pick action that moves AWAY from initial state). This fails on 0%
games because initial state may not be the right reference point — the game
might want the agent to EXPLORE (visit diverse states) not RETURN (minimize
distance).

v49 tests a fundamentally different goal: maximize IMMEDIATE observation change.
Pick the action that produces the largest encoding change between consecutive
steps. Zero learned params. Same reactive switching structure as v30 but the
fitness signal is different.

Architecture:
- enc = avgpool4 (256D) — same as v30
- Goal: maximize |enc_t - enc_{t-1}| (consecutive change)
  NOT minimize |enc_t - enc_0| (distance to initial)
- Progress = current change > previous change
- Same patience/switching logic as v30

This tests whether the ℓ₁ distance-to-initial goal is the bottleneck,
not the action selection mechanism.

KILL: ARC ≤ v30 (0.3319).
SUCCESS: ARC > 0 on previously 0% game type.
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
    """avgpool4: 64x64 -> 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class ChangeMaxReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_change = 0.0
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

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_change = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        # Consecutive change (NOT distance to initial)
        change = float(np.sum(np.abs(enc - self._prev_enc)))

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_change = change
            return action

        # Reactive switching — goal = maximize consecutive change
        # "progress" means current action is producing MORE change than before
        progress = change > self._prev_change + 1e-4
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
        self._prev_change = change
        return self._current_action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_change = 0.0
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
    "tag": "defense v49 (change-maximizing reactive: maximize |enc_t - enc_{t-1}| instead of minimize |enc_t - enc_0|)",
}

SUBSTRATE_CLASS = ChangeMaxReactiveSubstrate
