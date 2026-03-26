"""
sub1118_defense_v40.py — Raw pixel change detection (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1118 --substrate experiments/sub1118_defense_v40.py

FAMILY: Reactive action switching (raw pixel progress)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: avgpool4 encoding might MISS pixel-level changes that matter
for 0% games. Block averaging (4×4 → 1 value) smooths out single-pixel
changes. If a game changes 1 pixel per action, avgpool4 divides that signal
by 16, potentially dropping below detection threshold.

RAW PIXEL progress: compute L1 distance on the FULL 64×64 = 4096D observation,
not the 256D avgpool4 encoding. Same reactive switching logic. If pixel-level
changes exist but are sub-block, this will catch them.

ONE CHANGE FROM v30: distance computed on raw pixels (4096D), not avgpool4 (256D).

KILL: ARC ≤ 0 (pixel-level doesn't help — the games truly don't respond to keyboard).
SUCCESS: ARC > 0 on any previously-0% game (avgpool4 was hiding the signal).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20


class RawPixelReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._obs_0 = None
        self._prev_obs = None
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

    def _dist_to_initial(self, obs):
        if self._obs_0 is None:
            return 0.0
        return float(np.sum(np.abs(obs - self._obs_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1

        if self._obs_0 is None:
            self._obs_0 = obs.copy()
            self._prev_obs = obs.copy()
            self._prev_dist = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(obs)

        # Explore phase
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_obs = obs.copy()
            self._prev_dist = dist
            return action

        # Raw pixel progress: distance to initial decreased
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

        self._prev_obs = obs.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        self._obs_0 = None
        self._prev_obs = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0


CONFIG = {
    "n_kb": N_KB,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "reactive action switching",
    "tag": "defense v40 (ℓ₁ raw pixel 4096D progress — tests if avgpool4 hides signal)",
}

SUBSTRATE_CLASS = RawPixelReactiveSubstrate
