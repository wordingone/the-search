"""
sub1113_defense_v37.py — Random-sample explore (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1113 --substrate experiments/sub1113_defense_v37.py

FAMILY: Reactive action switching (v30 logic, random explore)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v30's sequential explore (0,1,2,...6,0,1,...) repeats 7
actions ~7 times in 50 steps. If only 1-2 actions produce change, the
explore phase finds them in ~7 steps but wastes the other 43 doing
nothing. Random sampling finds responsive actions with equal probability
but ALSO samples the 7 actions more uniformly across explore.

ONE CHANGE FROM v30: random action sampling during explore instead of
sequential cycling. After explore, track which actions produced ANY
change and restrict reactive switching to RESPONSIVE actions only.

KILL: ARC ≤ v30 (0.33) — random explore no better than sequential.
SUCCESS: ARC > v30 — random explore finds responsive actions faster.
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
CHANGE_THRESH = 0.5  # minimum L1 change to count as responsive


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class RandomExploreReactiveSubstrate:
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

        # Explore tracking
        self._explore_action = None
        self._action_responsiveness = np.zeros(N_KB, dtype=np.float32)
        self._responsive_actions = None  # set after explore

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
            # First explore action — random
            self._explore_action = self._rng.randint(self._n_actions)
            return self._explore_action

        dist = self._dist_to_initial(enc)

        # === Explore phase: random sampling ===
        if self.step_count <= EXPLORE_STEPS:
            # Track responsiveness of previous action
            if self._explore_action is not None and self._prev_enc is not None:
                change = float(np.sum(np.abs(enc - self._prev_enc)))
                self._action_responsiveness[self._explore_action] = max(
                    self._action_responsiveness[self._explore_action], change
                )

            # Random next action
            self._explore_action = self._rng.randint(self._n_actions)
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._explore_action

        # === Transition from explore to reactive ===
        if self._responsive_actions is None:
            # Build responsive action set
            responsive = []
            for a in range(self._n_actions):
                if self._action_responsiveness[a] > CHANGE_THRESH:
                    responsive.append(a)
            if len(responsive) < 2:
                # Not enough responsive — use all
                responsive = list(range(self._n_actions))
            self._responsive_actions = responsive
            self._current_action = 0
            self._steps_on_action = 0

        # === Reactive switching among responsive actions ===
        n_active = len(self._responsive_actions)
        progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
        no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

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

                if self._actions_tried_this_round >= n_active:
                    self._current_action = self._rng.randint(n_active)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action = (self._current_action + 1) % n_active

        action = self._responsive_actions[self._current_action % n_active]
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep responsive_actions across levels — same game type


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "change_thresh": CHANGE_THRESH,
    "family": "reactive action switching",
    "tag": "defense v37 (ℓ₁ random-sample explore + responsive action filtering)",
}

SUBSTRATE_CLASS = RandomExploreReactiveSubstrate
