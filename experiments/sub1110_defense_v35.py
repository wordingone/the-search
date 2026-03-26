"""
sub1110_defense_v35.py — Click-capable reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1110 --substrate experiments/sub1110_defense_v35.py

FAMILY: Reactive action switching (v30 logic, expanded action space)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: ~40% of ARC-AGI-3 games are click games (n_actions=4103).
Our substrates only use 7 keyboard actions. If a game REQUIRES clicking at
specific pixels, keyboard actions are all no-ops → permanent 0%. Expanding
the action space to include click targets should make click games solvable.

CLICK TARGET DESIGN: Full 4103-action space is intractable. avgpool4 block
centers = 256 click targets (16×16 grid). Each block center at pixel
(bx*4+2, by*4+2). Total: 7 keyboard + 256 click = 263 internal actions.

For keyboard-only games (n_actions=7): falls back to v30 behavior.
For click games (n_actions>7): uses 263 actions.

ONE CHANGE FROM v30: expanded action space for click games.
All reactive logic IDENTICAL to v21/v30.

KILL: same 0% pattern (click games not the bottleneck).
SUCCESS: L1 on any previously-0% game (click-game blindness was the barrier).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_BLOCKS = N_BLOCKS * N_BLOCKS  # 256 click targets
EXPLORE_STEPS_KB = 50
EXPLORE_STEPS_CLICK = 263  # one step per action when click-capable
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


def _block_to_click_action(block_idx):
    """Convert internal block index to environment click action.
    Block (bx, by) center pixel = (bx*4+2, by*4+2).
    Click action = 7 + px + py*64."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class ClickCapableReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_internal = N_KB
        self._has_clicks = False
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
        self._explore_steps = EXPLORE_STEPS_KB if not self._has_clicks else EXPLORE_STEPS_CLICK

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        if n_actions > N_KB:
            self._n_internal = N_KB + N_CLICK_BLOCKS  # 263
            self._has_clicks = True
        else:
            self._n_internal = min(n_actions, N_KB)
            self._has_clicks = False
        self._init_state()

    def _to_env_action(self, internal_action):
        """Map internal action index to environment action."""
        if internal_action < N_KB:
            return internal_action
        return _block_to_click_action(internal_action - N_KB)

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_internal))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._current_action = self._rng.randint(self._n_internal)
            return self._to_env_action(self._current_action)

        dist = self._dist_to_initial(enc)

        if self.step_count <= self._explore_steps:
            action = self.step_count % self._n_internal
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._to_env_action(action)

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

                if self._actions_tried_this_round >= self._n_internal:
                    self._current_action = self._rng.randint(self._n_internal)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action = (self._current_action + 1) % self._n_internal

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return self._to_env_action(self._current_action)

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
    "n_kb": N_KB,
    "n_click_blocks": N_CLICK_BLOCKS,
    "explore_steps_kb": EXPLORE_STEPS_KB,
    "explore_steps_click": EXPLORE_STEPS_CLICK,
    "max_patience": MAX_PATIENCE,
    "family": "reactive action switching",
    "tag": "defense v35 (ℓ₁ v30 + click-capable, 263 actions for click games)",
}

SUBSTRATE_CLASS = ClickCapableReactiveSubstrate
