"""
sub1108_defense_v34.py — Multi-frame observation handling (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1108 --substrate experiments/sub1108_defense_v34.py

FAMILY: Reactive action switching (v30 logic, robust observation handling)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: ARC-AGI-3 games return frame LISTS (one frame per sub-step
within an action). When a game's action involves animation (multiple sub-steps),
the frame list has length N > 1. np.array(frame_list) → shape (N, 64, 64).
Our substrates check obs.shape[0] == 1 and fall back to RANDOM for N > 1.
This means games with multi-frame actions get 0% L1 by construction — the
substrate never even runs its mechanism.

ONE CHANGE FROM v30: robust observation handling.
- obs.ndim == 3 and obs.shape[0] == 1 → squeeze (same as v30)
- obs.ndim == 3 and obs.shape[0] > 1 → take LAST frame (most recent state)
- obs.ndim == 2 and obs.shape == (64, 64) → use directly (same as v30)
All reactive logic IDENTICAL to v21/v30. Same avgpool4 encoding (256D).

KILL: same 0% pattern as v30 (multi-frame is not the bottleneck).
SUCCESS: L1 on ANY game where v30 gets 0% (format mismatch was the barrier).
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


class MultiFrameReactiveSubstrate:
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

    def _extract_frame(self, obs):
        """Robust observation extraction — handles any frame format."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3:
            # Multi-frame or single-frame: take last frame (most recent state)
            obs = obs[-1]
        if obs.ndim != 2 or obs.shape[0] != 64 or obs.shape[1] != 64:
            return None
        return obs

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def process(self, obs: np.ndarray) -> int:
        frame = self._extract_frame(obs)
        if frame is None:
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(frame)

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
    "tag": "defense v34 (ℓ₁ v30 + robust multi-frame observation handling)",
}

SUBSTRATE_CLASS = MultiFrameReactiveSubstrate
