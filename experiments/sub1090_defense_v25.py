"""
sub1090_defense_v25.py — Systematic search with tumble fallback + dual gradient

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1090 --substrate experiments/sub1090_defense_v25.py

FAMILY: Reactive action switching with bio-inspired fallback (defense-only)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Combining v21's systematic search (proven: ARC=0.2973) with
v24's random tumble fallback (proven: finds solutions v22/v23 miss) produces
a substrate that solves MORE games than either alone. Dual gradient covers
both "return to initial" and "explore new territory" game types.

SYNTHESIS OF v21 + v24:
- v21 (ARC=0.2973): systematic round-robin, single gradient (return only)
  → reliable on games where goal ≈ initial state
- v24 (ARC=0.0003): random tumble, dual gradient (return + explore)
  → finds occasional solutions v22/v23 miss

v25 ARCHITECTURE:
1. RUN: repeat action while EITHER gradient detected (dual gradient from v24)
2. SWITCH: if no gradient, try NEXT action in round-robin (systematic from v21)
3. TUMBLE: if ALL actions tried without gradient, random burst (tumble from v24)
4. ESCALATE: consecutive tumbles without improvement double burst length (Ashby)

This preserves v21's systematic guarantee (every action tried) while adding
v24's exploration (tumble when systematic fails). Dual gradient means more
games count as "making progress."

Zero learned parameters. avgpool8 (64D).

KILL: worse than v21 (0/3 games OR ARC < 0.2973 on same game type).
SUCCESS: solve v21's game type + any additional game.
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
TUMBLE_MIN = 7         # tumble burst length (> n_actions for diversity)
TUMBLE_MAX = 40
GRADIENT_THRESH = 1e-4


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class SystematicTumbleSubstrate:
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

        # Run phase (from v21)
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        # Tumble phase (from v24)
        self._tumbling = False
        self._tumble_steps_left = 0
        self._tumble_k = TUMBLE_MIN
        self._consecutive_tumbles = 0

        # Dual gradient tracking (from v24)
        self._best_dist_return = float('inf')
        self._best_dist_explore = 0.0

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

    def _detect_gradient(self, dist):
        """Dual gradient: return to initial OR explore beyond max."""
        return_grad = (self._prev_dist - dist) > GRADIENT_THRESH
        explore_grad = (dist - self._best_dist_explore) > GRADIENT_THRESH
        return return_grad or explore_grad

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        # Store initial encoding
        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._best_dist_return = 0.0
            self._best_dist_explore = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            self._best_dist_return = min(self._best_dist_return, dist)
            self._best_dist_explore = max(self._best_dist_explore, dist)
            return self.step_count % self._n_actions

        # Update best distances
        self._best_dist_return = min(self._best_dist_return, dist)
        self._best_dist_explore = max(self._best_dist_explore, dist)

        # TUMBLE PHASE
        if self._tumbling:
            self._tumble_steps_left -= 1
            if self._tumble_steps_left <= 0:
                self._tumbling = False
                self._current_action = self._rng.randint(self._n_actions)
                self._steps_on_action = 0
                self._actions_tried_this_round = 0
                improved = self._detect_gradient(dist)
                if improved:
                    self._tumble_k = TUMBLE_MIN
                    self._consecutive_tumbles = 0
                    self._consecutive_progress = 1
                    self._patience = 3
                else:
                    self._consecutive_tumbles += 1
                    self._tumble_k = min(self._tumble_k * 2, TUMBLE_MAX)
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._rng.randint(self._n_actions)

        # RUN PHASE: v21's reactive logic with dual gradient
        gradient = self._detect_gradient(dist)
        no_change = abs(self._prev_dist - dist) < 1e-6

        self._steps_on_action += 1

        if gradient:
            # Making progress — keep running
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
            self._consecutive_tumbles = 0
            self._tumble_k = TUMBLE_MIN
        else:
            # No gradient
            self._consecutive_progress = 0

            if self._steps_on_action >= self._patience or no_change:
                # Switch action (v21's systematic round-robin)
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = 3

                if self._actions_tried_this_round >= self._n_actions:
                    # ALL actions tried — TUMBLE (v24's fallback)
                    self._tumbling = True
                    self._tumble_steps_left = self._tumble_k
                    self._actions_tried_this_round = 0
                    self._prev_enc = enc.copy()
                    self._prev_dist = dist
                    return self._rng.randint(self._n_actions)
                else:
                    # Next action in round-robin
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
        self._tumbling = False
        self._tumble_steps_left = 0
        self._tumble_k = TUMBLE_MIN
        self._consecutive_tumbles = 0
        self._best_dist_return = float('inf')
        self._best_dist_explore = 0.0


CONFIG = {
    "n_dims": N_DIMS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "tumble_min": TUMBLE_MIN,
    "tumble_max": TUMBLE_MAX,
    "family": "reactive + tumble fallback",
    "tag": "defense v25 (ℓ₁ systematic + tumble + dual gradient, zero params)",
}

SUBSTRATE_CLASS = SystematicTumbleSubstrate
