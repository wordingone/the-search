"""
sub1089_defense_v24.py — Dual-gradient run-and-tumble (E. coli / Ashby inspired)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1089 --substrate experiments/sub1089_defense_v24.py

FAMILY: Bio-inspired reactive (defense-only architecture)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Biological run-and-tumble with dual gradient detection solves
more games than single-gradient reactive switching (v21). E. coli's chemotaxis:
run when gradient is favorable, random tumble burst when gradient vanishes.
Ashby's ultrastability: escalate tumble duration when stuck longer. If
model-free biological exploration outperforms model-free systematic cycling,
the issue with v21 is its SEARCH STRATEGY, not its lack of learned params.

INSPIRATION:
- E. coli chemotaxis: run-and-tumble is the most successful model-free
  exploration strategy in nature. Billions of years of evolution.
  Key: tumble is RANDOM (not systematic like round-robin).
- Ashby's homeostat: ultrastability via escalating random reconfiguration.
  The longer essential variables are out of bounds, the more aggressive
  the reconfiguration. Key: escalation.

BUILDS ON v21 (ARC=0.2973) but fundamentally different strategy:
- v21: systematic round-robin when stuck → limited exploration
- v24: random tumble burst when stuck → wider exploration
- v21: single gradient (closer to initial only)
- v24: dual gradient (closer to initial OR further than ever before)
  → catches games where goal ≠ initial state

ARCHITECTURE:
- avgpool8 (64D) encoding — same scale as v21
- Dual gradient detection:
  1. RETURN gradient: distance to initial DECREASED
  2. EXPLORE gradient: distance to initial EXCEEDED previous maximum
  Progress = EITHER gradient detected.
- RUN phase: repeat action while making progress (either gradient)
- TUMBLE phase: when stuck (patience exhausted), take K random actions
  - K starts at TUMBLE_MIN=5
  - Each consecutive tumble without improvement: K doubles (up to TUMBLE_MAX=40)
  - On any progress: K resets to TUMBLE_MIN
- Track best_dist_return (minimum distance, for return gradient)
- Track best_dist_explore (maximum distance, for explore gradient)
- Zero learned parameters

WHY DIFFERENT FROM PROSECUTION:
- No alpha, no W_pred, no attention, no trajectory buffer
- No learned parameters of any kind
- Fixed encoding, random tumble, threshold-based gradients

WHY DIFFERENT FROM v21-v23:
- v21: systematic round-robin → random tumble (different search strategy)
- v22: multi-scale → single scale (simpler)
- v23: oscillation detection → no oscillation detection (simpler)
- NEW: dual gradient (return + explore) vs single gradient (return only)

KILL: worse than v21 (0/3 games solved OR ARC < v21's 0.2973 on same game).
SUCCESS: solve v21's game + any additional game.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64
N_KB = 7
EXPLORE_STEPS = 30     # shorter initial exploration (tumble handles the rest)
MAX_PATIENCE = 15      # steps before tumble triggers
TUMBLE_MIN = 5         # initial tumble burst length
TUMBLE_MAX = 40        # maximum tumble burst length
GRADIENT_THRESH = 1e-4 # minimum change to count as gradient


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class RunAndTumbleSubstrate:
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

        # Run phase
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0

        # Tumble phase
        self._tumbling = False
        self._tumble_steps_left = 0
        self._tumble_k = TUMBLE_MIN  # current tumble burst length
        self._consecutive_tumbles = 0  # tumbles without improvement

        # Dual gradient tracking
        self._best_dist_return = float('inf')   # min distance (for return gradient)
        self._best_dist_explore = 0.0           # max distance (for explore gradient)

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
        """Dual gradient detection: return OR explore."""
        return_gradient = (self._prev_dist - dist) > GRADIENT_THRESH  # closer to initial
        explore_gradient = (dist - self._best_dist_explore) > GRADIENT_THRESH  # further than ever
        return return_gradient or explore_gradient

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

        # Initial exploration: cycle through actions
        if self.step_count <= EXPLORE_STEPS:
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            self._best_dist_return = min(self._best_dist_return, dist)
            self._best_dist_explore = max(self._best_dist_explore, dist)
            return self.step_count % self._n_actions

        # Update best distances
        self._best_dist_return = min(self._best_dist_return, dist)
        self._best_dist_explore = max(self._best_dist_explore, dist)

        # TUMBLE PHASE: random burst
        if self._tumbling:
            self._tumble_steps_left -= 1
            if self._tumble_steps_left <= 0:
                # Tumble finished — pick new action for run phase
                self._tumbling = False
                self._current_action = self._rng.randint(self._n_actions)
                self._steps_on_action = 0
                improved = self._detect_gradient(dist)
                if improved:
                    # Found new territory — reset tumble escalation
                    self._tumble_k = TUMBLE_MIN
                    self._consecutive_tumbles = 0
                    self._consecutive_progress = 1
                    self._patience = 3
                else:
                    # Still stuck — escalate tumble for next time
                    self._consecutive_tumbles += 1
                    self._tumble_k = min(self._tumble_k * 2, TUMBLE_MAX)
            # During tumble: random action
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._rng.randint(self._n_actions)

        # RUN PHASE: reactive switching
        gradient = self._detect_gradient(dist)

        self._steps_on_action += 1

        if gradient:
            # Making progress on either gradient — keep running
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._consecutive_tumbles = 0
            self._tumble_k = TUMBLE_MIN
        else:
            # No gradient detected
            self._consecutive_progress = 0

            if self._steps_on_action >= self._patience:
                # Patience exhausted — TUMBLE
                self._tumbling = True
                self._tumble_steps_left = self._tumble_k
                self._steps_on_action = 0
                self._patience = 3
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return self._rng.randint(self._n_actions)

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
    "family": "bio-inspired reactive (run-and-tumble)",
    "tag": "defense v24 (ℓ₁ dual-gradient run-and-tumble, E.coli/Ashby, zero params)",
}

SUBSTRATE_CLASS = RunAndTumbleSubstrate
