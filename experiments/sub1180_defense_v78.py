"""
sub1180_defense_v78.py — Multi-timescale drift detection (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1180 --substrate experiments/sub1180_defense_v78.py

FAMILY: Multi-timescale reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: All substrates detect change at ONE timescale:
- v30: enc vs enc_0 (cumulative from start)
- MI substrates: enc vs enc_prev (1-step delta)
Both miss SLOW DRIFT: a game that changes 0.01 per step shows no
immediate signal, but after 100 steps has drifted 1.0 — a real effect.

v78 detects change at 3 timescales simultaneously:
- SHORT (1 step): immediate response detection
- MEDIUM (20 steps): gradual drift detection
- LONG (100 steps): slow accumulation detection

For each action, score = max(short_change, medium_change, long_change).
An action is "responsive" if ANY timescale shows change above threshold.
This catches slow-acting games that immediate detection misses.

Inspired by biological multi-scale perception: retinal cells respond to
flicker (ms), adaptation (seconds), and circadian rhythm (hours).

ZERO learned parameters (defense: ℓ₁). Fixed timescale windows.

KILL: L1 ≤ random (3/5).
SUCCESS: Any 0% game detected via slow drift that single-timescale misses.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

# Timescale windows
SHORT_WINDOW = 1
MEDIUM_WINDOW = 20
LONG_WINDOW = 100

EPSILON = 0.2
RESPONSE_THRESH = 0.3
DRIFT_THRESH = 0.1       # lower threshold for slow drift (per-step is small)


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class MultiTimescaleReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._prev_action = 0

        # Observation history for multi-timescale comparison
        self._enc_history = []   # last LONG_WINDOW observations
        self._max_history = LONG_WINDOW + 1

        # Responsive action tracking with timescale scores
        self._responsive_actions = {}  # action -> max_score
        self._action_set = []

        # Reactive state
        self._current_idx = 0
        self._patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()
        n_kb = min(n_actions, N_KB)
        self._action_set = list(range(n_kb))

    def _multi_timescale_score(self, enc):
        """Compute change score across 3 timescales."""
        scores = []

        # SHORT: 1-step change
        if len(self._enc_history) >= SHORT_WINDOW:
            short_delta = float(np.sum(np.abs(enc - self._enc_history[-SHORT_WINDOW])))
            scores.append(short_delta)

        # MEDIUM: 20-step drift
        if len(self._enc_history) >= MEDIUM_WINDOW:
            medium_delta = float(np.sum(np.abs(enc - self._enc_history[-MEDIUM_WINDOW])))
            scores.append(medium_delta)

        # LONG: 100-step drift
        if len(self._enc_history) >= LONG_WINDOW:
            long_delta = float(np.sum(np.abs(enc - self._enc_history[-LONG_WINDOW])))
            scores.append(long_delta)

        return max(scores) if scores else 0.0

    def _add_responsive(self, action, score):
        if action in self._responsive_actions:
            self._responsive_actions[action] = max(
                self._responsive_actions[action], score
            )
        else:
            self._responsive_actions[action] = score
            self.r3_updates += 1
            self.att_updates_total += 1
        sorted_resp = sorted(
            self._responsive_actions.items(),
            key=lambda x: -x[1]
        )[:30]
        self._action_set = [a for a, _ in sorted_resp]
        if len(self._action_set) < N_KB:
            n_kb = min(self._n_actions_env, N_KB)
            for a in range(n_kb):
                if a not in self._action_set:
                    self._action_set.append(a)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._enc_history.append(enc.copy())
            self._prev_action = 0
            return 0

        # Multi-timescale change detection
        mt_score = self._multi_timescale_score(enc)
        if mt_score > DRIFT_THRESH:
            self._add_responsive(self._prev_action, mt_score)

        # Store in history (circular buffer)
        self._enc_history.append(enc.copy())
        if len(self._enc_history) > self._max_history:
            self._enc_history.pop(0)

        # Progress metric: also multi-timescale
        dist = float(np.sum(np.abs(enc - self._enc_0)))

        # Epsilon-greedy with reactive cycling
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
        else:
            if dist >= self._prev_dist:
                self._current_idx = (self._current_idx + 1) % len(self._action_set)
                self._patience = 0
            else:
                self._patience += 1
                if self._patience > 10:
                    self._patience = 0
                    self._current_idx = (self._current_idx + 1) % len(self._action_set)

            action = self._action_set[self._current_idx % len(self._action_set)]

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_idx = 0
        self._patience = 0
        self._enc_history = []
        # Keep responsive actions across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "short_window": SHORT_WINDOW,
    "medium_window": MEDIUM_WINDOW,
    "long_window": LONG_WINDOW,
    "epsilon": EPSILON,
    "response_thresh": RESPONSE_THRESH,
    "drift_thresh": DRIFT_THRESH,
    "family": "multi-timescale reactive",
    "tag": "defense v78 (ℓ₁ multi-timescale: detects change at 1/20/100 step windows. Tests if 0% games have slow drift invisible to single-timescale detection. Inspired by biological multi-scale perception.)",
}

SUBSTRATE_CLASS = MultiTimescaleReactiveSubstrate
