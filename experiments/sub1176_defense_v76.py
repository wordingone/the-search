"""
sub1176_defense_v76.py — Raw-pixel reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1176 --substrate experiments/sub1176_defense_v76.py

FAMILY: Raw-pixel reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: ALL 75+ debate experiments use avgpool4 (4×4 average pooling,
64×64 → 16×16 = 256D). This is an UNTESTED CONSTANT. avgpool4 DESTROYS
sub-4-pixel spatial information by averaging.

What if the 0% wall games respond to actions with SINGLE-PIXEL changes that
avgpool4 averages out? A game might change 1 pixel in a 4×4 block — the
block mean barely changes, below any detection threshold.

v76 tests: do raw 64×64 pixels (4096D) detect changes that avgpool4 misses?
Same v30 reactive logic. Only difference = encoding resolution.

If v76 breaks a 0% game: avgpool4 is the bottleneck, not mechanism.
If v76 ≈ v30: the 0% wall is truly unresponsive, not sub-resolution.

Combined with epsilon-greedy action sampling (30%) to also test coverage.

ZERO learned parameters (defense: ℓ₁). Fixed encoding + v30 reactive.

KILL: L1 ≤ v30 (2/5).
SUCCESS: L1 > v30. Any 0% game broken by higher resolution.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

N_KB = 7
N_DIMS = 64 * 64  # 4096D raw pixels
EPSILON = 0.2      # 20% random exploration for coverage
RESPONSE_THRESH = 0.1  # lower threshold — raw pixels have smaller per-pixel changes


def _obs_to_enc(obs):
    """Raw 64x64 flattened = 4096D. No pooling."""
    return obs.ravel().astype(np.float32)


class RawPixelReactiveSubstrate:
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

        # Responsive action tracking
        self._responsive_actions = {}
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

    def _add_responsive(self, action, change):
        if action in self._responsive_actions:
            self._responsive_actions[action] = max(
                self._responsive_actions[action], change
            )
        else:
            self._responsive_actions[action] = change
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
            self._prev_action = 0
            return 0

        # Check response at RAW pixel level
        delta = float(np.sum(np.abs(enc - self._prev_enc)))
        if delta > RESPONSE_THRESH:
            self._add_responsive(self._prev_action, delta)

        dist = float(np.sum(np.abs(enc - self._enc_0)))

        # Epsilon-greedy: explore full action space or exploit
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
        else:
            # v30 reactive cycling over action set
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
        # Keep responsive actions across levels


CONFIG = {
    "n_dims": N_DIMS,
    "encoding": "raw 64x64 (NO avgpool)",
    "epsilon": EPSILON,
    "response_thresh": RESPONSE_THRESH,
    "family": "raw-pixel reactive",
    "tag": "defense v76 (ℓ₁ raw-pixel: tests if avgpool4 encoding destroys critical sub-pixel game responses. 4096D raw pixels + v30 reactive + 20% epsilon-greedy. Addresses the one untested constant in 75+ experiments.)",
}

SUBSTRATE_CLASS = RawPixelReactiveSubstrate
