"""
sub1207_defense_v87.py — Online epsilon-greedy (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1207 --substrate experiments/sub1207_defense_v87.py

FAMILY: Online epsilon-greedy. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v80 (ℓ₁ ceiling at 3.3/5) uses a distinct explore-then-exploit
structure: 100 steps random → rank actions by change rate → cycle through ranks.
But 34 experiments show v80 is uniquely successful — what makes it work?

v80's key ingredients:
1. Change-rate ranking (which actions cause the most pixel change)
2. 20% epsilon-greedy (maintains coverage floor)
3. Deterministic cycling through ranked actions (exploitation)
4. Switching when change rate drops (adaptive)

v87 tests: does v80 NEED the exploration phase? What if we learn change
rates ONLINE from step 1? No phase transition. The substrate immediately
starts with uniform action selection and continuously updates rankings
as it observes change rates. As data accumulates, exploitation naturally
improves.

The advantage: no wasted exploration budget. Every step is both exploration
AND exploitation. The 20% epsilon maintains the coverage floor that v86
(softmax) showed is critical.

From step 1: 20% uniform random + 80% best-so-far action (by running
average change rate). Rankings update every step. No phase transition,
no switching criterion — just always pick the best known action and
occasionally try others.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.0/5 (= random).
SUCCESS: avg L1 ≥ 3.3/5 (= v80, proves exploration phase unnecessary).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EPSILON = 0.2


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class OnlineEpsilonGreedySubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0

        # Per-action running average change rate
        self._action_change_sum = {}
        self._action_change_count = {}

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _best_action(self):
        """Return the action with highest average change rate so far."""
        n_kb = min(self._n_actions_env, N_KB)
        best_a = 0
        best_rate = -1.0
        for a in range(n_kb):
            if a in self._action_change_sum:
                count = max(self._action_change_count.get(a, 1), 1)
                rate = self._action_change_sum[a] / count
                if rate > best_rate:
                    best_rate = rate
                    best_a = a
        return best_a

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._prev_enc is None:
            self._prev_enc = enc.copy()
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Measure change from previous action
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # Update running stats for previous action
        a = self._prev_action
        self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
        self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

        # Track R3 updates (new actions discovered)
        if self._action_change_count[a] == 1:
            self.r3_updates += 1
            self.att_updates_total += 1

        # Epsilon-greedy: 20% random, 80% best known action
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
        else:
            action = self._best_action()

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        # Keep action stats across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "epsilon": EPSILON,
    "family": "online epsilon-greedy",
    "tag": "defense v87 (ℓ₁ online: no exploration phase. From step 1: 20% random + 80% best-change-rate action. Rankings update every step. Tests if v80's exploration phase is necessary or if online learning suffices.)",
}

SUBSTRATE_CLASS = OnlineEpsilonGreedySubstrate
