"""
sub1186_defense_v80.py — Change-rate maximizing reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1186 --substrate experiments/sub1186_defense_v80.py

FAMILY: Change-rate maximizing. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: All 80+ defense substrates define progress as DECREASING
distance to some goal (initial obs, modal obs, predicted obs). But what
if game progress = INCREASING change? Many games reward ACTION — the
player who causes the most change wins.

v80 INVERTS the reactive signal: actions that produce MORE pixel change
are PREFERRED, not penalized. Instead of switching when change stops,
switch when change SLOWS DOWN. Hold actions that cause maximum churn.

Phase 1 (100 steps): sample random actions, measure change per action.
Phase 2 (exploit): cycle through actions in ORDER of change rate (highest
first). Switch when change rate drops below the action's measured average.

This tests the opposite hypothesis from v30: maybe the substrate should
MAXIMIZE disruption, not minimize distance.

Combined with epsilon-greedy (20%) for coverage.

ZERO learned parameters (defense: ℓ₁). Fixed protocol.

KILL: L1 ≤ random (3/5).
SUCCESS: Any game where maximizing change > minimizing distance.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EXPLORE_STEPS = 100
EPSILON = 0.2
CHANGE_THRESH = 0.1


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class ChangeRateMaximizingSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0

        # Per-action change statistics
        self._action_change_sum = {}   # action -> cumulative change
        self._action_change_count = {} # action -> count
        self._exploring = True

        # Exploit state
        self._ranked_actions = []  # actions sorted by avg change (highest first)
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _transition_to_exploit(self):
        self._exploring = False
        self.r3_updates += 1
        self.att_updates_total += 1

        # Rank actions by average change (HIGHEST first — maximize change)
        action_avgs = []
        for a, total in self._action_change_sum.items():
            count = self._action_change_count.get(a, 1)
            action_avgs.append((total / count, a))

        action_avgs.sort(reverse=True)  # highest change first
        self._ranked_actions = [a for _, a in action_avgs if _ > CHANGE_THRESH]

        if not self._ranked_actions:
            # No actions produced change — fall back to keyboard cycling
            n_kb = min(self._n_actions_env, N_KB)
            self._ranked_actions = list(range(n_kb))

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

        # Measure change from previous step
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # === EXPLORE: random sampling, record per-action change ===
        if self._exploring:
            a = self._prev_action
            self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
            self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # === EXPLOIT: hold high-change actions, switch when change drops ===

        # Epsilon-greedy exploration
        if self._rng.random() < EPSILON:
            # Record change for discovery
            a = self._prev_action
            self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
            self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
            self._prev_action = action
            return action

        # Track current action's change rate
        self._current_change_sum += delta
        self._current_hold_count += 1

        # Switch when change rate drops (action stopped causing change)
        current_avg = self._current_change_sum / max(self._current_hold_count, 1)
        if current_avg < CHANGE_THRESH and self._current_hold_count > 5:
            self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
            self._current_change_sum = 0.0
            self._current_hold_count = 0
            self._patience = 0
        elif self._current_hold_count > 20:
            # Patience: even if change continues, eventually try next action
            self._patience += 1
            if self._patience > 3:
                self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
                self._current_change_sum = 0.0
                self._current_hold_count = 0
                self._patience = 0

        action = self._ranked_actions[self._current_idx]
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0
        # Keep ranked actions across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "family": "change-rate maximizing",
    "tag": "defense v80 (ℓ₁ change-rate max: INVERTED signal — prefer actions that cause MOST pixel change. Tests if games reward disruption, not distance minimization.)",
}

SUBSTRATE_CLASS = ChangeRateMaximizingSubstrate
