"""
sub1177_defense_v77.py — Patient hold baseline (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1177 --substrate experiments/sub1177_defense_v77.py

FAMILY: Patient hold. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: All reactive substrates switch actions quickly (every 1-10
steps). But some games might need PATIENCE — holding the SAME action for
hundreds or thousands of steps while game-internal state evolves.

Example: a game with an internal timer that advances per-step. L1 might
solve after 500 steps of action 0 (wait). Every reactive substrate switches
away from action 0 within 10 steps because no pixel change is detected.

v77 tests: does PATIENCE (long holds) solve games that rapid switching misses?
- Hold each action for HOLD_DURATION steps before switching
- Cycle through all 7 KB actions
- With 10K budget and 1000-step holds: tests ~10 actions deeply
- With 10K budget and 500-step holds: tests ~20 actions with moderate depth

This is the OPPOSITE of v30 (fast reactive switching). If 0% games need
patience, v77 should break at least one.

Combined with 20% epsilon-greedy for action space coverage.

ZERO learned parameters (defense: ℓ₁). Fixed patience protocol.

KILL: L1 ≤ v30 (2/5).
SUCCESS: Any 0% game solved by patient holding.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

N_KB = 7
HOLD_DURATION = 500     # hold each action for 500 steps before switching
EPSILON = 0.1           # 10% random exploration (lower — patience is the point)


class PatientHoldSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._current_action = 0
        self._hold_counter = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def process(self, obs: np.ndarray) -> int:
        self.step_count += 1
        self._hold_counter += 1

        # Epsilon-greedy for coverage
        if self._rng.random() < EPSILON:
            return int(self._rng.randint(0, self._n_actions_env))

        # Hold current action for HOLD_DURATION steps
        if self._hold_counter >= HOLD_DURATION:
            self._hold_counter = 0
            self._current_action = (self._current_action + 1) % min(self._n_actions_env, N_KB)
            self.r3_updates += 1
            self.att_updates_total += 1

        return self._current_action

    def on_level_transition(self):
        # On level transition, restart patience cycle
        self._current_action = 0
        self._hold_counter = 0


CONFIG = {
    "hold_duration": HOLD_DURATION,
    "epsilon": EPSILON,
    "family": "patient hold",
    "tag": "defense v77 (ℓ₁ patient hold: hold each action 500 steps. Tests if 0% games need PATIENCE, not rapid switching. Opposite of v30's reactive cycling.)",
}

SUBSTRATE_CLASS = PatientHoldSubstrate
