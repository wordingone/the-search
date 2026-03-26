"""
sub1170_baseline_random.py — Pure random baseline

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1170 --substrate experiments/sub1170_baseline_random.py

FAMILY: Baseline. Tagged: neither (control).
R3 HYPOTHESIS: None — this is a CONTROL experiment. Establishes the floor
that any mechanism must beat. If random gets 2/5 L1 (same as most substrates),
then mechanism "intelligence" is adding nothing — the 2 responsive games
solve themselves regardless of action choice.

Architecture: purely random action selection. No observation processing,
no state tracking, no encoding. Each step returns a random valid action.

This is the simplest possible substrate — the theoretical minimum.

KILL: N/A (baseline).
SUCCESS: N/A (establishes floor).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np


class RandomBaselineSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = 7
        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._n_actions_env = n_actions

    def process(self, obs: np.ndarray) -> int:
        return int(self._rng.randint(0, self._n_actions_env))

    def on_level_transition(self):
        pass


CONFIG = {
    "family": "baseline",
    "tag": "random baseline (pure random action selection — establishes floor for mechanism comparison)",
}

SUBSTRATE_CLASS = RandomBaselineSubstrate
