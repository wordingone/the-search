"""
sub1154_defense_v60.py — Combination set scan (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1154 --substrate experiments/sub1154_defense_v60.py

FAMILY: Combination set scan (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: Mode 1 games produce zero pixel change to individual actions
AND to 2-action sequences (v57). What if the game tracks which actions have
been PERFORMED (in any order) and only reacts when the right SET is complete?
Like a combination lock — each key alone does nothing, but pressing all N
required keys (in any order) triggers a state change.

DISTINCT FROM v57 (sequences): v57 tested ORDERED pairs (a then b).
This tests UNORDERED SETS — execute all actions in a set, then check for change.

Architecture:
- Phase 1 (steps 0-50): baseline per-action change
- Phase 2 (steps 50-~2000): combination set scan
  - Test all 2-element subsets: {a, b} for a < b (21 pairs, execute both, check)
  - Test all 3-element subsets: {a, b, c} (35 combos × 3 actions = 105 steps)
  - Test all 4-element subsets: {a, b, c, d} (35 combos × 4 actions = 140 steps)
  - Test complete set {0,1,2,3,4,5,6} (7 actions, check change after all 7)
  - Between each set: execute 3 "reset" steps (action 0) to clear any state
- Phase 3 (remaining): exploit sets that produced change
  - Repeat sets with highest change, cycling through them

ZERO learned parameters (defense: ℓ₁). Fixed combinatorial scan.

KILL: No combination sets produce change AND ARC ≤ v30.
SUCCESS: Specific action sets trigger game response → combination mechanics exist.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from itertools import combinations

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
RESET_STEPS = 3  # steps between sets to clear game state


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class CombinationScanSubstrate:
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

        # Phase 1: individual action change
        self._individual_change = np.zeros(N_KB, dtype=np.float32)
        self._individual_counts = np.zeros(N_KB, dtype=np.int32)

        # Phase 2: combination scan
        self._combinations = []  # list of tuples (action_set,)
        self._combo_idx = 0
        self._combo_step = 0  # step within current combo execution
        self._combo_resetting = False  # executing reset steps
        self._reset_step = 0
        self._pre_combo_enc = None  # encoding before combo started
        self._combo_results = {}  # frozenset → change
        self._scanning = True
        self._combos_built = False

        # Phase 3: exploit
        self._best_combos = []
        self._exploit_combo_idx = 0
        self._exploit_step = 0
        self._exploit_patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _build_combinations(self):
        """Build all combination sets to test."""
        n_kb = min(self._n_actions_env, N_KB)
        self._combinations = []

        # Size 2 (21 combos for 7 actions)
        for combo in combinations(range(n_kb), 2):
            self._combinations.append(combo)
        # Size 3 (35 combos)
        for combo in combinations(range(n_kb), 3):
            self._combinations.append(combo)
        # Size 4 (35 combos)
        for combo in combinations(range(n_kb), 4):
            self._combinations.append(combo)
        # Size 5 (21 combos)
        for combo in combinations(range(n_kb), 5):
            self._combinations.append(combo)
        # Full set
        if n_kb >= 6:
            for combo in combinations(range(n_kb), 6):
                self._combinations.append(combo)
        self._combinations.append(tuple(range(n_kb)))

        self._combos_built = True

    def _build_exploit_set(self):
        """Find combinations that produced change."""
        self._scanning = False
        result_list = []

        for combo_key, change in self._combo_results.items():
            if change > 0.5:  # minimum threshold
                result_list.append((change, combo_key))

        result_list.sort(reverse=True)
        self._best_combos = [tuple(sorted(combo)) for _, combo in result_list[:10]]
        self.r3_updates += 1
        self.att_updates_total += 1

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
            return 0

        change = np.sum(np.abs(enc - self._prev_enc))

        # Phase 1: individual action scan (steps 1-50)
        if self.step_count <= 50:
            n_kb = min(self._n_actions_env, N_KB)
            action_idx = (self.step_count - 1) % n_kb
            prev_action = (self.step_count - 2) % n_kb
            if prev_action < N_KB:
                self._individual_change[prev_action] += change
                self._individual_counts[prev_action] += 1
            self._prev_enc = enc.copy()
            return action_idx

        # Build combinations on first entry to Phase 2
        if not self._combos_built:
            self._build_combinations()

        # Phase 2: combination set scan
        if self._scanning and self._combo_idx < len(self._combinations):
            # Reset steps between combos
            if self._combo_resetting:
                self._reset_step += 1
                if self._reset_step >= RESET_STEPS:
                    self._combo_resetting = False
                    self._reset_step = 0
                    self._pre_combo_enc = enc.copy()
                self._prev_enc = enc.copy()
                return 0  # reset action

            combo = self._combinations[self._combo_idx]

            if self._combo_step == 0:
                # Start of new combo — record pre-combo state
                if self._pre_combo_enc is None:
                    self._pre_combo_enc = enc.copy()

            if self._combo_step < len(combo):
                # Execute next action in the combo
                action = combo[self._combo_step]
                self._combo_step += 1
                self._prev_enc = enc.copy()
                return action
            else:
                # Combo complete — measure change from pre-combo
                total_change = np.sum(np.abs(enc - self._pre_combo_enc)) if self._pre_combo_enc is not None else 0
                self._combo_results[frozenset(combo)] = total_change

                # Move to next combo with reset
                self._combo_idx += 1
                self._combo_step = 0
                self._combo_resetting = True
                self._reset_step = 0
                self._pre_combo_enc = None
                self._prev_enc = enc.copy()
                return 0  # start reset

        # Transition to exploit
        if self._scanning:
            self._build_exploit_set()

        # Phase 3: exploit best combinations
        if self._best_combos:
            combo = self._best_combos[self._exploit_combo_idx]
            action = combo[self._exploit_step % len(combo)]
            self._exploit_step += 1

            if self._exploit_step >= len(combo) * 5:  # repeat each combo 5x
                self._exploit_step = 0
                self._exploit_combo_idx = (self._exploit_combo_idx + 1) % len(self._best_combos)

            self._prev_enc = enc.copy()
            return action

        # Fallback: keyboard cycling (no combos produced change)
        n_active = min(self._n_actions_env, N_KB)
        action = (self.step_count // 5) % n_active
        self._prev_enc = enc.copy()
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._pre_combo_enc = None
        self._combo_step = 0
        self._combo_resetting = False
        self._reset_step = 0
        self._exploit_combo_idx = 0
        self._exploit_step = 0
        # Keep combo results across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "reset_steps": RESET_STEPS,
    "family": "combination set scan",
    "tag": "defense v60 (ℓ₁ combination lock scan: test all 2/3/4/5/6/7-element action subsets, exploit responsive sets)",
}

SUBSTRATE_CLASS = CombinationScanSubstrate
