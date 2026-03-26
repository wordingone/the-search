"""
sub1201_defense_v85.py — CPG polyrhythmic + adaptive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1201 --substrate experiments/sub1201_defense_v85.py

FAMILY: CPG polyrhythmic. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: Inspired by biological Central Pattern Generators — neural
circuits that produce structured rhythmic movement WITHOUT sensory feedback.

v84 null hypothesis: pure cycling (0,1,2,...,6) = 2.0/5 (below random 3.0/5).
Random's advantage = stochastic diversity breaks correlations.

But CPGs suggest STRUCTURED rhythmic patterns might be better than simple
cycling. A polyrhythm with overlapping prime periods creates a rich temporal
pattern with period LCM(2,3,5,7) = 210 — much richer than cycling's period 7.

v85 DUAL-MODE:
- NO FEEDBACK MODE (CPG): When no action produces detectable pixel change
  in first 100 steps, use polyrhythmic CPG pattern for remaining 9900 steps.
  The CPG assigns each action a prime period and fires actions in an
  interference pattern, creating structured temporal diversity.

- FEEDBACK MODE (v80): When actions produce detectable change in first
  100 steps, switch to v80's change-rate maximizing (the best ℓ₁ mechanism
  at 3.3/5).

This tests: does STRUCTURED temporal pattern > random for zero-feedback
games? And does dual-mode (CPG for silent games + v80 for responsive
games) beat either alone?

ZERO learned parameters (defense: ℓ₁). CPG periods are fixed primes.

KILL: avg L1 ≤ 3.0/5 (no improvement over random).
SUCCESS: avg L1 > 3.3/5 (beats v80) OR CPG mode solves a 0% game.
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
CHANGE_THRESH = 0.05  # lower thresh for detecting ANY change

# CPG prime periods for each action
CPG_PERIODS = [2, 3, 5, 7, 11, 13, 17]  # first 7 primes


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class CPGPolyrhythmicSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0
        self._exploring = True
        self._cpg_mode = False  # True = no feedback detected, use CPG

        # Exploration stats
        self._action_change_sum = {}
        self._action_change_count = {}
        self._total_change = 0.0

        # v80-style exploit state
        self._ranked_actions = []
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _cpg_action(self, step):
        """Polyrhythmic CPG: fire the action whose prime period divides step.
        If multiple fire, pick the one with smallest period (highest frequency).
        If none fire, pick based on step mod n_actions (fallback cycling)."""
        n_kb = min(self._n_actions_env, N_KB)
        for i in range(n_kb):
            period = CPG_PERIODS[i]
            if step % period == 0:
                return i
        # Fallback: golden ratio stepping for irrational coverage
        golden = (1 + np.sqrt(5)) / 2
        return int((step * golden) % n_kb)

    def _transition_to_exploit(self):
        self._exploring = False
        self.r3_updates += 1
        self.att_updates_total += 1

        # Check if any action produced change
        if self._total_change < CHANGE_THRESH * EXPLORE_STEPS:
            # NO feedback detected — use CPG mode
            self._cpg_mode = True
            return

        # Feedback detected — use v80-style change-rate ranking
        self._cpg_mode = False
        action_avgs = []
        for a, total in self._action_change_sum.items():
            count = self._action_change_count.get(a, 1)
            action_avgs.append((total / count, a))
        action_avgs.sort(reverse=True)
        self._ranked_actions = [a for avg, a in action_avgs if avg > CHANGE_THRESH]
        if not self._ranked_actions:
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

        # Measure change
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # Record stats for previous action
        a = self._prev_action
        self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
        self._action_change_count[a] = self._action_change_count.get(a, 0) + 1
        self._total_change += delta

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # === CPG MODE: no feedback, use polyrhythmic pattern ===
        if self._cpg_mode:
            # Still check for change — maybe CPG pattern unlocks something
            if delta > CHANGE_THRESH * 10:
                # Feedback appeared! Switch to v80 mode
                self._cpg_mode = False
                self._transition_to_exploit()
            else:
                # Epsilon-random even in CPG mode
                if self._rng.random() < EPSILON:
                    action = int(self._rng.randint(0, self._n_actions_env))
                else:
                    action = self._cpg_action(self.step_count)
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # === v80 MODE: feedback detected, change-rate exploitation ===

        # Epsilon-greedy
        if self._rng.random() < EPSILON:
            # Track change for discovery
            self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
            self._action_change_count[a] = self._action_change_count.get(a, 0) + 1
            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
            self._prev_action = action
            return action

        # Track current action's change rate
        self._current_change_sum += delta
        self._current_hold_count += 1

        # Switch when change rate drops
        current_avg = self._current_change_sum / max(self._current_hold_count, 1)
        if current_avg < CHANGE_THRESH and self._current_hold_count > 5:
            self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
            self._current_change_sum = 0.0
            self._current_hold_count = 0
        elif self._current_hold_count > 20:
            self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
            self._current_change_sum = 0.0
            self._current_hold_count = 0

        action = self._ranked_actions[self._current_idx]
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        # Keep mode and rankings across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "cpg_periods": CPG_PERIODS[:N_KB],
    "family": "CPG polyrhythmic + adaptive",
    "tag": "defense v85 (ℓ₁ CPG: biological central pattern generators for zero-feedback games. Polyrhythmic prime periods create structured temporal diversity. Dual-mode: CPG when no feedback, v80 when feedback detected.)",
}

SUBSTRATE_CLASS = CPGPolyrhythmicSubstrate
