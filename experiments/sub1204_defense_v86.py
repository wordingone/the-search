"""
sub1204_defense_v86.py — Weighted random sampling (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1204 --substrate experiments/sub1204_defense_v86.py

FAMILY: Weighted random. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v84 null hypothesis proved randomness adds +1.0 to L1
(dominant factor). v80 adds +0.3 via change-rate ranking (marginal).
But v80's deterministic cycling REDUCES randomness in exploit phase.

v86 tests: what if we keep FULL randomness but BIAS it toward high-change
actions? Instead of deterministic cycling (v80), sample from softmax
distribution weighted by observed change rates.

This should capture BOTH factors:
- Full stochastic coverage (the +1.0 entropy factor)
- Change-rate bias (the +0.3 observation processing factor)

If v86 ≥ v80 (3.3/5): deterministic cycling unnecessary, bias + entropy
is sufficient. Simplest possible combination.
If v86 < v80 but > random (3.0-3.3): partial benefit, cycling adds value.
If v86 ≈ random (3.0/5): bias adds nothing when randomness is preserved.

Phase 1 (100 steps): uniform random exploration, measure change per action.
Phase 2: sample from softmax(change_rate / temperature) distribution.
No cycling, no switching criterion, no patience, no epsilon.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.0/5 (= random, bias adds nothing).
SUCCESS: avg L1 > 3.3/5 (beats v80, proves entropy + bias > cycling).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EXPLORE_STEPS = 100
TEMPERATURE = 1.0  # softmax temperature for action sampling


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class WeightedRandomSubstrate:
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

        # Per-action change statistics
        self._action_change_sum = {}
        self._action_change_count = {}

        # Sampling distribution
        self._action_probs = None  # softmax over change rates
        self._n_kb = min(self._n_actions_env, N_KB)

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _build_distribution(self):
        """Build softmax sampling distribution from change rates."""
        n_kb = min(self._n_actions_env, N_KB)
        rates = np.zeros(n_kb, dtype=np.float32)

        for a in range(n_kb):
            if a in self._action_change_sum:
                count = max(self._action_change_count.get(a, 1), 1)
                rates[a] = self._action_change_sum[a] / count
            else:
                rates[a] = 0.0

        # If all rates are zero, use uniform
        if np.max(rates) < 1e-6:
            self._action_probs = np.ones(n_kb, dtype=np.float32) / n_kb
        else:
            # Softmax with temperature
            logits = rates / TEMPERATURE
            logits -= np.max(logits)  # numerical stability
            exp_logits = np.exp(logits)
            self._action_probs = exp_logits / np.sum(exp_logits)

    def _transition_to_exploit(self):
        self._exploring = False
        self._build_distribution()
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

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # Periodically rebuild distribution (incorporate new observations)
        if self.step_count % 500 == 0:
            self._build_distribution()

        # === EXPLOIT: weighted random sampling ===
        # Sample from change-rate-weighted distribution
        # This maintains full stochastic coverage while biasing
        # toward actions that produce observable change
        n_kb = min(self._n_actions_env, N_KB)
        action = int(self._rng.choice(n_kb, p=self._action_probs))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        # Keep distribution across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "temperature": TEMPERATURE,
    "family": "weighted random",
    "tag": "defense v86 (ℓ₁ weighted random: softmax sampling from change-rate distribution. Maintains full stochastic coverage (+1.0 entropy) while biasing toward high-change actions (+0.3 obs processing). Tests if cycling is necessary or if bias + entropy suffices.)",
}

SUBSTRATE_CLASS = WeightedRandomSubstrate
