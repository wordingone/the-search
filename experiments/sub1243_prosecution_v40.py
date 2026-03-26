"""
sub1243_prosecution_v40.py — Online state-conditioned change-rate (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1243 --substrate experiments/sub1243_prosecution_v40.py

FAMILY: State-conditioned (ℓ_π). Tagged: prosecution.
R3 HYPOTHESIS: Online per-(state,action) EMA from step 1 (no exploration phase)
solves v93's budget fragmentation problem. The substrate LEARNS which actions
work in which states without a dedicated explore phase.

v93: 100-step explore, 4 quadrants × 7 actions = 28 buckets, ~3.6 samples each.
v40: online from step 1, EMA accumulates signal DURING exploitation, not just
during explore. After 500 steps: ~70 samples per bucket vs v93's 3.6.

WHY THIS IS ℓ_π: Action ranking self-modifies per state (state-dependent policy
from learned EMAs). Defense v80: same ranking in ALL states (global stats).

KILL: avg L1 < v80 (3.3/5) → state conditioning still hurts.
SUCCESS: avg L1 > 3.3/5 → prosecution ℓ_π > ℓ₁.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EMA_ALPHA = 0.1
EPSILON = 0.2
CHANGE_THRESH = 0.1
WARMUP_STEPS = 50
N_COARSE_STATES = 4  # top-2 dims binary partition


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _coarse_state(enc):
    """Map 256D encoding to one of 4 coarse states.

    Uses top-2 variance dims (precomputed from running stats) with
    median split. Fixed partition — no learned parameters in the
    partitioning itself.
    """
    # Use dims 0 and 1 of the encoding (top-left corner) as proxy
    # for state. Binary split at 0.5 for each.
    d0 = 1 if enc[0] > 0.5 else 0
    d1 = 1 if enc[1] > 0.5 else 0
    return d0 * 2 + d1


class OnlineStateConditionedSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0
        self._prev_state = 0

        # Per-(state, action) EMA — THE ℓ_π component
        # Learns from step 1, no dedicated explore phase
        self._state_action_ema = {}  # (state, action) -> float

        # Global EMA as fallback (same as v80)
        self._global_ema = {}  # action -> float

        # Per-state ranked action lists (rebuilt periodically)
        self._state_rankings = {}  # state -> [actions]
        self._global_ranking = list(range(min(self._n_actions_env, N_KB)))

        # Exploit state
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold = 0
        self._patience = 0

        # Running stats for adaptive state partitioning
        self._enc_sum = np.zeros(N_DIMS, dtype=np.float64)
        self._enc_count = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _update_ema(self, state, action, delta):
        """Update both state-conditioned and global EMAs."""
        # State-conditioned EMA (ℓ_π)
        key = (state, action)
        if key in self._state_action_ema:
            self._state_action_ema[key] = (1 - EMA_ALPHA) * self._state_action_ema[key] + EMA_ALPHA * delta
        else:
            self._state_action_ema[key] = delta

        # Global EMA (fallback)
        if action in self._global_ema:
            self._global_ema[action] = (1 - EMA_ALPHA) * self._global_ema[action] + EMA_ALPHA * delta
        else:
            self._global_ema[action] = delta

        self.r3_updates += 1
        self.att_updates_total += 1

    def _rebuild_rankings(self):
        """Build per-state rankings from accumulated EMAs."""
        n_kb = min(self._n_actions_env, N_KB)

        # Per-state rankings
        for s in range(N_COARSE_STATES):
            state_avgs = []
            for a in range(n_kb):
                key = (s, a)
                if key in self._state_action_ema:
                    state_avgs.append((self._state_action_ema[key], a))

            if state_avgs:
                state_avgs.sort(reverse=True)
                ranked = [a for v, a in state_avgs if v > CHANGE_THRESH]
                if ranked:
                    self._state_rankings[s] = ranked
                    continue

            # Fallback: use global EMA
            global_avgs = [(self._global_ema.get(a, 0.0), a) for a in range(n_kb)]
            global_avgs.sort(reverse=True)
            ranked = [a for v, a in global_avgs if v > CHANGE_THRESH]
            if ranked:
                self._state_rankings[s] = ranked
            else:
                self._state_rankings[s] = list(range(n_kb))

        # Global ranking
        global_avgs = [(self._global_ema.get(a, 0.0), a) for a in range(n_kb)]
        global_avgs.sort(reverse=True)
        self._global_ranking = [a for v, a in global_avgs if v > CHANGE_THRESH]
        if not self._global_ranking:
            self._global_ranking = list(range(n_kb))

    def _get_ranked(self, state):
        """Get ranked actions for current state."""
        if state in self._state_rankings:
            return self._state_rankings[state]
        return self._global_ranking

    def _adaptive_state(self, enc):
        """Coarse state from running statistics — adapts as encoding stats stabilize."""
        if self._enc_count < 10:
            return _coarse_state(enc)

        # Use running mean as split point (adaptive median)
        mean = self._enc_sum / self._enc_count
        d0 = 1 if enc[0] > mean[0] else 0
        d1 = 1 if enc[1] > mean[1] else 0
        return d0 * 2 + d1

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        # Update running stats
        self._enc_sum += enc.astype(np.float64)
        self._enc_count += 1

        state = self._adaptive_state(enc)

        if self._prev_enc is None:
            self._prev_enc = enc.copy()
            self._prev_state = state
            self._prev_action = int(self._rng.randint(0, min(self._n_actions_env, N_KB)))
            return self._prev_action

        # Measure change
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # ONLINE LEARNING: update EMAs from step 1 (no explore phase)
        self._update_ema(self._prev_state, self._prev_action, delta)

        # Rebuild rankings periodically
        if self.step_count % 100 == 0:
            self._rebuild_rankings()
            self._current_idx = 0

        # WARMUP: random actions for first N steps to seed EMAs
        if self.step_count <= WARMUP_STEPS:
            action = int(self._rng.randint(0, min(self._n_actions_env, N_KB)))
            self._prev_enc = enc.copy()
            self._prev_state = state
            self._prev_action = action
            return action

        # First ranking build after warmup
        if self.step_count == WARMUP_STEPS + 1:
            self._rebuild_rankings()

        # EXPLOIT: v80-style cycling with STATE-CONDITIONED rankings
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, min(self._n_actions_env, N_KB)))
            self._prev_enc = enc.copy()
            self._prev_state = state
            self._prev_action = action
            return action

        ranked = self._get_ranked(state)
        self._current_idx = self._current_idx % len(ranked)

        self._current_change_sum += delta
        self._current_hold += 1

        current_avg = self._current_change_sum / max(self._current_hold, 1)
        if current_avg < CHANGE_THRESH and self._current_hold > 5:
            self._current_idx = (self._current_idx + 1) % len(ranked)
            self._current_change_sum = 0.0
            self._current_hold = 0
            self._patience = 0
        elif self._current_hold > 20:
            self._patience += 1
            if self._patience > 3:
                self._current_idx = (self._current_idx + 1) % len(ranked)
                self._current_change_sum = 0.0
                self._current_hold = 0
                self._patience = 0

        action = ranked[self._current_idx]
        self._prev_enc = enc.copy()
        self._prev_state = state
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold = 0
        self._patience = 0
        # Keep EMAs and rankings across levels (ℓ_π: learned state persists)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "ema_alpha": EMA_ALPHA,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "warmup_steps": WARMUP_STEPS,
    "n_coarse_states": N_COARSE_STATES,
    "family": "online state-conditioned",
    "tag": "prosecution v40 (ℓ_π online state-conditioned: EMA from step 1, 4 states via top-2 dim binary split, concurrent learning. Fixes v93 budget fragmentation.)",
}

SUBSTRATE_CLASS = OnlineStateConditionedSubstrate
