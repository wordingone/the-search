"""
sub1156_defense_v62.py — Online empowerment reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1156 --substrate experiments/sub1156_defense_v62.py

FAMILY: Online empowerment + reactive hybrid. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v59's estimation phase uses 350 steps before acting.
v30 acts from step 1. What if responsive games reward FAST action and
the estimation phase wastes the critical early window?

This substrate computes empowerment ONLINE — no separate estimation phase.
Every step updates the Welford stats for the current action AND selects
the next action using the current best empowerment + distance-to-initial.

KEY DIFFERENCE FROM v59 (empowerment):
- v59: 350-step estimation → exploit. Two phases.
- v62: single phase. Reactive from step 1, empowerment refines continuously.

KEY DIFFERENCE FROM v30 (reactive):
- v30: cycles all actions uniformly, switches on distance improvement.
- v62: biases toward actions with highest current empowerment estimate,
  with exploration bonus for under-sampled actions.

Architecture:
- From step 1: act + update
  - Welford online mean+var for current action
  - Action score = empowerment[a] + explore_bonus/(count[a]+1) - dist_penalty
    where dist_penalty favors actions that moved closer to initial
  - Pick highest-scoring action

ZERO learned parameters (defense: ℓ₁). Fixed online estimation.

KILL: ARC ≤ v30.
SUCCESS: Online empowerment > phased empowerment, approaching v30.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
EXPLORE_BONUS = 5.0
EMPOWERMENT_UPDATE_INTERVAL = 50  # recompute empowerment every N steps
DISTINGUISHABILITY_EPS = 1e-6


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class OnlineEmpowermentSubstrate:
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
        self._prev_action_idx = 0

        self._n_active = N_KB

        # Per-action Welford stats
        self._action_means = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_m2 = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_vars = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_counts = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.int32)

        # Empowerment scores (updated periodically)
        self._empowerment = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)
        self._last_empowerment_update = 0

        # Reactive state
        self._prev_dist = 0.0
        self._action_improved = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)  # times action improved dist
        self._action_tried = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)  # times action tried in reactive

        # Click regions
        self._click_actions = []
        self._regions_set = False

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _discover_regions(self, enc):
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in click_regions]
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(self._n_actions_env, N_KB)
        self._regions_set = True

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _update_action_stats(self, action_idx, enc):
        n = self._action_counts[action_idx] + 1
        self._action_counts[action_idx] = n
        delta = enc.astype(np.float64) - self._action_means[action_idx]
        self._action_means[action_idx] += delta / n
        delta2 = enc.astype(np.float64) - self._action_means[action_idx]
        self._action_m2[action_idx] += delta * delta2
        if n > 1:
            self._action_vars[action_idx] = self._action_m2[action_idx] / (n - 1)

    def _update_empowerment(self):
        """Recompute pairwise distinguishability."""
        self._last_empowerment_update = self.step_count
        self.r3_updates += 1
        self.att_updates_total += 1

        for a in range(self._n_active):
            if self._action_counts[a] < 3:
                self._empowerment[a] = 0.0
                continue
            total_dist = 0.0
            n_comp = 0
            for b in range(self._n_active):
                if b == a or self._action_counts[b] < 3:
                    continue
                mean_diff = self._action_means[a] - self._action_means[b]
                pooled_var = (self._action_vars[a] + self._action_vars[b]) / 2.0 + DISTINGUISHABILITY_EPS
                d_sq = np.sum(mean_diff ** 2 / pooled_var)
                total_dist += np.sqrt(d_sq)
                n_comp += 1
            if n_comp > 0:
                self._empowerment[a] = total_dist / n_comp

    def _select_action(self, enc):
        """Combined scoring: empowerment + exploration + reactive improvement rate."""
        dist_to_initial = np.sum(np.abs(enc - self._enc_0))
        scores = np.zeros(self._n_active, dtype=np.float32)

        for a in range(self._n_active):
            # Empowerment component (normalized)
            emp = self._empowerment[a]

            # Exploration bonus for under-sampled actions
            explore = EXPLORE_BONUS / (self._action_counts[a] + 1)

            # Reactive improvement rate
            if self._action_tried[a] > 0:
                improvement_rate = self._action_improved[a] / self._action_tried[a]
            else:
                improvement_rate = 0.5  # neutral prior

            scores[a] = emp + explore + improvement_rate

        # Pick best action
        best = int(np.argmax(scores[:self._n_active]))
        return best

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
            self._discover_regions(enc)
            self._prev_action_idx = 0
            return 0

        # Update stats for previous action
        self._update_action_stats(self._prev_action_idx, enc)

        # Update reactive improvement tracking
        dist_to_initial = np.sum(np.abs(enc - self._enc_0))
        self._action_tried[self._prev_action_idx] += 1
        if dist_to_initial < self._prev_dist:
            self._action_improved[self._prev_action_idx] += 1
        self._prev_dist = dist_to_initial

        # Periodically recompute empowerment
        if self.step_count - self._last_empowerment_update >= EMPOWERMENT_UPDATE_INTERVAL:
            self._update_empowerment()

        # First few steps: round-robin to seed statistics
        if self.step_count <= self._n_active * 3:
            action_idx = (self.step_count - 1) % self._n_active
        else:
            action_idx = self._select_action(enc)

        self._prev_enc = enc.copy()
        self._prev_action_idx = action_idx
        return self._idx_to_env_action(action_idx)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = 0.0
        # Keep empowerment and improvement stats across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_bonus": EXPLORE_BONUS,
    "empowerment_update_interval": EMPOWERMENT_UPDATE_INTERVAL,
    "family": "online empowerment reactive",
    "tag": "defense v62 (ℓ₁ online empowerment: no estimation phase, act+learn from step 1, empowerment+exploration+reactive improvement rate)",
}

SUBSTRATE_CLASS = OnlineEmpowermentSubstrate
