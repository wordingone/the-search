"""
sub1155_defense_v61.py — Empowerment-filtered reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1155 --substrate experiments/sub1155_defense_v61.py

FAMILY: Empowerment + reactive hybrid. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v59 (empowerment) got 3/5 L1 with GAME_4 SOLVED on one draw.
v30 (reactive argmin) gets ARC=0.33 consistently. What if we COMBINE them?
- Use empowerment estimation (v59) to identify which actions are CONTROLLABLE
- Use v30's distance-to-initial reactive goal on ONLY the controllable actions
- This concentrates the reactive mechanism on actions that actually affect the game

CONTROLLED COMPARISON vs v30:
- SAME: distance-to-initial goal, avgpool4 encoding, reactive switching
- DIFFERENT: v30 cycles all actions uniformly. v61 concentrates on empowered actions.
  If v61 > v30 → action filtering matters (some actions are noise).

Architecture:
- Phase 1 (steps 0-350): empowerment estimation (same as v59)
  - Per-action: 50 observations, Welford mean+variance
  - Pairwise distinguishability → empowerment scores
- Phase 2 (remaining): empowerment-filtered reactive
  - Only cycle through top-K empowered actions (K=3)
  - Apply distance-to-initial scoring: prefer action that moves enc toward enc_0
  - If all empowerment = 0 (Mode 1): fall back to full action cycling

ZERO learned parameters (defense: ℓ₁). Fixed estimation + fixed reactive goal.

KILL: ARC ≤ v30.
SUCCESS: Filtering by empowerment improves reactive navigation.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
SAMPLES_PER_ACTION = 50
DISTINGUISHABILITY_EPS = 1e-6
TOP_K = 3  # use only top-K empowered actions


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class EmpowermentReactiveSubstrate:
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

        # Empowerment estimation
        self._estimating = True
        self._current_action_idx = 0
        self._action_sample_count = 0
        self._n_active = N_KB
        self._action_means = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_m2 = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_vars = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_counts = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.int32)
        self._empowerment = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)
        self._filtered_actions = []  # top-K empowered action indices

        # Reactive phase
        self._reactive_idx = 0  # index into _filtered_actions
        self._prev_action_idx = 0
        self._prev_dist = 0.0

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

    def _compute_empowerment(self):
        self._estimating = False
        self.r3_updates += 1
        self.att_updates_total += 1

        for a in range(self._n_active):
            if self._action_counts[a] < 2:
                self._empowerment[a] = 0.0
                continue
            total_dist = 0.0
            n_comp = 0
            for b in range(self._n_active):
                if b == a or self._action_counts[b] < 2:
                    continue
                mean_diff = self._action_means[a] - self._action_means[b]
                pooled_var = (self._action_vars[a] + self._action_vars[b]) / 2.0 + DISTINGUISHABILITY_EPS
                d_sq = np.sum(mean_diff ** 2 / pooled_var)
                total_dist += np.sqrt(d_sq)
                n_comp += 1
            if n_comp > 0:
                self._empowerment[a] = total_dist / n_comp

        # Filter to top-K empowered actions
        scored = [(self._empowerment[a], a) for a in range(self._n_active)]
        scored.sort(reverse=True)

        # Only keep actions with nonzero empowerment
        self._filtered_actions = [a for emp, a in scored[:TOP_K] if emp > 0.01]

        # Fallback: if no empowered actions, use all keyboard actions
        if not self._filtered_actions:
            self._filtered_actions = list(range(min(self._n_actions_env, N_KB)))

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
            return 0

        # Phase 1: Empowerment estimation
        if self._estimating:
            self._update_action_stats(self._current_action_idx, enc)
            self._action_sample_count += 1

            if self._action_sample_count >= SAMPLES_PER_ACTION:
                self._action_sample_count = 0
                self._current_action_idx += 1
                if self._current_action_idx >= self._n_active:
                    self._compute_empowerment()
                    self._prev_enc = enc.copy()
                    if self._filtered_actions:
                        return self._idx_to_env_action(self._filtered_actions[0])
                    return 0

            self._prev_enc = enc.copy()
            return self._idx_to_env_action(self._current_action_idx)

        # Phase 2: Empowerment-filtered reactive
        dist_to_initial = np.sum(np.abs(enc - self._enc_0))

        # Reactive switching: if last action moved us closer to initial, keep it
        # If it moved us further, try next filtered action
        if dist_to_initial >= self._prev_dist:
            # No improvement — switch to next filtered action
            self._reactive_idx = (self._reactive_idx + 1) % len(self._filtered_actions)

        self._prev_dist = dist_to_initial
        action_idx = self._filtered_actions[self._reactive_idx]

        self._prev_enc = enc.copy()
        self._prev_action_idx = action_idx
        return self._idx_to_env_action(action_idx)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._reactive_idx = 0
        self._prev_dist = 0.0
        # Keep empowerment estimates and filtered actions across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "samples_per_action": SAMPLES_PER_ACTION,
    "top_k": TOP_K,
    "family": "empowerment-filtered reactive",
    "tag": "defense v61 (ℓ₁ empowerment-filtered reactive: estimate empowerment, then distance-to-initial on top-3 controllable actions only)",
}

SUBSTRATE_CLASS = EmpowermentReactiveSubstrate
