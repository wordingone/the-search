"""
sub1158_defense_v64.py — Interleaved reactive + empowerment (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1158 --substrate experiments/sub1158_defense_v64.py

FAMILY: Interleaved dual-strategy. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v63 branched keyboard/click strategies and got 3/5.
But branching means each game gets ONLY ONE strategy. What if we
INTERLEAVE both strategies on every game? Alternate between v30-style
reactive keyboard actions and empowerment-guided click targeting.
Each strategy gets half the budget. If a game responds to keyboard,
the reactive half handles it. If it responds to clicks, the
empowerment half handles it.

KEY DIFFERENCE FROM v63 (branching):
- v63: one strategy per game based on n_actions.
- v64: both strategies on every game, interleaved.

Architecture:
- Steps 0-200: keyboard estimation (cycle all keyboard actions, track change per action)
  + click estimation (cycle salient click regions, track empowerment)
  Interleaved: even steps = keyboard, odd steps = clicks
- Steps 200+: exploit phase
  - Even steps: best keyboard action (highest observed change, reactive switching)
  - Odd steps: best click action (highest empowerment)

ZERO learned parameters (defense: ℓ₁). Fixed interleaving protocol.

KILL: ARC ≤ v30.
SUCCESS: Interleaving > branching (v63) or pure reactive (v30).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 12
EST_PHASE_END = 200


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class InterleavedSubstrate:
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

        # Keyboard tracking (change magnitude per action)
        self._kb_change = np.zeros(N_KB, dtype=np.float32)
        self._kb_counts = np.zeros(N_KB, dtype=np.int32)
        self._kb_best = 0
        self._kb_patience = 0
        self._prev_kb_dist = float('inf')

        # Click tracking (empowerment via Welford)
        self._click_actions = []
        self._click_means = np.zeros((N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._click_m2 = np.zeros((N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._click_vars = np.zeros((N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._click_counts = np.zeros(N_CLICK_REGIONS, dtype=np.int32)
        self._click_empowerment = np.zeros(N_CLICK_REGIONS, dtype=np.float32)
        self._click_best = []
        self._click_exploit_idx = 0
        self._click_patience = 0

        self._regions_set = False
        self._estimation_done = False

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
        self._regions_set = True

    def _finalize_estimation(self):
        self._estimation_done = True
        self.r3_updates += 1
        self.att_updates_total += 1

        # Keyboard: rank by average change
        kb_n = min(self._n_actions_env, N_KB)
        best_change = -1.0
        for a in range(kb_n):
            if self._kb_counts[a] > 0:
                avg = self._kb_change[a] / self._kb_counts[a]
                if avg > best_change:
                    best_change = avg
                    self._kb_best = a

        # Clicks: compute empowerment if clicks available
        if self._has_clicks:
            eps = 1e-6
            for a in range(min(len(self._click_actions), N_CLICK_REGIONS)):
                if self._click_counts[a] < 2:
                    continue
                total_dist = 0.0
                n_comp = 0
                var_a = self._click_m2[a] / max(self._click_counts[a] - 1, 1)
                for b in range(min(len(self._click_actions), N_CLICK_REGIONS)):
                    if b == a or self._click_counts[b] < 2:
                        continue
                    var_b = self._click_m2[b] / max(self._click_counts[b] - 1, 1)
                    mean_diff = self._click_means[a] - self._click_means[b]
                    pooled_var = (var_a + var_b) / 2.0 + eps
                    d_sq = np.sum(mean_diff ** 2 / pooled_var)
                    total_dist += np.sqrt(d_sq)
                    n_comp += 1
                if n_comp > 0:
                    self._click_empowerment[a] = total_dist / n_comp

            scored = [(self._click_empowerment[a], a) for a in range(min(len(self._click_actions), N_CLICK_REGIONS))]
            scored.sort(reverse=True)
            self._click_best = [a for emp, a in scored[:4] if emp > 0.01]

        if not self._click_best:
            self._click_best = list(range(min(3, len(self._click_actions)))) if self._click_actions else [0]

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

        change = np.sum(np.abs(enc - self._prev_enc))
        is_even = (self.step_count % 2 == 0)

        # Estimation phase: interleave keyboard and click sampling
        if self.step_count <= EST_PHASE_END and not self._estimation_done:
            if is_even or not self._has_clicks:
                # Keyboard action
                kb_n = min(self._n_actions_env, N_KB)
                action_idx = ((self.step_count // 2) - 1) % kb_n
                # Record change from previous keyboard action
                prev_kb = ((self.step_count // 2) - 2) % kb_n
                self._kb_change[prev_kb] += change
                self._kb_counts[prev_kb] += 1
                self._prev_enc = enc.copy()
                return action_idx
            else:
                # Click action
                n_clicks = min(len(self._click_actions), N_CLICK_REGIONS)
                if n_clicks == 0:
                    self._prev_enc = enc.copy()
                    return 0
                click_idx = ((self.step_count // 2) - 1) % n_clicks
                # Update Welford stats
                n = self._click_counts[click_idx] + 1
                self._click_counts[click_idx] = n
                delta = enc.astype(np.float64) - self._click_means[click_idx]
                self._click_means[click_idx] += delta / n
                delta2 = enc.astype(np.float64) - self._click_means[click_idx]
                self._click_m2[click_idx] += delta * delta2

                self._prev_enc = enc.copy()
                return self._click_actions[click_idx]

        # Transition to exploit
        if not self._estimation_done:
            self._finalize_estimation()

        # Exploit phase: interleave keyboard reactive + click empowerment
        dist = np.sum(np.abs(enc - self._enc_0))

        if is_even or not self._has_clicks:
            # Keyboard: reactive argmin (switch on distance improvement)
            if dist >= self._prev_kb_dist:
                kb_n = min(self._n_actions_env, N_KB)
                self._kb_best = (self._kb_best + 1) % kb_n
                self._kb_patience = 0
            else:
                self._kb_patience += 1
                if self._kb_patience > 8:
                    kb_n = min(self._n_actions_env, N_KB)
                    self._kb_best = (self._kb_best + 1) % kb_n
                    self._kb_patience = 0

            self._prev_kb_dist = dist
            self._prev_enc = enc.copy()
            return self._kb_best
        else:
            # Click: cycle through best empowered clicks
            click_idx = self._click_best[self._click_exploit_idx]
            self._click_patience += 1
            if self._click_patience >= 10:
                self._click_patience = 0
                self._click_exploit_idx = (self._click_exploit_idx + 1) % len(self._click_best)

            self._prev_enc = enc.copy()
            if click_idx < len(self._click_actions):
                return self._click_actions[click_idx]
            return self._kb_best

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_kb_dist = float('inf')
        self._kb_patience = 0
        self._click_exploit_idx = 0
        self._click_patience = 0
        # Keep estimation results across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "est_phase_end": EST_PHASE_END,
    "family": "interleaved dual-strategy",
    "tag": "defense v64 (ℓ₁ interleaved: even=keyboard reactive, odd=click empowerment. Both strategies every game.)",
}

SUBSTRATE_CLASS = InterleavedSubstrate
