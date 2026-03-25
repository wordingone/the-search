"""
sub1049_defense_v7.py — Defense v7: entropy-guided goal + SPSA (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1049 --substrate experiments/sub1049_defense_v7.py

FAMILY: parametric-goal
R3 HYPOTHESIS: ℓ₁ R3 with entropy-guided goal.
  v2-v6 all use freq_mode (most frequent pixel value) as primary goal.
  freq_mode = background state. For games where target ≠ background, it's wrong.

  v7 fix: compute per-pixel entropy from frequency histogram.
  High entropy = pixel changes often = likely interactive.
  Low entropy = pixel is constant = background.

  Goal blends:
    - For LOW entropy pixels: freq_mode (we know their stable value)
    - For HIGH entropy pixels: most RECENT value (not most frequent)

  SPSA tunes: w_entropy (blend between freq_mode and recent-value goal),
  sigma (smoothing), alpha (novelty blend).

  If LF52/SC25 have interactive pixels that freq_mode incorrectly locks to
  background, entropy-guided goal finds the correct target.
  Falsified if: same 0% (interactive pixels don't have higher entropy,
  or the target isn't the recent value either).

Phase 1 (0-210): Keyboard scan.
Phase 2 (210-2500): Random warmup (coarse grid — proven better than full-pixel).
Phase 3 (2500+): SPSA-guided exploitation with entropy goal.

KILL: AR25 < 40% (v2 baseline)
SUCCESS: L1 > 0 on LF52 or SC25
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
KB_PHASE_END = 210       # 7 actions × 30 = 210 steps
WARMUP_STEPS = 2500      # total warmup (including KB phase)
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
ZONE_RADIUS = 3
CF_CHANGE_THRESH = 0.1

# R3 SPSA parameters
SIGMA_INIT = 1.0
ALPHA_INIT = 0.999
W_ENTROPY_INIT = 0.5     # blend between freq_mode goal and entropy-recent goal

SPSA_DELTA_SIGMA = 1.0
SPSA_DELTA_ALPHA = 0.01
SPSA_DELTA_W = 0.1
SPSA_LR = 0.02

# Action encoding
N_KB = 7
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def _click_action(x, y):
    return N_KB + y * 64 + x


def _decode_click(action):
    if action < N_KB:
        return None
    idx = action - N_KB
    return (idx % 64, idx // 64)


class DefenseV7Substrate:
    """
    ℓ₁ R3 v7: entropy-guided goal + SPSA.
    High-entropy pixels get recent-value goal, low-entropy get freq_mode goal.
    SPSA tunes the blend weight.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._recent_obs = None  # EMA of recent observations
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_INIT
            self.alpha = ALPHA_INIT
            self.w_entropy = W_ENTROPY_INIT
            self.r3_updates = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._recent_obs = None

    def _pixel_entropy(self):
        """Per-pixel entropy from frequency histogram. Returns (64,64) in [0,1]."""
        total = np.maximum(self.freq_hist.sum(axis=2, keepdims=True), 1).astype(np.float32)
        probs = self.freq_hist.astype(np.float32) / total
        # Avoid log(0)
        log_probs = np.where(probs > 0, np.log2(probs + 1e-10), 0.0)
        entropy = -np.sum(probs * log_probs, axis=2)
        # Normalize to [0, 1] — max entropy for 16 bins = log2(16) = 4.0
        return np.clip(entropy / 4.0, 0.0, 1.0)

    def _goal(self, obs, sigma=None, alpha=None, w_entropy=None):
        """Entropy-guided goal: blend freq_mode and recent-value based on pixel entropy."""
        if sigma is None: sigma = self.sigma
        if alpha is None: alpha = self.alpha
        if w_entropy is None: w_entropy = self.w_entropy

        # Freq-mode goal (smoothed)
        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        s = max(0.3, sigma)
        for c_idx in range(16):
            smoothed[:, :, c_idx] = gaussian_filter(
                self.freq_hist[:, :, c_idx].astype(np.float32), sigma=s)
        freq_goal = np.argmax(smoothed, axis=2).astype(np.float32)

        if self.novelty_map is not None:
            freq_goal = alpha * freq_goal + (1.0 - alpha) * self.novelty_map

        # Recent-value goal (EMA of recent observations)
        if self._recent_obs is None:
            return freq_goal

        # Entropy map: high entropy → interactive pixel
        entropy = self._pixel_entropy()

        # Blend: w_entropy controls how much we trust recent values for high-entropy pixels
        w_e = np.clip(abs(w_entropy), 0.0, 1.0)
        # Entropy-weighted blend: high entropy pixels → recent_obs, low entropy → freq_mode
        blend_weight = w_e * entropy  # per-pixel weight toward recent_obs
        return (1.0 - blend_weight) * freq_goal + blend_weight * self._recent_obs

    def _r3_spsa_update(self, obs_before, obs_after, cx, cy):
        """SPSA gradient on 3 parameters."""
        r = ZONE_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)

        local_change = float(np.abs(
            obs_after[y0:y1, x0:x1] - obs_before[y0:y1, x0:x1]).mean())
        zone_changed = local_change > CF_CHANGE_THRESH

        def _spsa_step(param_val, delta, lo, hi, goal_kwarg):
            p_plus = np.clip(param_val + delta, lo, hi)
            p_minus = np.clip(param_val - delta, lo, hi)
            if p_plus == p_minus:
                return param_val
            goal_p = self._goal(obs_before, **{goal_kwarg: p_plus})
            goal_m = self._goal(obs_before, **{goal_kwarg: p_minus})
            mm_p = float(np.abs(obs_before[y0:y1, x0:x1] - goal_p[y0:y1, x0:x1]).mean())
            mm_m = float(np.abs(obs_before[y0:y1, x0:x1] - goal_m[y0:y1, x0:x1]).mean())
            s_p = mm_p if zone_changed else -mm_p
            s_m = mm_m if zone_changed else -mm_m
            grad = (s_p - s_m) / (p_plus - p_minus)
            return float(np.clip(param_val + SPSA_LR * grad, lo, hi))

        self.sigma = _spsa_step(self.sigma, SPSA_DELTA_SIGMA, 0.5, 16.0, 'sigma')
        self.alpha = _spsa_step(self.alpha, SPSA_DELTA_ALPHA, 0.9, 0.999, 'alpha')
        self.w_entropy = _spsa_step(self.w_entropy, SPSA_DELTA_W, 0.0, 1.0, 'w_entropy')
        self.r3_updates += 1

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))
        arr = obs
        obs_int = obs.astype(np.int32)
        self.step_count += 1
        self.suppress = np.maximum(0, self.suppress - 1)

        # Update recent observation EMA
        if self._recent_obs is None:
            self._recent_obs = arr.copy()
        else:
            self._recent_obs = 0.99 * self._recent_obs + 0.01 * arr

        # R3 SPSA update
        if self._prev_obs_arr is not None and self._prev_action is not None:
            click_xy = _decode_click(self._prev_action)
            if click_xy is not None:
                cx, cy = click_xy
                self._r3_spsa_update(self._prev_obs_arr, arr, cx, cy)

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self.novelty_map = arr.copy()
            r, c = np.arange(64)[:, None], np.arange(64)[None, :]
            self.freq_hist[r, c, obs_int] += 1
            action = 0
            self.prev_action_type = 'kb'
            self.prev_kb_idx = 0
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        r_idx, c_idx = np.arange(64)[:, None], np.arange(64)[None, :]
        self.freq_hist[r_idx, c_idx, obs_int] += 1
        self.novelty_map = 0.999 * self.novelty_map + 0.001 * arr

        goal = self._goal(arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # ── Phase 1: keyboard scan (0-210) ──
        if self.step_count < KB_PHASE_END:
            kb = (self.step_count - 1) % N_KB
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Phase 2: random warmup (210-2500) ──
        if self.step_count < WARMUP_STEPS:
            if self._supports_click and self._rng.random() < 0.9:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
            else:
                kb = self._rng.randint(N_KB)
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Phase 3: SPSA-guided exploitation ──
        click_score = float(np.max(smoothed)) if self._supports_click else -1.0
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.kb_influence[k] * smoothed)
        best_kb = int(np.argmax(kb_scores))
        best_kb_score = kb_scores[best_kb]

        if self._rng.random() < 0.1:
            if self._supports_click and self._rng.random() < 0.5:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
            else:
                kb = self._rng.randint(N_KB)
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        if self._supports_click and click_score >= best_kb_score:
            idx = np.argmax(smoothed)
            y, x = np.unravel_index(idx, (64, 64))
            action = _click_action(int(x), int(y))
            y0 = max(0, y - SUPPRESS_RADIUS)
            y1 = min(64, y + SUPPRESS_RADIUS + 1)
            x0 = max(0, x - SUPPRESS_RADIUS)
            x1 = min(64, x + SUPPRESS_RADIUS + 1)
            self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
            self.prev_action_type = 'click'
        else:
            action = best_kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = best_kb

        self._prev_obs_arr = arr.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        """Reset exploration state for new level."""
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.freq_hist[:] = 0
        self.novelty_map = None
        self.prev_obs = None
        self._recent_obs = None
        # Keep R3 parameters + kb_influence


CONFIG = {
    "kb_phase": KB_PHASE_END,
    "warmup": WARMUP_STEPS,
    "sigma_init": SIGMA_INIT,
    "spsa_lr": SPSA_LR,
    "w_entropy_init": W_ENTROPY_INIT,
    "v7_features": "entropy-guided goal + keyboard-first warmup + SPSA",
}

SUBSTRATE_CLASS = DefenseV7Substrate
