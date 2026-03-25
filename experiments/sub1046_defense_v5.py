"""
sub1046_defense_v5.py — Defense v5: transition memory + SPSA + multi-signal goal (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1046 --substrate experiments/sub1046_defense_v5.py

FAMILY: parametric-goal
R3 HYPOTHESIS: ℓ₁ transition-conditioned parameter memory.
  Level transition boosts SPSA parameters toward pre-transition values.
  On transition: snapshot {sigma, alpha, w_init, w_change} + last N actions.
  Post-transition: bias SPSA toward snapshot params (exploit mode).
  Decay bias over time → return to explore if no new transition.

  Defense counterpart to prosecution v4. Same signal (transition), different
  level of self-modification: parameters (ℓ₁) vs attention encoding (ℓ_π).

  v3 lesson: systematic scan HURTS. v4 kept random warmup. v5 adds
  transition memory on top of v4's keyboard-first + random warmup design.

  If parameter snapshot transfers "what goal to use" across levels, L1 rate
  improves on games where v2 got 0%.
  Falsified if: same 0% (parameter knowledge doesn't transfer).

Includes transition-reset (shared). Multi-signal goal from v2.

KILL: AR25 < 40% (v2 baseline regression)
SUCCESS: L1 > 0 on any game where v2 got 0%
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
W_INIT_INIT = 0.0
W_CHANGE_INIT = 0.0

SPSA_DELTA_SIGMA = 1.0
SPSA_DELTA_ALPHA = 0.01
SPSA_DELTA_W = 0.1
SPSA_LR = 0.02

# Transition memory parameters
TRANSITION_SNAPSHOT_N = 20      # remember last N actions before transition
EXPLOIT_BIAS_INIT = 0.8         # initial bias strength toward snapshot
EXPLOIT_BIAS_DECAY = 0.995      # per-step decay of exploit bias
EXPLOIT_BIAS_MIN = 0.05         # minimum bias
ACTION_BIAS_STRENGTH = 0.3      # probability of replaying snapshot action

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


class DefenseV5Substrate:
    """
    ℓ₁ R3 v5: transition memory + keyboard-first warmup + SPSA + multi-signal goal.

    On level transition: snapshot SPSA params + recent actions.
    Post-transition: bias SPSA toward snapshot params (exploit mode).
    Decay bias → return to explore if no transition.
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
        self._initial_obs = None
        self._change_target = None
        self._action_history = []
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_INIT
            self.alpha = ALPHA_INIT
            self.w_init = W_INIT_INIT
            self.w_change = W_CHANGE_INIT
            self.r3_updates = 0
            # Transition memory
            self._param_snapshot = None
            self._action_snapshot = None
            self._exploit_bias = 0.0
            self._exploit_mode = False
            self.transitions_seen = 0

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
        self._initial_obs = None
        self._change_target = None
        self._action_history = []

    def _goal(self, obs, sigma=None, alpha=None, w_init=None, w_change=None):
        """Multi-signal parametric goal (from v2)."""
        if sigma is None: sigma = self.sigma
        if alpha is None: alpha = self.alpha
        if w_init is None: w_init = self.w_init
        if w_change is None: w_change = self.w_change

        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        s = max(0.3, sigma)
        for c_idx in range(16):
            smoothed[:, :, c_idx] = gaussian_filter(
                self.freq_hist[:, :, c_idx].astype(np.float32), sigma=s)
        freq_goal = np.argmax(smoothed, axis=2).astype(np.float32)

        if self.novelty_map is not None:
            freq_goal = alpha * freq_goal + (1.0 - alpha) * self.novelty_map

        if self._initial_obs is None:
            return freq_goal

        change_goal = self._change_target if self._change_target is not None else self._initial_obs

        w_f = max(0.0, 1.0 - abs(w_init) - abs(w_change))
        w_i = max(0.0, min(1.0, abs(w_init)))
        w_c = max(0.0, min(1.0, abs(w_change)))
        total = w_f + w_i + w_c
        if total > 0:
            w_f /= total; w_i /= total; w_c /= total
        else:
            w_f = 1.0

        return w_f * freq_goal + w_i * self._initial_obs + w_c * change_goal

    def _r3_spsa_update(self, obs_before, obs_after, cx, cy):
        """SPSA gradient on 4 parameters."""
        r = ZONE_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)

        local_change = float(np.abs(
            obs_after[y0:y1, x0:x1] - obs_before[y0:y1, x0:x1]).mean())
        zone_changed = local_change > CF_CHANGE_THRESH

        # Update change_target
        diff = np.abs(obs_after - obs_before)
        changed_mask = diff > CF_CHANGE_THRESH
        if np.any(changed_mask):
            if self._change_target is None:
                self._change_target = obs_after.copy()
            else:
                self._change_target[changed_mask] = (
                    0.8 * self._change_target[changed_mask] +
                    0.2 * obs_after[changed_mask])

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
        self.w_init = _spsa_step(self.w_init, SPSA_DELTA_W, -1.0, 1.0, 'w_init')
        self.w_change = _spsa_step(self.w_change, SPSA_DELTA_W, -1.0, 1.0, 'w_change')
        self.r3_updates += 1

    def _apply_param_snapshot_bias(self):
        """In exploit mode, blend current params toward snapshot."""
        if self._param_snapshot is None or not self._exploit_mode:
            return
        bias = self._exploit_bias
        self.sigma = (1.0 - bias) * self.sigma + bias * self._param_snapshot['sigma']
        self.alpha = (1.0 - bias) * self.alpha + bias * self._param_snapshot['alpha']
        self.w_init = (1.0 - bias) * self.w_init + bias * self._param_snapshot['w_init']
        self.w_change = (1.0 - bias) * self.w_change + bias * self._param_snapshot['w_change']
        self._exploit_bias = max(EXPLOIT_BIAS_MIN, self._exploit_bias * EXPLOIT_BIAS_DECAY)

    def _get_exploit_action(self):
        """In exploit mode, replay an action from the snapshot."""
        if (self._action_snapshot is not None
                and self._rng.random() < ACTION_BIAS_STRENGTH * self._exploit_bias):
            idx = self._rng.randint(len(self._action_snapshot))
            action = self._action_snapshot[idx]
            if action < self._n_actions:
                return action
        return None

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

        if self._initial_obs is None:
            self._initial_obs = arr.copy()

        # Apply parameter snapshot bias (exploit mode)
        self._apply_param_snapshot_bias()

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
            self._action_history.append(action)
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
            self._action_history.append(action)
            return action

        # ── Phase 2: deep random warmup (210-2500) ──
        if self.step_count < WARMUP_STEPS:
            exploit_action = self._get_exploit_action() if self._exploit_mode else None
            if exploit_action is not None:
                action = exploit_action
                if action < N_KB:
                    self.prev_action_type = 'kb'
                    self.prev_kb_idx = action
                else:
                    self.prev_action_type = 'click'
                    self.prev_kb_idx = None
            elif self._supports_click and self._rng.random() < 0.9:
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
            self._action_history.append(action)
            return action

        # ── Phase 3: SPSA-guided exploitation ──
        exploit_action = self._get_exploit_action() if self._exploit_mode else None
        if exploit_action is not None and self._rng.random() < 0.3:
            action = exploit_action
            if action < N_KB:
                self.prev_action_type = 'kb'
                self.prev_kb_idx = action
            else:
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            self._action_history.append(action)
            return action

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
            self._action_history.append(action)
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
        self._action_history.append(action)
        return action

    def on_level_transition(self):
        """Transition memory: snapshot SPSA params + actions, then reset."""
        # ── Transition memory (NEW in v5) ──
        self._param_snapshot = {
            'sigma': self.sigma,
            'alpha': self.alpha,
            'w_init': self.w_init,
            'w_change': self.w_change,
        }
        n = min(TRANSITION_SNAPSHOT_N, len(self._action_history))
        if n > 0:
            self._action_snapshot = list(self._action_history[-n:])
        self._exploit_mode = True
        self._exploit_bias = EXPLOIT_BIAS_INIT
        self.transitions_seen += 1

        # ── Standard reset (shared) ──
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.freq_hist[:] = 0
        self.novelty_map = None
        self.prev_obs = None
        self._initial_obs = None
        self._change_target = None
        self._action_history = []
        # Keep R3 parameters — biased by snapshot in exploit mode
        # Keep kb_influence — keyboard effects shared


CONFIG = {
    "kb_phase": KB_PHASE_END,
    "warmup": WARMUP_STEPS,
    "sigma_init": SIGMA_INIT,
    "spsa_lr": SPSA_LR,
    "exploit_bias_init": EXPLOIT_BIAS_INIT,
    "exploit_bias_decay": EXPLOIT_BIAS_DECAY,
    "action_bias_strength": ACTION_BIAS_STRENGTH,
    "transition_snapshot_n": TRANSITION_SNAPSHOT_N,
    "v5_features": "transition memory + keyboard-first warmup + SPSA + multi-signal goal",
}

SUBSTRATE_CLASS = DefenseV5Substrate
