"""
sub1043_prosecution_v2.py — Prosecution v2: attention-gated action selection (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1043 --substrate experiments/sub1043_prosecution_v2.py

FAMILY: attention-gated
R3 HYPOTHESIS: ℓ_π attention-gated action selection self-modifies the exploration policy.
  Actions are scored by change in HIGH-ATTENTION dimensions. The substrate preferentially
  explores actions that affect informative pixels, adapting per-game.
  If attention gating, then action selection differs between games because attention
  weights learn per-game informativeness from prediction error.
  Falsified if: action scores are uniform across actions (attention has no effect on selection).

Extension of 1040b: adds attention-weighted action scoring on top of attention-gated encoding.
Includes cold-start fix + transition-reset (shared with defense).

KILL: chain_score < 1040b or no L1 improvement
SUCCESS: L2+ on any game OR consistent L1 improvement on random-pool games
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
WARMUP_STEPS = 3000
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
CF_CHANGE_THRESH = 0.1

# Attention R3 parameters
ATT_INIT = 0.5
ATT_LR = 0.02
ATT_MIN = 0.01
ATT_MAX = 1.0

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


class ProsecutionV2Substrate:
    """
    ℓ_π R3 v2: attention-gated encoding + attention-gated action selection.

    New in v2: action scoring uses attention as importance filter.
    - For clicks: score = sum(attention * |predicted_change|) in the zone
    - For keyboard: score = sum(attention * kb_influence[k])
    - Actions affecting high-attention dims are preferred
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.gated_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.attention = np.full((64, 64), ATT_INIT, dtype=np.float32)
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._raw_goal = None
        self._gated_goal = None
        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.gated_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.attention = np.full((64, 64), ATT_INIT, dtype=np.float32)
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._raw_goal = None
        self._gated_goal = None

    def _r3_attention_update(self, obs_before, obs_after):
        """Counterfactual advantage: update attention at changed pixels."""
        diff = np.abs(obs_after - obs_before)
        changed = diff > CF_CHANGE_THRESH
        n_changed = int(np.sum(changed))
        if n_changed == 0:
            return
        if self._raw_goal is None or self._gated_goal is None:
            return

        raw_error = np.abs(self._raw_goal - obs_after)
        gated_error = np.abs(self._gated_goal - obs_after)
        advantage = raw_error - gated_error

        self.attention[changed] += ATT_LR * advantage[changed]
        self.attention = np.clip(self.attention, ATT_MIN, ATT_MAX)

        self.r3_updates += 1
        self.att_updates_total += n_changed

    def _attention_action_score_click(self, smoothed):
        """Score click actions by attention-weighted mismatch. Returns (score, x, y)."""
        # smoothed already has attention baked in via mismatch computation
        # But we re-weight by attention for action scoring specifically
        att_smoothed = smoothed * self.attention
        suppress_mask = (self.suppress == 0).astype(np.float32)
        att_smoothed *= suppress_mask
        best_idx = np.argmax(att_smoothed)
        y, x = np.unravel_index(best_idx, (64, 64))
        return float(att_smoothed[y, x]), int(x), int(y)

    def _attention_action_score_kb(self, smoothed):
        """Score keyboard actions by attention-weighted influence. Returns (score, action)."""
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            # v2: weight kb influence by attention
            kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * smoothed)
        best = int(np.argmax(kb_scores))
        return kb_scores[best], best

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

        # R3 update from previous action
        if self._prev_obs_arr is not None and self._prev_action is not None:
            click_xy = _decode_click(self._prev_action)
            if click_xy is not None:
                self._r3_attention_update(self._prev_obs_arr, arr)

        # Update frequency histograms
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0
        self.gated_freq[r, c, obs_int] += self.attention

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
            self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)
            if self._supports_click:
                action = _click_action(32, 32)
            else:
                action = 0
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        # Compute goals
        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)

        # Mismatch using gated goal and attention
        mismatch = self.attention * np.abs(arr - self._gated_goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # Warmup: random
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

        # Cold-start fix
        if self.step_count == WARMUP_STEPS:
            cm_max = self.change_map.max()
            if cm_max > 0:
                self.attention = np.clip(
                    0.1 + 0.8 * (self.change_map / cm_max), ATT_MIN, ATT_MAX)
            self.gated_freq[:] = 0

        # v2: attention-gated action selection
        click_score, cx, cy = self._attention_action_score_click(smoothed) if self._supports_click else (-1.0, 0, 0)
        kb_score, best_kb = self._attention_action_score_kb(smoothed)

        # Epsilon-greedy exploration
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

        if self._supports_click and click_score >= kb_score:
            action = _click_action(cx, cy)
            y0 = max(0, cy - SUPPRESS_RADIUS)
            y1 = min(64, cy + SUPPRESS_RADIUS + 1)
            x0 = max(0, cx - SUPPRESS_RADIUS)
            x1 = min(64, cx + SUPPRESS_RADIUS + 1)
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
        self.raw_freq[:] = 0
        self.gated_freq[:] = 0
        self.prev_obs = None
        self._raw_goal = None
        self._gated_goal = None
        # Keep attention — carries learned priors
        # Keep kb_influence — keyboard effects shared across levels


CONFIG = {
    "warmup": WARMUP_STEPS,
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "att_min": ATT_MIN,
    "att_max": ATT_MAX,
    "v2_feature": "attention-gated action selection",
}

SUBSTRATE_CLASS = ProsecutionV2Substrate
