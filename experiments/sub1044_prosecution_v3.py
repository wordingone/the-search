"""
sub1044_prosecution_v3.py — Prosecution v3: coarse-to-fine scan + multi-signal goal (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1044 --substrate experiments/sub1044_prosecution_v3.py

FAMILY: attention-gated
R3 HYPOTHESIS: ℓ_π coarse-to-fine exploration with attention-gated refinement.
  Phase 1 (0-210): Try ALL 7 keyboard actions 30 times each — guaranteed keyboard coverage.
  Phase 2 (210-1000): Coarse click grid (4x4=16 positions, 50 clicks each) — guaranteed spatial coverage.
  Phase 3 (1000+): Attention-gated refinement — focus exploration on discovered interactive zones.
  Goal: attention-weighted multi-signal (freq_mode + initial_obs + change_target).

  If coarse-to-fine discovers interactive zones that random warmup misses, L1 rate
  improves on games where v2 got 0%.
  Falsified if: same 0% on LF52/SC25 (zones not discoverable at any resolution).

Includes transition-reset (shared with defense). Incorporates defense v2 multi-signal goal.

KILL: chain_score < v2 or regression on AR25
SUCCESS: L1 > 0 on any game where v2 got 0%
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
KB_PHASE_END = 210       # 7 actions × 30 = 210 steps
COARSE_PHASE_END = 1000  # 16 positions × ~50 = 800 steps
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
# Coarse grid: 4x4 = 16 positions, centered in each 16x16 block
COARSE_GRID = [(gx * 16 + 8, gy * 16 + 8) for gy in range(4) for gx in range(4)]
# Fine grid for exploration after warmup
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def _click_action(x, y):
    return N_KB + y * 64 + x


def _decode_click(action):
    if action < N_KB:
        return None
    idx = action - N_KB
    return (idx % 64, idx // 64)


class ProsecutionV3Substrate:
    """
    ℓ_π R3 v3: coarse-to-fine scan + attention-gated refinement + multi-signal goal.

    Phase 1: keyboard scan (ensures keyboard-only games get explored)
    Phase 2: coarse spatial scan (ensures every region gets visited)
    Phase 3: attention-gated refinement (focus on informative regions)
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
        self._initial_obs = None
        self._change_target = None
        self._coarse_idx = 0
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
        self._initial_obs = None
        self._change_target = None
        self._coarse_idx = 0

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

        # Update change_target
        if self._change_target is None:
            self._change_target = obs_after.copy()
        else:
            self._change_target[changed] = (
                0.8 * self._change_target[changed] + 0.2 * obs_after[changed])

        self.r3_updates += 1
        self.att_updates_total += n_changed

    def _multi_signal_goal(self):
        """Multi-signal goal: blend gated freq mode + initial + change target."""
        gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)

        if self._initial_obs is None:
            return gated_goal

        change_goal = self._change_target if self._change_target is not None else self._initial_obs

        # Use attention to weight the blend: high attention → more gated_goal,
        # low attention → more initial/change signals
        att_norm = self.attention / max(ATT_MAX, 1e-6)
        return att_norm * gated_goal + (1.0 - att_norm) * (0.5 * self._initial_obs + 0.5 * change_goal)

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

        # R3 update from previous action
        if self._prev_obs_arr is not None and self._prev_action is not None:
            self._r3_attention_update(self._prev_obs_arr, arr)

        # Update frequency histograms
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0
        self.gated_freq[r, c, obs_int] += self.attention

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
            self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)
            # Start with keyboard action
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

        # Compute goals
        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)

        # Multi-signal goal with attention weighting
        goal = self._multi_signal_goal()
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # ── Phase 1: keyboard scan (0 to KB_PHASE_END) ──
        if self.step_count < KB_PHASE_END:
            kb = (self.step_count - 1) % N_KB
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Phase 2: coarse spatial scan (KB_PHASE_END to COARSE_PHASE_END) ──
        if self.step_count < COARSE_PHASE_END and self._supports_click:
            cx, cy = COARSE_GRID[self._coarse_idx % len(COARSE_GRID)]
            self._coarse_idx += 1
            action = _click_action(cx, cy)
            self.prev_action_type = 'click'
            self.prev_kb_idx = None
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # Cold-start fix: initialize attention from change_map at phase 3 start
        if self.step_count == COARSE_PHASE_END or (not self._supports_click and self.step_count == KB_PHASE_END):
            cm_max = self.change_map.max()
            if cm_max > 0:
                self.attention = np.clip(
                    0.1 + 0.8 * (self.change_map / cm_max), ATT_MIN, ATT_MAX)
            self.gated_freq[:] = 0

        # ── Phase 3: attention-gated refinement ──
        att_smoothed = smoothed * self.attention
        click_score = float(np.max(att_smoothed)) if self._supports_click else -1.0
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * smoothed)
        best_kb = int(np.argmax(kb_scores))
        best_kb_score = kb_scores[best_kb]

        # Epsilon-greedy
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
            idx = np.argmax(att_smoothed)
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
        self.raw_freq[:] = 0
        self.gated_freq[:] = 0
        self.prev_obs = None
        self._raw_goal = None
        self._gated_goal = None
        self._initial_obs = None
        self._change_target = None
        self._coarse_idx = 0
        # Keep attention — learned priors
        # Keep kb_influence — keyboard effects shared


CONFIG = {
    "kb_phase": KB_PHASE_END,
    "coarse_phase": COARSE_PHASE_END,
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "v3_features": "coarse-to-fine scan + multi-signal goal + attention refinement",
}

SUBSTRATE_CLASS = ProsecutionV3Substrate
