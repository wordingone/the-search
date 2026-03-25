"""
sub1046_prosecution_v4.py — Prosecution v4: transition memory + deep warmup (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1046 --substrate experiments/sub1046_prosecution_v4.py

FAMILY: attention-gated
R3 HYPOTHESIS: ℓ_π transition-conditioned attention shift.
  Level transition permanently modifies attention weights.
  On transition: snapshot attention weights + last N actions.
  Post-transition: bias attention toward snapshot (exploit mode).
  Decay bias over time → return to explore if no new transition.

  v3 lesson: systematic scan HURTS (depth > breadth for change_map).
  v4 uses: deep random warmup (proven), prosecution v2 attention core,
  defense v2 multi-signal goal (borrowed), transition memory (NEW).

  If attention snapshot transfers "what to look at" across levels, L1 rate
  improves on games where v2 got 0%.
  Falsified if: same 0% (transition knowledge doesn't transfer).

Includes transition-reset (shared). Multi-signal goal from defense v2.

KILL: AR25 < 30% (v2 baseline regression)
SUCCESS: L1 > 0 on any game where v2 got 0%
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
KB_PHASE_END = 210       # 7 actions × 30 = 210 steps
WARMUP_STEPS = 2500      # total warmup (including KB phase)
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

# Transition memory parameters
TRANSITION_SNAPSHOT_N = 20      # remember last N actions before transition
EXPLOIT_BIAS_INIT = 0.8         # initial bias strength toward snapshot
EXPLOIT_BIAS_DECAY = 0.995      # per-step decay of exploit bias
EXPLOIT_BIAS_MIN = 0.05         # minimum bias (never fully forget)
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


class ProsecutionV4Substrate:
    """
    ℓ_π R3 v4: transition memory + deep warmup + attention-gated + multi-signal goal.

    On level transition: snapshot attention weights + recent actions.
    Post-transition: bias attention toward snapshot (exploit mode).
    Decay bias → return to explore if exploit doesn't produce another transition.
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
        self._action_history = []
        # Transition memory (persists across levels)
        if not hasattr(self, '_attention_snapshot'):
            self._attention_snapshot = None
            self._action_snapshot = None
            self._exploit_bias = 0.0
            self._exploit_mode = False
            self.r3_updates = 0
            self.att_updates_total = 0
            self.transitions_seen = 0

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
        self._action_history = []

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

        att_norm = self.attention / max(ATT_MAX, 1e-6)
        return att_norm * gated_goal + (1.0 - att_norm) * (0.5 * self._initial_obs + 0.5 * change_goal)

    def _get_exploit_action(self):
        """In exploit mode, replay an action from the snapshot with some probability."""
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

        # Apply attention snapshot bias (exploit mode)
        if self._exploit_mode and self._attention_snapshot is not None:
            bias = self._exploit_bias
            self.attention = (1.0 - bias) * self.attention + bias * self._attention_snapshot
            self.attention = np.clip(self.attention, ATT_MIN, ATT_MAX)
            self._exploit_bias = max(EXPLOIT_BIAS_MIN, self._exploit_bias * EXPLOIT_BIAS_DECAY)

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

        # Compute goals
        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)

        goal = self._multi_signal_goal()
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
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
            # In exploit mode, try replaying snapshot actions
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

        # ── Phase 3: attention-gated exploitation ──
        # In exploit mode, sometimes replay snapshot actions
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
            self._action_history.append(action)
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
        self._action_history.append(action)
        return action

    def on_level_transition(self):
        """Transition memory: snapshot attention + actions, then reset exploration."""
        # ── Transition memory (NEW in v4) ──
        # Snapshot current attention weights
        self._attention_snapshot = self.attention.copy()
        # Snapshot recent actions
        n = min(TRANSITION_SNAPSHOT_N, len(self._action_history))
        if n > 0:
            self._action_snapshot = list(self._action_history[-n:])
        # Enter exploit mode with fresh bias
        self._exploit_mode = True
        self._exploit_bias = EXPLOIT_BIAS_INIT
        self.transitions_seen += 1

        # ── Standard reset (shared) ──
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.raw_freq[:] = 0
        self.gated_freq[:] = 0
        self.prev_obs = None
        self._raw_goal = None
        self._gated_goal = None
        self._initial_obs = None
        self._change_target = None
        self._action_history = []
        # Keep attention — biased by snapshot in exploit mode
        # Keep kb_influence — keyboard effects shared across levels


CONFIG = {
    "kb_phase": KB_PHASE_END,
    "warmup": WARMUP_STEPS,
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "exploit_bias_init": EXPLOIT_BIAS_INIT,
    "exploit_bias_decay": EXPLOIT_BIAS_DECAY,
    "action_bias_strength": ACTION_BIAS_STRENGTH,
    "transition_snapshot_n": TRANSITION_SNAPSHOT_N,
    "v4_features": "transition memory + deep warmup + attention-gated + multi-signal goal",
}

SUBSTRATE_CLASS = ProsecutionV4Substrate
