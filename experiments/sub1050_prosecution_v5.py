"""
sub1050_prosecution_v5.py — Prosecution v5: action pattern exploration (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1050 --substrate experiments/sub1050_prosecution_v5.py

FAMILY: attention-gated
R3 HYPOTHESIS: ℓ_π action pattern discovery with attention-gated exploitation.
  Single actions may produce no detectable effect on some games.
  Multi-step action sequences (pairs/triples) may be required.

  Phase 1 (0-2000): Standard random warmup (single actions, build change_map).
  Phase 2 (2000-5000): Action PAIR probing.
    - Track pair_delta = |obs(t+2) - obs(t)| across the PAIR, not per-step.
    - Keyboard pairs: 7×7 = 49 pairs.
    - Click pairs: 200 random pixel pairs from coarse grid.
    - Keep top-K pairs by pair_delta.
  Phase 3 (5000+): Exploit discovered patterns + attention-gated.
    - Repeat top-K patterns, evaluate via attention-weighted mismatch.

  If LF52/SC25 require multi-step sequences, pair probing discovers them
  where single-action exploration cannot.
  Falsified if: same 0% (games don't require action patterns).

Includes transition-reset (shared). Attention-gated from v2.

KILL: LF52/SC25 still 0% with patterns
SUCCESS: L1 > 0 on LF52 or SC25
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
KB_PHASE_END = 210       # keyboard scan within Phase 1
WARMUP_STEPS = 2000      # single-action warmup
PAIR_PHASE_END = 5000    # pair probing phase
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

# Pattern parameters
TOP_K_PATTERNS = 20       # keep top K pairs by pair_delta
N_CLICK_PAIRS = 200       # random click pairs to probe

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


class ProsecutionV5Substrate:
    """
    ℓ_π R3 v5: action pattern discovery + attention-gated exploitation.

    Phase 1: Single-action warmup (build change_map, attention).
    Phase 2: Pair probing (discover multi-step patterns).
    Phase 3: Exploit discovered patterns with attention gating.
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
        # Pattern state
        self._pair_queue = []        # list of (action1, action2) to probe
        self._pair_scores = {}       # (action1, action2) -> EMA of pair_delta
        self._pair_obs_before = None # obs before first action of pair
        self._pair_step = 0          # 0 = first action, 1 = second action
        self._current_pair = None
        self._top_patterns = []      # top-K discovered patterns
        self._pair_idx = 0
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
        self._pair_queue = []
        self._pair_scores = {}
        self._pair_obs_before = None
        self._pair_step = 0
        self._current_pair = None
        self._top_patterns = []
        self._pair_idx = 0

    def _build_pair_queue(self):
        """Build list of action pairs to probe."""
        pairs = []
        # All keyboard pairs: 7×7 = 49
        for a1 in range(N_KB):
            for a2 in range(N_KB):
                pairs.append((a1, a2))
        # Random click pairs (if click game)
        if self._supports_click:
            for _ in range(N_CLICK_PAIRS):
                g1 = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                g2 = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                a1 = _click_action(g1[0], g1[1])
                a2 = _click_action(g2[0], g2[1])
                pairs.append((a1, a2))
            # Also keyboard→click and click→keyboard pairs
            for _ in range(50):
                kb = self._rng.randint(N_KB)
                g = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                click = _click_action(g[0], g[1])
                pairs.append((kb, click))
                pairs.append((click, kb))
        self._rng.shuffle(pairs)
        return pairs

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
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)

        goal = self._gated_goal
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # ── Phase 1: warmup (0-2000) — keyboard scan + random ──
        if self.step_count < WARMUP_STEPS:
            if self.step_count < KB_PHASE_END:
                kb = (self.step_count - 1) % N_KB
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
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
            return action

        # ── Phase 2: pair probing (2000-5000) ──
        if self.step_count < PAIR_PHASE_END:
            # Initialize pair queue on first entry
            if not self._pair_queue and self._pair_idx == 0:
                self._pair_queue = self._build_pair_queue()

            if self._pair_step == 0:
                # First action of pair
                if self._pair_idx < len(self._pair_queue):
                    self._current_pair = self._pair_queue[self._pair_idx]
                else:
                    # Exhausted queue, cycle
                    self._pair_idx = 0
                    self._current_pair = self._pair_queue[self._pair_idx]
                self._pair_obs_before = arr.copy()
                action = self._current_pair[0]
                if action >= self._n_actions:
                    action = self._rng.randint(self._n_actions)
                self._pair_step = 1
            else:
                # Second action of pair
                action = self._current_pair[1]
                if action >= self._n_actions:
                    action = self._rng.randint(self._n_actions)

                # Score the pair: change across BOTH actions
                if self._pair_obs_before is not None:
                    pair_delta = float(np.abs(arr - self._pair_obs_before).mean())
                    key = self._current_pair
                    if key in self._pair_scores:
                        self._pair_scores[key] = 0.7 * self._pair_scores[key] + 0.3 * pair_delta
                    else:
                        self._pair_scores[key] = pair_delta

                self._pair_step = 0
                self._pair_idx += 1

            if action < N_KB:
                self.prev_action_type = 'kb'
                self.prev_kb_idx = action
            else:
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Phase 3: exploit discovered patterns ──
        # Build top patterns on first entry
        if not self._top_patterns and self._pair_scores:
            sorted_pairs = sorted(self._pair_scores.items(), key=lambda x: -x[1])
            self._top_patterns = [p for p, s in sorted_pairs[:TOP_K_PATTERNS] if s > 0.01]
            # Bootstrap attention from change_map
            cm_max = self.change_map.max()
            if cm_max > 0:
                self.attention = np.clip(
                    0.1 + 0.8 * (self.change_map / cm_max), ATT_MIN, ATT_MAX)
                self.gated_freq[:] = 0

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

        # Exploit: try top patterns or attention-gated single actions
        if self._top_patterns and self._rng.random() < 0.5:
            # Execute a discovered pattern (first action of pair)
            pattern = self._top_patterns[self._rng.randint(len(self._top_patterns))]
            action = pattern[0] if self._pair_step == 0 else pattern[1]
            if action >= self._n_actions:
                action = self._rng.randint(self._n_actions)
            self._pair_step = 1 - self._pair_step  # alternate
        else:
            # Attention-gated single action
            att_smoothed = smoothed * self.attention
            click_score = float(np.max(att_smoothed)) if self._supports_click else -1.0
            kb_scores = np.zeros(N_KB)
            for k in range(N_KB):
                kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * smoothed)
            best_kb = int(np.argmax(kb_scores))
            best_kb_score = kb_scores[best_kb]

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

        if action < N_KB:
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
        else:
            self.prev_action_type = 'click'
            self.prev_kb_idx = None
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
        self._pair_queue = []
        self._pair_scores = {}
        self._pair_obs_before = None
        self._pair_step = 0
        self._current_pair = None
        self._pair_idx = 0
        # Keep attention — learned priors
        # Keep top_patterns — discovered across levels
        # Keep kb_influence


CONFIG = {
    "warmup": WARMUP_STEPS,
    "pair_phase": PAIR_PHASE_END,
    "top_k_patterns": TOP_K_PATTERNS,
    "n_click_pairs": N_CLICK_PAIRS,
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "v5_features": "action pattern discovery + attention-gated exploitation",
}

SUBSTRATE_CLASS = ProsecutionV5Substrate
