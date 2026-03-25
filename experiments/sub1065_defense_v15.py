"""
sub1065_defense_v15.py — Defense v15: sustained probing + per-action tracking (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1065 --substrate experiments/sub1065_defense_v15.py

FAMILY: parametric-goal
R3 HYPOTHESIS: The 0% wall is caused by insufficient PROBING, not insufficient
  self-modification level. Current probes cycle actions every step — if a game
  requires sustained interaction (hold action for N steps before response),
  single-step cycling misses it entirely. ℓ₁ SPSA is sufficient IF discovery
  gives the goal function signal to learn from.

  Changes from v11:
  1. SUSTAINED KB probes: hold each action 50 steps, measure per-action response
  2. Per-action response tracking: identify WHICH actions are responsive
  3. Lower detection threshold (0.01 vs 0.03) — catch subtle responses
  4. Per-pixel MAX change detection (not just mean) — catch localized responses
  5. Longer evo sequences (max 15 vs 7) — some games need longer combos
  6. RESTART cascade: if first pass finds nothing, second pass with even lower threshold

KILL: ALL games 0%
SUCCESS: L1 > 0 on any game, no 0% games (or improvement over v11/v14)
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
CF_CHANGE_THRESH = 0.1

# Probe boundaries — two passes
PASS1_KB_END = 400      # 7 actions × ~57 steps each
PASS1_CLICK_END = 800
PASS1_SEQ_END = 1800
# Pass 2 (if pass 1 finds nothing)
PASS2_KB_END = 2200     # 400 steps
PASS2_CLICK_END = 2600
PASS2_SEQ_END = 3600

# Signal detection — lower than v11
PROBE_SIGNAL_THRESH = 0.01      # was 0.03
PROBE_SIGNAL_THRESH_P2 = 0.003  # even lower for pass 2
MAX_PIXEL_THRESH = 3            # any single pixel change > this = signal

# Sustained probe settings
KB_HOLD_STEPS = 50              # hold each KB action for 50 steps

# SPSA R3 parameters
SIGMA_INIT = 1.0
ALPHA_INIT = 0.999
SPSA_DELTA_SIGMA = 1.0
SPSA_DELTA_ALPHA = 0.01
SPSA_LR = 0.02

# Evolutionary parameters
POP_SIZE = 12
SEQ_MIN = 3
SEQ_MAX = 15  # was 7 in v11
MUTATE_EVERY = 10

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


class DefenseV15Substrate:
    """
    ℓ₁ R3 v15: sustained probing + per-action tracking.
    Hold each action for K steps during probes. Track per-action response.
    Two-pass cascade: first with normal threshold, second with ultra-low.
    SPSA goal adaptation throughout.
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

        # Cascade state
        self._detected_type = None
        self._best_action = None  # best responsive action index
        self._pass = 1  # which probe pass we're on
        self._cascade_done = False

        # Per-action response tracking (sustained probing)
        self._kb_per_action_change = np.zeros(N_KB, dtype=np.float32)
        self._kb_per_action_max = np.zeros(N_KB, dtype=np.float32)
        self._kb_current_action = 0
        self._kb_hold_counter = 0
        self._kb_hold_obs_start = None  # obs at start of hold period

        # Click probing
        self._click_per_pos_change = {}
        self._click_probe_idx = 0
        self._click_hold_counter = 0
        self._click_hold_obs_start = None
        self._best_click_regions = []

        # Evolution state
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False
        self._archive = []
        self._archive_max = 20
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0

        # R3 SPSA state (persists across levels)
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_INIT
            self.alpha = ALPHA_INIT
            self.r3_updates = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()

    def _random_sequence(self):
        length = self._rng.randint(SEQ_MIN, SEQ_MAX + 1)
        seq = []
        for _ in range(length):
            if self._supports_click and self._rng.random() < 0.7:
                if self._best_click_regions and self._rng.random() < 0.5:
                    cx, cy = self._best_click_regions[
                        self._rng.randint(len(self._best_click_regions))]
                    seq.append(_click_action(cx, cy))
                else:
                    cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                    seq.append(_click_action(cx, cy))
            else:
                seq.append(self._rng.randint(N_KB))
        return seq

    def _mutate_sequence(self, seq):
        seq = list(seq)
        mut = self._rng.randint(4)
        if mut == 0 and len(seq) > SEQ_MIN:
            seq.pop(self._rng.randint(len(seq)))
        elif mut == 1 and len(seq) < SEQ_MAX:
            idx = self._rng.randint(len(seq) + 1)
            if self._supports_click and self._rng.random() < 0.7:
                g = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                seq.insert(idx, _click_action(g[0], g[1]))
            else:
                seq.insert(idx, self._rng.randint(N_KB))
        elif mut == 2:
            idx = self._rng.randint(len(seq))
            if self._supports_click and self._rng.random() < 0.7:
                g = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                seq[idx] = _click_action(g[0], g[1])
            else:
                seq[idx] = self._rng.randint(N_KB)
        elif mut == 3:
            idx = self._rng.randint(len(seq))
            xy = _decode_click(seq[idx])
            if xy is not None:
                cx = max(0, min(63, xy[0] + self._rng.randint(-4, 5)))
                cy = max(0, min(63, xy[1] + self._rng.randint(-4, 5)))
                seq[idx] = _click_action(cx, cy)
        return seq

    def _init_population(self):
        if self._archive:
            n_from = min(POP_SIZE // 2, len(self._archive))
            archive_sorted = sorted(self._archive, key=lambda x: -x[0])
            pop = [self._mutate_sequence(archive_sorted[i][1]) for i in range(n_from)]
            pop += [self._random_sequence() for _ in range(POP_SIZE - n_from)]
            self._evo_pop = pop
        else:
            self._evo_pop = [self._random_sequence() for _ in range(POP_SIZE)]
        self._evo_scores = [0.0] * POP_SIZE
        self._evo_counts = [0] * POP_SIZE
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_total_evals = 0
        self._evo_initialized = True

    def _goal(self, obs, sigma=None, alpha=None):
        if sigma is None:
            sigma = self.sigma
        if alpha is None:
            alpha = self.alpha
        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        s = max(0.3, sigma)
        for c_idx in range(16):
            smoothed[:, :, c_idx] = gaussian_filter(
                self.freq_hist[:, :, c_idx].astype(np.float32), sigma=s)
        freq_goal = np.argmax(smoothed, axis=2).astype(np.float32)
        if self.novelty_map is not None:
            freq_goal = alpha * freq_goal + (1.0 - alpha) * self.novelty_map
        return freq_goal

    def _fitness(self, obs_start, obs_end):
        goal = self._goal(obs_end)
        mismatch_start = np.abs(obs_start - goal)
        mismatch_end = np.abs(obs_end - goal)
        reduction = mismatch_start - mismatch_end
        return float(np.sum(self.change_map * reduction))

    def _r3_spsa_update(self, obs_before, obs_after, cx, cy):
        r = 3
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)
        local_change = float(np.abs(
            obs_after[y0:y1, x0:x1] - obs_before[y0:y1, x0:x1]).mean())
        zone_changed = local_change > CF_CHANGE_THRESH

        def _step(val, delta, lo, hi, kwarg):
            p_plus = np.clip(val + delta, lo, hi)
            p_minus = np.clip(val - delta, lo, hi)
            if p_plus == p_minus:
                return val
            g_p = self._goal(obs_before, **{kwarg: p_plus})
            g_m = self._goal(obs_before, **{kwarg: p_minus})
            mm_p = float(np.abs(
                obs_before[y0:y1, x0:x1] - g_p[y0:y1, x0:x1]).mean())
            mm_m = float(np.abs(
                obs_before[y0:y1, x0:x1] - g_m[y0:y1, x0:x1]).mean())
            s_p = mm_p if zone_changed else -mm_p
            s_m = mm_m if zone_changed else -mm_m
            grad = (s_p - s_m) / (p_plus - p_minus)
            return float(np.clip(val + SPSA_LR * grad, lo, hi))

        self.sigma = _step(self.sigma, SPSA_DELTA_SIGMA, 0.5, 16.0, 'sigma')
        self.alpha = _step(self.alpha, SPSA_DELTA_ALPHA, 0.9, 0.999, 'alpha')
        self.r3_updates += 1

    # ── Exploitation strategies ──

    def _do_kb_bootloader(self, arr):
        goal = self._goal(arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.kb_influence[k] * mismatch)
        action = int(np.argmax(kb_scores))
        if self._rng.random() < 0.1:
            action = self._rng.randint(N_KB)
        self.prev_action_type = 'kb'
        self.prev_kb_idx = action
        return action

    def _do_click_exploit(self, arr):
        goal = self._goal(arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)
        if self._rng.random() < 0.1:
            if self._best_click_regions:
                cx, cy = self._best_click_regions[
                    self._rng.randint(len(self._best_click_regions))]
            else:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
            return _click_action(cx, cy)
        idx = np.argmax(smoothed)
        y, x = np.unravel_index(idx, (64, 64))
        action = _click_action(int(x), int(y))
        y0, y1 = max(0, y - SUPPRESS_RADIUS), min(64, y + SUPPRESS_RADIUS + 1)
        x0, x1 = max(0, x - SUPPRESS_RADIUS), min(64, x + SUPPRESS_RADIUS + 1)
        self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
        return action

    def _do_evolution(self, arr):
        if not self._evo_initialized:
            self._init_population()
        if self._evo_exec_idx == 0:
            self._evo_obs_start = arr.copy()
        seq = self._evo_pop[self._evo_current]
        action = seq[self._evo_exec_idx]
        if action >= self._n_actions:
            action = self._rng.randint(self._n_actions)
        self._evo_exec_idx += 1
        if self._evo_exec_idx >= len(seq):
            score = self._fitness(self._evo_obs_start, arr)
            idx = self._evo_current
            self._evo_counts[idx] += 1
            a = 0.3 if self._evo_counts[idx] > 1 else 1.0
            self._evo_scores[idx] = (1 - a) * self._evo_scores[idx] + a * score
            self._evo_total_evals += 1
            if score > 0:
                self._archive.append((score, list(seq)))
                if len(self._archive) > self._archive_max:
                    self._archive.sort(key=lambda x: -x[0])
                    self._archive = self._archive[:self._archive_max]
            if (self._evo_total_evals % MUTATE_EVERY == 0
                    and self._evo_total_evals > POP_SIZE):
                worst = int(np.argmin(self._evo_scores))
                best = int(np.argmax(self._evo_scores))
                if worst != best:
                    self._evo_pop[worst] = self._mutate_sequence(
                        self._evo_pop[best])
                    self._evo_scores[worst] = self._evo_scores[best] * 0.5
                    self._evo_counts[worst] = 0
            self._evo_current = (self._evo_current + 1) % POP_SIZE
            self._evo_exec_idx = 0
        return action

    def _do_exploit_sequences(self, arr):
        if not self._top_sequences:
            if self._archive:
                self._archive.sort(key=lambda x: -x[0])
                self._top_sequences = [s for _, s in self._archive[:5]]
            if not self._top_sequences:
                self._top_sequences = [self._random_sequence()]
        if self._rng.random() < 0.1:
            return self._rng.randint(self._n_actions)
        seq = self._top_sequences[self._exploit_current]
        action = seq[self._exploit_exec_idx]
        if action >= self._n_actions:
            action = self._rng.randint(self._n_actions)
        self._exploit_exec_idx += 1
        if self._exploit_exec_idx >= len(seq):
            self._exploit_exec_idx = 0
            self._exploit_current = (self._exploit_current + 1) % len(
                self._top_sequences)
        return action

    # ── Probe phase boundary evaluation ──

    def _get_probe_boundaries(self):
        if self._pass == 1:
            return PASS1_KB_END, PASS1_CLICK_END, PASS1_SEQ_END
        else:
            return PASS2_KB_END, PASS2_CLICK_END, PASS2_SEQ_END

    def _get_threshold(self):
        return PROBE_SIGNAL_THRESH if self._pass == 1 else PROBE_SIGNAL_THRESH_P2

    def _eval_kb_probe(self):
        """Evaluate sustained KB probe results."""
        thresh = self._get_threshold()
        mean_response = float(np.mean(self._kb_per_action_change))
        max_response = float(np.max(self._kb_per_action_change))
        max_pixel_response = float(np.max(self._kb_per_action_max))

        if mean_response > thresh or max_pixel_response > MAX_PIXEL_THRESH:
            self._detected_type = 'kb'
            self._best_action = int(np.argmax(self._kb_per_action_change))
            return True
        return False

    def _eval_click_probe(self):
        """Evaluate click probe results."""
        thresh = self._get_threshold()
        if not self._click_per_pos_change:
            return False
        max_change = max(self._click_per_pos_change.values())
        if max_change > thresh:
            self._detected_type = 'click'
            # Build best click regions from high-response positions
            smoothed = uniform_filter(self.change_map, size=8)
            threshold = float(np.percentile(smoothed, 70))
            responsive = np.argwhere(smoothed > threshold)
            if len(responsive) > 0:
                indices = self._rng.choice(
                    len(responsive),
                    size=min(25, len(responsive)), replace=False)
                for i in indices:
                    y, x = responsive[i]
                    self._best_click_regions.append((int(x), int(y)))
            return True
        return False

    def _eval_seq_probe(self):
        """Evaluate sequence probe results."""
        if self._archive and max(s for s, _ in self._archive) > 0:
            self._detected_type = 'seq'
            return True
        return False

    def _start_pass2(self):
        """Reset probing for second pass with lower thresholds."""
        self._pass = 2
        self._kb_per_action_change[:] = 0
        self._kb_per_action_max[:] = 0
        self._kb_current_action = 0
        self._kb_hold_counter = 0
        self._kb_hold_obs_start = None
        self._click_per_pos_change = {}
        self._click_probe_idx = 0
        self._click_hold_counter = 0
        self._click_hold_obs_start = None
        # Keep evo archive — build on pass 1's exploration

    # ── Main process ──

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

        # R3 SPSA update
        if self._prev_obs_arr is not None and self._prev_action is not None:
            click_xy = _decode_click(self._prev_action)
            if click_xy is not None:
                self._r3_spsa_update(
                    self._prev_obs_arr, arr, click_xy[0], click_xy[1])

        # First observation
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
        self.prev_obs = arr.copy()

        kb_end, click_end, seq_end = self._get_probe_boundaries()

        # ── Already detected → exploit ──
        if self._cascade_done or self._detected_type is not None:
            action = self._exploit(arr)
        # ── SUSTAINED KB PROBE ──
        elif self.step_count < kb_end:
            action = self._sustained_kb_probe(arr, diff)
        elif self.step_count == kb_end:
            if self._eval_kb_probe():
                action = self._do_kb_bootloader(arr)
            else:
                action = self._rng.randint(self._n_actions)
        # ── CLICK PROBE ──
        elif self.step_count < click_end:
            if self._supports_click:
                action = self._sustained_click_probe(arr, diff)
            else:
                self._detected_type = 'kb'
                action = self._do_kb_bootloader(arr)
        elif self.step_count == click_end:
            if self._detected_type is None:
                if self._eval_click_probe():
                    action = self._do_click_exploit(arr)
                else:
                    action = self._rng.randint(self._n_actions)
            else:
                action = self._exploit(arr)
        # ── SEQUENCE PROBE ──
        elif self.step_count < seq_end:
            action = self._do_evolution(arr)
        elif self.step_count == seq_end:
            if self._detected_type is None:
                if self._eval_seq_probe():
                    action = self._do_evolution(arr)
                elif self._pass == 1:
                    # First pass found nothing — try pass 2
                    self._start_pass2()
                    action = self._rng.randint(self._n_actions)
                else:
                    # Pass 2 also found nothing — declare unknown
                    self._detected_type = 'unknown'
                    self._cascade_done = True
                    action = self._exploit(arr)
            else:
                action = self._exploit(arr)
        # ── Post-probe fallthrough ──
        else:
            if self._detected_type is None:
                self._detected_type = 'unknown'
            self._cascade_done = True
            action = self._exploit(arr)

        if action < N_KB:
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
        else:
            self.prev_action_type = 'click'
            self.prev_kb_idx = None
        self._prev_obs_arr = arr.copy()
        self._prev_action = action
        return action

    def _sustained_kb_probe(self, arr, diff):
        """Hold each KB action for KB_HOLD_STEPS, measure response."""
        action = self._kb_current_action

        # Track per-action response
        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        self._kb_per_action_change[action] += mean_diff
        self._kb_per_action_max[action] = max(
            self._kb_per_action_max[action], max_diff)

        self._kb_hold_counter += 1
        if self._kb_hold_counter >= KB_HOLD_STEPS:
            # Move to next action
            self._kb_current_action = (self._kb_current_action + 1) % N_KB
            self._kb_hold_counter = 0

        self.prev_action_type = 'kb'
        self.prev_kb_idx = action
        return action

    def _sustained_click_probe(self, arr, diff):
        """Click probe with per-position tracking."""
        idx = self._click_probe_idx % len(CLICK_GRID)
        cx, cy = CLICK_GRID[idx]
        action = _click_action(cx, cy)

        # Track response
        key = (cx, cy)
        mean_diff = float(diff.mean())
        if key not in self._click_per_pos_change:
            self._click_per_pos_change[key] = 0.0
        self._click_per_pos_change[key] += mean_diff

        self._click_probe_idx += 1
        self.prev_action_type = 'click'
        return action

    def _exploit(self, arr):
        """Exploit detected strategy."""
        if self._detected_type == 'kb':
            return self._do_kb_bootloader(arr)
        elif self._detected_type == 'click':
            return self._do_click_exploit(arr)
        elif self._detected_type == 'seq':
            if self.step_count < (self._get_probe_boundaries()[2] + 3000):
                return self._do_evolution(arr)
            else:
                return self._do_exploit_sequences(arr)
        else:  # unknown — split budget
            if self.step_count % 3 == 0:
                return self._do_kb_bootloader(arr)
            elif self._supports_click and self.step_count % 3 == 1:
                return self._do_click_exploit(arr)
            else:
                return self._do_evolution(arr)

    def on_level_transition(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.freq_hist[:] = 0
        self.novelty_map = None
        self.prev_obs = None
        self._detected_type = None
        self._best_action = None
        self._pass = 1
        self._cascade_done = False
        self._kb_per_action_change[:] = 0
        self._kb_per_action_max[:] = 0
        self._kb_current_action = 0
        self._kb_hold_counter = 0
        self._kb_hold_obs_start = None
        self._click_per_pos_change = {}
        self._click_probe_idx = 0
        self._click_hold_counter = 0
        self._click_hold_obs_start = None
        self._best_click_regions = []
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False
        self._archive = []
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0
        # Keep SPSA params + kb_influence across levels


CONFIG = {
    "probes": "sustained KB(0-400)/click(400-800)/seq(800-1800) x2 passes",
    "probe_signal_thresh": PROBE_SIGNAL_THRESH,
    "probe_signal_thresh_p2": PROBE_SIGNAL_THRESH_P2,
    "kb_hold_steps": KB_HOLD_STEPS,
    "pop_size": POP_SIZE,
    "seq_range": f"{SEQ_MIN}-{SEQ_MAX}",
    "sigma_init": SIGMA_INIT,
    "spsa_lr": SPSA_LR,
    "v15_features": "sustained probing + per-action tracking + two-pass cascade + lower thresholds (l_1)",
}

SUBSTRATE_CLASS = DefenseV15Substrate
