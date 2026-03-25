"""
sub1063_prosecution_v13.py — Prosecution v13: multi-timescale + opaque-game fallback (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1063 --substrate experiments/sub1063_prosecution_v13.py

FAMILY: attention-gated
R3 HYPOTHESIS: ℓ_π adaptive cascade with multi-timescale change detection +
  opaque-game fallback. When cascade detects NO signal at any timescale after
  2000 steps, switch to MAXIMUM DIVERSITY mode: cycle ALL action types
  systematically. No goal function, no exploitation — just maximum coverage.
  Honest strategy for zero-feedback games. If level transition occurs during
  diversity, cascade re-probes at new level.

KILL: ALL games 0%
SUCCESS: L1 > 0 on any game, no 0% games
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
CF_CHANGE_THRESH = 0.1

# Probe boundaries
KB_PROBE_END = 500
CLICK_PROBE_END = 1000
SEQ_PROBE_END = 2000

# Signal detection threshold
PROBE_SIGNAL_THRESH = 0.03  # min change_map mean to declare "signal found"

# Attention R3 parameters
ATT_INIT = 0.5
ATT_LR = 0.02
ATT_MIN = 0.01
ATT_MAX = 1.0

# Evolutionary parameters
POP_SIZE = 10
SEQ_MIN = 3
SEQ_MAX = 7
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


class ProsecutionV13Substrate:
    """
    ℓ_π R3 v10: adaptive cascading strategy.
    Short probes detect game type → full budget to detected strategy.
    Attention transfers across probes.
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

        # Cascade state
        self._detected_type = None  # 'kb', 'click', 'seq', or 'unknown'
        self._kb_change_accum = 0.0
        self._click_change_accum = 0.0
        self._best_click_regions = []

        # Multi-timescale detection
        self._obs_history = []  # ring buffer of recent obs (max 60)
        self._obs_history_max = 60
        self._early_mean = None  # mean obs over steps 0-200
        self._late_mean = None   # mean obs over steps 300-500
        self._early_count = 0
        self._late_count = 0
        self._medium_change_accum = 0  # pixels changed at 5-step delay
        self._long_change_accum = 0    # pixels changed at 50-step delay

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
        self._archive_max = 15
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

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

    def _fitness(self, obs_start, obs_end):
        return float(np.sum(self.attention * np.abs(obs_end - obs_start)))

    def _r3_attention_update(self, obs_before, obs_after):
        diff = np.abs(obs_after - obs_before)
        changed = diff > CF_CHANGE_THRESH
        n_changed = int(np.sum(changed))
        if n_changed == 0 or self._raw_goal is None or self._gated_goal is None:
            return
        raw_error = np.abs(self._raw_goal - obs_after)
        gated_error = np.abs(self._gated_goal - obs_after)
        advantage = raw_error - gated_error
        self.attention[changed] += ATT_LR * advantage[changed]
        self.attention = np.clip(self.attention, ATT_MIN, ATT_MAX)
        self.r3_updates += 1
        self.att_updates_total += n_changed

    def _do_kb_bootloader(self, arr):
        """Full keyboard exploitation — running mean bootloader."""
        goal = self._gated_goal if self._gated_goal is not None else arr
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * mismatch)
        action = int(np.argmax(kb_scores))
        if self._rng.random() < 0.1:
            action = self._rng.randint(N_KB)
        self.prev_action_type = 'kb'
        self.prev_kb_idx = action
        return action

    def _do_click_exploit(self, arr):
        """Click exploitation on detected responsive regions."""
        goal = self._gated_goal if self._gated_goal is not None else arr
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        if self._rng.random() < 0.1:
            if self._best_click_regions:
                cx, cy = self._best_click_regions[self._rng.randint(len(self._best_click_regions))]
            else:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
            action = _click_action(cx, cy)
            self.prev_action_type = 'click'
        else:
            idx = np.argmax(smoothed)
            y, x = np.unravel_index(idx, (64, 64))
            action = _click_action(int(x), int(y))
            y0, y1 = max(0, y - SUPPRESS_RADIUS), min(64, y + SUPPRESS_RADIUS + 1)
            x0, x1 = max(0, x - SUPPRESS_RADIUS), min(64, x + SUPPRESS_RADIUS + 1)
            self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
            self.prev_action_type = 'click'
        return action

    def _do_evolution(self, arr):
        """Evolutionary sequence search."""
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
            if self._evo_total_evals % MUTATE_EVERY == 0 and self._evo_total_evals > POP_SIZE:
                worst = int(np.argmin(self._evo_scores))
                best = int(np.argmax(self._evo_scores))
                if worst != best:
                    self._evo_pop[worst] = self._mutate_sequence(self._evo_pop[best])
                    self._evo_scores[worst] = self._evo_scores[best] * 0.5
                    self._evo_counts[worst] = 0
            self._evo_current = (self._evo_current + 1) % POP_SIZE
            self._evo_exec_idx = 0
        return action

    def _do_exploit_sequences(self, arr):
        """Exploit best archived sequences."""
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
            self._exploit_current = (self._exploit_current + 1) % len(self._top_sequences)
        return action

    def _do_max_diversity(self, arr):
        """Maximum action diversity for opaque games — no signal to exploit."""
        if self._rng.random() < 0.1:
            return self._rng.randint(self._n_actions)
        cycle_len = N_KB + (len(CLICK_GRID) if self._supports_click else 0)
        idx = self.step_count % cycle_len
        if idx < N_KB:
            action = idx
        elif self._supports_click:
            cx, cy = CLICK_GRID[(idx - N_KB) % len(CLICK_GRID)]
            action = _click_action(cx, cy)
        else:
            action = idx % N_KB
        return action

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

        if self._prev_obs_arr is not None and self._prev_action is not None:
            self._r3_attention_update(self._prev_obs_arr, arr)

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
        self.prev_obs = arr.copy()

        # Multi-timescale tracking
        self._obs_history.append(arr.copy())
        if len(self._obs_history) > self._obs_history_max:
            self._obs_history.pop(0)
        # Distribution drift tracking
        if self.step_count <= 200:
            if self._early_mean is None:
                self._early_mean = arr.copy()
            else:
                self._early_mean = (self._early_mean * self._early_count + arr) / (self._early_count + 1)
            self._early_count += 1
        elif self.step_count >= 300 and self.step_count <= 500:
            if self._late_mean is None:
                self._late_mean = arr.copy()
            else:
                self._late_mean = (self._late_mean * self._late_count + arr) / (self._late_count + 1)
            self._late_count += 1
        # Medium/long timescale change (during probe phase)
        if self.step_count < KB_PROBE_END and len(self._obs_history) >= 6:
            med_diff = np.abs(arr - self._obs_history[-6])
            self._medium_change_accum += int(np.sum(med_diff > 1.0))
        if self.step_count < KB_PROBE_END and len(self._obs_history) >= 51:
            long_diff = np.abs(arr - self._obs_history[-51])
            self._long_change_accum += int(np.sum(long_diff > 1.0))

        # ── Already detected → exploit ──
        if self._detected_type is not None:
            if self._detected_type == 'kb':
                action = self._do_kb_bootloader(arr)
            elif self._detected_type == 'click':
                action = self._do_click_exploit(arr)
            elif self._detected_type == 'seq':
                if self.step_count < SEQ_PROBE_END + 3000:
                    action = self._do_evolution(arr)
                else:
                    action = self._do_exploit_sequences(arr)
            else:  # unknown/opaque — maximum diversity
                action = self._do_max_diversity(arr)
        # ── Keyboard probe (0-500) ──
        elif self.step_count < KB_PROBE_END:
            kb = (self.step_count - 1) % N_KB
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = kb
            self._kb_change_accum += float(diff.mean())
        # ── Evaluate KB probe (multi-timescale) ──
        elif self.step_count == KB_PROBE_END:
            short_signal = self._kb_change_accum / KB_PROBE_END > PROBE_SIGNAL_THRESH
            medium_signal = self._medium_change_accum > 50
            long_signal = self._long_change_accum > 100
            drift_signal = False
            if self._early_mean is not None and self._late_mean is not None:
                drift = float(np.abs(self._late_mean - self._early_mean).mean())
                drift_signal = drift > 0.5
            if short_signal or medium_signal or long_signal or drift_signal:
                self._detected_type = 'kb'
            action = self._do_kb_bootloader(arr) if self._detected_type == 'kb' else self._rng.randint(self._n_actions)
        # ── Click probe (500-1000) ──
        elif self.step_count < CLICK_PROBE_END:
            if self._supports_click:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
                self._click_change_accum += float(diff.mean())
            else:
                action = self._do_kb_bootloader(arr)
                self._detected_type = 'kb'  # no clicks available → must be KB
        # ── Evaluate click probe ──
        elif self.step_count == CLICK_PROBE_END:
            if self._detected_type is None:
                avg_click = self._click_change_accum / max(1, CLICK_PROBE_END - KB_PROBE_END)
                if avg_click > PROBE_SIGNAL_THRESH:
                    self._detected_type = 'click'
                    smoothed = uniform_filter(self.change_map, size=8)
                    threshold = float(np.percentile(smoothed, 75))
                    responsive = np.argwhere(smoothed > threshold)
                    if len(responsive) > 0:
                        indices = self._rng.choice(len(responsive), size=min(20, len(responsive)), replace=False)
                        for i in indices:
                            y, x = responsive[i]
                            self._best_click_regions.append((int(x), int(y)))
            action = self._do_click_exploit(arr) if self._detected_type == 'click' else self._rng.randint(self._n_actions)
        # ── Sequence probe (1000-2000) ──
        elif self.step_count < SEQ_PROBE_END:
            action = self._do_evolution(arr)
        # ── Evaluate sequence probe ──
        elif self.step_count == SEQ_PROBE_END:
            if self._detected_type is None:
                if self._archive and max(s for s, _ in self._archive) > 0:
                    self._detected_type = 'seq'
                else:
                    self._detected_type = 'unknown'
            action = self._do_evolution(arr)
        # ── Post-probe exploitation ──
        else:
            if self._detected_type == 'kb':
                action = self._do_kb_bootloader(arr)
            elif self._detected_type == 'click':
                action = self._do_click_exploit(arr)
            elif self._detected_type == 'seq':
                if self.step_count < SEQ_PROBE_END + 3000:
                    action = self._do_evolution(arr)
                else:
                    action = self._do_exploit_sequences(arr)
            else:  # unknown/opaque — maximum diversity
                action = self._do_max_diversity(arr)

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
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.raw_freq[:] = 0
        self.gated_freq[:] = 0
        self.prev_obs = None
        self._raw_goal = None
        self._gated_goal = None
        self._detected_type = None
        self._kb_change_accum = 0.0
        self._click_change_accum = 0.0
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
        # Keep attention + kb_influence across levels


CONFIG = {
    "probes": "kb(0-500)/click(500-1000)/seq(1000-2000)/exploit(2000+)",
    "probe_signal_thresh": PROBE_SIGNAL_THRESH,
    "pop_size": POP_SIZE,
    "seq_range": f"{SEQ_MIN}-{SEQ_MAX}",
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "v13_features": "multi-timescale cascade + opaque-game max diversity fallback (l_pi)",
}

SUBSTRATE_CLASS = ProsecutionV13Substrate
