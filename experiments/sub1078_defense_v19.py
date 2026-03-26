"""
sub1078_defense_v19.py — Defense v19: temporal bisection for opaque games (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1078 --substrate experiments/sub1078_defense_v19.py

FAMILY: temporal-bisection
R3 HYPOTHESIS: Opaque games (0% wall) respond to LONG action sequences, not short probes.
  Every prior substrate probes short-horizon (1-3 step) actions and checks for immediate
  observation change. Opaque games may only respond after 50-200 steps of correct input —
  like a combination lock with no intermediate feedback.

  Temporal bisection: execute 200-step random walk, compare obs[0] vs obs[200].
  If change detected → identify which ~20-step window had the biggest delta.
  Record that window's action sequence. Seed evolution with it.

  SPSA goal on block-reduced features (64D). No encoding modification (ℓ₁).
  MI detection used for per-action scoring within identified windows — read-only,
  no attention weight modification.

KILL: Bisection finds no change even in 200-step walks on 0% games
SUCCESS: Any currently-opaque game breaks 0%
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS

# Temporal bisection parameters
WALK_LEN = 200         # steps per random walk
SNAPSHOT_INTERVAL = 20  # save obs every N steps (10 snapshots per walk)
N_WALKS = 3            # max walks before declaring truly opaque
CHANGE_THRESH = 0.5    # minimum block-space L1 to count as "changed"

# Probe phase boundaries
BISECT_END = WALK_LEN * N_WALKS + 100  # ~700 steps for bisection
KB_PROBE_END = 1000
CLICK_PROBE_END = 1600
SEQ_PROBE_END = 2600

# MI detection (read-only — no attention modification)
MI_THRESH = 0.05
MI_EMA = 0.95
MI_EPSILON = 1e-8
SUSTAIN_STEPS = 10

# SPSA goal (ℓ₁ — parameter update only)
SPSA_LR = 0.02
SPSA_PERTURB = 0.1
SPSA_DIMS = N_DIMS

POP_SIZE = 12
SEQ_MIN = 3
SEQ_MAX = 15
MUTATE_EVERY = 10

N_KB = 7
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def _click_action(x, y):
    return N_KB + y * 64 + x

def _decode_click(action):
    if action < N_KB:
        return None
    idx = action - N_KB
    return (idx % 64, idx // 64)

def _obs_to_blocks(obs):
    blocks = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            blocks[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return blocks


class DefenseV19Substrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        # MI stats (read-only, no attention modification)
        self._mi_mu = None
        self._mi_var = None
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_count = None
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        self._prev_blocks = None

        # Temporal bisection state
        self._walk_number = 0
        self._walk_step = 0
        self._walk_snapshots = []     # (step, blocks) at each snapshot
        self._walk_actions = []       # action sequence during walk
        self._walk_initial_obs = None
        self._responsive_windows = [] # (window_start, window_end, delta, action_subseq)
        self._bisect_complete = False
        self._any_response_found = False

        # Observation tracking
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)

        # Frequency tables for goal
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self._raw_goal = None

        # SPSA goal (ℓ₁)
        self._spsa_goal = None  # block-space goal vector
        self._spsa_direction = None
        self._spsa_plus_score = 0.0
        self._spsa_minus_score = 0.0
        self._spsa_phase = 'plus'  # 'plus' or 'minus'
        self._spsa_step = 0

        # KB influence
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None

        self._detected_type = None
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
        self._archive_max = 20
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def _init_mi_stats(self, n_actions):
        self._mi_mu = np.zeros((n_actions, N_DIMS), dtype=np.float32)
        self._mi_var = np.full((n_actions, N_DIMS), 1e-4, dtype=np.float32)
        self._mi_count = np.zeros(n_actions, dtype=np.float32)

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()
        self._init_mi_stats(n_actions)

    def _update_mi(self, action, delta_blocks):
        if self._mi_mu is None:
            self._init_mi_stats(self._n_actions)
        if action >= len(self._mi_mu):
            return
        a = action
        self._mi_count[a] += 1
        alpha = 1.0 - MI_EMA
        self._mi_mu[a] = MI_EMA * self._mi_mu[a] + alpha * delta_blocks
        residual = delta_blocks - self._mi_mu[a]
        self._mi_var[a] = MI_EMA * self._mi_var[a] + alpha * (residual ** 2)
        self._mi_var_total = MI_EMA * self._mi_var_total + alpha * (delta_blocks ** 2)

    def _compute_mi(self):
        if self._mi_mu is None:
            return
        active = self._mi_count > 5
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)

    def _mi_action_score(self, action):
        if self._mi_mu is None or action >= len(self._mi_mu):
            return 0.0
        return float(np.sum(self._mi_values * np.abs(self._mi_mu[action])))

    # ── TEMPORAL BISECTION ──

    def _start_walk(self, blocks):
        """Begin a new random walk for bisection."""
        self._walk_step = 0
        self._walk_snapshots = [(0, blocks.copy())]
        self._walk_actions = []
        self._walk_initial_obs = blocks.copy()

    def _do_bisect_step(self, blocks):
        """Execute one step of the bisection walk. Returns action."""
        self._walk_step += 1

        # Save snapshot every SNAPSHOT_INTERVAL steps
        if self._walk_step % SNAPSHOT_INTERVAL == 0:
            self._walk_snapshots.append((self._walk_step, blocks.copy()))

        # Generate random action
        if self._supports_click and self._rng.random() < 0.5:
            cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
            action = _click_action(cx, cy)
        else:
            action = self._rng.randint(N_KB)
        self._walk_actions.append(action)

        # End of walk — analyze
        if self._walk_step >= WALK_LEN:
            self._analyze_walk(blocks)
            self._walk_number += 1
            if self._walk_number >= N_WALKS or self._any_response_found:
                self._bisect_complete = True

        return action

    def _analyze_walk(self, final_blocks):
        """Compare snapshots to find responsive windows."""
        initial = self._walk_initial_obs
        total_delta = float(np.sum(np.abs(final_blocks - initial)))

        if total_delta < CHANGE_THRESH:
            return  # No response in this walk

        self._any_response_found = True

        # Find which snapshot window had the biggest change
        best_delta = 0.0
        best_window = None
        for i in range(len(self._walk_snapshots) - 1):
            step_a, blocks_a = self._walk_snapshots[i]
            step_b, blocks_b = self._walk_snapshots[i + 1]
            window_delta = float(np.sum(np.abs(blocks_b - blocks_a)))
            if window_delta > best_delta:
                best_delta = window_delta
                best_window = (step_a, step_b)

        if best_window is not None and best_delta > CHANGE_THRESH * 0.3:
            start, end = best_window
            subseq = self._walk_actions[start:end]
            self._responsive_windows.append((start, end, best_delta, subseq))

        # Also find the 2nd-best window for diversity
        second_delta = 0.0
        second_window = None
        for i in range(len(self._walk_snapshots) - 1):
            step_a, blocks_a = self._walk_snapshots[i]
            step_b, blocks_b = self._walk_snapshots[i + 1]
            window_delta = float(np.sum(np.abs(blocks_b - blocks_a)))
            if window_delta > second_delta and (step_a, step_b) != best_window:
                second_delta = window_delta
                second_window = (step_a, step_b)

        if second_window is not None and second_delta > CHANGE_THRESH * 0.3:
            start, end = second_window
            subseq = self._walk_actions[start:end]
            self._responsive_windows.append((start, end, second_delta, subseq))

    # ── EVOLUTION ──

    def _random_sequence(self):
        length = self._rng.randint(SEQ_MIN, SEQ_MAX + 1)
        seq = []
        # Seed from responsive windows if available
        if self._responsive_windows and self._rng.random() < 0.6:
            _, _, _, subseq = self._responsive_windows[
                self._rng.randint(len(self._responsive_windows))]
            # Use a slice of the responsive window
            start = self._rng.randint(max(1, len(subseq) - SEQ_MAX))
            seq = list(subseq[start:start + length])
            return seq[:SEQ_MAX]

        for _ in range(length):
            if self._supports_click and self._rng.random() < 0.7:
                if self._best_click_regions and self._rng.random() < 0.5:
                    cx, cy = self._best_click_regions[self._rng.randint(len(self._best_click_regions))]
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
        """ℓ₁ fitness: block-space L1 distance (no attention weighting)."""
        blocks_start = _obs_to_blocks(obs_start)
        blocks_end = _obs_to_blocks(obs_end)
        return float(np.sum(np.abs(blocks_end - blocks_start)))

    def _do_kb_bootloader(self, arr):
        goal = self._raw_goal if self._raw_goal is not None else arr
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.kb_influence[k] * mismatch) + 0.5 * self._mi_action_score(k)
        action = int(np.argmax(kb_scores))
        if self._rng.random() < 0.1:
            action = self._rng.randint(N_KB)
        self.prev_action_type = 'kb'
        self.prev_kb_idx = action
        return action

    def _do_click_exploit(self, arr):
        goal = self._raw_goal if self._raw_goal is not None else arr
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)
        if self._rng.random() < 0.1:
            if self._best_click_regions:
                cx, cy = self._best_click_regions[self._rng.randint(len(self._best_click_regions))]
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
        blocks = _obs_to_blocks(arr)

        # MI tracking (read-only — no attention modification)
        if self._prev_blocks is not None and self._prev_action is not None:
            delta_blocks = blocks - self._prev_blocks
            self._update_mi(self._prev_action, delta_blocks)
            if self.step_count % 100 == 0:
                self._compute_mi()

        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self._prev_blocks = blocks.copy()
            self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
            self._start_walk(blocks)
            action = self._rng.randint(N_KB)
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff
        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)
        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self.prev_obs = arr.copy()
        self._prev_blocks = blocks.copy()

        # ── PHASE 0: TEMPORAL BISECTION (steps 1 to ~700) ──
        if not self._bisect_complete:
            action = self._do_bisect_step(blocks)
            # If walk just ended and we need another, start it
            if self._walk_step >= WALK_LEN and not self._bisect_complete:
                self._start_walk(blocks)
            self.prev_action_type = 'kb' if action < N_KB else 'click'
            self.prev_kb_idx = action if action < N_KB else None
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Post-bisection: decide detection type ──
        if self._detected_type is not None:
            action = self._exploit(arr)
        elif self.step_count < KB_PROBE_END:
            # Short-horizon probe for responsive games (standard)
            action = self._rng.randint(N_KB)
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
        elif self.step_count == KB_PROBE_END:
            self._compute_mi()
            if float(self._mi_values.max()) > MI_THRESH:
                self._detected_type = 'kb'
            elif self._responsive_windows:
                # Bisection found something — seed evolution with window sequences
                self._detected_type = 'seq'
            action = self._do_kb_bootloader(arr) if self._detected_type == 'kb' else self._rng.randint(self._n_actions)
        elif self.step_count < CLICK_PROBE_END:
            if self._supports_click:
                click_phase = (self.step_count - KB_PROBE_END) // SUSTAIN_STEPS
                grid_idx = click_phase % len(CLICK_GRID)
                cx, cy = CLICK_GRID[grid_idx]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
            else:
                self._detected_type = 'kb' if float(self._mi_values.max()) > MI_THRESH else 'seq'
                action = self._do_kb_bootloader(arr)
        elif self.step_count == CLICK_PROBE_END:
            if self._detected_type is None:
                self._compute_mi()
                if float(self._mi_values.max()) > MI_THRESH:
                    self._detected_type = 'click'
                    high_mi = np.argwhere(self._mi_values > np.percentile(self._mi_values, 75))
                    for bi in high_mi:
                        by, bx = bi[0] // N_BLOCKS, bi[0] % N_BLOCKS
                        self._best_click_regions.append((bx * BLOCK_SIZE + 4, by * BLOCK_SIZE + 4))
                elif self._responsive_windows:
                    self._detected_type = 'seq'
                else:
                    self._detected_type = 'unknown'
            action = self._do_click_exploit(arr) if self._detected_type == 'click' else self._rng.randint(self._n_actions)
        elif self.step_count < SEQ_PROBE_END:
            action = self._do_evolution(arr)
        elif self.step_count == SEQ_PROBE_END:
            if self._detected_type is None:
                if self._archive and max(s for s, _ in self._archive) > 0:
                    self._detected_type = 'seq'
                else:
                    self._detected_type = 'unknown'
            action = self._do_evolution(arr)
        else:
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

    def _exploit(self, arr):
        if self._detected_type == 'kb':
            return self._do_kb_bootloader(arr)
        elif self._detected_type == 'click':
            return self._do_click_exploit(arr)
        elif self._detected_type == 'seq':
            if self.step_count < SEQ_PROBE_END + 3000:
                return self._do_evolution(arr)
            else:
                return self._do_exploit_sequences(arr)
        else:
            # Unknown — cycle through strategies
            if self.step_count % 3 == 0:
                return self._do_kb_bootloader(arr)
            elif self._supports_click and self.step_count % 3 == 1:
                return self._do_click_exploit(arr)
            else:
                return self._do_evolution(arr)

    def on_level_transition(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.raw_freq[:] = 0
        self.prev_obs = None
        self._prev_blocks = None
        self._raw_goal = None
        self._detected_type = None
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
        self._init_mi_stats(self._n_actions)
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        # Keep bisection findings across levels — responsive windows persist
        # Reset walk state for new level
        self._walk_number = 0
        self._walk_step = 0
        self._walk_snapshots = []
        self._walk_actions = []
        self._walk_initial_obs = None
        self._bisect_complete = False if not self._any_response_found else True


CONFIG = {
    "walk_len": WALK_LEN,
    "snapshot_interval": SNAPSHOT_INTERVAL,
    "n_walks": N_WALKS,
    "change_thresh": CHANGE_THRESH,
    "mi_thresh": MI_THRESH,
    "spsa_lr": SPSA_LR,
    "v19_features": "temporal bisection (long walks) + window seeding + MI detection (read-only) + SPSA goal (l1)",
}

SUBSTRATE_CLASS = DefenseV19Substrate
