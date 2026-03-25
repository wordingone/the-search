"""
sub1069_defense_v17.py — Defense v17: block-reduced SPSA goal (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1069 --substrate experiments/sub1069_defense_v17.py

FAMILY: parametric-goal
R3 HYPOTHESIS: Defense v16 failed (ALL zeros) because per-pixel SPSA on 4096 dims
  is intractable — too many parameters for finite-difference gradient estimation.
  Block-reduced SPSA on 64 dims (8×8 block means) is tractable. The hypothesis:
  ℓ₁ parameter adaptation IS sufficient when the parameter space is appropriately
  reduced. Same multivariate Mahalanobis detection as prosecution v15, but goal
  adaptation via 64-dim block SPSA instead of per-block attention.

  If defense v17 matches prosecution v15: ℓ₁ works when dimensionality is manageable.
  If defense v17 < prosecution v15: ℓ_π attention has inherent advantage beyond dims.

KILL: ALL games 0%
SUCCESS: Match or beat prosecution v15
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

# Block reduction: 64x64 → 8x8 blocks → 64 dims
BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64

# Probe boundaries
BASELINE_END = 200
KB_PROBE_END = 700
CLICK_PROBE_END = 1200
SEQ_PROBE_END = 2200

# Multivariate detection (same as prosecution v15)
MAHAL_THRESH = 3.0
Z_THRESH = 3.0
EPSILON = 1e-6
PROBE_SIGNAL_THRESH = 0.03
PCA_DIMS = 32

# Sustained probe
SUSTAIN_STEPS = 10

# Block SPSA parameters (ℓ₁) — 64 dims, not 4096
SPSA_BLOCK_C = 0.5       # perturbation size
SPSA_BLOCK_A = 0.1       # step size
SPSA_DECAY = 0.999       # step size decay
SPSA_EVAL_WINDOW = 20    # steps between SPSA updates

# Evolutionary parameters
POP_SIZE = 12
SEQ_MIN = 3
SEQ_MAX = 15
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


def _obs_to_blocks(obs):
    """Reduce 64x64 observation to 64-dim block means."""
    blocks = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            blocks[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return blocks


def _blocks_to_pixel_goal(block_goal):
    """Upsample 64-dim block goal to 64×64 pixel goal."""
    goal = np.zeros((64, 64), dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            goal[y0:y1, x0:x1] = block_goal[by * N_BLOCKS + bx]
    return goal


class DefenseV17Substrate:
    """
    ℓ₁ R3 v17: block-reduced SPSA goal + multivariate Mahalanobis detection.
    Same detection as prosecution v15 (PCA whitening, Mahalanobis distance).
    Goal adaptation via 64-dim block SPSA (tractable) instead of attention (ℓ_π).
    Uniform weighting across blocks — no attention encoding.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        # Multivariate baseline (block-reduced)
        self._block_samples = []
        self._block_mean = None
        self._pca_components = None
        self._pca_eigenvals = None
        self._baseline_done = False
        self._baseline_count = 0

        # Per-pixel baseline (fallback)
        self._pixel_mean = None
        self._pixel_var = None

        # Change detection
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.block_change = np.zeros(N_DIMS, dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)

        # Block SPSA goal (ℓ₁) — 64 parameters
        self.block_goal = np.full(N_DIMS, 7.5, dtype=np.float32)  # neutral init
        self._pixel_goal = _blocks_to_pixel_goal(self.block_goal)
        self._spsa_step_size = SPSA_BLOCK_A
        self._spsa_perturbation = None
        self._spsa_eval_count = 0
        self._spsa_loss_plus = 0.0
        self._spsa_loss_minus = 0.0
        self._spsa_phase = 'plus'  # 'plus' or 'minus'

        # Frequency tracking
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self._raw_goal = None
        self._freq_goal = None

        # KB influence
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None

        # Sustained probe state
        self._sustain_action = None
        self._sustain_remaining = 0

        # Cascade detection
        self._detected_type = None
        self._kb_change_accum = 0.0
        self._kb_mahal_accum = 0.0
        self._click_change_accum = 0.0
        self._click_mahal_accum = 0.0
        self._best_click_regions = []

        # Evolution
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
            self.spsa_updates_total = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()

    def _compute_pca(self):
        """Compute PCA whitening from baseline block samples."""
        if len(self._block_samples) < PCA_DIMS + 5:
            return False
        X = np.array(self._block_samples, dtype=np.float64)
        self._block_mean = X.mean(axis=0).astype(np.float32)
        X_centered = X - self._block_mean
        cov = X_centered.T @ X_centered / max(len(X) - 1, 1)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        k = min(PCA_DIMS, np.sum(eigenvals > EPSILON))
        if k < 2:
            return False
        self._pca_components = eigenvecs[:, :k].T.astype(np.float32)
        self._pca_eigenvals = np.maximum(eigenvals[:k], EPSILON).astype(np.float32)
        return True

    def _compute_mahalanobis(self, obs):
        """Compute Mahalanobis distance of block-reduced obs from baseline."""
        if self._pca_components is None or self._block_mean is None:
            return 0.0, np.zeros(N_DIMS, dtype=np.float32)
        blocks = _obs_to_blocks(obs)
        centered = blocks - self._block_mean
        projected = self._pca_components @ centered
        z_scores_pca = projected / np.sqrt(self._pca_eigenvals)
        mahal_dist = float(np.sqrt(np.sum(z_scores_pca ** 2)))
        block_contrib = np.abs(self._pca_components.T @ z_scores_pca)
        return mahal_dist, block_contrib.astype(np.float32)

    def _update_pixel_baseline(self, obs):
        self._baseline_count += 1
        if self._pixel_mean is None:
            self._pixel_mean = obs.copy()
            self._pixel_var = np.zeros((64, 64), dtype=np.float32)
        else:
            delta = obs - self._pixel_mean
            self._pixel_mean += delta / self._baseline_count
            delta2 = obs - self._pixel_mean
            self._pixel_var += (delta * delta2 - self._pixel_var) / self._baseline_count

    def _spsa_goal_loss(self, obs):
        """ℓ₁ loss: block-level mismatch between obs and current goal, weighted by change_map."""
        blocks = _obs_to_blocks(obs)
        block_change = _obs_to_blocks(self.change_map)
        return float(np.sum(block_change * np.abs(blocks - self.block_goal)))

    def _spsa_update(self, obs):
        """One SPSA step on block goal (64 dims)."""
        self._spsa_eval_count += 1

        if self._spsa_eval_count % SPSA_EVAL_WINDOW == 0:
            if self._spsa_perturbation is None:
                # Start new perturbation cycle
                self._spsa_perturbation = self._rng.choice([-1.0, 1.0], size=N_DIMS).astype(np.float32)
                self._spsa_loss_plus = 0.0
                self._spsa_loss_minus = 0.0
                self._spsa_phase = 'plus'
                # Apply + perturbation
                self.block_goal += SPSA_BLOCK_C * self._spsa_perturbation
                self.block_goal = np.clip(self.block_goal, 0, 15)
                self._pixel_goal = _blocks_to_pixel_goal(self.block_goal)
            elif self._spsa_phase == 'plus':
                # Measured plus, now measure minus
                self._spsa_loss_plus = self._spsa_goal_loss(obs)
                self._spsa_phase = 'minus'
                # Undo plus, apply minus
                self.block_goal -= 2 * SPSA_BLOCK_C * self._spsa_perturbation
                self.block_goal = np.clip(self.block_goal, 0, 15)
                self._pixel_goal = _blocks_to_pixel_goal(self.block_goal)
            else:
                # Measured minus, compute gradient
                self._spsa_loss_minus = self._spsa_goal_loss(obs)
                # Undo minus perturbation (back to original)
                self.block_goal += SPSA_BLOCK_C * self._spsa_perturbation
                # SPSA gradient estimate
                grad = (self._spsa_loss_plus - self._spsa_loss_minus) / (
                    2 * SPSA_BLOCK_C * self._spsa_perturbation + EPSILON)
                # Gradient descent on goal
                self.block_goal -= self._spsa_step_size * grad
                self.block_goal = np.clip(self.block_goal, 0, 15)
                self._pixel_goal = _blocks_to_pixel_goal(self.block_goal)
                self._spsa_step_size *= SPSA_DECAY
                self._spsa_perturbation = None
                self.r3_updates += 1
                self.spsa_updates_total += 1

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
        """Goal-mismatch fitness (ℓ₁): reduction in block mismatch."""
        blocks_start = _obs_to_blocks(obs_start)
        blocks_end = _obs_to_blocks(obs_end)
        block_cm = _obs_to_blocks(self.change_map)
        mismatch_before = np.sum(block_cm * np.abs(blocks_start - self.block_goal))
        mismatch_after = np.sum(block_cm * np.abs(blocks_end - self.block_goal))
        return float(mismatch_before - mismatch_after)

    def _do_kb_bootloader(self, arr):
        goal = self._pixel_goal
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
        goal = self._pixel_goal
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

        # SPSA goal update (ℓ₁)
        if self._baseline_done and self._prev_obs_arr is not None:
            self._spsa_update(arr)

        # Frequency tracking
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
            self._freq_goal = self._raw_goal.copy()
            self._update_pixel_baseline(arr)
            self._block_samples.append(_obs_to_blocks(arr))
            action = 0
            self.prev_action_type = 'kb'
            self.prev_kb_idx = 0
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        # Multivariate block change
        if self._baseline_done:
            mahal_dist, block_contrib = self._compute_mahalanobis(arr)
            self.block_change = ALPHA_CHANGE * self.block_change + (1 - ALPHA_CHANGE) * block_contrib

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._freq_goal = self._raw_goal.copy()
        self.prev_obs = arr.copy()

        # ── BASELINE PHASE (0-200) ──
        if self.step_count <= BASELINE_END:
            self._update_pixel_baseline(arr)
            self._block_samples.append(_obs_to_blocks(arr))
            if self.step_count == BASELINE_END:
                pca_ok = self._compute_pca()
                self._baseline_done = True
                if not pca_ok:
                    self._pca_components = None
                # Initialize block goal from frequency mode
                self.block_goal = _obs_to_blocks(self._raw_goal)
                self._pixel_goal = _blocks_to_pixel_goal(self.block_goal)
            kb = (self.step_count - 1) // SUSTAIN_STEPS % N_KB
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
            self._kb_change_accum += float(diff.mean())
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Already detected → exploit ──
        if self._detected_type is not None:
            action = self._exploit(arr)
        # ── KB PROBE (200-700): sustained holds ──
        elif self.step_count < KB_PROBE_END:
            kb = ((self.step_count - BASELINE_END) // SUSTAIN_STEPS) % N_KB
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = kb
            self._kb_change_accum += float(diff.mean())
            mahal_dist, _ = self._compute_mahalanobis(arr)
            if mahal_dist > MAHAL_THRESH:
                self._kb_mahal_accum += 1.0
        elif self.step_count == KB_PROBE_END:
            abs_signal = self._kb_change_accum / KB_PROBE_END > PROBE_SIGNAL_THRESH
            mahal_signal = self._kb_mahal_accum > 30
            if abs_signal or mahal_signal:
                self._detected_type = 'kb'
            action = self._do_kb_bootloader(arr) if self._detected_type == 'kb' else self._rng.randint(self._n_actions)
        # ── CLICK PROBE (700-1200): sustained holds ──
        elif self.step_count < CLICK_PROBE_END:
            if self._supports_click:
                click_phase = (self.step_count - KB_PROBE_END) // SUSTAIN_STEPS
                grid_idx = click_phase % len(CLICK_GRID)
                cx, cy = CLICK_GRID[grid_idx]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
                self._click_change_accum += float(diff.mean())
                mahal_dist, _ = self._compute_mahalanobis(arr)
                if mahal_dist > MAHAL_THRESH:
                    self._click_mahal_accum += 1.0
            else:
                self._detected_type = 'kb'
                action = self._do_kb_bootloader(arr)
        elif self.step_count == CLICK_PROBE_END:
            if self._detected_type is None:
                abs_signal = self._click_change_accum / max(1, CLICK_PROBE_END - KB_PROBE_END) > PROBE_SIGNAL_THRESH
                mahal_signal = self._click_mahal_accum > 20
                if abs_signal or mahal_signal:
                    self._detected_type = 'click'
                    responsive_blocks = np.argwhere(self.block_change > np.percentile(self.block_change, 75))
                    for bi in responsive_blocks:
                        by, bx = bi[0] // N_BLOCKS, bi[0] % N_BLOCKS
                        cx = bx * BLOCK_SIZE + BLOCK_SIZE // 2
                        cy = by * BLOCK_SIZE + BLOCK_SIZE // 2
                        self._best_click_regions.append((cx, cy))
            action = self._do_click_exploit(arr) if self._detected_type == 'click' else self._rng.randint(self._n_actions)
        # ── SEQUENCE PROBE (1200-2200) ──
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
            if self.step_count % 3 == 0:
                return self._do_kb_bootloader(arr)
            elif self._supports_click and self.step_count % 3 == 1:
                return self._do_click_exploit(arr)
            else:
                return self._do_evolution(arr)

    def on_level_transition(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.block_change = np.zeros(N_DIMS, dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.raw_freq[:] = 0
        self.prev_obs = None
        self._raw_goal = None
        self._freq_goal = None
        self._detected_type = None
        self._kb_change_accum = 0.0
        self._kb_mahal_accum = 0.0
        self._click_change_accum = 0.0
        self._click_mahal_accum = 0.0
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
        self._sustain_action = None
        self._sustain_remaining = 0
        # Reset baseline for new level
        self._block_samples = []
        self._block_mean = None
        self._pca_components = None
        self._pca_eigenvals = None
        self._baseline_done = False
        self._baseline_count = 0
        self._pixel_mean = None
        self._pixel_var = None
        # Reset SPSA state
        self._spsa_perturbation = None
        self._spsa_eval_count = 0
        self._spsa_step_size = SPSA_BLOCK_A
        # Re-initialize block goal from freq mode on next baseline completion
        # Keep kb_influence across levels (transfer)


CONFIG = {
    "baseline_steps": BASELINE_END,
    "probes": f"baseline(0-{BASELINE_END})/kb({BASELINE_END}-{KB_PROBE_END})/click({KB_PROBE_END}-{CLICK_PROBE_END})/seq({CLICK_PROBE_END}-{SEQ_PROBE_END})/exploit({SEQ_PROBE_END}+)",
    "block_size": BLOCK_SIZE,
    "n_dims": N_DIMS,
    "pca_dims": PCA_DIMS,
    "mahal_thresh": MAHAL_THRESH,
    "sustain_steps": SUSTAIN_STEPS,
    "spsa_block_c": SPSA_BLOCK_C,
    "spsa_block_a": SPSA_BLOCK_A,
    "spsa_eval_window": SPSA_EVAL_WINDOW,
    "pop_size": POP_SIZE,
    "seq_range": f"{SEQ_MIN}-{SEQ_MAX}",
    "v17_features": "block-reduced SPSA goal (64 dims) + multivariate Mahalanobis + sustained probes + adaptive cascade (l_1)",
}

SUBSTRATE_CLASS = DefenseV17Substrate
