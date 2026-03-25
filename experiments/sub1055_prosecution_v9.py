"""
sub1055_prosecution_v9.py — Prosecution v9: strategy-level internal restarts (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1055 --substrate experiments/sub1055_prosecution_v9.py

FAMILY: attention-gated
R3 HYPOTHESIS: ℓ_π with STRATEGY-level restarts, not weight-level.
  Jun insight: substrate should explore different starting CONDITIONS, not
  starting WEIGHTS. Each epoch tries a fundamentally different action strategy.
  Attention learned in one strategy transfers to inform the next (ℓ_π R3).

Epoch 1 (0-2500): KEYBOARD STRATEGY — intensive keyboard exploration.
  Self-evaluate: which dims respond to keyboard actions?
Epoch 2 (2500-5000): CLICK STRATEGY — pixel click exploration.
  Self-evaluate: which regions respond to clicks?
Epoch 3 (5000-7500): SEQUENCE STRATEGY — evolutionary sequences using
  BEST actions from epochs 1+2 (only actions that showed signal).
  Attention weights carry across epochs → informed sequence construction.
Epoch 4 (7500-10000): EXPLOIT — best strategy from epochs 1-3.

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

# Epoch boundaries
EPOCH_1_END = 2500   # keyboard strategy
EPOCH_2_END = 5000   # click strategy
EPOCH_3_END = 7500   # sequence strategy
# 7500-10000 = exploit

# Attention R3 parameters
ATT_INIT = 0.5
ATT_LR = 0.02
ATT_MIN = 0.01
ATT_MAX = 1.0

# Evolutionary parameters (epoch 3)
POP_SIZE = 10
SEQ_MIN = 3
SEQ_MAX = 7
MUTATE_EVERY = 10

# Strategy evaluation thresholds
CHANGE_SIGNAL_THRESH = 0.05  # min change_map mean to count as "signal"

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


class ProsecutionV9Substrate:
    """
    ℓ_π R3 v9: strategy-level internal restarts.
    Each epoch tries a different action strategy.
    Attention weights transfer across strategies (ℓ_π R3).
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

        # Strategy evaluation
        self._kb_change_map = np.zeros((64, 64), dtype=np.float32)
        self._click_change_map = np.zeros((64, 64), dtype=np.float32)
        self._kb_has_signal = False
        self._click_has_signal = False

        # Best actions discovered per epoch
        self._best_kb_actions = []   # keyboard actions that caused change
        self._best_click_regions = []  # click regions that caused change

        # Evolution state (epoch 3)
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False

        # Archive
        self._archive = []
        self._archive_max = 15

        # Exploit state
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0
        self._best_strategy = None  # 'kb', 'click', 'seq', or 'mixed'

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()

    def _random_sequence(self):
        """Build sequence from discovered effective actions."""
        length = self._rng.randint(SEQ_MIN, SEQ_MAX + 1)
        seq = []
        for _ in range(length):
            # Use discovered actions if available
            if self._kb_has_signal and self._click_has_signal:
                # Mixed: use both
                if self._rng.random() < 0.5 and self._best_click_regions:
                    cx, cy = self._best_click_regions[
                        self._rng.randint(len(self._best_click_regions))]
                    seq.append(_click_action(cx, cy))
                elif self._best_kb_actions:
                    seq.append(self._best_kb_actions[
                        self._rng.randint(len(self._best_kb_actions))])
                else:
                    seq.append(self._rng.randint(N_KB))
            elif self._kb_has_signal and self._best_kb_actions:
                seq.append(self._best_kb_actions[
                    self._rng.randint(len(self._best_kb_actions))])
            elif self._click_has_signal and self._best_click_regions:
                cx, cy = self._best_click_regions[
                    self._rng.randint(len(self._best_click_regions))]
                seq.append(_click_action(cx, cy))
            else:
                # No signal from either — use random
                if self._supports_click and self._rng.random() < 0.7:
                    cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                    seq.append(_click_action(cx, cy))
                else:
                    seq.append(self._rng.randint(N_KB))
        return seq

    def _mutate_sequence(self, seq):
        seq = list(seq)
        mut = self._rng.randint(4)
        if mut == 0 and len(seq) > SEQ_MIN:
            idx = self._rng.randint(len(seq))
            seq.pop(idx)
        elif mut == 1 and len(seq) < SEQ_MAX:
            idx = self._rng.randint(len(seq) + 1)
            new_action = self._random_sequence()[0] if self._rng.random() < 0.5 else self._rng.randint(self._n_actions)
            seq.insert(idx, new_action)
        elif mut == 2:
            idx = self._rng.randint(len(seq))
            seq[idx] = self._random_sequence()[0] if self._rng.random() < 0.5 else self._rng.randint(self._n_actions)
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
            n_from_archive = min(POP_SIZE // 2, len(self._archive))
            archive_sorted = sorted(self._archive, key=lambda x: -x[0])
            pop = [self._mutate_sequence(archive_sorted[i][1])
                   for i in range(n_from_archive)]
            pop += [self._random_sequence() for _ in range(POP_SIZE - n_from_archive)]
            self._evo_pop = pop
        else:
            self._evo_pop = [self._random_sequence() for _ in range(POP_SIZE)]
        self._evo_scores = [0.0] * POP_SIZE
        self._evo_counts = [0] * POP_SIZE
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_total_evals = 0
        self._evo_initialized = True

    def _archive_sequence(self, score, seq):
        self._archive.append((score, list(seq)))
        if len(self._archive) > self._archive_max:
            self._archive.sort(key=lambda x: -x[0])
            self._archive = self._archive[:self._archive_max]

    def _fitness(self, obs_start, obs_end):
        """Attention-weighted total change (ℓ_π fitness)."""
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

    def _evaluate_epoch1(self):
        """Self-evaluate keyboard strategy."""
        mean_change = float(self._kb_change_map.mean())
        self._kb_has_signal = mean_change > CHANGE_SIGNAL_THRESH
        if self._kb_has_signal:
            # Find which keyboard actions caused the most change
            for k in range(N_KB):
                if float(self.kb_influence[k].mean()) > CHANGE_SIGNAL_THRESH:
                    self._best_kb_actions.append(k)
            if not self._best_kb_actions:
                self._best_kb_actions = list(range(N_KB))

    def _evaluate_epoch2(self):
        """Self-evaluate click strategy."""
        mean_change = float(self._click_change_map.mean())
        self._click_has_signal = mean_change > CHANGE_SIGNAL_THRESH
        if self._click_has_signal:
            # Find click regions that caused change
            smoothed = uniform_filter(self._click_change_map, size=8)
            threshold = float(np.percentile(smoothed, 75))
            responsive = np.argwhere(smoothed > threshold)
            if len(responsive) > 0:
                # Sample up to 20 responsive positions
                indices = self._rng.choice(len(responsive),
                                           size=min(20, len(responsive)),
                                           replace=False)
                for i in indices:
                    y, x = responsive[i]
                    self._best_click_regions.append((int(x), int(y)))
            if not self._best_click_regions:
                self._best_click_regions = CLICK_GRID[:20]

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

        # R3 attention update
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
        self.prev_obs = arr.copy()

        # ── Epoch 1: KEYBOARD STRATEGY (0-2500) ──
        if self.step_count < EPOCH_1_END:
            # Intensive keyboard exploration — 350 steps per key
            kb = (self.step_count - 1) % N_KB
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = kb
            self._kb_change_map = ALPHA_CHANGE * self._kb_change_map + (1 - ALPHA_CHANGE) * diff

        # ── Epoch 2: CLICK STRATEGY (2500-5000) ──
        elif self.step_count < EPOCH_2_END:
            if self.step_count == EPOCH_1_END:
                self._evaluate_epoch1()

            if not self._supports_click:
                # No clicks available — re-explore keyboard with attention
                goal = self._gated_goal if self._gated_goal is not None else arr
                mismatch = self.attention * np.abs(arr - goal) * self.change_map
                suppress_mask = (self.suppress == 0).astype(np.float32)
                mismatch *= suppress_mask
                kb_scores = np.zeros(N_KB)
                for k in range(N_KB):
                    kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * mismatch)
                action = int(np.argmax(kb_scores))
                if self._rng.random() < 0.3:
                    action = self._rng.randint(N_KB)
                self.prev_action_type = 'kb'
                self.prev_kb_idx = action
            else:
                # Random pixel clicks to discover click-responsive regions
                if self._rng.random() < 0.9:
                    cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                    action = _click_action(cx, cy)
                    self.prev_action_type = 'click'
                    self.prev_kb_idx = None
                else:
                    kb = self._rng.randint(N_KB)
                    action = kb
                    self.prev_action_type = 'kb'
                    self.prev_kb_idx = kb
                self._click_change_map = ALPHA_CHANGE * self._click_change_map + (1 - ALPHA_CHANGE) * diff

        # ── Epoch 3: SEQUENCE STRATEGY (5000-7500) ──
        elif self.step_count < EPOCH_3_END:
            if self.step_count == EPOCH_2_END:
                self._evaluate_epoch2()

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
                    self._archive_sequence(score, seq)

                if self._evo_total_evals % MUTATE_EVERY == 0 and self._evo_total_evals > POP_SIZE:
                    worst = int(np.argmin(self._evo_scores))
                    best = int(np.argmax(self._evo_scores))
                    if worst != best:
                        self._evo_pop[worst] = self._mutate_sequence(self._evo_pop[best])
                        self._evo_scores[worst] = self._evo_scores[best] * 0.5
                        self._evo_counts[worst] = 0

                self._evo_current = (self._evo_current + 1) % POP_SIZE
                self._evo_exec_idx = 0

        # ── Epoch 4: EXPLOIT (7500-10000) ──
        else:
            if not self._top_sequences:
                if self._archive:
                    self._archive.sort(key=lambda x: -x[0])
                    self._top_sequences = [s for _, s in self._archive[:5]]
                if not self._top_sequences:
                    self._top_sequences = [self._random_sequence()]

            if self._rng.random() < 0.1:
                if self._supports_click and self._rng.random() < 0.5:
                    cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                    action = _click_action(cx, cy)
                    self.prev_action_type = 'click'
                else:
                    action = self._rng.randint(N_KB)
                    self.prev_action_type = 'kb'
                    self.prev_kb_idx = action
            elif self._rng.random() < 0.7:
                seq = self._top_sequences[self._exploit_current]
                action = seq[self._exploit_exec_idx]
                if action >= self._n_actions:
                    action = self._rng.randint(self._n_actions)
                self._exploit_exec_idx += 1
                if self._exploit_exec_idx >= len(seq):
                    self._exploit_exec_idx = 0
                    self._exploit_current = (self._exploit_current + 1) % len(self._top_sequences)
            else:
                goal = self._gated_goal if self._gated_goal is not None else arr
                mismatch = self.attention * np.abs(arr - goal) * self.change_map
                suppress_mask = (self.suppress == 0).astype(np.float32)
                mismatch *= suppress_mask
                smoothed = uniform_filter(mismatch, size=KERNEL)

                click_score = float(np.max(smoothed)) if self._supports_click else -1.0
                kb_scores = np.zeros(N_KB)
                for k in range(N_KB):
                    kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * smoothed)
                best_kb = int(np.argmax(kb_scores))

                if self._supports_click and click_score >= kb_scores[best_kb]:
                    idx = np.argmax(smoothed)
                    y, x = np.unravel_index(idx, (64, 64))
                    action = _click_action(int(x), int(y))
                    y0, y1 = max(0, y - SUPPRESS_RADIUS), min(64, y + SUPPRESS_RADIUS + 1)
                    x0, x1 = max(0, x - SUPPRESS_RADIUS), min(64, x + SUPPRESS_RADIUS + 1)
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
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.raw_freq[:] = 0
        self.gated_freq[:] = 0
        self.prev_obs = None
        self._raw_goal = None
        self._gated_goal = None
        self._kb_change_map = np.zeros((64, 64), dtype=np.float32)
        self._click_change_map = np.zeros((64, 64), dtype=np.float32)
        self._kb_has_signal = False
        self._click_has_signal = False
        self._best_kb_actions = []
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
        self._best_strategy = None
        # Keep attention + kb_influence across levels


CONFIG = {
    "epochs": "keyboard(0-2500)/click(2500-5000)/sequence(5000-7500)/exploit(7500-10000)",
    "pop_size": POP_SIZE,
    "seq_range": f"{SEQ_MIN}-{SEQ_MAX}",
    "change_signal_thresh": CHANGE_SIGNAL_THRESH,
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "v9_features": "strategy-level restarts — kb/click/seq epochs + attention transfer (l_pi)",
}

SUBSTRATE_CLASS = ProsecutionV9Substrate
