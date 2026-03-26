"""
sub1133_defense_v47.py — Multi-strategy meta-reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1133 --substrate experiments/sub1133_defense_v47.py

FAMILY: Multi-strategy meta-reactive (NEW defense concept from rethink)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: 30 experiments show no SINGLE strategy works across all
game types. Meta-reactive: try multiple strategies sequentially, switching
when a strategy fails to produce progress within a budget.

Strategies (tried in order):
A. Keyboard reactive (v30-style, patience cycling) — 2000 steps
B. Saliency click targeting (v46-style, click salient blocks) — 2000 steps
C. Ultra-patient keyboard (patience=100, for games needing sustained action) — 2000 steps
D. Random click sweep (probe random pixels) — 2000 steps
E. Brute-force all actions with max patience — remaining budget

Each strategy gets 2000 steps. If progress is detected (distance-from-
initial increases by >5.0 during the budget), LOCK that strategy and use
remaining budget. If no progress, move to next strategy.

Zero learned params. No alpha, no W_pred. Pure systematic fallback.

KILL: ARC ≤ v30 (0.3319). Multi-strategy overhead must not hurt keyboard games.
SUCCESS: ARC > v30. Meta-fallback helps on draws where v30 gets 0%.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
STRATEGY_BUDGET = 2000
PROGRESS_THRESH = 5.0  # total distance increase needed to lock strategy
MAX_PATIENCE = 20
CHANGE_THRESH = 0.5
TOP_K_SALIENT = 16


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


def _block_to_click_action(block_idx):
    """Block center → click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class MultiStrategySubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._max_dist = 0.0

        # Strategy management
        self._strategies = ["keyboard_reactive", "saliency_click",
                           "ultra_patient_kb", "random_click", "brute_force"]
        self._strategy_idx = 0
        self._strategy_start_step = 0
        self._strategy_start_dist = 0.0
        self._locked = False  # once locked, don't switch

        # Reactive switching state (shared across strategies)
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        # Saliency state
        self._salient_blocks = []
        self._salient_probe_idx = 0
        self._salient_probe_step = 0
        self._salient_pre_enc = None
        self._responsive_clicks = []
        self._saliency_computed = False

        # Active action set
        self._active_actions = list(range(min(self._n_actions_env, N_KB)))

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()
        self._active_actions = list(range(min(n_actions, N_KB)))

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _current_strategy(self):
        if self._strategy_idx < len(self._strategies):
            return self._strategies[self._strategy_idx]
        return "brute_force"

    def _should_switch_strategy(self, dist):
        """Check if current strategy has made enough progress."""
        if self._locked:
            return False
        steps_in_strategy = self.step_count - self._strategy_start_step
        if steps_in_strategy < STRATEGY_BUDGET:
            return False
        # Check progress during this strategy's budget
        progress = dist - self._strategy_start_dist
        if progress > PROGRESS_THRESH:
            self._locked = True  # This strategy works, lock it
            return False
        return True  # No progress, switch

    def _advance_strategy(self, dist):
        """Move to next strategy."""
        self._strategy_idx += 1
        self._strategy_start_step = self.step_count
        self._strategy_start_dist = dist
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._saliency_computed = False
        self._salient_probe_idx = 0
        self._salient_probe_step = 0
        self._responsive_clicks = []

    def _reactive_step(self, dist, n_active):
        """Shared reactive switching logic."""
        progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
        no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

        self._steps_on_action += 1
        if progress:
            self._consecutive_progress += 1
            patience_max = MAX_PATIENCE if self._current_strategy() != "ultra_patient_kb" else 100
            self._patience = min(3 + self._consecutive_progress, patience_max)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0
            patience_base = 3 if self._current_strategy() != "ultra_patient_kb" else 30
            if self._steps_on_action >= self._patience or no_change:
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = patience_base
                if self._actions_tried_this_round >= n_active:
                    self._current_action = self._rng.randint(n_active)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action = (self._current_action + 1) % n_active

        return self._active_actions[self._current_action % n_active]

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._strategy_start_step = 1
            self._current_action = self._rng.randint(min(self._n_actions_env, N_KB))
            return self._current_action

        dist = self._dist_to_initial(enc)
        self._max_dist = max(self._max_dist, dist)

        # Check if we should switch strategy
        if self._should_switch_strategy(dist):
            self._advance_strategy(dist)

        strategy = self._current_strategy()

        # === Strategy A: Keyboard reactive (v30-style) ===
        if strategy == "keyboard_reactive":
            self._active_actions = list(range(min(self._n_actions_env, N_KB)))
            action = self._reactive_step(dist, len(self._active_actions))
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # === Strategy B: Saliency click targeting ===
        if strategy == "saliency_click":
            if not self._has_clicks:
                # No clicks available, skip to next strategy
                self._advance_strategy(dist)
                strategy = self._current_strategy()
            else:
                # Compute saliency on first step of this strategy
                if not self._saliency_computed:
                    mean_val = enc.mean()
                    saliency = np.abs(enc - mean_val)
                    sorted_desc = np.argsort(saliency)[::-1]
                    self._salient_blocks = list(sorted_desc[:TOP_K_SALIENT].astype(int))
                    self._saliency_computed = True
                    self._salient_probe_idx = 0
                    self._salient_probe_step = 0
                    self._salient_pre_enc = enc.copy()

                # Probing phase
                if self._salient_probe_idx < len(self._salient_blocks):
                    if self._salient_probe_step > 0 and self._salient_pre_enc is not None:
                        if self._salient_probe_step == 2:
                            change = float(np.sum(np.abs(enc - self._salient_pre_enc)))
                            if change > CHANGE_THRESH:
                                self._responsive_clicks.append(
                                    self._salient_blocks[self._salient_probe_idx]
                                )
                            self._salient_probe_idx += 1
                            self._salient_probe_step = 0
                            self._salient_pre_enc = enc.copy()

                    if self._salient_probe_idx < len(self._salient_blocks):
                        self._salient_probe_step += 1
                        block = self._salient_blocks[self._salient_probe_idx]
                        self._prev_enc = enc.copy()
                        self._prev_dist = dist
                        return _block_to_click_action(block)

                # Probing done — build active set and reactive switch
                if self._responsive_clicks:
                    self._active_actions = []
                    for block in self._responsive_clicks:
                        self._active_actions.append(_block_to_click_action(block))
                else:
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))

                action = self._reactive_step(dist, len(self._active_actions))
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

        # === Strategy C: Ultra-patient keyboard ===
        if strategy == "ultra_patient_kb":
            self._active_actions = list(range(min(self._n_actions_env, N_KB)))
            # Use higher base patience (30) and max patience (100)
            action = self._reactive_step(dist, len(self._active_actions))
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # === Strategy D: Random click sweep ===
        if strategy == "random_click":
            if not self._has_clicks:
                self._advance_strategy(dist)
                strategy = self._current_strategy()
            else:
                # Random click each step, track what produces change
                block_idx = self._rng.randint(N_BLOCKS * N_BLOCKS)
                action = _block_to_click_action(block_idx)
                if self._prev_enc is not None:
                    change = float(np.sum(np.abs(enc - self._prev_enc)))
                    if change > CHANGE_THRESH:
                        # Found responsive click — but keep exploring
                        pass
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

        # === Strategy E: Brute force all actions ===
        if strategy == "brute_force":
            n_kb = min(self._n_actions_env, N_KB)
            if self._has_clicks:
                # Alternate between keyboard and random clicks
                if self.step_count % 3 == 0:
                    action = self._rng.randint(n_kb)
                else:
                    block_idx = self._rng.randint(N_BLOCKS * N_BLOCKS)
                    action = _block_to_click_action(block_idx)
            else:
                action = self.step_count % n_kb
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Fallback
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._max_dist = 0.0
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep strategy and locked state — same game type


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "strategy_budget": STRATEGY_BUDGET,
    "progress_thresh": PROGRESS_THRESH,
    "max_patience": MAX_PATIENCE,
    "change_thresh": CHANGE_THRESH,
    "family": "multi-strategy meta-reactive",
    "tag": "defense v47 (ℓ₁ multi-strategy: keyboard → saliency click → ultra-patient → random click → brute force)",
}

SUBSTRATE_CLASS = MultiStrategySubstrate
