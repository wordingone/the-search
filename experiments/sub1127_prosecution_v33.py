"""
sub1127_prosecution_v33.py — Click-first alpha targeting (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1127 --substrate experiments/sub1127_prosecution_v33.py

FAMILY: Alpha-gated click targeting (fixing v29's chicken-and-egg).
Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: Random click warmup discovers responsive regions, THEN
alpha concentrates on them for efficient targeting. Fixes v29 where
keyboard warmup taught alpha nothing useful for click games.

Phase 1 (0-50): keyboard probe — detect if game responds to keyboard.
Phase 2 (50-250): random click exploration — W_pred trains → alpha learns.
Phase 3 (250+): alpha-gated targeting — click top-16 blocks by alpha weight.
If keyboard responsive → v30-style reactive on keyboard.

KILL: ARC < v35 defense (0.015).
SUCCESS: ARC > 0.05 on click games.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
TOP_K_CLICKS = 16
PRED_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0
PHASE_1_END = 50
PHASE_2_END = 250
MAX_PATIENCE = 20
CHANGE_THRESH = 0.5


def _obs_to_enc(obs):
    """avgpool4 + center: 64x64 → 16x16 = 256D, zero-centered."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    enc -= enc.mean()
    return enc


def _block_to_click_action(block_idx):
    """Block center pixel → click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class ClickFirstAlphaSubstrate:
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

        # Alpha (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        # Phase 1 tracking
        self._kb_total_change = 0.0
        self._kb_responsive = False

        # Click targets
        self._click_targets = []
        self._click_idx = 0

        # Reactive switching state
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _update_alpha(self, pred_error):
        error_mag = np.abs(pred_error)
        emax = error_mag.max()
        if emax < 1e-8:
            return
        logits = ALPHA_CONC * error_mag / emax
        logits -= logits.max()
        exp_l = np.exp(logits)
        target = exp_l / (exp_l.sum() + 1e-8)
        self._alpha = (1 - ALPHA_LR) * self._alpha + ALPHA_LR * target
        self.r3_updates += 1
        self.att_updates_total += 1

    def _update_w_pred(self, enc, prev_enc):
        pred = self._w_pred @ prev_enc
        error = enc - pred
        norm_sq = np.dot(prev_enc, prev_enc) + 1e-8
        self._w_pred += PRED_LR * np.outer(error, prev_enc) / norm_sq
        return error

    def _update_click_targets(self):
        """Top-K blocks by alpha weight."""
        top = np.argsort(self._alpha)[-TOP_K_CLICKS:]
        self._click_targets = list(top.astype(int))
        self._click_idx = 0

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

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
            self._current_action = self._rng.randint(min(self._n_actions_env, N_KB))
            return self._current_action

        dist = self._dist_to_initial(enc)

        # Update W_pred and alpha from ANY observation
        if self._prev_enc is not None:
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

        # Track keyboard change during Phase 1
        if self.step_count <= PHASE_1_END and self._prev_enc is not None:
            change = float(np.sum(np.abs(enc - self._prev_enc)))
            self._kb_total_change += change

        # === Phase 1: Keyboard probe ===
        if self.step_count <= PHASE_1_END:
            action = self.step_count % min(self._n_actions_env, N_KB)
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Detect keyboard responsiveness at Phase 1 end
        if self.step_count == PHASE_1_END + 1:
            self._kb_responsive = self._kb_total_change > (CHANGE_THRESH * PHASE_1_END / 10)

        # === Phase 2: Random click warmup (click games only) ===
        if self.step_count <= PHASE_2_END and self._has_clicks and not self._kb_responsive:
            block_idx = self._rng.randint(N_BLOCKS * N_BLOCKS)
            action = _block_to_click_action(block_idx)
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # === Phase 3: Targeted action ===
        # Build click targets at phase 3 start
        if self.step_count == PHASE_2_END + 1 and self._has_clicks and not self._kb_responsive:
            self._update_click_targets()

        # Determine action set
        if self._kb_responsive:
            # Keyboard game → v30-style reactive on keyboard
            n_active = min(self._n_actions_env, N_KB)

            progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
            no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

            self._steps_on_action += 1
            if progress:
                self._consecutive_progress += 1
                self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
                self._actions_tried_this_round = 0
            else:
                self._consecutive_progress = 0
                if self._steps_on_action >= self._patience or no_change:
                    self._actions_tried_this_round += 1
                    self._steps_on_action = 0
                    self._patience = 3
                    if self._actions_tried_this_round >= n_active:
                        self._current_action = self._rng.randint(n_active)
                        self._actions_tried_this_round = 0
                    else:
                        self._current_action = (self._current_action + 1) % n_active

            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._current_action
        else:
            # Click game → alpha-gated targeting
            if not self._click_targets:
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return self._rng.randint(min(self._n_actions_env, N_KB))

            n_targets = len(self._click_targets)

            progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
            no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

            self._steps_on_action += 1
            if progress:
                self._consecutive_progress += 1
                self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
                self._actions_tried_this_round = 0
            else:
                self._consecutive_progress = 0
                if self._steps_on_action >= self._patience or no_change:
                    self._actions_tried_this_round += 1
                    self._steps_on_action = 0
                    self._patience = 3
                    if self._actions_tried_this_round >= n_targets:
                        self._click_idx = self._rng.randint(n_targets)
                        self._actions_tried_this_round = 0
                        # Refresh targets periodically
                        self._update_click_targets()
                    else:
                        self._click_idx = (self._click_idx + 1) % n_targets

            block = self._click_targets[self._click_idx % n_targets]
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return _block_to_click_action(block)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._click_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep alpha and W_pred across levels (ℓ_π)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "top_k_clicks": TOP_K_CLICKS,
    "pred_lr": PRED_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "phase_1_end": PHASE_1_END,
    "phase_2_end": PHASE_2_END,
    "max_patience": MAX_PATIENCE,
    "family": "alpha-gated click targeting",
    "tag": "prosecution v33 (ℓ_π click-first warmup → alpha-gated targeting, fixes v29 chicken-and-egg)",
}

SUBSTRATE_CLASS = ClickFirstAlphaSubstrate
