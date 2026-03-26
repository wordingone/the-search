"""
sub1111_prosecution_v29.py — Alpha-gated click targeting

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1111 --substrate experiments/sub1111_prosecution_v29.py

FAMILY: Alpha-gated action (prosecution branch). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: Alpha discovers which avgpool4 blocks are game-responsive.
Click those blocks. ℓ_π because click target selection is encoding-modified.
Defense v35 clicks ALL block centers equally; prosecution v29 clicks the
blocks alpha identifies as INFORMATIVE.

Phase 1 (steps 0-100): keyboard exploration, learn alpha.
Phase 2 (steps 100+): alternate keyboard/click phases (200 steps each).
  - Click targets = top-16 blocks by alpha weight.
  - Stick with whichever phase shows progress.

KILL: 0/3 ARC AND no click-game signal.
SUCCESS: ANY previously-0% game shows signal from click actions.
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
WARMUP = 100
PHASE_LENGTH = 200
MAX_PATIENCE = 20


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


class AlphaClickTargetingSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._enc_0 = None
        self._prev_dist = 0.0

        # Alpha (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        # Phase management
        self._phase = "keyboard"  # "keyboard" or "click"
        self._phase_steps = 0
        self._phase_progress_count = 0

        # Keyboard action state
        self._kb_action = 0
        self._kb_patience = 3
        self._kb_steps_on = 0
        self._kb_consecutive = 0

        # Click target state
        self._click_targets = []
        self._click_idx = 0
        self._click_patience = 3
        self._click_steps_on = 0
        self._click_consecutive = 0

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

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _keyboard_action(self, dist):
        """Reactive switching over keyboard actions (v30 logic)."""
        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(self._prev_dist - dist) < 1e-6
        self._kb_steps_on += 1

        if progress:
            self._kb_consecutive += 1
            self._kb_patience = min(3 + self._kb_consecutive, MAX_PATIENCE)
            self._phase_progress_count += 1
        else:
            self._kb_consecutive = 0
            if self._kb_steps_on >= self._kb_patience or no_change:
                self._kb_action = (self._kb_action + 1) % N_KB
                self._kb_steps_on = 0
                self._kb_patience = 3

        return self._kb_action

    def _click_action(self, dist):
        """Reactive switching over alpha-selected click targets."""
        if not self._click_targets:
            return self._rng.randint(N_KB)

        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(self._prev_dist - dist) < 1e-6
        self._click_steps_on += 1

        if progress:
            self._click_consecutive += 1
            self._click_patience = min(3 + self._click_consecutive, MAX_PATIENCE)
            self._phase_progress_count += 1
        else:
            self._click_consecutive = 0
            if self._click_steps_on >= self._click_patience or no_change:
                self._click_idx = (self._click_idx + 1) % len(self._click_targets)
                self._click_steps_on = 0
                self._click_patience = 3

        block = self._click_targets[self._click_idx % len(self._click_targets)]
        return _block_to_click_action(block)

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
            self._kb_action = self._rng.randint(N_KB)
            return self._kb_action

        dist = self._dist_to_initial(enc)

        # Update alpha from prediction error
        if self._prev_enc is not None:
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

        # Phase 1: keyboard warmup
        if self.step_count <= WARMUP:
            action = self.step_count % N_KB
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Update click targets from alpha
        if self._has_clicks and self.step_count == WARMUP + 1:
            self._update_click_targets()

        # Phase switching
        self._phase_steps += 1
        if self._phase_steps >= PHASE_LENGTH and self._has_clicks:
            old_progress = self._phase_progress_count
            if self._phase == "keyboard":
                self._phase = "click"
                self._update_click_targets()
            else:
                self._phase = "keyboard"
            self._phase_steps = 0
            self._phase_progress_count = 0

        # Select action based on phase
        if not self._has_clicks or self._phase == "keyboard":
            action = self._keyboard_action(dist)
        else:
            action = self._click_action(dist)

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = 0.0
        self._kb_action = 0
        self._kb_patience = 3
        self._kb_steps_on = 0
        self._kb_consecutive = 0
        self._click_idx = 0
        self._click_patience = 3
        self._click_steps_on = 0
        self._click_consecutive = 0
        self._phase_steps = 0
        self._phase_progress_count = 0
        # Keep alpha and W_pred across levels (ℓ_π)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "top_k_clicks": TOP_K_CLICKS,
    "pred_lr": PRED_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "warmup": WARMUP,
    "phase_length": PHASE_LENGTH,
    "max_patience": MAX_PATIENCE,
    "family": "alpha-gated click targeting",
    "tag": "prosecution v29 (ℓ_π alpha-gated click targets + keyboard/click phases)",
}

SUBSTRATE_CLASS = AlphaClickTargetingSubstrate
