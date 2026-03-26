"""
sub1109_prosecution_v28.py — Alpha-gated action elimination

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1109 --substrate experiments/sub1109_prosecution_v28.py

FAMILY: Alpha-gated elimination (NEW prosecution family)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: Alpha discovers which dims are game-informative. Actions that
don't change informative dims are eliminated. Fewer candidate actions = faster
convergence = higher ARC efficiency. Zero warmup — learning concurrent.

DIFFERENT FROM DEFENSE: alpha-gated elimination + alpha-weighted change check.
Defense cycles ALL actions with raw L1 distance-to-initial.

DIFFERENT FROM PROSECUTION v24: no forward model, no warmup. Simpler.
PB30-compliant.

KILL: ARC < 0 on all games.
SUCCESS: ARC > 0.01 on any game.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
PRED_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0
CHANGE_DECAY = 0.95
ELIM_START = 20  # start elimination after this many steps
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


class AlphaGatedEliminationSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None

        # Alpha (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS

        # W_pred: predict next enc from current (for alpha updates)
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        # Per-action per-dim change tracking
        self._action_change = np.zeros((self._n_actions, N_DIMS), dtype=np.float32)

        # Action selection state
        self._active_actions = list(range(self._n_actions))
        self._current_idx = 0
        self._steps_on_action = 0
        self._patience = 3
        self._consecutive_progress = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
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

    def _update_elimination(self):
        """Recompute active actions based on alpha-gated informativeness."""
        if self.step_count < ELIM_START:
            self._active_actions = list(range(self._n_actions))
            return

        # Informativeness: how much does each action change alpha-weighted dims?
        informative = self._action_change[:self._n_actions] @ self._alpha
        positives = informative[informative > 0]
        if len(positives) < 2:
            self._active_actions = list(range(self._n_actions))
            return

        threshold = float(np.median(positives))
        active = [a for a in range(self._n_actions) if informative[a] >= threshold]
        if len(active) < 2:
            self._active_actions = list(range(self._n_actions))
        else:
            self._active_actions = active

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._prev_enc is not None:
            prev_action = self._active_actions[self._current_idx % len(self._active_actions)]

            # Update action_change EMA
            delta = np.abs(enc - self._prev_enc)
            self._action_change[prev_action] = (
                CHANGE_DECAY * self._action_change[prev_action] +
                (1 - CHANGE_DECAY) * delta
            )

            # Update W_pred and alpha
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

            # Update elimination
            self._update_elimination()

            # Check alpha-weighted progress
            alpha_change = float(np.sum(self._alpha * delta))
            progress = alpha_change > 1e-4

            self._steps_on_action += 1

            if progress:
                self._consecutive_progress += 1
                self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            else:
                self._consecutive_progress = 0
                if self._steps_on_action >= self._patience:
                    # Move to next active action
                    self._current_idx = (self._current_idx + 1) % len(self._active_actions)
                    self._steps_on_action = 0
                    self._patience = 3
        else:
            self._current_idx = self._rng.randint(len(self._active_actions))

        self._prev_enc = enc.copy()
        action = self._active_actions[self._current_idx % len(self._active_actions)]
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._steps_on_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        # Keep alpha and action_change across levels (ℓ_π cross-level transfer)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "pred_lr": PRED_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "change_decay": CHANGE_DECAY,
    "elim_start": ELIM_START,
    "max_patience": MAX_PATIENCE,
    "family": "alpha-gated elimination",
    "tag": "prosecution v28 (ℓ_π alpha-gated action elimination + avgpool4 256D)",
}

SUBSTRATE_CLASS = AlphaGatedEliminationSubstrate
