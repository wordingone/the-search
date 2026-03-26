"""
sub1097_prosecution_v25.py — Forward model with action pruning

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1097 --substrate experiments/sub1097_prosecution_v25.py

FAMILY: Forward-model action selection (same as v23/v24)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: After warmup, W_fwd knows which actions produce change and
which don't. Pruning to top-K (K=3) actions eliminates epsilon waste on
useless actions. v24 wastes 20% of post-warmup steps on all 7 actions
equally. v25 restricts epsilon to top-3 predicted-change actions.
Re-rank every 500 steps to adapt to game dynamics.

CHANGES FROM v24:
1. After warmup, rank actions by mean predicted change, keep top-K (K=3)
2. Epsilon exploration only samples from top-K, not all 7
3. Re-evaluate rankings every 500 steps
Everything else IDENTICAL to v24.

KILL: ARC < v24 (0.0045/game) → pruning hurts.
SUCCESS: ARC > 0.05/game (10x improvement, closing toward defense).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
INPUT_DIMS = N_DIMS + N_KB    # 263
EPSILON = 0.20
PRED_LR = 0.001
FWD_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0
TOP_K = 3
RERANK_INTERVAL = 500


def _obs_to_enc(obs):
    """avgpool16 + center: 64x64 → 16x16 = 256D, zero-centered."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    enc -= enc.mean()
    return enc


class ForwardModelPrunedSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = None

        # Alpha (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS

        # W_pred: predict next enc from current (for alpha updates)
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        # W_fwd: action-conditioned forward model (256 × 263)
        self._w_fwd = np.zeros((N_DIMS, INPUT_DIMS), dtype=np.float32)
        self._w_fwd[:N_DIMS, :N_DIMS] = np.eye(N_DIMS) * 0.99

        # Action pruning state
        self._active_actions = None  # top-K actions (set after warmup)
        self._last_rerank_step = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _alpha_weight(self, enc):
        return enc * self._alpha

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

    def _make_input(self, enc_w, action):
        """Create forward model input: [enc_weighted, one_hot_action]."""
        x = np.zeros(INPUT_DIMS, dtype=np.float32)
        x[:N_DIMS] = enc_w
        x[N_DIMS + action] = 1.0
        return x

    def _update_w_fwd(self, enc_w_actual, prev_enc_w, prev_action):
        """Update action-conditioned forward model."""
        x = self._make_input(prev_enc_w, prev_action)
        pred = self._w_fwd @ x
        error = enc_w_actual - pred
        norm_sq = np.dot(x, x) + 1e-8
        self._w_fwd += FWD_LR * np.outer(error, x) / norm_sq

    def _rank_actions(self, enc_w):
        """Rank actions by predicted change magnitude, return top-K indices."""
        scores = np.zeros(self._n_actions, dtype=np.float32)
        for a in range(self._n_actions):
            x = self._make_input(enc_w, a)
            predicted_next = self._w_fwd @ x
            scores[a] = float(np.sqrt(np.sum((predicted_next - enc_w) ** 2)))
        k = min(TOP_K, self._n_actions)
        top_k = np.argsort(scores)[-k:]
        return top_k

    def _select_action(self, enc_w):
        """Select action with pruning: only evaluate/explore top-K actions."""
        if self.step_count < 100:  # warmup (same as v24)
            return self._rng.randint(self._n_actions)

        # Rank actions after warmup or every RERANK_INTERVAL steps
        if (self._active_actions is None or
                self.step_count - self._last_rerank_step >= RERANK_INTERVAL):
            self._active_actions = self._rank_actions(enc_w)
            self._last_rerank_step = self.step_count

        # Epsilon: explore within top-K only
        if self._rng.random() < EPSILON:
            return int(self._rng.choice(self._active_actions))

        # Greedy: argmax over top-K
        best_score = -1.0
        best_action = int(self._active_actions[0])
        for a in self._active_actions:
            x = self._make_input(enc_w, int(a))
            predicted_next = self._w_fwd @ x
            predicted_change = float(np.sqrt(np.sum((predicted_next - enc_w) ** 2)))
            if predicted_change > best_score:
                best_score = predicted_change
                best_action = int(a)

        return best_action

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        enc_w = self._alpha_weight(enc)

        if self._prev_enc is not None and self._prev_action is not None:
            prev_enc_w = self._alpha_weight(self._prev_enc)

            # Update forward model
            self._update_w_fwd(enc_w, prev_enc_w, self._prev_action)

            # Update W_pred and alpha
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

        action = self._select_action(enc_w)

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        # Keep forward model and alpha across levels
        self._prev_enc = None
        self._prev_action = None


CONFIG = {
    "n_dims": N_DIMS,
    "input_dims": INPUT_DIMS,
    "epsilon": EPSILON,
    "pred_lr": PRED_LR,
    "fwd_lr": FWD_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "top_k": TOP_K,
    "rerank_interval": RERANK_INTERVAL,
    "family": "forward-model action selection",
    "tag": "prosecution v25 (ℓ_π forward model + action pruning, top-3)",
}

SUBSTRATE_CLASS = ForwardModelPrunedSubstrate
