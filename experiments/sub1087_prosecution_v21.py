"""
sub1087_prosecution_v21.py — Alpha-projected attention-trajectory

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1087 --substrate experiments/sub1087_prosecution_v21.py

FAMILY: Attention-trajectory (prosecution iteration)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: Alpha-projected attention in low-D (16D) sharpens state similarity
for action retrieval, improving action efficiency. ℓ_π²: alpha determines both
what "change" matters AND what "similar state" means. The substrate learns its
own attention dimensionality.

CHANGES FROM v20 (Step 1085):
1. Top-K dim selection: after ~200 steps, select top_k dims by alpha magnitude
2. Projected attention: q/K in top_k-D instead of 256D → sharper similarity
3. Adaptive K: starts at 64 (conservative), narrows to 16 as alpha concentrates
4. Directional delta: signed, not magnitude. Pick action with highest |predicted direction|.

KILL: ARC efficiency < v20 (0.0037) → projection hurts.
SUCCESS: ARC > 0.05 on any solved game (15x improvement).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
BUFFER_MAX = 2000
EPSILON = 0.20
PRED_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0
TOP_K_MAX = 64    # initial broad projection
TOP_K_MIN = 16    # sharp projection after alpha concentrates
WARMUP_STEPS = 200  # steps before projection kicks in


def _obs_to_enc(obs):
    """avgpool16 + center: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    enc -= enc.mean()
    return enc


class AlphaProjectedAttentionSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = None

        # Trajectory buffer: parallel arrays
        self._buf_enc = np.zeros((BUFFER_MAX, N_DIMS), dtype=np.float32)
        self._buf_action = np.zeros(BUFFER_MAX, dtype=np.int32)
        self._buf_delta = np.zeros(BUFFER_MAX, dtype=np.float32)  # signed delta magnitude
        self._buf_size = 0

        # Alpha (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS

        # Top-K projection indices
        self._top_k = TOP_K_MAX
        self._top_k_idx = np.arange(N_DIMS)  # all dims initially

        # W_pred
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _update_alpha(self, pred_error):
        """Update alpha from prediction error."""
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

    def _update_projection(self):
        """Update top-K projection based on alpha concentration."""
        # Adaptive K: measure alpha concentration via entropy
        alpha_norm = self._alpha / (self._alpha.sum() + 1e-8)
        entropy = -np.sum(alpha_norm * np.log(alpha_norm + 1e-8))
        max_entropy = np.log(N_DIMS)
        # concentration: 0 = uniform, 1 = fully concentrated
        concentration = 1.0 - (entropy / max_entropy)
        # Map concentration to top_k: high concentration → fewer dims
        self._top_k = int(TOP_K_MAX - concentration * (TOP_K_MAX - TOP_K_MIN))
        self._top_k = max(TOP_K_MIN, min(TOP_K_MAX, self._top_k))
        # Select top-K dims by alpha magnitude
        self._top_k_idx = np.argsort(self._alpha)[-self._top_k:]

    def _update_w_pred(self, enc, prev_enc):
        """Update prediction matrix. Returns error."""
        pred = self._w_pred @ prev_enc
        error = enc - pred
        norm_sq = np.dot(prev_enc, prev_enc) + 1e-8
        self._w_pred += PRED_LR * np.outer(error, prev_enc) / norm_sq
        return error

    def _add_to_buffer(self, enc_w, action, delta):
        """Add to trajectory buffer. Drop least informative if full."""
        if self._buf_size < BUFFER_MAX:
            idx = self._buf_size
            self._buf_size += 1
        else:
            idx = int(np.argmin(np.abs(self._buf_delta[:self._buf_size])))
            if abs(delta) <= abs(self._buf_delta[idx]):
                return
        self._buf_enc[idx] = enc_w
        self._buf_action[idx] = action
        self._buf_delta[idx] = delta  # signed!

    def _select_action(self, enc_w):
        """Alpha-projected attention-based action selection."""
        if self._buf_size < self._n_actions * 3:
            return self._rng.randint(self._n_actions)

        if self._rng.random() < EPSILON:
            return self._rng.randint(self._n_actions)

        # Project query to top-K dims
        proj = self._top_k_idx
        q = enc_w[proj]  # (top_k,)
        attn_temp = np.sqrt(float(len(proj)))

        scores = np.zeros(self._n_actions, dtype=np.float32)
        valid = 0

        for a in range(self._n_actions):
            mask = self._buf_action[:self._buf_size] == a
            n_entries = int(mask.sum())
            if n_entries < 2:
                continue
            valid += 1

            # Project keys to top-K dims
            K_a = self._buf_enc[:self._buf_size][mask][:, proj]  # (n, top_k)
            V_a = self._buf_delta[:self._buf_size][mask]  # (n,) signed

            # Scaled dot-product attention
            logits = K_a @ q / attn_temp
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum() + 1e-8

            # Score = weighted mean of |signed delta| (direction-agnostic magnitude)
            scores[a] = float(weights @ np.abs(V_a))

        if valid < 2:
            return self._rng.randint(self._n_actions)

        return int(np.argmax(scores))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        enc_w = enc * self._alpha

        if self._prev_enc is not None and self._prev_action is not None:
            prev_enc_w = self._prev_enc * self._alpha
            # Signed delta: positive = overall change magnitude
            delta_vec = enc_w - prev_enc_w
            delta = float(np.sum(delta_vec))  # signed sum

            self._add_to_buffer(prev_enc_w, self._prev_action, delta)

            # Update W_pred and alpha
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

            # Update projection periodically after warmup
            if self.step_count > WARMUP_STEPS and self.step_count % 50 == 0:
                self._update_projection()

        action = self._select_action(enc_w)

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        # Keep buffer, alpha, projection across levels
        self._prev_enc = None
        self._prev_action = None


CONFIG = {
    "buffer_max": BUFFER_MAX,
    "epsilon": EPSILON,
    "top_k_range": f"{TOP_K_MIN}-{TOP_K_MAX}",
    "warmup": WARMUP_STEPS,
    "pred_lr": PRED_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "family": "attention-trajectory (projected)",
    "tag": "prosecution v21 (ℓ_π² alpha-projected attention, adaptive-K)",
}

SUBSTRATE_CLASS = AlphaProjectedAttentionSubstrate
