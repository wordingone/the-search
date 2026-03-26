"""
sub1085_prosecution_v20.py — Attention-over-trajectory action retrieval

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1085 --substrate experiments/sub1085_prosecution_v20.py

FAMILY: Attention-trajectory (prosecution-only architecture)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: State-conditioned action retrieval via attention over trajectory
buffer provides state-LOCAL action values. Alpha-weighted encoding determines
"similar state" — self-modifies from prediction error. The similarity metric
is learned, not fixed → R3.

WHY STATE-LOCAL MATTERS (Step 1083 failure analysis):
- Global EMA averages out oscillation: mean_progress[action_A] ≈ 0 because
  it's positive in state X and negative in state Y.
- Need: "from THIS state, which action worked?" not "globally, which action works?"
- Attention over trajectory: retrieve past experiences from SIMILAR states,
  weighted by similarity. State-local by construction.

ARCHITECTURE:
- enc = avgpool16 + centered (256D)
- Alpha from W_pred error (softmax concentration). enc_weighted = alpha * enc.
- Trajectory buffer (max 2000 entries): (enc_weighted, action, delta_magnitude)
- Action selection via attention:
  - q = current enc_weighted
  - For each action a: find buffer entries where action == a
  - Compute attention weights: softmax(q @ K_a^T / sqrt(256))
  - Score[a] = weighted mean of delta values from similar states
  - action = argmax(score) with epsilon exploration

KILL: 0/3 ARC games AND no oscillation improvement.
SUCCESS: any responsive game L1 > 0 OR sign changes < 30/74.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4  # avgpool16: 64/4 = 16 blocks
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
BUFFER_MAX = 2000
EPSILON = 0.20
PRED_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0
ATTN_TEMP = np.sqrt(256.0)  # sqrt(d) for scaled dot product


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


class AttentionTrajectorySubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc_w = None  # previous alpha-weighted encoding
        self._prev_action = None

        # Trajectory buffer: parallel arrays for speed
        self._buf_enc = np.zeros((BUFFER_MAX, N_DIMS), dtype=np.float32)
        self._buf_action = np.zeros(BUFFER_MAX, dtype=np.int32)
        self._buf_delta = np.zeros(BUFFER_MAX, dtype=np.float32)
        self._buf_size = 0

        # Alpha: attention weights over encoding dims (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS

        # W_pred: predict next enc from current
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _alpha_weight(self, enc):
        """Apply alpha attention weighting."""
        return enc * self._alpha

    def _update_alpha(self, pred_error):
        """Update alpha from prediction error. ℓ_π component."""
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
        """Update prediction matrix. Returns prediction error."""
        pred = self._w_pred @ prev_enc
        error = enc - pred
        norm_sq = np.dot(prev_enc, prev_enc) + 1e-8
        self._w_pred += PRED_LR * np.outer(error, prev_enc) / norm_sq
        return error

    def _add_to_buffer(self, enc_w, action, delta):
        """Add entry to trajectory buffer. Drop least informative if full."""
        if self._buf_size < BUFFER_MAX:
            idx = self._buf_size
            self._buf_size += 1
        else:
            # Drop entry with lowest delta (least informative moment)
            idx = int(np.argmin(self._buf_delta[:self._buf_size]))
            if delta <= self._buf_delta[idx]:
                return  # new entry is less informative than worst in buffer
        self._buf_enc[idx] = enc_w
        self._buf_action[idx] = action
        self._buf_delta[idx] = delta

    def _select_action(self, enc_w):
        """Attention-based action selection over trajectory buffer."""
        if self._buf_size < self._n_actions * 3:
            # Not enough data — explore
            return self._rng.randint(self._n_actions)

        if self._rng.random() < EPSILON:
            return self._rng.randint(self._n_actions)

        q = enc_w  # query: current state (256D)
        scores = np.zeros(self._n_actions, dtype=np.float32)
        valid_actions = 0

        for a in range(self._n_actions):
            # Find buffer entries for action a
            mask = self._buf_action[:self._buf_size] == a
            n_entries = mask.sum()
            if n_entries < 2:
                scores[a] = 0.0
                continue
            valid_actions += 1

            # Keys: encoded states where action a was taken
            K_a = self._buf_enc[:self._buf_size][mask]  # (n_entries, 256)
            V_a = self._buf_delta[:self._buf_size][mask]  # (n_entries,)

            # Scaled dot-product attention
            attn_logits = K_a @ q / ATTN_TEMP  # (n_entries,)
            attn_logits -= attn_logits.max()  # numerical stability
            attn_weights = np.exp(attn_logits)
            attn_weights /= attn_weights.sum() + 1e-8

            # Weighted delta: expected effect of action a from similar states
            scores[a] = float(attn_weights @ V_a)

        if valid_actions < 2:
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
        enc_w = self._alpha_weight(enc)

        # Update from previous action
        if self._prev_enc_w is not None and self._prev_action is not None:
            # Delta: magnitude of change in alpha-weighted space
            delta = float(np.sqrt(np.sum((enc_w - self._prev_enc_w) ** 2)))

            # Add previous (state, action, effect) to trajectory buffer
            self._add_to_buffer(self._prev_enc_w, self._prev_action, delta)

            # Update W_pred and alpha (ℓ_π self-modification)
            pred_error = self._update_w_pred(enc, _obs_to_enc(obs))
            self._update_alpha(pred_error)

        # Select action via attention over trajectory
        action = self._select_action(enc_w)

        self._prev_enc_w = enc_w.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        # Keep trajectory buffer and alpha across levels
        # Only reset per-episode tracking
        self._prev_enc_w = None
        self._prev_action = None


CONFIG = {
    "buffer_max": BUFFER_MAX,
    "epsilon": EPSILON,
    "pred_lr": PRED_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "n_dims": N_DIMS,
    "family": "attention-trajectory",
    "tag": "prosecution v20 (ℓ_π attention-over-trajectory, state-local values)",
}

SUBSTRATE_CLASS = AttentionTrajectorySubstrate
