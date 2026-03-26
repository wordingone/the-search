"""
sub1083_prosecution_v19.py — State-dependent action-value with progress encoding

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1083 --substrate experiments/sub1083_prosecution_v19.py

FAMILY: State-dependent action selection (new architecture)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: State-conditioned action selection with progress-direction encoding
breaks the oscillation bottleneck. The encoding of "progress" is learned from
interaction (alpha discovers which dims carry signal) — self-modification of
operations, not just storage.

ARCHITECTURE:
- enc = avgpool16 + centered (256D)
- Track obs_0 = initial observation per episode
- Per-action running stats (EMA, decay=0.99):
  - mean_progress[a] = does action a move toward or away from start?
  - mean_delta[a] = 256D vector of typical change when a is taken
- State-dependent selection:
  - predicted_change[a] = mean_delta[a]
  - predicted_progress[a] = ||enc + predicted_change - enc_0|| - ||enc - enc_0||
  - action = argmax(predicted_progress) with epsilon exploration
- ℓ_π: alpha-weighted encoding of deltas. Alpha updates from W_pred error.
  Dims with high prediction error → alpha concentrates → delta tracking in
  alpha-weighted space. The encoding of "what changed" is self-modified.

WHY THIS ADDRESSES OSCILLATION (Step 1082):
- 1082: kb1=+0.37 toward, kb6=-0.50 away. Substrate KNOWS direction per action.
- Old cascade: evolves random sequences, ignores per-action stats → oscillates.
- New: predicted_progress from CURRENT state. If game oscillates A→B→A→B,
  the predicted progress alternates between actions → substrate alternates.

KILL: chain_score < Step 1081 or 0/3 responsive games.
SUCCESS: oscillation reduced (sign changes < 30/74) AND any responsive game shows L1.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4  # avgpool16 equivalent: 64/4 = 16 blocks → but 4x4 = 16 blocks per dim
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
EMA_DECAY = 0.99
EPSILON = 0.20
PRED_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0


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


class StateDepActionValueSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None  # initial observation encoding
        self._prev_enc = None
        self._prev_action = None

        # Per-action running statistics (EMA)
        self._mean_delta = np.zeros((self._n_actions, N_DIMS), dtype=np.float32)
        self._mean_progress = np.zeros(self._n_actions, dtype=np.float32)
        self._action_count = np.zeros(self._n_actions, dtype=np.float32)

        # Alpha: attention weights over encoding dims (ℓ_π component)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS

        # W_pred: predict next enc from current (256×256)
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)  # only keyboard actions for now
        self._init_state()

    def _alpha_weight(self, vec):
        """Apply alpha attention weighting to a vector."""
        return vec * self._alpha

    def _update_alpha(self, pred_error):
        """Update alpha from prediction error magnitude per dim.
        Dims with high prediction error → alpha concentrates there.
        This IS the ℓ_π component: encoding of "what changed" is self-modified."""
        error_mag = np.abs(pred_error)
        # Softmax-like concentration
        logits = ALPHA_CONC * error_mag / (error_mag.max() + 1e-8)
        logits -= logits.max()
        exp_logits = np.exp(logits)
        target_alpha = exp_logits / (exp_logits.sum() + 1e-8)
        self._alpha = (1 - ALPHA_LR) * self._alpha + ALPHA_LR * target_alpha
        self.r3_updates += 1
        self.att_updates_total += 1

    def _update_w_pred(self, enc, prev_enc):
        """Update prediction matrix: W @ prev_enc ≈ enc."""
        pred = self._w_pred @ prev_enc
        error = enc - pred
        # Gradient step: W += lr * error @ prev_enc^T (rank-1 update)
        norm_sq = np.dot(prev_enc, prev_enc) + 1e-8
        self._w_pred += PRED_LR * np.outer(error, prev_enc) / norm_sq
        return error

    def _select_action(self, enc):
        """State-dependent action selection using predicted progress."""
        if self._enc_0 is None:
            return self._rng.randint(self._n_actions)

        dist_current = np.sqrt(np.sum((self._alpha_weight(enc - self._enc_0)) ** 2))

        scores = np.zeros(self._n_actions, dtype=np.float32)
        valid = 0
        for a in range(self._n_actions):
            if self._action_count[a] < 3:
                scores[a] = 0.0  # not enough data
                continue
            valid += 1
            # Predict state after action a
            predicted_enc = enc + self._alpha_weight(self._mean_delta[a])
            dist_after = np.sqrt(np.sum((self._alpha_weight(predicted_enc - self._enc_0)) ** 2))
            # Progress = change in distance from start
            # Negative = moving toward start (which we want for solving)
            scores[a] = -(dist_after - dist_current)

        if valid < 2 or self._rng.random() < EPSILON:
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

        # Store initial encoding
        if self._enc_0 is None:
            self._enc_0 = enc.copy()

        # Update from previous action
        if self._prev_enc is not None and self._prev_action is not None:
            a = self._prev_action
            delta = enc - self._prev_enc
            alpha_delta = self._alpha_weight(delta)

            # Update per-action statistics
            if a < self._n_actions:
                self._action_count[a] += 1
                alpha = 1.0 - EMA_DECAY
                self._mean_delta[a] = EMA_DECAY * self._mean_delta[a] + alpha * alpha_delta

                # Progress: did this action move us toward or away from start?
                dist_before = np.sqrt(np.sum((self._alpha_weight(self._prev_enc - self._enc_0)) ** 2))
                dist_after = np.sqrt(np.sum((self._alpha_weight(enc - self._enc_0)) ** 2))
                progress = dist_before - dist_after  # positive = moved toward start
                self._mean_progress[a] = EMA_DECAY * self._mean_progress[a] + alpha * progress

            # Update W_pred and alpha (ℓ_π: self-modification)
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

        # Select action
        action = self._select_action(enc)

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        # Reset episode-specific state but KEEP learned statistics
        self._enc_0 = None
        self._prev_enc = None
        self._prev_action = None


CONFIG = {
    "ema_decay": EMA_DECAY,
    "epsilon": EPSILON,
    "pred_lr": PRED_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "n_dims": N_DIMS,
    "family": "state-dependent action-value",
    "tag": "prosecution v19 (ℓ_π progress encoding)",
}

SUBSTRATE_CLASS = StateDepActionValueSubstrate
