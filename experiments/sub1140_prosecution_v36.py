"""
sub1140_prosecution_v36.py — Alpha-weighted novelty seeking (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1140 --substrate experiments/sub1140_prosecution_v36.py

FAMILY: Novelty-seeking exploration (NEW). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: Defense v51 measures novelty in FIXED encoding space.
Prosecution v36 measures novelty in ALPHA-WEIGHTED space where alpha
is the prediction error per dimension. Alpha compresses 256D to the
informative subset — novelty is measured in MEANING space, not pixel space.

CONTROLLED COMPARISON:
- Defense v51: hash(enc) → visit counts → novelty signal
- Prosecution v36: hash(alpha * enc) → visit counts → novelty signal
- ONLY DIFFERENCE: alpha weighting before hash

If alpha-weighted novelty outperforms raw novelty → encoding modification
adds value for exploration (prosecution wins on novelty axis).
If raw novelty matches → ℓ₁ is sufficient for novelty (defense wins).

Architecture:
- enc = avgpool4 (256D)
- W_pred (256x256): predicts enc from prev_enc. Updated with EMA.
- alpha = |enc - W_pred @ prev_enc| / (|enc| + eps) — prediction error per dim
- alpha_enc = alpha * enc — reweight encoding by informativeness
- LSH: 8 random hyperplanes (FROZEN) → 8-bit hash → 256 buckets
  Applied to alpha_enc, not enc
- Visit count per bucket: counts[bucket] += 1
- Novelty signal: novelty = 1.0 / (counts[bucket] + 1)
- Same reactive switching as defense v51

KILL: ARC ≤ defense v51 AND ≤ v30.
SUCCESS: alpha-weighted novelty outperforms raw novelty.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
N_LSH_BITS = 8
EXPLORE_STEPS = 50
MAX_PATIENCE = 20
PRED_LR = 0.01


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


def _block_to_click_action(block_idx):
    """Block center -> click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class AlphaNoveltySubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0

        # FROZEN random hyperplanes for LSH (R1 compliant)
        self._hyperplanes = self._rng.randn(N_LSH_BITS, N_DIMS).astype(np.float32)

        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_novelty = 1.0

        # Prediction model (ℓ_π component)
        self._W_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99
        self._alpha = np.ones(N_DIMS, dtype=np.float32)

        # Visit counts per LSH bucket (in alpha-weighted space)
        self._visit_counts = {}

        # Click regions
        self._click_regions = []
        self._click_actions = []
        self._n_active = N_KB
        self._regions_set = False

        # Reactive switching
        self._current_action_idx = 0
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

    def _update_prediction(self, prev_enc, enc):
        """Update W_pred and compute alpha (prediction error per dim)."""
        pred = self._W_pred @ prev_enc
        error = enc - pred
        norm_sq = np.dot(prev_enc, prev_enc) + 1e-8
        self._W_pred += PRED_LR * np.outer(error, prev_enc) / norm_sq

        # Alpha = normalized prediction error per dimension
        enc_norm = np.abs(enc) + 1e-8
        self._alpha = np.abs(error) / enc_norm
        # Normalize alpha to [0, 1] range
        alpha_max = self._alpha.max() + 1e-8
        self._alpha = self._alpha / alpha_max

        self.r3_updates += 1
        self.att_updates_total += 1

    def _lsh_hash(self, weighted_enc):
        """LSH: 8 random hyperplanes -> 8-bit hash -> bucket index."""
        bits = (self._hyperplanes @ weighted_enc > 0).astype(np.int32)
        bucket = 0
        for i in range(N_LSH_BITS):
            bucket |= (bits[i] << i)
        return bucket

    def _get_novelty(self, enc):
        """Novelty in alpha-weighted space."""
        alpha_enc = self._alpha * enc
        bucket = self._lsh_hash(alpha_enc)
        count = self._visit_counts.get(bucket, 0)
        self._visit_counts[bucket] = count + 1
        return 1.0 / (count + 1)

    def _discover_regions(self, enc):
        """Find top-16 salient blocks for click targeting."""
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        self._click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in self._click_regions]
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(self._n_actions_env, N_KB)
        self._regions_set = True

    def _action_idx_to_env_action(self, idx):
        """Convert internal index to PRISM action."""
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

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
            self._discover_regions(enc)
            self._prev_novelty = self._get_novelty(enc)
            self._current_action_idx = self._rng.randint(min(self._n_active, N_KB))
            return self._action_idx_to_env_action(self._current_action_idx)

        # Update prediction model and alpha
        if self._prev_enc is not None:
            self._update_prediction(self._prev_enc, enc)

        # Compute novelty in alpha-weighted space
        novelty = self._get_novelty(enc)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action_idx = self.step_count % self._n_active
            self._current_action_idx = action_idx
            self._prev_enc = enc.copy()
            self._prev_novelty = novelty
            return self._action_idx_to_env_action(action_idx)

        # Reactive switching with NOVELTY as progress signal
        progress = novelty > self._prev_novelty + 1e-6
        no_change = abs(novelty - self._prev_novelty) < 1e-8

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
                if self._actions_tried_this_round >= self._n_active:
                    self._current_action_idx = self._rng.randint(self._n_active)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action_idx = (self._current_action_idx + 1) % self._n_active

        self._prev_enc = enc.copy()
        self._prev_novelty = novelty
        return self._action_idx_to_env_action(self._current_action_idx)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_novelty = 1.0
        self._current_action_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep W_pred, alpha, visit counts, hyperplanes across levels (ℓ_π)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_lsh_bits": N_LSH_BITS,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "pred_lr": PRED_LR,
    "family": "novelty-seeking exploration",
    "tag": "prosecution v36 (ℓ_π alpha-weighted LSH novelty: W_pred → alpha → hash(alpha*enc) → visit counts)",
}

SUBSTRATE_CLASS = AlphaNoveltySubstrate
