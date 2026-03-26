"""
sub1138_prosecution_v35.py — Spatial click region discovery + forward model (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1138 --substrate experiments/sub1138_prosecution_v35.py

FAMILY: Click-region forward model (NEW). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: Instead of modeling 4096 individual click targets, discover
~16 interactive REGIONS from initial observation saliency, then build
per-region forward models. The forward model DISCOVERS which actions
(keyboard or click) produce change — auto-detecting game type.

Architecture:
- enc = avgpool4 (256D)
- Region discovery: top-16 salient blocks from initial observation
  sal[i] = |block_mean[i] - screen_mean|
- Per-action forward model: W_fwd[a] = EMA of encoding change
  W_fwd[a] = 0.95 * W_fwd[a] + 0.05 * (enc_new - enc_old) when action a taken
  23 actions total: 7 keyboard + 16 click regions
- Action selection: argmax(||W_fwd[a]||) with reactive switching

WHY ℓ_π: W_fwd learns per-action effects (encoding modification from interaction).
Zero-param defense can't do this.

KILL: ARC ≤ v30 (0.3319) on keyboard AND ARC ≤ v35_click (0.015) on click.
SUCCESS: ARC > 0 on a click-only game from a NEW draw.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
N_TOTAL_ACTIONS = N_KB + N_CLICK_REGIONS  # 23
WARMUP_STEPS = 100
EMA_ALPHA = 0.05
MAX_PATIENCE = 20


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
    """Block center -> click action index in PRISM action space."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class SpatialClickForwardSubstrate:
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

        # Click region indices (set from initial observation saliency)
        self._click_regions = []  # block indices of salient regions
        self._click_actions = []  # PRISM action indices for click regions
        self._regions_set = False

        # Per-action forward model: EMA of encoding change
        self._n_active = N_KB  # updated when regions discovered
        self._w_fwd = [np.zeros(N_DIMS, dtype=np.float32) for _ in range(N_TOTAL_ACTIONS)]
        self._action_counts = np.zeros(N_TOTAL_ACTIONS, dtype=np.int32)

        # Reactive switching
        self._current_action_idx = 0  # index into active action list
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

    def _discover_regions(self, enc):
        """Find top-16 salient blocks from initial encoding."""
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
        """Convert internal action index to PRISM action."""
        if idx < N_KB:
            return idx  # keyboard action
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _update_forward_model(self, action_idx, enc_old, enc_new):
        """Update EMA of encoding change for this action."""
        if action_idx >= N_TOTAL_ACTIONS:
            return
        delta = enc_new - enc_old
        self._w_fwd[action_idx] = (1 - EMA_ALPHA) * self._w_fwd[action_idx] + EMA_ALPHA * delta
        self._action_counts[action_idx] += 1
        self.r3_updates += 1
        self.att_updates_total += 1

    def _predict_effect_magnitude(self):
        """Return predicted effect magnitude for each active action."""
        magnitudes = np.zeros(self._n_active, dtype=np.float32)
        for i in range(self._n_active):
            magnitudes[i] = float(np.sum(np.abs(self._w_fwd[i])))
        return magnitudes

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
            self._discover_regions(enc)
            self._current_action_idx = self._rng.randint(min(self._n_active, N_KB))
            return self._action_idx_to_env_action(self._current_action_idx)

        dist = self._dist_to_initial(enc)

        # Update forward model with observed transition
        if self._prev_enc is not None:
            self._update_forward_model(self._current_action_idx, self._prev_enc, enc)

        # === Warmup: cycle through all active actions ===
        if self.step_count <= WARMUP_STEPS:
            action_idx = self.step_count % self._n_active
            self._current_action_idx = action_idx
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._action_idx_to_env_action(action_idx)

        # === Post-warmup: forward-model-guided reactive switching ===
        magnitudes = self._predict_effect_magnitude()

        progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
        no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

        self._steps_on_action += 1

        if progress:
            # Current action making progress — keep it
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0
            if self._steps_on_action >= self._patience or no_change:
                self._steps_on_action = 0
                self._patience = 3
                self._actions_tried_this_round += 1

                if self._actions_tried_this_round >= self._n_active:
                    # All tried — pick predicted best
                    self._current_action_idx = int(np.argmax(magnitudes))
                    self._actions_tried_this_round = 0
                else:
                    # Try next by predicted magnitude (descending)
                    sorted_actions = np.argsort(magnitudes)[::-1]
                    idx = self._actions_tried_this_round % self._n_active
                    self._current_action_idx = int(sorted_actions[idx])

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return self._action_idx_to_env_action(self._current_action_idx)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep W_fwd and regions across levels (ℓ_π)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "warmup_steps": WARMUP_STEPS,
    "ema_alpha": EMA_ALPHA,
    "max_patience": MAX_PATIENCE,
    "family": "click-region forward model",
    "tag": "prosecution v35 (ℓ_π spatial click region discovery + EMA forward model, 16 salient regions + 7 kb = 23 actions)",
}

SUBSTRATE_CLASS = SpatialClickForwardSubstrate
