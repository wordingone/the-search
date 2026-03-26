"""
sub1142_defense_v52.py — Sobel edge encoding + reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1142 --substrate experiments/sub1142_defense_v52.py

FAMILY: Edge-based encoding (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: All 40 prior experiments used avgpool4 encoding (block-mean
brightness). avgpool4 is BLIND to edges, boundaries, and shapes — it only
sees average luminance per 4x4 block. If 0% games have interactive elements
with distinct edges (buttons, borders, tiles), avgpool4 cannot perceive them.

Sobel edge detection captures GRADIENT MAGNITUDE — the strength of edges at
each location. This is fundamentally different information than average brightness.
If edge encoding breaks the 0% wall → avgpool4 was the perception bottleneck.
If same wall → the bottleneck is below encoding resolution entirely.

Architecture:
- enc = sobel_edge_magnitude per 4x4 block (256D)
  - Sobel Gx = [[-1,0,1],[-2,0,2],[-1,0,1]], Gy = Gx.T
  - gradient_magnitude = sqrt(Gx² + Gy²) per pixel
  - block mean of gradient magnitudes → 256D
- Same reactive switching as v30 (distance-to-initial, patience, action cycling)
- Same saliency-based click regions (16 most salient blocks)
- ZERO learned parameters (defense: ℓ₁)

CONTROLLED COMPARISON: vs v30 (avgpool4 encoding, same switching).
ONLY DIFFERENCE: encoding function (sobel edge vs average pool).

KILL: ARC ≤ v30 (0.3319).
SUCCESS: Edge encoding breaks 0% wall on games where avgpool4 fails.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
EXPLORE_STEPS = 50
MAX_PATIENCE = 20

# Sobel kernels (3x3)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _sobel_edge_enc(obs):
    """Sobel edge magnitude, block-averaged: 64x64 -> 16x16 = 256D.
    Vectorized — no Python loops for Sobel convolution."""
    # Vectorized Sobel: Gx and Gy via array slicing
    gx = (obs[:-2, 2:] - obs[:-2, :-2]
          + 2.0 * (obs[1:-1, 2:] - obs[1:-1, :-2])
          + obs[2:, 2:] - obs[2:, :-2])
    gy = (obs[2:, :-2] - obs[:-2, :-2]
          + 2.0 * (obs[2:, 1:-1] - obs[:-2, 1:-1])
          + obs[2:, 2:] - obs[:-2, 2:])

    # Gradient magnitude (62x62, padded back to 64x64)
    grad_mag = np.zeros((64, 64), dtype=np.float32)
    grad_mag[1:63, 1:63] = np.sqrt(gx * gx + gy * gy)

    # Block-average: reshape to (16, 4, 16, 4) then mean over block dims
    enc = grad_mag.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel()
    return enc.astype(np.float32)


def _block_to_click_action(block_idx):
    """Block center -> click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class SobelEdgeSubstrate:
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
        self._prev_dist = 0.0

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
        enc = _sobel_edge_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._discover_regions(enc)
            self._current_action_idx = self._rng.randint(min(self._n_active, N_KB))
            return self._action_idx_to_env_action(self._current_action_idx)

        # Distance to initial (same goal as v30)
        dist = np.sum(np.abs(enc - self._enc_0))

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action_idx = self.step_count % self._n_active
            self._current_action_idx = action_idx
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._action_idx_to_env_action(action_idx)

        # Reactive switching (same as v30)
        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(dist - self._prev_dist) < 1e-8

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
        self._prev_dist = dist
        return self._action_idx_to_env_action(self._current_action_idx)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = 0.0
        self._current_action_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Regions reset on level transition (new visual layout)
        self._regions_set = False


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "encoding": "sobel_edge_magnitude_block_mean",
    "family": "edge-based encoding",
    "tag": "defense v52 (ℓ₁ Sobel edge encoding: gradient magnitude per block, same reactive switching as v30)",
}

SUBSTRATE_CLASS = SobelEdgeSubstrate
