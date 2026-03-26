"""
sub1144_defense_v54.py — Raw pixel change detection + reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1144 --substrate experiments/sub1144_defense_v54.py

FAMILY: Raw pixel encoding (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: avgpool4 averages 4x4=16 pixels per block. If a game changes
1-2 pixels per action, the change is diluted 8-16x by averaging. Step 1141
pixel scan used avgpool4 to DETECT changes — if changes were sub-block,
the scan would report "no responsive pixels" even if pixels DID change.

This substrate uses the FULL 4096-dim raw pixel vector (64x64 flattened).
If raw pixels detect changes that avgpool4 misses → encoding resolution
was the bottleneck all along.

Architecture:
- enc = obs.ravel() (4096D raw pixels, no averaging)
- Distance-to-initial: ℓ₁ on full 4096D vector
- Reactive switching: same as v30 but on raw pixels
- Click regions: top-16 blocks by raw pixel variance (not avgpool4 saliency)
- ZERO learned parameters (defense: ℓ₁)

CONTROLLED COMPARISON: vs v30 (avgpool4 256D, same switching).
ONLY DIFFERENCE: encoding dimensionality (4096 raw vs 256 averaged).

KILL: ARC ≤ v30 (0.3319).
SUCCESS: Raw pixel detection breaks 0% wall.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_KB = 7
N_CLICK_REGIONS = 16
EXPLORE_STEPS = 50
MAX_PATIENCE = 20
RAW_DIM = 64 * 64  # 4096


def _block_to_click_action(block_idx):
    """Block center -> click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class RawPixelSubstrate:
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

    def _discover_regions(self, obs):
        """Find top-16 blocks by pixel variance for click targeting."""
        variances = np.zeros(N_BLOCKS * N_BLOCKS, dtype=np.float32)
        for by in range(N_BLOCKS):
            for bx in range(N_BLOCKS):
                y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
                x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
                variances[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].var()
        sorted_blocks = np.argsort(variances)[::-1]
        self._click_actions = [_block_to_click_action(int(b)) for b in sorted_blocks[:N_CLICK_REGIONS]]
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
        enc = obs.ravel()  # 4096D raw pixels

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._discover_regions(obs)
            self._current_action_idx = self._rng.randint(min(self._n_active, N_KB))
            return self._action_idx_to_env_action(self._current_action_idx)

        # Distance to initial on RAW 4096D pixels
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
        self._regions_set = False


CONFIG = {
    "raw_dim": RAW_DIM,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "raw pixel encoding",
    "tag": "defense v54 (ℓ₁ raw 4096D pixels: no averaging, full resolution change detection)",
}

SUBSTRATE_CLASS = RawPixelSubstrate
