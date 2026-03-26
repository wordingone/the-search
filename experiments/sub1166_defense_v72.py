"""
sub1166_defense_v72.py — Modal goal reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1166 --substrate experiments/sub1166_defense_v72.py

FAMILY: Modal goal reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v30 uses the INITIAL observation as the goal (progress =
return to start). But what if the goal state is NOT the initial state?
Many games have goal states that are ATTRACTORS — the game naturally
spends more time near the goal than away from it.

If the goal is an attractor, the MOST COMMON observation over time is
the best estimate of the goal. The MODE of the observation distribution
(per block) converges to the goal state.

This substrate tracks per-block value frequencies and uses the modal
(most common) observation as the goal target. Progress = decreasing
distance to the modal observation. Reactive cycling switches actions
when distance-to-mode stops decreasing.

CONTROLLED COMPARISON vs v30:
- SAME: reactive cycling, zero params, same encoding
- DIFFERENT: v30 goal = initial obs (fixed). v72 goal = modal obs (evolving).

ZERO learned parameters (defense: ℓ₁). Frequency histogram = counter, not weight.

KILL: ARC ≤ v30.
SUCCESS: Modal goal outperforms initial-goal on games where start ≠ target.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
N_BINS = 16  # discretize block values into bins
MODE_UPDATE_INTERVAL = 50  # recompute modal goal every N steps


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class ModalGoalReactiveSubstrate:
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
        self._prev_dist = float('inf')

        self._n_active = N_KB

        # Per-block frequency histogram
        self._block_hist = np.zeros((N_DIMS, N_BINS), dtype=np.float32)
        self._bin_edges = None  # set from first observation
        self._modal_goal = None  # modal observation (estimated goal)

        # Reactive state
        self._current_action = 0
        self._patience = 0

        # Click regions
        self._click_actions = []
        self._regions_set = False

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _discover_regions(self, enc):
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in click_regions]
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(self._n_actions_env, N_KB)
        self._regions_set = True

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _update_histogram(self, enc):
        """Update per-block frequency histogram."""
        for d in range(N_DIMS):
            # Find which bin this value falls into
            val = enc[d]
            bin_idx = np.searchsorted(self._bin_edges[d], val, side='right') - 1
            bin_idx = max(0, min(N_BINS - 1, bin_idx))
            self._block_hist[d, bin_idx] += 1

    def _compute_modal_goal(self):
        """Compute modal (most frequent) value per block."""
        modal = np.zeros(N_DIMS, dtype=np.float32)
        for d in range(N_DIMS):
            best_bin = int(np.argmax(self._block_hist[d]))
            # Modal value = center of the most frequent bin
            modal[d] = (self._bin_edges[d][best_bin] + self._bin_edges[d][best_bin + 1]) / 2.0
        self._modal_goal = modal

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
            # Initialize bin edges from first observation (range per dim)
            self._bin_edges = np.zeros((N_DIMS, N_BINS + 1), dtype=np.float32)
            for d in range(N_DIMS):
                # Use fixed range 0-15 (pixel values) or adaptive
                self._bin_edges[d] = np.linspace(0, 15, N_BINS + 1)
            self._update_histogram(enc)
            self._modal_goal = enc.copy()  # initial goal = initial obs
            return 0

        # Update histogram
        self._update_histogram(enc)

        # Periodically recompute modal goal
        if self.step_count % MODE_UPDATE_INTERVAL == 0:
            self._compute_modal_goal()

        # Distance to modal goal (evolving target)
        goal = self._modal_goal if self._modal_goal is not None else self._enc_0
        dist = np.sum(np.abs(enc - goal))

        # Reactive cycling based on distance-to-mode
        if dist >= self._prev_dist:
            self._current_action = (self._current_action + 1) % self._n_active
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._current_action = (self._current_action + 1) % self._n_active

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        return self._idx_to_env_action(self._current_action)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_action = 0
        self._patience = 0
        # Reset histogram for new level
        self._block_hist[:] = 0
        self._modal_goal = None


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "n_bins": N_BINS,
    "mode_update_interval": MODE_UPDATE_INTERVAL,
    "family": "modal goal reactive",
    "tag": "defense v72 (ℓ₁ modal goal: goal = most common observation per block. Tests whether goal state is an attractor revealed by frequency counting.)",
}

SUBSTRATE_CLASS = ModalGoalReactiveSubstrate
