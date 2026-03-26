"""
sub1139_defense_v51.py — Novelty-seeking with LSH visit counts (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1139 --substrate experiments/sub1139_defense_v51.py

FAMILY: Novelty-seeking exploration (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: All 35 prior defense experiments used distance-to-initial as
the progress signal. Novelty seeking uses a fundamentally different signal:
visit counts in observation hash space. Go where you haven't been.

The debate reframe: it's not ℓ₁ vs ℓ_π for goal function — it's WHETHER
learned encoding improves novelty-based exploration. This is the defense
control: novelty in FIXED encoding space.

Architecture:
- enc = avgpool4 (256D)
- LSH: 8 random hyperplanes (FROZEN) → 8-bit hash → 256 buckets
  Locality-preserving: similar observations → same bucket
- Visit count per bucket: counts[bucket] += 1
- Novelty signal: novelty = 1.0 / (counts[bucket] + 1)
- Progress = current novelty > previous novelty (visiting rarer states)
- Same reactive switching as v30 but with novelty as progress signal
- Actions: keyboard (7) + salient click regions (16) = 23 total (for click games)

KILL: ARC ≤ v30 (0.3319).
SUCCESS: novelty seeking breaks 0% wall on games where distance-to-initial fails.
CONTROLLED COMPARISON: vs prosecution v36 (alpha-weighted novelty).
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


class NoveltySeekingSubstrate:
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

        # Visit counts per LSH bucket
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

    def _lsh_hash(self, enc):
        """LSH: 8 random hyperplanes -> 8-bit hash -> bucket index."""
        bits = (self._hyperplanes @ enc > 0).astype(np.int32)
        bucket = 0
        for i in range(N_LSH_BITS):
            bucket |= (bits[i] << i)
        return bucket

    def _get_novelty(self, enc):
        """Novelty = 1/(visit_count + 1). Higher for rare states."""
        bucket = self._lsh_hash(enc)
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

        # Compute novelty (visit count based)
        novelty = self._get_novelty(enc)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action_idx = self.step_count % self._n_active
            self._current_action_idx = action_idx
            self._prev_enc = enc.copy()
            self._prev_novelty = novelty
            return self._action_idx_to_env_action(action_idx)

        # Reactive switching with NOVELTY as progress signal
        # "progress" = current state is more novel than previous
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
        # Keep visit counts and hyperplanes across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_lsh_bits": N_LSH_BITS,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "novelty-seeking exploration",
    "tag": "defense v51 (ℓ₁ LSH novelty: 8-bit hash → 256 buckets, visit counts, fixed encoding)",
}

SUBSTRATE_CLASS = NoveltySeekingSubstrate
