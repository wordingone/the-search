"""
sub1160_defense_v66.py — Sustained-hold reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1160 --substrate experiments/sub1160_defense_v66.py

FAMILY: Sustained-hold reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v30 switches actions EVERY STEP on non-improvement. But some
games may require HOLDING an action for multiple frames before the effect
appears (e.g., button must be pressed for N frames, or click-and-drag needs
sustained contact). v30 would never discover these because it switches away
after 1 frame of no improvement.

RETHINK INSIGHT: prosecution v16 uses SUSTAIN_STEPS=15 (holds each action
for 15 steps during probing). No defense substrate has ever tested sustained
holds. This is a different interaction modality, not a different detection
method.

This substrate tests SUSTAINED HOLDS:
- When trying a new action, hold it for HOLD_DURATION steps before evaluating
- Track total distance change over the full hold period
- If hold produced improvement (dist decreased): keep action, reset hold
- If no improvement after full hold: switch to next action

CONTROLLED COMPARISON vs v30:
- SAME: reactive distance-to-initial, action cycling, zero params
- DIFFERENT: v30 evaluates every step. v66 evaluates every HOLD_DURATION steps.

ZERO learned parameters (defense: ℓ₁). Fixed hold protocol.

KILL: ARC ≤ v30.
SUCCESS: Sustained holds > rapid switching (some games need prolonged interaction).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
SCAN_END = 100
HOLD_DURATION = 10  # hold each action for 10 steps before evaluating


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class SustainedHoldReactiveSubstrate:
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

        # Sustained hold state
        self._current_action = 0
        self._hold_counter = 0       # steps held on current action
        self._hold_start_dist = float('inf')  # dist when hold began

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
            return 0

        dist = np.sum(np.abs(enc - self._enc_0))

        # Phase 1: scan all actions (one step each, quick survey)
        if self.step_count <= SCAN_END:
            action = (self.step_count - 1) % self._n_active
            self._prev_dist = dist
            self._prev_enc = enc.copy()
            return self._idx_to_env_action(action)

        # Phase 2: sustained-hold reactive
        self._hold_counter += 1

        # Initialize hold tracking on first exploit step
        if self._hold_counter == 1:
            self._hold_start_dist = dist

        # Evaluate after full hold period
        if self._hold_counter >= HOLD_DURATION:
            # Did this hold produce improvement?
            if dist < self._hold_start_dist:
                # Yes — keep this action, start new hold
                self._hold_counter = 0
                self._hold_start_dist = dist
            else:
                # No improvement — switch to next action
                self._current_action = (self._current_action + 1) % self._n_active
                self._hold_counter = 0
                self._hold_start_dist = dist

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        return self._idx_to_env_action(self._current_action)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_action = 0
        self._hold_counter = 0
        self._hold_start_dist = float('inf')


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "scan_end": SCAN_END,
    "hold_duration": HOLD_DURATION,
    "family": "sustained-hold reactive",
    "tag": "defense v66 (ℓ₁ sustained-hold: hold each action for 10 steps before evaluating. Tests whether rapid switching misses delayed-effect games.)",
}

SUBSTRATE_CLASS = SustainedHoldReactiveSubstrate
