"""
sub1141_prosecution_v37.py — Systematic pixel scan diagnostic (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1141 --substrate experiments/sub1141_prosecution_v37.py

FAMILY: Pixel scan diagnostic (NEW). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: 40 experiments hit the same 2/5 wall. All used block-center
clicks (16 or 256 points). If 0% games have responsive pixels at non-block-center
positions, the bottleneck is action targeting. The substrate discovers responsive
pixels through systematic scanning and MODIFIES its action policy to exploit them
(Phase 3 = R3 self-modification). If responsive pixels exist and Phase 3 improves
score → the 0% wall was action targeting, not perception.

Architecture:
- Phase 1 (steps 0-50): keyboard scan, 7 actions cycled
- Phase 2 (steps 50-4146): systematic pixel scan, 64x64 = 4096 pixels
  - Row by row: click pixel (0,0), (1,0), ... (63,0), (0,1), ...
  - After each click: compare enc_after vs enc_before (avgpool4 ℓ₁)
  - Record heat_map[pixel_index] = change magnitude
- Phase 3 (steps 4146+): exploit heat map
  - Sort pixels by responsiveness, take top-K responsive
  - Click ONLY responsive pixels with reactive switching
  - If no responsive pixels found → keyboard-only reactive switching

DIAGNOSTIC OUTPUT: The heat map itself is the finding. If ALL entries ≈ 0,
0% games are genuinely inert. If some entries > 0, we know WHERE to click.

KILL: No responsive pixels found AND ARC ≤ v30.
SUCCESS: Responsive pixels found, Phase 3 exploits them, breaks 0% wall.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
SCAN_START = 50
SCAN_PIXELS = 64 * 64  # 4096
SCAN_END = SCAN_START + SCAN_PIXELS  # 4146
TOP_K = 32  # top responsive pixels to exploit
MAX_PATIENCE = 20
RESPONSIVE_THRESH = 0.5  # minimum ℓ₁ change to count as responsive


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class PixelScanSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = None

        # Heat map: responsiveness per pixel
        self._heat_map = np.zeros(SCAN_PIXELS, dtype=np.float32)
        self._scan_complete = False

        # Phase 3: exploit responsive pixels
        self._responsive_pixels = []
        self._responsive_actions = []
        self._n_responsive = 0
        self._exploit_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._prev_dist = 0.0
        self._enc_0 = None

        # Diagnostic counters
        self._total_change = 0.0
        self._max_change = 0.0
        self._nonzero_pixels = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _pixel_to_action(self, pixel_idx):
        """Convert pixel index to click action. pixel_idx = px + py*64."""
        px = pixel_idx % 64
        py = pixel_idx // 64
        return N_KB + px + py * 64

    def _build_exploit_set(self):
        """After scan: find responsive pixels, build exploit action set."""
        self._scan_complete = True

        # Count diagnostics
        self._nonzero_pixels = int(np.sum(self._heat_map > RESPONSIVE_THRESH))
        self._total_change = float(np.sum(self._heat_map))
        self._max_change = float(np.max(self._heat_map))

        # Sort by responsiveness, take top-K
        sorted_indices = np.argsort(self._heat_map)[::-1]
        responsive = []
        for idx in sorted_indices[:TOP_K]:
            if self._heat_map[idx] > RESPONSIVE_THRESH:
                responsive.append(int(idx))

        if responsive and self._has_clicks:
            self._responsive_pixels = responsive
            self._responsive_actions = [self._pixel_to_action(p) for p in responsive]
            self._n_responsive = len(responsive)
        else:
            # No responsive pixels found — keyboard only
            self._responsive_pixels = []
            self._responsive_actions = []
            self._n_responsive = 0

        self.r3_updates += 1
        self.att_updates_total += 1

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        # Record initial encoding
        if self._enc_0 is None:
            self._enc_0 = enc.copy()

        # Record heat map entry from previous action
        if self._prev_enc is not None and self._prev_action is not None:
            change = np.sum(np.abs(enc - self._prev_enc))
            # If we were in scan phase, record the change for that pixel
            scan_step = self.step_count - 1 - SCAN_START
            if 0 <= scan_step < SCAN_PIXELS:
                self._heat_map[scan_step] = change

        # Phase 1: keyboard scan (steps 1-50)
        if self.step_count <= SCAN_START:
            action = (self.step_count - 1) % min(self._n_actions_env, N_KB)
            self._prev_enc = enc.copy()
            self._prev_action = action
            return action

        # Phase 2: systematic pixel scan (steps 51-4146)
        if not self._scan_complete:
            scan_step = self.step_count - SCAN_START - 1  # 0-indexed pixel
            if scan_step < SCAN_PIXELS and self._has_clicks:
                pixel_idx = scan_step
                action = self._pixel_to_action(pixel_idx)
                # Clamp to valid action range
                if action >= self._n_actions_env:
                    action = self._rng.randint(min(self._n_actions_env, N_KB))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action
            else:
                # Scan complete or no clicks — build exploit set
                self._build_exploit_set()

        # Phase 3: exploit responsive pixels
        if self._n_responsive > 0:
            # Reactive switching among responsive pixels
            dist = np.sum(np.abs(enc - self._enc_0))
            progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist > 0 else False

            self._steps_on_action += 1

            if progress:
                self._consecutive_progress += 1
                self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
                self._actions_tried_this_round = 0
            else:
                self._consecutive_progress = 0
                if self._steps_on_action >= self._patience:
                    self._actions_tried_this_round += 1
                    self._steps_on_action = 0
                    self._patience = 3
                    if self._actions_tried_this_round >= self._n_responsive:
                        self._exploit_idx = self._rng.randint(self._n_responsive)
                        self._actions_tried_this_round = 0
                    else:
                        self._exploit_idx = (self._exploit_idx + 1) % self._n_responsive

            self._prev_dist = dist
            action = self._responsive_actions[self._exploit_idx]
            self._prev_enc = enc.copy()
            self._prev_action = action
            return action

        # Fallback: keyboard-only reactive switching (no responsive pixels)
        n_active = min(self._n_actions_env, N_KB)
        dist = np.sum(np.abs(enc - self._enc_0))
        progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist > 0 else False

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0
            if self._steps_on_action >= self._patience:
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = 3
                if self._actions_tried_this_round >= n_active:
                    self._exploit_idx = self._rng.randint(n_active)
                    self._actions_tried_this_round = 0
                else:
                    self._exploit_idx = (self._exploit_idx + 1) % n_active

        self._prev_dist = dist
        action = self._exploit_idx
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_action = None
        self._prev_dist = 0.0
        self._exploit_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep heat map and responsive pixels across levels (ℓ_π: learned)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "scan_pixels": SCAN_PIXELS,
    "scan_start": SCAN_START,
    "scan_end": SCAN_END,
    "top_k": TOP_K,
    "responsive_thresh": RESPONSIVE_THRESH,
    "max_patience": MAX_PATIENCE,
    "family": "pixel scan diagnostic",
    "tag": "prosecution v37 (ℓ_π systematic pixel scan: 4096 pixels → heat map → exploit responsive)",
}

SUBSTRATE_CLASS = PixelScanSubstrate
