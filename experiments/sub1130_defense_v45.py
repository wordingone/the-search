"""
sub1130_defense_v45.py — Hybrid encoding click search (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1130 --substrate experiments/sub1130_defense_v45.py

FAMILY: Hierarchical click exploration (defense v45, hybrid encoding)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v43/v44 used avgpool4 encoding (4×4 block averages) for
probe change detection. Click responses may be localized to specific pixels
WITHIN a 4×4 block — avgpool4 smooths them out below CHANGE_THRESH. v45
uses RAW PIXEL comparison for probe phase (catches sub-block changes) and
avgpool4 for reactive switching (filters noise during exploitation).

Architecture:
- Phase 1 (steps 1-50): keyboard explore (avgpool4 for change detection)
- Phase 2 (steps 51-98): probe 16 macro-blocks using RAW PIXEL change
  detection (64×64 = 4096D comparison). Lower effective threshold because
  raw pixels catch localized changes that avgpool4 misses.
- Phase 3 (steps 99-146): refine best macro-block using raw pixel detection
- Phase 4 (step 147+): reactive switching using avgpool4 (noise-filtered)

Budget: same as v43 (146 explore steps).

KILL: ARC ≤ v43 (0.0041) if on comparable draw.
SUCCESS: ARC > v43 — raw pixel probing finds more responsive targets.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
KB_EXPLORE = 50
MACRO_SIZE = 4
PROBE_REPS = 3
MAX_PATIENCE = 20
CHANGE_THRESH_KB = 0.5      # keyboard probe: avgpool4
CHANGE_THRESH_CLICK = 2.0   # click probe: raw pixel (higher absolute, lower relative)


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


def _pixel_to_click_action(px, py):
    """Pixel coordinate → click action index."""
    return N_KB + px + py * 64


class HybridEncodingClickSubstrate:
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
        self._prev_raw = None  # raw 64×64 for probe phase

        # Phase management
        self._phase = "keyboard"
        self._kb_total_change = 0.0
        self._kb_responsive = []

        # Hierarchical click probing
        self._macro_probes = []
        self._macro_idx = 0
        self._macro_step = 0
        self._macro_pre_raw = None  # raw pixel for probe change detection
        self._macro_response = {}

        self._fine_probes = []
        self._fine_idx = 0
        self._fine_step = 0
        self._fine_pre_raw = None
        self._responsive_clicks = []

        # Reactive switching
        self._active_actions = []
        self._current_idx = 0
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
        self._macro_probes = []
        macro_step = 64 // MACRO_SIZE
        for my in range(MACRO_SIZE):
            for mx in range(MACRO_SIZE):
                cx = mx * macro_step + macro_step // 2
                cy = my * macro_step + macro_step // 2
                self._macro_probes.append((cx, cy))
        self._rng.shuffle(self._macro_probes)

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _raw_change(self, raw_a, raw_b):
        """L1 distance between raw 64×64 observations."""
        return float(np.sum(np.abs(raw_a - raw_b)))

    def _build_fine_probes(self, best_macro_idx):
        """Build 16 fine-grid probes within the best macro-block."""
        cx, cy = self._macro_probes[best_macro_idx]
        macro_step = 64 // MACRO_SIZE
        fine_step = macro_step // 4
        base_x = cx - macro_step // 2
        base_y = cy - macro_step // 2
        self._fine_probes = []
        for fy in range(4):
            for fx in range(4):
                px = base_x + fx * fine_step + fine_step // 2
                py = base_y + fy * fine_step + fine_step // 2
                px = max(0, min(63, px))
                py = max(0, min(63, py))
                self._fine_probes.append((px, py))
        self._rng.shuffle(self._fine_probes)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        raw = obs.copy()  # keep raw for probe phase

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._prev_raw = raw.copy()
            self._current_idx = self._rng.randint(min(self._n_actions_env, N_KB))
            return self._current_idx

        dist = self._dist_to_initial(enc)

        # === Phase 1: Keyboard explore (avgpool4 detection) ===
        if self._phase == "keyboard":
            if self._prev_enc is not None:
                change = float(np.sum(np.abs(enc - self._prev_enc)))
                self._kb_total_change += change
                prev_a = (self.step_count - 1) % min(self._n_actions_env, N_KB)
                if change > CHANGE_THRESH_KB and prev_a not in self._kb_responsive:
                    self._kb_responsive.append(prev_a)

            if self.step_count <= KB_EXPLORE:
                action = self.step_count % min(self._n_actions_env, N_KB)
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                self._prev_raw = raw.copy()
                return action

            if len(self._kb_responsive) >= 2:
                self._phase = "reactive"
                self._active_actions = self._kb_responsive[:]
                self._current_idx = 0
            elif self._has_clicks:
                self._phase = "macro_probe"
                self._macro_idx = 0
                self._macro_step = 0
                self._macro_pre_raw = raw.copy()
            else:
                self._phase = "reactive"
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._current_idx = 0

        # === Phase 2: Macro-block probing (RAW PIXEL detection) ===
        if self._phase == "macro_probe":
            if self._macro_step > 0 and self._macro_pre_raw is not None:
                if self._macro_step == PROBE_REPS:
                    # Use RAW pixel change for probe detection
                    probe_change = self._raw_change(raw, self._macro_pre_raw)
                    self._macro_response[self._macro_idx] = probe_change
                    self._macro_idx += 1
                    self._macro_step = 0
                    self._macro_pre_raw = raw.copy()

            if self._macro_idx >= len(self._macro_probes):
                if self._macro_response:
                    best_idx = max(self._macro_response, key=self._macro_response.get)
                    best_change = self._macro_response[best_idx]
                    if best_change > CHANGE_THRESH_CLICK:
                        self._build_fine_probes(best_idx)
                        self._phase = "fine_probe"
                        self._fine_idx = 0
                        self._fine_step = 0
                        self._fine_pre_raw = raw.copy()
                        cx, cy = self._macro_probes[best_idx]
                        self._responsive_clicks.append((cx, cy))
                    else:
                        self._phase = "reactive"
                        self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                        self._current_idx = 0
                else:
                    self._phase = "reactive"
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                    self._current_idx = 0
            else:
                self._macro_step += 1
                cx, cy = self._macro_probes[self._macro_idx]
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                self._prev_raw = raw.copy()
                return _pixel_to_click_action(cx, cy)

        # === Phase 3: Fine-grid probing (RAW PIXEL detection) ===
        if self._phase == "fine_probe":
            if self._fine_step > 0 and self._fine_pre_raw is not None:
                if self._fine_step == PROBE_REPS:
                    probe_change = self._raw_change(raw, self._fine_pre_raw)
                    if probe_change > CHANGE_THRESH_CLICK:
                        px, py = self._fine_probes[self._fine_idx]
                        self._responsive_clicks.append((px, py))
                    self._fine_idx += 1
                    self._fine_step = 0
                    self._fine_pre_raw = raw.copy()

            if self._fine_idx >= len(self._fine_probes):
                self._active_actions = self._kb_responsive[:]
                for px, py in self._responsive_clicks:
                    self._active_actions.append(_pixel_to_click_action(px, py))
                if not self._active_actions:
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._phase = "reactive"
                self._current_idx = 0
            else:
                self._fine_step += 1
                px, py = self._fine_probes[self._fine_idx]
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                self._prev_raw = raw.copy()
                return _pixel_to_click_action(px, py)

        # === Phase 4: Reactive switching (avgpool4 for noise filtering) ===
        if self._phase == "reactive":
            n_active = len(self._active_actions)
            if n_active == 0:
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                n_active = len(self._active_actions)

            progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
            no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

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
                    if self._actions_tried_this_round >= n_active:
                        self._current_idx = self._rng.randint(n_active)
                        self._actions_tried_this_round = 0
                    else:
                        self._current_idx = (self._current_idx + 1) % n_active

            action = self._active_actions[self._current_idx % n_active]
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            self._prev_raw = raw.copy()
            return action

        # Fallback
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        self._prev_raw = raw.copy()
        return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._prev_raw = None
        self._current_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_kb": N_KB,
    "kb_explore": KB_EXPLORE,
    "macro_size": MACRO_SIZE,
    "probe_reps": PROBE_REPS,
    "max_patience": MAX_PATIENCE,
    "change_thresh_kb": CHANGE_THRESH_KB,
    "change_thresh_click": CHANGE_THRESH_CLICK,
    "family": "hierarchical click exploration",
    "tag": "defense v45 (ℓ₁ hybrid encoding: raw pixel probe detection + avgpool4 reactive)",
}

SUBSTRATE_CLASS = HybridEncodingClickSubstrate
