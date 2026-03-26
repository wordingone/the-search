"""
sub1128_defense_v44.py — Multi-region hierarchical click search (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1128 --substrate experiments/sub1128_defense_v44.py

FAMILY: Hierarchical click exploration (defense v44, multi-region variant)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v43 found one click game solvable (GAME_3 ARC=0.0207) but
only refined the SINGLE best macro-block. If responsive areas span multiple
regions (e.g., game has buttons in different quadrants), v43 misses them.
v44 refines ALL responsive macro-blocks (up to 3), building a richer
reactive action set.

Architecture:
- Phase 1 (steps 1-50): keyboard explore (same as v43)
- If keyboard works → pure reactive on keyboard
- Phase 2 (steps 51-98): probe 16 macro-blocks (4×4 grid, 3 reps each)
- Phase 3 (steps 99+): refine ALL responsive macro-blocks (up to 3),
  16 sub-blocks each, 2 reps each. Max fine probes: 3×16×2 = 96 steps.
- Phase 4: reactive switching among all responsive click targets.

Budget: 50 + 48 + 96 = 194 explore steps (v43=146, v35=263).
More coverage than v43, still more efficient than v35.

KILL: ARC ≤ v43 (0.0041).
SUCCESS: ARC > v43 — multi-region finds more click targets.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
KB_EXPLORE = 50
MACRO_SIZE = 4  # 4×4 macro-blocks = 16 regions of 16×16 pixels
PROBE_REPS = 3  # steps per macro probe
FINE_REPS = 2   # steps per fine probe (fewer to fit budget)
MAX_REFINE = 3  # refine up to 3 responsive macro-blocks
MAX_PATIENCE = 20
CHANGE_THRESH = 0.5


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


class MultiRegionClickSubstrate:
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

        # Phase management
        self._phase = "keyboard"
        self._kb_total_change = 0.0
        self._kb_responsive = []

        # Macro probing
        self._macro_probes = []
        self._macro_idx = 0
        self._macro_step = 0
        self._macro_pre_enc = None
        self._macro_response = {}

        # Fine probing (multi-region)
        self._fine_regions = []  # list of (region_probes, region_center)
        self._region_idx = 0
        self._fine_probes = []
        self._fine_idx = 0
        self._fine_step = 0
        self._fine_pre_enc = None
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
        # Build macro-block probes
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

    def _build_fine_probes_for_region(self, macro_idx):
        """Build 16 fine-grid probes within a macro-block."""
        cx, cy = self._macro_probes[macro_idx]
        macro_step = 64 // MACRO_SIZE
        fine_step = macro_step // 4
        base_x = cx - macro_step // 2
        base_y = cy - macro_step // 2
        probes = []
        for fy in range(4):
            for fx in range(4):
                px = base_x + fx * fine_step + fine_step // 2
                py = base_y + fy * fine_step + fine_step // 2
                px = max(0, min(63, px))
                py = max(0, min(63, py))
                probes.append((px, py))
        self._rng.shuffle(probes)
        return probes

    def _setup_fine_phase(self):
        """Set up fine probing for top responsive macro-blocks."""
        if not self._macro_response:
            return False
        # Sort macro-blocks by response magnitude, take top MAX_REFINE
        sorted_macros = sorted(
            self._macro_response.items(), key=lambda x: x[1], reverse=True
        )
        responsive = [(idx, change) for idx, change in sorted_macros
                       if change > CHANGE_THRESH]
        if not responsive:
            return False
        # Build fine probes for each responsive region
        self._fine_regions = []
        for idx, _ in responsive[:MAX_REFINE]:
            probes = self._build_fine_probes_for_region(idx)
            cx, cy = self._macro_probes[idx]
            self._fine_regions.append((probes, (cx, cy)))
            # Add macro center as responsive
            self._responsive_clicks.append((cx, cy))
        # Start with first region
        self._region_idx = 0
        self._fine_probes = self._fine_regions[0][0]
        self._fine_idx = 0
        self._fine_step = 0
        return True

    def _advance_fine_region(self):
        """Move to next region or finish fine probing."""
        self._region_idx += 1
        if self._region_idx < len(self._fine_regions):
            self._fine_probes = self._fine_regions[self._region_idx][0]
            self._fine_idx = 0
            self._fine_step = 0
            return True
        return False

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
            self._current_idx = self._rng.randint(min(self._n_actions_env, N_KB))
            return self._current_idx

        dist = self._dist_to_initial(enc)

        # === Phase 1: Keyboard explore ===
        if self._phase == "keyboard":
            if self._prev_enc is not None:
                change = float(np.sum(np.abs(enc - self._prev_enc)))
                self._kb_total_change += change
                prev_a = (self.step_count - 1) % min(self._n_actions_env, N_KB)
                if change > CHANGE_THRESH and prev_a not in self._kb_responsive:
                    self._kb_responsive.append(prev_a)

            if self.step_count <= KB_EXPLORE:
                action = self.step_count % min(self._n_actions_env, N_KB)
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

            # Keyboard done
            if len(self._kb_responsive) >= 2:
                self._phase = "reactive"
                self._active_actions = self._kb_responsive[:]
                self._current_idx = 0
            elif self._has_clicks:
                self._phase = "macro_probe"
                self._macro_idx = 0
                self._macro_step = 0
                self._macro_pre_enc = enc.copy()
            else:
                self._phase = "reactive"
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._current_idx = 0

        # === Phase 2: Macro-block probing ===
        if self._phase == "macro_probe":
            if self._macro_step > 0 and self._macro_pre_enc is not None:
                if self._macro_step == PROBE_REPS:
                    probe_change = float(np.sum(np.abs(enc - self._macro_pre_enc)))
                    self._macro_response[self._macro_idx] = probe_change
                    self._macro_idx += 1
                    self._macro_step = 0
                    self._macro_pre_enc = enc.copy()

            if self._macro_idx >= len(self._macro_probes):
                # Macro done — set up fine probing for multiple regions
                if self._setup_fine_phase():
                    self._phase = "fine_probe"
                    self._fine_pre_enc = enc.copy()
                else:
                    self._phase = "reactive"
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                    self._current_idx = 0
            else:
                self._macro_step += 1
                cx, cy = self._macro_probes[self._macro_idx]
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return _pixel_to_click_action(cx, cy)

        # === Phase 3: Fine-grid probing (multi-region) ===
        if self._phase == "fine_probe":
            if self._fine_step > 0 and self._fine_pre_enc is not None:
                if self._fine_step == FINE_REPS:
                    probe_change = float(np.sum(np.abs(enc - self._fine_pre_enc)))
                    if probe_change > CHANGE_THRESH:
                        px, py = self._fine_probes[self._fine_idx]
                        self._responsive_clicks.append((px, py))
                    self._fine_idx += 1
                    self._fine_step = 0
                    self._fine_pre_enc = enc.copy()

            if self._fine_idx >= len(self._fine_probes):
                # Current region done — try next region
                if not self._advance_fine_region():
                    # All regions done — build action set
                    self._active_actions = self._kb_responsive[:]
                    for px, py in self._responsive_clicks:
                        self._active_actions.append(_pixel_to_click_action(px, py))
                    if not self._active_actions:
                        self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                    self._phase = "reactive"
                    self._current_idx = 0
                else:
                    self._fine_pre_enc = enc.copy()
            else:
                self._fine_step += 1
                px, py = self._fine_probes[self._fine_idx]
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return _pixel_to_click_action(px, py)

        # === Phase 4: Reactive switching ===
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
            return action

        # Fallback
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep phase and active_actions — same game type


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_kb": N_KB,
    "kb_explore": KB_EXPLORE,
    "macro_size": MACRO_SIZE,
    "probe_reps": PROBE_REPS,
    "fine_reps": FINE_REPS,
    "max_refine": MAX_REFINE,
    "max_patience": MAX_PATIENCE,
    "change_thresh": CHANGE_THRESH,
    "family": "hierarchical click exploration",
    "tag": "defense v44 (ℓ₁ multi-region hierarchical: refine top-3 responsive macro-blocks)",
}

SUBSTRATE_CLASS = MultiRegionClickSubstrate
