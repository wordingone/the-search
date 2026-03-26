"""
sub1079_diagnostic_opaque.py — Diagnostic: why are opaque games opaque?

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1079 --substrate experiments/sub1079_diagnostic_opaque.py

FAMILY: diagnostic (not a substrate experiment)
R3 HYPOTHESIS: N/A — this is a measurement, not an R3 test.

PURPOSE: Understand the 0% wall. For each game in the draw, measure:
  1. Raw pixel variance across 1000 random actions
  2. Per-action conditioned pixel variance (100 applications per action)
  3. Frame differencing: |frame_t - frame_{t-1}| magnitude per step
  4. Avgpool16 retention ratio: how much pixel variance survives block averaging

Output: per-game diagnostic table printed to log.

KILL: N/A — measurement only
SUCCESS: Clear diagnosis of where the 0% wall originates
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS
N_KB = 7
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]

# Measurement windows
MEASURE_STEPS = 2000  # plenty of steps for statistics


def _click_action(x, y):
    return N_KB + y * 64 + x


def _obs_to_blocks(obs):
    """Avgpool with 8x8 blocks → 64 dims."""
    blocks = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            blocks[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return blocks


class DiagnosticOpaqueSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self.prev_obs = None
        self._prev_action = None

        # Diagnostic accumulators
        self._all_pixels = []          # raw obs arrays (subsample)
        self._per_action_pixels = {}   # action → list of obs arrays
        self._frame_diffs = []         # |frame_t - frame_{t-1}| L1 norms
        self._frame_diffs_max = []     # max pixel diff per frame
        self._frame_diffs_nonzero = [] # count of non-zero diff pixels per frame
        self._block_variances = []     # block-space obs vectors

        # Track action counts
        self._action_counts = {}

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        # Print diagnostic for previous game if we have data
        if self._all_pixels:
            self._print_diagnostic()

        self._game_number += 1
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()

    def _print_diagnostic(self):
        """Print per-game diagnostic summary."""
        print(f"\n{'='*70}", flush=True)
        print(f"DIAGNOSTIC — GAME_{self._game_number} (n_actions={self._n_actions})", flush=True)
        print(f"{'='*70}", flush=True)

        # 1. Raw pixel variance
        if len(self._all_pixels) > 1:
            pixel_stack = np.stack(self._all_pixels, axis=0)  # (N, 64, 64)
            pixel_var = pixel_stack.var(axis=0)  # (64, 64) variance per pixel
            total_pixel_var = float(pixel_var.mean())
            max_pixel_var = float(pixel_var.max())
            nonzero_pixels = int(np.sum(pixel_var > 1e-6))
            print(f"\n1. RAW PIXEL VARIANCE (across {len(self._all_pixels)} observations):", flush=True)
            print(f"   Mean pixel variance:    {total_pixel_var:.6f}", flush=True)
            print(f"   Max pixel variance:     {max_pixel_var:.6f}", flush=True)
            print(f"   Non-zero variance pixels: {nonzero_pixels}/{64*64} ({100*nonzero_pixels/(64*64):.1f}%)", flush=True)

            # Spatial distribution of variance
            high_var_mask = pixel_var > (total_pixel_var * 2)
            print(f"   High-variance region:   {int(np.sum(high_var_mask))} pixels", flush=True)

            # 4. Avgpool16 retention ratio
            block_stack = np.stack(self._block_variances, axis=0)  # (N, 64)
            block_var = block_stack.var(axis=0)
            total_block_var = float(block_var.mean())
            retention = total_block_var / max(total_pixel_var, 1e-10)
            print(f"\n4. AVGPOOL8 RETENTION RATIO:", flush=True)
            print(f"   Block-space variance:   {total_block_var:.6f}", flush=True)
            print(f"   Retention ratio:        {retention:.4f} ({100*retention:.1f}%)", flush=True)
            nonzero_blocks = int(np.sum(block_var > 1e-6))
            print(f"   Non-zero variance blocks: {nonzero_blocks}/{N_DIMS}", flush=True)
        else:
            print("\n1. RAW PIXEL VARIANCE: insufficient data", flush=True)

        # 2. Per-action conditioned variance
        print(f"\n2. ACTION-CONDITIONED PIXEL VARIANCE:", flush=True)
        any_action_response = False
        action_responses = []
        for a in sorted(self._per_action_pixels.keys()):
            obs_list = self._per_action_pixels[a]
            if len(obs_list) < 3:
                continue
            obs_stack = np.stack(obs_list, axis=0)
            action_var = float(obs_stack.var(axis=0).mean())
            count = len(obs_list)
            action_responses.append((a, action_var, count))
            if action_var > 1e-5:
                any_action_response = True

        if action_responses:
            # Show top 10 most responsive actions
            action_responses.sort(key=lambda x: -x[1])
            print(f"   Total actions tracked: {len(action_responses)}", flush=True)
            print(f"   Any action with non-trivial variance: {'YES' if any_action_response else 'NO'}", flush=True)
            print(f"   Top 10 most variable actions:", flush=True)
            for a, v, c in action_responses[:10]:
                label = f"kb{a}" if a < N_KB else f"click({(a-N_KB)%64},{(a-N_KB)//64})"
                print(f"     action={label:>20s}: var={v:.6f} (n={c})", flush=True)
            # Also show bottom 5
            if len(action_responses) > 10:
                print(f"   Bottom 5 least variable:", flush=True)
                for a, v, c in action_responses[-5:]:
                    label = f"kb{a}" if a < N_KB else f"click({(a-N_KB)%64},{(a-N_KB)//64})"
                    print(f"     action={label:>20s}: var={v:.6f} (n={c})", flush=True)
        else:
            print("   No actions with sufficient samples", flush=True)

        # 3. Frame differencing
        if self._frame_diffs:
            diffs = np.array(self._frame_diffs)
            maxes = np.array(self._frame_diffs_max)
            nonzeros = np.array(self._frame_diffs_nonzero)
            print(f"\n3. FRAME DIFFERENCING ({len(diffs)} transitions):", flush=True)
            print(f"   Mean |frame_t - frame_{{t-1}}| L1: {diffs.mean():.6f}", flush=True)
            print(f"   Max frame diff:                   {diffs.max():.6f}", flush=True)
            print(f"   Frames with ZERO diff:            {int(np.sum(diffs < 1e-8))}/{len(diffs)} ({100*np.mean(diffs < 1e-8):.1f}%)", flush=True)
            print(f"   Mean max-pixel-diff per frame:    {maxes.mean():.4f}", flush=True)
            print(f"   Mean non-zero-diff pixels/frame:  {nonzeros.mean():.1f}", flush=True)

            # Distribution of diff magnitudes
            if diffs.max() > 0:
                pcts = [10, 25, 50, 75, 90, 95, 99]
                print(f"   Diff magnitude percentiles:", flush=True)
                for p in pcts:
                    print(f"     p{p:>2d}: {np.percentile(diffs, p):.6f}", flush=True)
        else:
            print("\n3. FRAME DIFFERENCING: no transitions recorded", flush=True)

        # Summary verdict
        print(f"\n--- VERDICT ---", flush=True)
        if len(self._all_pixels) < 2:
            print("INSUFFICIENT DATA", flush=True)
        elif float(np.stack(self._all_pixels).var(axis=0).mean()) < 1e-6:
            print("TRULY INERT: observation never changes regardless of action", flush=True)
        elif not any_action_response:
            print("UNIFORM RESPONSE: observation changes but NO action differentiates (all actions equivalent)", flush=True)
        elif retention < 0.01:
            print(f"ENCODING BOTTLENECK: pixel variance exists but avgpool8 destroys {100*(1-retention):.1f}% of it", flush=True)
        else:
            print("RESPONSIVE: game responds to actions with detectable signal through encoding", flush=True)
        print(f"{'='*70}\n", flush=True)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))
        arr = obs
        self.step_count += 1

        # ── Collect measurements ──

        # Store obs (subsample every 2 steps to save memory)
        if self.step_count % 2 == 0 or self.step_count <= 10:
            self._all_pixels.append(arr.copy())
            self._block_variances.append(_obs_to_blocks(arr))

        # Frame diff
        if self.prev_obs is not None:
            diff = np.abs(arr - self.prev_obs)
            self._frame_diffs.append(float(diff.mean()))
            self._frame_diffs_max.append(float(diff.max()))
            self._frame_diffs_nonzero.append(int(np.sum(diff > 0)))

        # Per-action tracking (store obs AFTER action was taken)
        if self._prev_action is not None:
            a = self._prev_action
            if a not in self._per_action_pixels:
                self._per_action_pixels[a] = []
            # Limit per-action storage
            if len(self._per_action_pixels[a]) < 150:
                self._per_action_pixels[a].append(arr.copy())

        self.prev_obs = arr.copy()

        # ── Take random action ──
        # Cycle through actions systematically for good coverage
        if self.step_count <= N_KB * 30:
            # First ~210 steps: cycle through KB actions for coverage
            action = (self.step_count - 1) % N_KB
        elif self._supports_click and self.step_count <= N_KB * 30 + len(CLICK_GRID) * 10:
            # Next ~640 steps: cycle through click grid
            idx = (self.step_count - N_KB * 30 - 1) % len(CLICK_GRID)
            cx, cy = CLICK_GRID[idx]
            action = _click_action(cx, cy)
        else:
            # Random mix
            if self._supports_click and self._rng.random() < 0.7:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
            else:
                action = self._rng.randint(N_KB)

        self._prev_action = action
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        return action

    def on_level_transition(self):
        # Print diagnostic for the level that just ended
        if self._all_pixels:
            self._print_diagnostic()
        self._init_state()


CONFIG = {
    "measure_steps": MEASURE_STEPS,
    "v79_features": "diagnostic-only: pixel variance, action-conditioned variance, frame diffs, avgpool retention",
}

SUBSTRATE_CLASS = DiagnosticOpaqueSubstrate
