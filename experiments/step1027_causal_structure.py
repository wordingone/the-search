"""
Step 1027 — Causal Structure (Cross-Zone HOW)

D2-grounded experiment: WHERE → cross-zone causal mapping.
R3 hypothesis: Clicking zone A produces measurable change at zone B's location.
The causal matrix (which zones affect which) encodes the game's mechanical
structure — e.g., Lights-Out adjacency in FT09, or canal-lock sequencing in VC33.

Method:
  Phase 1 (WHERE, 5000 steps): Random clicking → mode map → zones.
  Phase 2 (Causal probe): For each zone A, click it 10x. After each click,
    measure pixel change at EVERY other zone B's location. Build NxN causal
    matrix: M[A,B] = mean |diff at B| when A was clicked.
  Phase 3 (Analysis): Report causal matrix, identify: self-only zones (diagonal),
    spread zones (affect neighbors), and asymmetric pairs (A→B but not B→A).

Games: FT09 + VC33 (click games). 5 seeds.

Kill: Causal matrix = pure diagonal (each zone only affects itself) → no cross-zone mechanics.
Signal: Off-diagonal entries > 0 → game has cross-zone causal structure.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import label as ndlabel

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
MODE_WARMUP = 5000
CAUSAL_CLICKS = 10    # clicks per zone for causal probing
MIN_CLUSTER = 4
MAX_CLUSTER = 60
TIME_CAP = 90
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
ZONE_RADIUS = 4       # pixels around zone center to measure change

# ─── Mode map (from 576/1023) ───
def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r, c = np.arange(64)[:, None], np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1

def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)

def find_isolated_clusters(mode_arr):
    clusters = []
    for color in range(1, 16):
        mask = (mode_arr == color)
        if not mask.any(): continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if MIN_CLUSTER <= sz <= MAX_CLUSTER:
                ys, xs = np.where(region)
                clusters.append({
                    'cx_int': int(round(xs.mean())),
                    'cy_int': int(round(ys.mean())),
                    'color': int(color), 'size': sz
                })
    return clusters

# ─── Causal measurement ───
def measure_zone_change(frame_before, frame_after, cx, cy, radius=ZONE_RADIUS):
    """Measure absolute pixel change in a radius around (cx, cy)."""
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    y0 = max(0, cy - radius)
    y1 = min(64, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(64, cx + radius + 1)
    patch_diff = np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1])
    return float(patch_diff.mean())


# ─── Run ───
def run_seed(arc, game_id, seed):
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    obs = env.reset()
    steps = 0
    t0 = time.time()
    levels = 0

    # Phase 1: WHERE
    for _ in range(MODE_WARMUP):
        if time.time() - t0 > TIME_CAP: break
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        update_freq(freq, obs.frame)
        a = np.random.randint(N_GRID)
        cx, cy = CLICK_GRID[a]
        obs = env.step(action6, data={"x": cx, "y": cy})
        steps += 1
        if obs and obs.levels_completed > levels:
            levels = obs.levels_completed

    mode = compute_mode(freq)
    clusters = find_isolated_clusters(mode)
    n_zones = len(clusters)

    if n_zones < 2:
        return {
            'seed': seed, 'n_zones': n_zones,
            'causal_matrix': None, 'self_only': 0, 'spread': 0,
            'asymmetric_pairs': 0, 'levels': levels,
            'steps': steps, 'elapsed': round(time.time() - t0, 1)
        }

    # Phase 2: Causal probe — click zone A, measure change at all zones
    # causal_accum[A][B] = list of change magnitudes at B when A clicked
    causal_accum = [[[] for _ in range(n_zones)] for _ in range(n_zones)]

    for zi in range(n_zones):
        cl_a = clusters[zi]
        for click_idx in range(CAUSAL_CLICKS):
            if time.time() - t0 > TIME_CAP: break
            if obs is None or obs.state == GameState.GAME_OVER:
                obs = env.reset(); continue
            if obs.state == GameState.WIN: break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            frame_before = obs.frame
            obs = env.step(action6, data={"x": cl_a['cx_int'], "y": cl_a['cy_int']})
            steps += 1

            if obs is not None and obs.frame and len(obs.frame) > 0:
                for zj in range(n_zones):
                    cl_b = clusters[zj]
                    change = measure_zone_change(frame_before, obs.frame,
                                                  cl_b['cx_int'], cl_b['cy_int'])
                    causal_accum[zi][zj].append(change)
                if obs.levels_completed > levels:
                    levels = obs.levels_completed

    # Phase 3: Build causal matrix
    causal_matrix = np.zeros((n_zones, n_zones), dtype=np.float32)
    for i in range(n_zones):
        for j in range(n_zones):
            if causal_accum[i][j]:
                causal_matrix[i][j] = np.mean(causal_accum[i][j])

    # Classify zones
    diag_threshold = 0.1  # minimum self-change to count as interactive
    offdiag_threshold = 0.05  # minimum cross-change to count as causal link

    self_only = 0
    spread = 0
    for i in range(n_zones):
        self_change = causal_matrix[i, i]
        off_diag = np.array([causal_matrix[i, j] for j in range(n_zones) if j != i])
        max_off = float(off_diag.max()) if len(off_diag) > 0 else 0
        if self_change > diag_threshold and max_off < offdiag_threshold:
            self_only += 1
        elif max_off >= offdiag_threshold:
            spread += 1

    # Count asymmetric pairs
    asymmetric = 0
    for i in range(n_zones):
        for j in range(i + 1, n_zones):
            ab = causal_matrix[i, j]
            ba = causal_matrix[j, i]
            if (ab > offdiag_threshold) != (ba > offdiag_threshold):
                asymmetric += 1

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'n_zones': n_zones,
        'causal_matrix': causal_matrix.tolist(),
        'self_only': self_only,
        'spread': spread,
        'asymmetric_pairs': asymmetric,
        'levels': levels,
        'steps': steps,
        'elapsed': round(elapsed, 1),
        'zones': [(cl['cx_int'], cl['cy_int'], cl['color'], cl['size']) for cl in clusters]
    }


def print_causal_matrix(r):
    """Print a compact causal matrix."""
    if r['causal_matrix'] is None:
        print("    (no matrix — <2 zones)")
        return
    M = np.array(r['causal_matrix'])
    n = M.shape[0]
    # Print header
    header = "     " + "".join(f"  z{j:<3d}" for j in range(min(n, 12)))
    print(header)
    for i in range(min(n, 12)):
        row = f"  z{i}: " + "".join(f"{M[i,j]:5.2f} " for j in range(min(n, 12)))
        print(row)
    if n > 12:
        print(f"    ... ({n} zones total, showing first 12)")


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()

    games = {}
    for e in envs:
        if 'vc33' in e.game_id.lower():
            games['VC33'] = e.game_id
        elif 'ft09' in e.game_id.lower():
            games['FT09'] = e.game_id

    print("=== Step 1027: Causal Structure (Cross-Zone HOW) ===")
    print(f"WHERE={MODE_WARMUP} steps, CAUSAL={CAUSAL_CLICKS} clicks/zone")
    print(f"Zone radius={ZONE_RADIUS}px\n")

    for game_name in ['FT09', 'VC33']:
        game_id = games.get(game_name)
        if not game_id:
            print(f"SKIP {game_name}: not found"); continue

        print(f"--- {game_name} ({game_id}) ---")
        all_results = []
        for seed in range(5):
            r = run_seed(arc, game_id, seed)
            print(f"  s{seed}: zones={r['n_zones']}  self_only={r['self_only']}  "
                  f"spread={r['spread']}  asymmetric={r['asymmetric_pairs']}  "
                  f"L={r['levels']}  {r['elapsed']}s")
            print_causal_matrix(r)
            all_results.append(r)

        # Summary
        avg_spread = np.mean([r['spread'] for r in all_results])
        avg_asym = np.mean([r['asymmetric_pairs'] for r in all_results])
        any_spread = any(r['spread'] > 0 for r in all_results)
        print(f"\n  {game_name} summary: avg_spread={avg_spread:.1f}  "
              f"avg_asymmetric={avg_asym:.1f}")
        if any_spread:
            print(f"  SIGNAL: Cross-zone causal structure detected")
        else:
            print(f"  NO SIGNAL: All zones are causally independent (self-only)")
        print()

    print("Step 1027 DONE")


if __name__ == "__main__":
    main()
