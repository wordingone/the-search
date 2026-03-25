"""
Step 1029 — VC33: Causal-Guided Action for L2+

D2-grounded experiment: WHERE → HOW → ACT pipeline.
R3 hypothesis: Sorting zones by causal influence (off-diagonal sum from causal
matrix) and clicking hub-first produces L2+ progression on VC33.

Method:
  Phase 1 (WHERE, 5000 steps): Mode map → zones.
  Phase 2 (HOW, 2000 steps): Click each zone, record magnitude + causal matrix.
  Phase 3 (ACT, remaining): Sort by descending causal influence. Click in order.
    Re-click if frame change is small. On level transition, re-discover (mini WHERE+HOW).

Kill: L1 < 5/5 (regression) or L2 = 0/5.
Budget: 5 min per seed.
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
WHERE_STEPS = 5000
HOW_STEPS = 2000
REDISCOVER_WHERE = 1000  # mini WHERE after level transition
REDISCOVER_HOW = 500     # mini HOW after level transition
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 120
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
ZONE_RADIUS = 4
ACT_BURST = 5  # clicks per zone in ACT phase
RECLICK_THRESHOLD = 0.05  # min frame change to move to next zone

# ─── Mode map ───
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

def measure_zone_change(frame_before, frame_after, cx, cy, radius=ZONE_RADIUS):
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    return float(np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1]).mean())

def frame_diff_magnitude(frame_before, frame_after):
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    return float(np.abs(b - a).mean())


# ─── Discovery phases ───
def discover_where(env, action6, obs, n_steps, rng):
    """Run WHERE discovery. Returns (clusters, obs, steps_used)."""
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    steps = 0
    for _ in range(n_steps):
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue
        update_freq(freq, obs.frame)
        a = rng.randint(N_GRID)
        cx, cy = CLICK_GRID[a]
        obs = env.step(action6, data={"x": cx, "y": cy})
        steps += 1
    mode = compute_mode(freq)
    clusters = find_isolated_clusters(mode)
    return clusters, obs, steps

def discover_how(env, action6, obs, clusters, n_steps_per_zone, rng):
    """Run HOW discovery. Returns (causal_influence, magnitude, obs, steps_used)."""
    n = len(clusters)
    if n == 0:
        return [], [], obs, 0

    clicks_per_zone = max(3, n_steps_per_zone // n)
    magnitudes = [0.0] * n
    causal_matrix = np.zeros((n, n), dtype=np.float32)
    steps = 0

    for zi in range(n):
        cl = clusters[zi]
        for _ in range(clicks_per_zone):
            if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                obs = env.reset() if obs is None or obs.state == GameState.GAME_OVER else obs
                if obs is None or obs.state == GameState.WIN: break
                continue
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            frame_before = obs.frame
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            steps += 1

            if obs and obs.frame and len(obs.frame) > 0:
                # Magnitude
                diff = frame_diff_magnitude(frame_before, obs.frame)
                magnitudes[zi] = max(magnitudes[zi], diff)
                # Causal: measure change at all zones
                for zj in range(n):
                    cl_b = clusters[zj]
                    change = measure_zone_change(frame_before, obs.frame,
                                                  cl_b['cx_int'], cl_b['cy_int'])
                    causal_matrix[zi][zj] += change

    # Normalize
    causal_matrix /= max(clicks_per_zone, 1)

    # Causal influence = sum of off-diagonal effects
    causal_influence = []
    for i in range(n):
        off_diag = sum(causal_matrix[i][j] for j in range(n) if j != i)
        causal_influence.append(off_diag)

    return causal_influence, magnitudes, obs, steps


# ─── Run ───
def run_seed(arc, game_id, seed):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]
    obs = env.reset()
    total_steps = 0
    t0 = time.time()
    max_levels = 0
    level_steps = {}  # level_num -> step at which it was reached

    # Initial discovery
    clusters, obs, s = discover_where(env, action6, obs, WHERE_STEPS, rng)
    total_steps += s
    causal_inf, magnitudes, obs, s = discover_how(env, action6, obs, clusters,
                                                    HOW_STEPS, rng)
    total_steps += s

    if obs and obs.levels_completed > max_levels:
        max_levels = obs.levels_completed

    # ACT phase
    act_rounds = 0
    while total_steps < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if not clusters:
            break

        # Sort zones by causal influence (descending)
        if causal_inf:
            order = sorted(range(len(clusters)), key=lambda i: causal_inf[i], reverse=True)
        else:
            order = list(range(len(clusters)))

        # Execute in causal order
        for zi in order:
            if total_steps >= MAX_STEPS or time.time() - t0 >= TIME_CAP:
                break
            cl = clusters[zi]

            for _ in range(ACT_BURST):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset(); continue
                if obs.state == GameState.WIN: break
                if not obs.frame or len(obs.frame) == 0:
                    obs = env.reset(); continue

                frame_before = obs.frame
                levels_before = obs.levels_completed
                obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
                total_steps += 1

                if obs is None: break

                # Level transition detection
                if obs.levels_completed > levels_before:
                    if obs.levels_completed > max_levels:
                        max_levels = obs.levels_completed
                        level_steps[max_levels] = total_steps

                    # Re-discover for new level
                    clusters, obs, s = discover_where(env, action6, obs,
                                                       REDISCOVER_WHERE, rng)
                    total_steps += s
                    causal_inf, magnitudes, obs, s = discover_how(
                        env, action6, obs, clusters, REDISCOVER_HOW, rng)
                    total_steps += s
                    break  # restart ACT with new zones

                # Check if click had effect
                if obs.frame and len(obs.frame) > 0:
                    diff = frame_diff_magnitude(frame_before, obs.frame)
                    if diff > RECLICK_THRESHOLD:
                        break  # significant change, move to next zone
            else:
                continue
            break  # re-enter ACT loop with potentially new zones

        act_rounds += 1
        if act_rounds > 200:  # safety cap
            break

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'max_levels': max_levels,
        'level_steps': level_steps,
        'n_zones_final': len(clusters),
        'total_steps': total_steps,
        'act_rounds': act_rounds,
        'elapsed': round(elapsed, 1)
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if not vc33:
        print("SKIP: VC33 not found"); return

    print("=== Step 1029: VC33 Causal-Guided Action for L2+ ===")
    print(f"WHERE={WHERE_STEPS}, HOW={HOW_STEPS}")
    print(f"Rediscover: WHERE={REDISCOVER_WHERE}, HOW={REDISCOVER_HOW}")
    print(f"Game: {vc33.game_id}\n")

    results = []
    for seed in range(5):
        r = run_seed(arc, vc33.game_id, seed)
        print(f"  s{seed}: L={r['max_levels']}  steps={r['total_steps']}  "
              f"zones={r['n_zones_final']}  rounds={r['act_rounds']}  "
              f"{r['elapsed']}s")
        if r['level_steps']:
            for lvl, step in sorted(r['level_steps'].items()):
                print(f"    L{lvl} @ step {step}")
        results.append(r)

    l1 = sum(1 for r in results if r['max_levels'] >= 1)
    l2 = sum(1 for r in results if r['max_levels'] >= 2)
    l3 = sum(1 for r in results if r['max_levels'] >= 3)
    max_l = max(r['max_levels'] for r in results)
    print(f"\n  VC33: {l1}/5 L1, {l2}/5 L2, {l3}/5 L3+, max_level={max_l}")
    print(f"  Comparison: 1023 mode-map-only = 5/5 L1, 0/5 L2+")
    if l2 > 0:
        print(f"  SIGNAL: Causal-guided action produces L2+ progression!")
    elif l1 < 5:
        print(f"  REGRESSION: L1 < 5/5 — causal-guided action hurts basic discovery")
    else:
        print(f"  NO L2 PROGRESS: causal ordering doesn't help beyond mode map")

    print("\nStep 1029 DONE")


if __name__ == "__main__":
    main()
