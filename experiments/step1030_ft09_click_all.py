"""
Step 1030 — FT09: Click-All-Interactive for L2+

D2-grounded experiment: WHERE → HOW → ACT pipeline for FT09.
R3 hypothesis: FT09 L1-L4 are independent wall toggles. Clicking ALL interactive
zones (magnitude > threshold) solves the level. Repeated sweeps handle multi-click
walls (3-color cycle, L4). On level transition, re-discover zones.

Method:
  Phase 1 (WHERE+HOW, 3000 steps): Mode map → zones. Click each → magnitude.
    Threshold mag > 0.1 = interactive.
  Phase 2 (ACT): Click ALL interactive zones once. Check level transition.
    If no transition, sweep again (some walls need 2+ clicks). Repeat.
  On level transition: re-run WHERE+HOW for new level layout.

Kill: L1 < 5/5 (regression).
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
WHERE_STEPS = 3000
HOW_CLICKS = 5         # clicks per zone to measure magnitude
REDISCOVER_STEPS = 1500
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 120
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
MAG_THRESHOLD = 0.1    # minimum response magnitude to count as interactive
MAX_SWEEPS = 20        # max full sweeps before giving up on a level

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

ZONE_RADIUS = 4

def frame_diff_magnitude(frame_before, frame_after):
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    return float(np.abs(b - a).mean())

def local_zone_change(frame_before, frame_after, cx, cy, radius=ZONE_RADIUS):
    """Measure pixel change LOCAL to a zone (not global frame)."""
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    return float(np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1]).mean())


# ─── Discovery ───
def discover(env, action6, obs, n_where_steps, rng):
    """Run WHERE + HOW discovery. Returns (interactive_zones, all_zones, obs, steps)."""
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    steps = 0

    # WHERE phase
    for _ in range(n_where_steps):
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

    # HOW phase: measure response magnitude per zone
    magnitudes = [0.0] * len(clusters)
    for zi, cl in enumerate(clusters):
        for _ in range(HOW_CLICKS):
            if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset()
                break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            frame_before = obs.frame
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            steps += 1
            if obs and obs.frame and len(obs.frame) > 0:
                mag = local_zone_change(frame_before, obs.frame,
                                         cl['cx_int'], cl['cy_int'])
                magnitudes[zi] = max(magnitudes[zi], mag)

    # Filter interactive zones
    interactive = [(zi, clusters[zi]) for zi in range(len(clusters))
                   if magnitudes[zi] > MAG_THRESHOLD]

    return interactive, clusters, magnitudes, obs, steps


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
    level_steps = {}

    # Initial discovery
    interactive, all_zones, mags, obs, s = discover(env, action6, obs, WHERE_STEPS, rng)
    total_steps += s

    if obs and obs.levels_completed > max_levels:
        max_levels = obs.levels_completed

    # ACT phase: sweep all interactive zones
    sweep_count = 0
    while total_steps < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if not interactive:
            break

        sweep_count += 1
        if sweep_count > MAX_SWEEPS:
            # Re-discover (might be stuck on wrong zones)
            interactive, all_zones, mags, obs, s = discover(
                env, action6, obs, REDISCOVER_STEPS, rng)
            total_steps += s
            sweep_count = 0
            continue

        # Click all interactive zones
        level_changed = False
        for zi, cl in interactive:
            if total_steps >= MAX_STEPS or time.time() - t0 >= TIME_CAP:
                break
            if obs is None or obs.state == GameState.GAME_OVER:
                obs = env.reset(); continue
            if obs.state == GameState.WIN: break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            levels_before = obs.levels_completed
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            total_steps += 1

            if obs is None: break

            if obs.levels_completed > levels_before:
                if obs.levels_completed > max_levels:
                    max_levels = obs.levels_completed
                    level_steps[max_levels] = total_steps
                level_changed = True

                # Re-discover for new level
                interactive, all_zones, mags, obs, s = discover(
                    env, action6, obs, REDISCOVER_STEPS, rng)
                total_steps += s
                sweep_count = 0
                break

        if obs and obs.state == GameState.WIN:
            break

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'max_levels': max_levels,
        'level_steps': level_steps,
        'n_interactive': len(interactive),
        'n_total_zones': len(all_zones),
        'total_steps': total_steps,
        'elapsed': round(elapsed, 1)
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if not ft09:
        print("SKIP: FT09 not found"); return

    print("=== Step 1030: FT09 Click-All-Interactive for L2+ ===")
    print(f"WHERE={WHERE_STEPS}, HOW_CLICKS={HOW_CLICKS}")
    print(f"MAG_THRESHOLD={MAG_THRESHOLD}, MAX_SWEEPS={MAX_SWEEPS}")
    print(f"Game: {ft09.game_id}\n")

    results = []
    for seed in range(5):
        r = run_seed(arc, ft09.game_id, seed)
        print(f"  s{seed}: L={r['max_levels']}  interactive={r['n_interactive']}/"
              f"{r['n_total_zones']}  steps={r['total_steps']}  {r['elapsed']}s")
        if r['level_steps']:
            for lvl, step in sorted(r['level_steps'].items()):
                print(f"    L{lvl} @ step {step}")
        results.append(r)

    l1 = sum(1 for r in results if r['max_levels'] >= 1)
    l2 = sum(1 for r in results if r['max_levels'] >= 2)
    l3 = sum(1 for r in results if r['max_levels'] >= 3)
    max_l = max(r['max_levels'] for r in results)
    print(f"\n  FT09: {l1}/5 L1, {l2}/5 L2, {l3}/5 L3+, max_level={max_l}")
    print(f"  Comparison: 1023b mode-map-only = 5/5 L1, 0/5 L2+")
    if l2 > 0:
        print(f"  SIGNAL: Click-all-interactive produces L2+ progression!")
    elif l1 < 5:
        print(f"  REGRESSION: L1 < 5/5")
    else:
        print(f"  NO L2 PROGRESS")

    print("\nStep 1030 DONE")


if __name__ == "__main__":
    main()
