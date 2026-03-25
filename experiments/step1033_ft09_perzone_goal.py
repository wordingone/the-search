"""
Step 1033 — FT09: Per-Zone Goal Inference via Trial-and-Error

D2-grounded: WHERE → HOW → STATE → per-zone GOAL → ACT.
R3 hypothesis: The substrate discovers each wall's correct state by
clicking it and measuring whether the game state improves (frame becomes
more "solved-looking") or worsens. Click → measure → undo if bad.

Method:
  1. WHERE+HOW (3000 steps): Mode map + local magnitude → interactive zones
  2. TRIAL phase: For each interactive zone:
     a. Record frame BEFORE click
     b. Click the zone
     c. Record frame AFTER click
     d. Click again to UNDO (toggle back) — FT09 walls cycle, 2 clicks = undo for 2-color
     e. Measure: did the click make other zones' local state closer to uniformity?
        - Score = count of zone pairs with same color AFTER click minus BEFORE click
        - If score > 0: clicking this zone is PRODUCTIVE (brings game closer to solved)
        - If score < 0: this zone is already correct, don't click it
     f. Record: zone_i → {should_click: bool}
  3. ACT: Click only zones marked should_click=True. One sweep.
  4. On level transition: re-discover + re-trial.

FT09, 5 seeds, 120s cap.
Kill: L1 < 5/5 or L2 = 0/5.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import label as ndlabel
from collections import Counter

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
WHERE_STEPS = 3000
HOW_CLICKS = 5
REDISCOVER_STEPS = 1500
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 120
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
ZONE_RADIUS = 4
MAG_THRESHOLD = 0.1
MAX_ACT_SWEEPS = 10

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

def local_zone_change(frame_before, frame_after, cx, cy, radius=ZONE_RADIUS):
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    return float(np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1]).mean())

def read_zone_color(frame, cx, cy, radius=2):
    arr = np.array(frame[0], dtype=np.int32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    patch = arr[y0:y1, x0:x1].flatten()
    if len(patch) == 0: return 0
    return Counter(patch.tolist()).most_common(1)[0][0]

def uniformity_score(frame, zones):
    """Count zone pairs with matching colors = how 'solved' the layout looks."""
    colors = [read_zone_color(frame, cl['cx_int'], cl['cy_int']) for _, cl in zones]
    score = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if colors[i] == colors[j]:
                score += 1
    return score


# ─── Discovery ───
def discover(env, action6, obs, n_steps, rng):
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
                mag = local_zone_change(frame_before, obs.frame, cl['cx_int'], cl['cy_int'])
                magnitudes[zi] = max(magnitudes[zi], mag)

    interactive = [(zi, clusters[zi]) for zi in range(len(clusters))
                   if magnitudes[zi] > MAG_THRESHOLD]
    return interactive, clusters, magnitudes, obs, steps


# ─── Trial-and-error goal inference ───
def trial_phase(env, action6, obs, interactive, clusters):
    """For each interactive zone: click → measure uniformity change → undo (click again)."""
    should_click = {}
    steps = 0

    for zi, cl in interactive:
        if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
            if obs is None or obs.state == GameState.GAME_OVER:
                obs = env.reset()
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        # Measure uniformity BEFORE
        score_before = uniformity_score(obs.frame, interactive)

        # Click
        obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
        steps += 1
        if obs is None or not obs.frame: break

        # Measure uniformity AFTER
        score_after = uniformity_score(obs.frame, interactive)

        # Undo: click again (toggle back for 2-color walls)
        obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
        steps += 1

        # Decision: was clicking productive?
        delta = score_after - score_before
        should_click[zi] = (delta > 0)  # clicking improved uniformity

    return should_click, obs, steps


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

    while total_steps < MAX_STEPS and time.time() - t0 < TIME_CAP:
        # Discover
        n_disc = WHERE_STEPS if total_steps == 0 else REDISCOVER_STEPS
        interactive, clusters, mags, obs, s = discover(env, action6, obs, n_disc, rng)
        total_steps += s
        if obs and obs.levels_completed > max_levels:
            max_levels = obs.levels_completed

        if not interactive:
            break

        # Trial: which zones should be clicked?
        should_click, obs, s = trial_phase(env, action6, obs, interactive, clusters)
        total_steps += s

        to_click = [(zi, cl) for zi, cl in interactive if should_click.get(zi, False)]
        n_productive = len(to_click)

        if obs and obs.levels_completed > max_levels:
            max_levels = obs.levels_completed
            level_steps[max_levels] = total_steps

        # ACT: click the productive zones
        level_gained = False
        for sweep in range(MAX_ACT_SWEEPS):
            if not to_click: break
            for zi, cl in to_click:
                if total_steps >= MAX_STEPS or time.time() - t0 >= TIME_CAP: break
                if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                    if obs is None or obs.state == GameState.GAME_OVER:
                        obs = env.reset()
                    break
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
                    level_gained = True
                    break
            if level_gained: break

        if not level_gained:
            # Try clicking ALL interactive (fallback)
            for zi, cl in interactive:
                if total_steps >= MAX_STEPS or time.time() - t0 >= TIME_CAP: break
                if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN): break
                if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
                levels_before = obs.levels_completed
                obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
                total_steps += 1
                if obs and obs.levels_completed > levels_before:
                    if obs.levels_completed > max_levels:
                        max_levels = obs.levels_completed
                        level_steps[max_levels] = total_steps
                    level_gained = True
                    break
            if not level_gained:
                break  # stuck

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'max_levels': max_levels,
        'level_steps': level_steps,
        'n_interactive': len(interactive),
        'total_steps': total_steps,
        'elapsed': round(elapsed, 1)
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if not ft09:
        print("SKIP"); return

    print("=== Step 1033: FT09 Per-Zone Goal Inference (Trial-and-Error) ===")
    print(f"WHERE={WHERE_STEPS}, HOW_CLICKS={HOW_CLICKS}")
    print(f"Trial: click → uniformity score → undo → decide")
    print(f"Game: {ft09.game_id}\n")

    results = []
    for seed in range(5):
        r = run_seed(arc, ft09.game_id, seed)
        print(f"  s{seed}: L={r['max_levels']}  interactive={r['n_interactive']}  "
              f"steps={r['total_steps']}  {r['elapsed']}s")
        if r['level_steps']:
            for lvl, step in sorted(r['level_steps'].items()):
                print(f"    L{lvl} @ step {step}")
        results.append(r)

    l1 = sum(1 for r in results if r['max_levels'] >= 1)
    l2 = sum(1 for r in results if r['max_levels'] >= 2)
    l3 = sum(1 for r in results if r['max_levels'] >= 3)
    max_l = max(r['max_levels'] for r in results)
    print(f"\n  FT09: {l1}/5 L1, {l2}/5 L2, {l3}/5 L3+, max_level={max_l}")
    if l2 > 0:
        print(f"  SIGNAL: Per-zone goal inference produces L2+!")
    print("\nStep 1033 DONE")


if __name__ == "__main__":
    main()
