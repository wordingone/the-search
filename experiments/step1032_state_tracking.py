"""
Step 1032 — State-Aware Action: Current vs Goal Detection

D2-grounded experiment: WHERE → HOW → STATE → ACT pipeline.
R3 hypothesis: The substrate can infer which zones are "wrong" by comparing
each zone's local appearance to a majority-vote target, and selectively
click only mismatched zones. This prevents toggle oscillation (1030's failure).

Method:
  1. WHERE+HOW (3000 steps): Mode map + magnitude → interactive zones
  2. STATE READ: Extract each zone's local color (mode of center pixels)
  3. GOAL INFERENCE: Majority vote across interactive zones = target color
  4. ACT: Click only zones whose color ≠ target. Re-read after each click.
  5. On level transition: re-discover (new level = new layout).

Both games, 5 seeds. One config.
Kill: L1 < 3/5 on either game or L2 = 0/5 on both.
Success: L2 on either game = first L2+ from discovered interaction.
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
MAX_ACT_ROUNDS = 100

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
    """Read the dominant color in a small patch around zone center."""
    arr = np.array(frame[0], dtype=np.int32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    patch = arr[y0:y1, x0:x1].flatten()
    if len(patch) == 0:
        return 0
    counts = Counter(patch.tolist())
    return counts.most_common(1)[0][0]


# ─── Discovery ───
def discover(env, action6, obs, n_steps, rng):
    """WHERE+HOW discovery. Returns interactive zones + all zones."""
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

    # HOW: measure local magnitude
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


# ─── State-aware action ───
def infer_target_and_act(env, action6, obs, interactive, clusters, rng,
                          budget_steps, time_limit):
    """Read zone states, infer target by majority vote, click mismatched zones."""
    steps = 0
    max_levels = obs.levels_completed if obs else 0
    level_steps = {}
    rediscover_needed = False
    act_rounds = 0
    n_targeted_clicks = 0
    n_wasted_clicks = 0

    while steps < budget_steps and time.time() < time_limit:
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue
        if not interactive:
            break

        act_rounds += 1
        if act_rounds > MAX_ACT_ROUNDS:
            rediscover_needed = True
            break

        # STATE READ: get current color of each interactive zone
        zone_colors = {}
        for zi, cl in interactive:
            zone_colors[zi] = read_zone_color(obs.frame, cl['cx_int'], cl['cy_int'])

        # GOAL INFERENCE: majority vote
        color_counts = Counter(zone_colors.values())
        target_color = color_counts.most_common(1)[0][0]

        # Find mismatched zones
        mismatched = [(zi, cl) for zi, cl in interactive
                      if zone_colors.get(zi, target_color) != target_color]

        if not mismatched:
            # All zones match target — try clicking ALL once more (maybe target is wrong)
            # Or try the SECOND most common as target
            if len(color_counts) > 1:
                alt_target = color_counts.most_common(2)[1][0]
                mismatched = [(zi, cl) for zi, cl in interactive
                              if zone_colors.get(zi, alt_target) != alt_target]
            if not mismatched:
                # All uniform — done with this level, but no transition
                # Try clicking the most causally influential zone once
                if interactive:
                    zi, cl = interactive[0]
                    if obs.frame:
                        levels_before = obs.levels_completed
                        obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
                        steps += 1
                        n_wasted_clicks += 1
                        if obs and obs.levels_completed > levels_before:
                            if obs.levels_completed > max_levels:
                                max_levels = obs.levels_completed
                                level_steps[max_levels] = steps
                            rediscover_needed = True
                            break
                continue

        # ACT: click only mismatched zones
        for zi, cl in mismatched:
            if steps >= budget_steps or time.time() >= time_limit:
                break
            if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset()
                break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            levels_before = obs.levels_completed
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            steps += 1
            n_targeted_clicks += 1

            if obs is None: break
            if obs.levels_completed > levels_before:
                if obs.levels_completed > max_levels:
                    max_levels = obs.levels_completed
                    level_steps[max_levels] = steps
                rediscover_needed = True
                break

        if rediscover_needed:
            break

    return obs, steps, max_levels, level_steps, rediscover_needed, {
        'act_rounds': act_rounds,
        'targeted_clicks': n_targeted_clicks,
        'wasted_clicks': n_wasted_clicks
    }


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
    all_stats = []

    while total_steps < MAX_STEPS and time.time() - t0 < TIME_CAP:
        # Discover
        n_disc = WHERE_STEPS if total_steps == 0 else REDISCOVER_STEPS
        interactive, clusters, mags, obs, s = discover(env, action6, obs, n_disc, rng)
        total_steps += s

        if obs and obs.levels_completed > max_levels:
            max_levels = obs.levels_completed

        # State-aware action
        budget = min(10000, MAX_STEPS - total_steps)
        obs, s, ml, ls, rediscover, stats = infer_target_and_act(
            env, action6, obs, interactive, clusters, rng,
            budget, t0 + TIME_CAP)
        total_steps += s
        if ml > max_levels:
            max_levels = ml
        level_steps.update(ls)
        all_stats.append(stats)

        if not rediscover:
            break  # no level transition and no more to try

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'max_levels': max_levels,
        'level_steps': level_steps,
        'n_interactive': len(interactive),
        'n_total_zones': len(clusters),
        'total_steps': total_steps,
        'elapsed': round(elapsed, 1),
        'stats': all_stats
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()

    games = {}
    for e in envs:
        if 'vc33' in e.game_id.lower():
            games['VC33'] = e.game_id
        elif 'ft09' in e.game_id.lower():
            games['FT09'] = e.game_id

    print("=== Step 1032: State-Aware Action (Current vs Goal) ===")
    print(f"WHERE={WHERE_STEPS}, HOW_CLICKS={HOW_CLICKS}")
    print(f"State tracking: majority-vote goal inference, selective clicking")
    print(f"MAG_THRESHOLD={MAG_THRESHOLD}\n")

    for game_name in ['VC33', 'FT09']:
        game_id = games.get(game_name)
        if not game_id:
            print(f"SKIP {game_name}"); continue

        print(f"--- {game_name} ({game_id}) ---")
        results = []
        for seed in range(5):
            r = run_seed(arc, game_id, seed)
            stats_str = ""
            if r['stats']:
                s = r['stats'][0]
                stats_str = f"  targeted={s['targeted_clicks']}  wasted={s['wasted_clicks']}"
            print(f"  s{seed}: L={r['max_levels']}  interactive={r['n_interactive']}/"
                  f"{r['n_total_zones']}  steps={r['total_steps']}{stats_str}  "
                  f"{r['elapsed']}s")
            if r['level_steps']:
                for lvl, step in sorted(r['level_steps'].items()):
                    print(f"    L{lvl} @ step {step}")
            results.append(r)

        l1 = sum(1 for r in results if r['max_levels'] >= 1)
        l2 = sum(1 for r in results if r['max_levels'] >= 2)
        l3 = sum(1 for r in results if r['max_levels'] >= 3)
        max_l = max(r['max_levels'] for r in results)
        print(f"\n  {game_name}: {l1}/5 L1, {l2}/5 L2, {l3}/5 L3+, max_level={max_l}")
        if l2 > 0:
            print(f"  SIGNAL: State-aware action produces L2+!")
        print()

    print("Step 1032 DONE")


if __name__ == "__main__":
    main()
