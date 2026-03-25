"""
Step 1031 — Unified WHERE→HOW→ACT Pipeline (Both Games)

D2-grounded experiment: one substrate, one config, discovers strategy from interaction.
R3 hypothesis: The substrate can detect game type from HOW data and select the
appropriate action strategy — hub-first (VC33-like) vs click-all (FT09-like).

Decision rule: if >80% of interactive zones have similar magnitude → click-all.
If magnitude varies widely (max/min > 5x) → hub-first ordering.

Both games, 5 seeds. One config. Spirit clause: one system discovers from interaction.

Success: L2+ on EITHER game with same substrate.
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

# ─── Constants (same for both games) ───
WHERE_STEPS = 4000
HOW_CLICKS = 10
REDISCOVER_WHERE = 1000
REDISCOVER_HOW = 500
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 120
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
ZONE_RADIUS = 4
MAG_THRESHOLD = 0.1
ACT_BURST = 5
MAX_SWEEPS = 20
MAG_RATIO_THRESHOLD = 5.0   # max/min ratio for hub-first vs click-all decision
SIMILAR_FRACTION = 0.8       # fraction of zones that must be similar for click-all

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

def frame_diff_magnitude(frame_before, frame_after):
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    return float(np.abs(b - a).mean())

def measure_zone_change(frame_before, frame_after, cx, cy, radius=ZONE_RADIUS):
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    return float(np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1]).mean())


# ─── Unified discovery ───
def discover(env, action6, obs, n_where, n_how_per_zone, rng):
    """Full WHERE+HOW discovery. Returns discovery result dict."""
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    steps = 0
    levels = obs.levels_completed if obs else 0

    # WHERE
    for _ in range(n_where):
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
        if obs and obs.levels_completed > levels:
            levels = obs.levels_completed

    mode = compute_mode(freq)
    clusters = find_isolated_clusters(mode)
    n = len(clusters)

    # HOW: magnitude + causal
    magnitudes = [0.0] * n
    causal_influence = [0.0] * n

    for zi in range(n):
        cl = clusters[zi]
        for _ in range(n_how_per_zone):
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
                mag = measure_zone_change(frame_before, obs.frame,
                                           cl['cx_int'], cl['cy_int'])
                magnitudes[zi] = max(magnitudes[zi], mag)
                # Causal: measure cross-zone effects
                off_diag = 0.0
                for zj in range(n):
                    if zj == zi: continue
                    off_diag += measure_zone_change(frame_before, obs.frame,
                                                     clusters[zj]['cx_int'],
                                                     clusters[zj]['cy_int'])
                causal_influence[zi] += off_diag
                if obs.levels_completed > levels:
                    levels = obs.levels_completed

    # Normalize causal influence
    causal_influence = [c / max(n_how_per_zone, 1) for c in causal_influence]

    # Classify interactive zones
    interactive_idx = [i for i in range(n) if magnitudes[i] > MAG_THRESHOLD]
    interactive_mags = [magnitudes[i] for i in interactive_idx]

    # Decide strategy
    strategy = 'click-all'  # default
    if len(interactive_mags) >= 2:
        max_m = max(interactive_mags)
        min_m = min(m for m in interactive_mags if m > 0)
        ratio = max_m / min_m if min_m > 0 else 999
        if ratio > MAG_RATIO_THRESHOLD:
            strategy = 'hub-first'

    return {
        'clusters': clusters,
        'magnitudes': magnitudes,
        'causal_influence': causal_influence,
        'interactive_idx': interactive_idx,
        'strategy': strategy,
        'obs': obs,
        'steps': steps,
        'levels': levels
    }


# ─── Act strategies ───
def act_hub_first(env, action6, obs, disc, rng, budget_steps, time_limit):
    """Hub-first: click zones in descending causal influence order."""
    clusters = disc['clusters']
    causal = disc['causal_influence']
    interactive = disc['interactive_idx']
    if not interactive:
        interactive = list(range(len(clusters)))

    order = sorted(interactive, key=lambda i: causal[i], reverse=True)
    steps = 0
    max_levels = disc['levels']
    level_steps = {}
    rediscovered = False

    rounds = 0
    while steps < budget_steps and time.time() < time_limit:
        rounds += 1
        if rounds > 200: break

        for zi in order:
            if steps >= budget_steps or time.time() >= time_limit: break
            cl = clusters[zi]
            for _ in range(ACT_BURST):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset(); continue
                if obs.state == GameState.WIN: return obs, steps, max_levels, level_steps
                if not obs.frame or len(obs.frame) == 0:
                    obs = env.reset(); continue

                levels_before = obs.levels_completed
                obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
                steps += 1

                if obs is None: break
                if obs.levels_completed > levels_before:
                    if obs.levels_completed > max_levels:
                        max_levels = obs.levels_completed
                        level_steps[max_levels] = steps
                    rediscovered = True
                    break
            if rediscovered:
                break
        if rediscovered:
            break

    return obs, steps, max_levels, level_steps


def act_click_all(env, action6, obs, disc, rng, budget_steps, time_limit):
    """Click-all: sweep all interactive zones, repeat."""
    clusters = disc['clusters']
    interactive = disc['interactive_idx']
    if not interactive:
        interactive = list(range(len(clusters)))

    steps = 0
    max_levels = disc['levels']
    level_steps = {}
    rediscovered = False

    for sweep in range(MAX_SWEEPS):
        if steps >= budget_steps or time.time() >= time_limit: break

        for zi in interactive:
            if steps >= budget_steps or time.time() >= time_limit: break
            cl = clusters[zi]
            if obs is None or obs.state == GameState.GAME_OVER:
                obs = env.reset(); continue
            if obs.state == GameState.WIN: return obs, steps, max_levels, level_steps
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            levels_before = obs.levels_completed
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            steps += 1

            if obs is None: break
            if obs.levels_completed > levels_before:
                if obs.levels_completed > max_levels:
                    max_levels = obs.levels_completed
                    level_steps[max_levels] = steps
                rediscovered = True
                break
        if rediscovered:
            break

    return obs, steps, max_levels, level_steps


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
    strategies_used = []

    while total_steps < MAX_STEPS and time.time() - t0 < TIME_CAP:
        # Discover
        n_where = WHERE_STEPS if total_steps == 0 else REDISCOVER_WHERE
        n_how = HOW_CLICKS if total_steps == 0 else max(3, REDISCOVER_HOW // max(1, len(strategies_used)))

        disc = discover(env, action6, obs, n_where, n_how, rng)
        obs = disc['obs']
        total_steps += disc['steps']
        if disc['levels'] > max_levels:
            max_levels = disc['levels']

        strategy = disc['strategy']
        strategies_used.append(strategy)

        # Act
        budget = min(10000, MAX_STEPS - total_steps)
        time_limit = t0 + TIME_CAP

        if strategy == 'hub-first':
            obs, s, ml, ls = act_hub_first(env, action6, obs, disc, rng, budget, time_limit)
        else:
            obs, s, ml, ls = act_click_all(env, action6, obs, disc, rng, budget, time_limit)

        total_steps += s
        if ml > max_levels:
            max_levels = ml
        level_steps.update(ls)

        # If no level gained in this cycle, break (avoid infinite loop)
        if not ls:
            break

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'max_levels': max_levels,
        'level_steps': level_steps,
        'strategies': strategies_used,
        'total_steps': total_steps,
        'elapsed': round(elapsed, 1)
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

    print("=== Step 1031: Unified WHERE→HOW→ACT Pipeline ===")
    print(f"WHERE={WHERE_STEPS}, HOW_CLICKS={HOW_CLICKS}")
    print(f"MAG_RATIO_THRESHOLD={MAG_RATIO_THRESHOLD} (hub-first vs click-all)")
    print(f"One substrate, one config, discovers strategy from interaction.\n")

    for game_name in ['VC33', 'FT09']:
        game_id = games.get(game_name)
        if not game_id:
            print(f"SKIP {game_name}"); continue

        print(f"--- {game_name} ({game_id}) ---")
        results = []
        for seed in range(5):
            r = run_seed(arc, game_id, seed)
            strat_str = '→'.join(r['strategies'][:3])
            print(f"  s{seed}: L={r['max_levels']}  strategy={strat_str}  "
                  f"steps={r['total_steps']}  {r['elapsed']}s")
            if r['level_steps']:
                for lvl, step in sorted(r['level_steps'].items()):
                    print(f"    L{lvl} @ step {step}")
            results.append(r)

        l1 = sum(1 for r in results if r['max_levels'] >= 1)
        l2 = sum(1 for r in results if r['max_levels'] >= 2)
        max_l = max(r['max_levels'] for r in results)
        print(f"\n  {game_name}: {l1}/5 L1, {l2}/5 L2+, max_level={max_l}")
        print()

    print("Step 1031 DONE")


if __name__ == "__main__":
    main()
