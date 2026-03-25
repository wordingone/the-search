"""
Step 1028 — Ordering Sensitivity (WHEN Discovery)

D2-grounded experiment: WHERE → HOW → WHEN pipeline.
R3 hypothesis: Click ordering matters for level progression. Some orderings of
discovered zones produce level transitions, others don't. This is WHEN discovery —
finding the temporal structure of game mechanics.

Method:
  Phase 1 (WHERE, 5000 steps): Random clicking → mode map → zones.
  Phase 2 (WHEN probe, 100 orderings): Generate 100 random permutations of
    discovered zones. For each ordering, execute the full sequence (click each zone
    BURST=3 times in order). Record: did a level transition occur? At which step?
  Phase 3 (Analysis): Compare success rate across orderings. If ordering matters,
    success rate should vary significantly across permutations.

Focus: VC33 L1→L2 contrast (L1 is order-free, L2+ may require ordering).
Also: FT09 (expected order-free for simple levels). 5 seeds.

Kill: All orderings succeed or fail identically → ordering is irrelevant, WHEN = trivial.
Signal: Success rate varies >30% across orderings → ordering is load-bearing.
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
N_ORDERINGS = 100
BURST = 3             # clicks per zone per ordering attempt
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 120
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]

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


# ─── Run ───
def run_seed(arc, game_id, seed):
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    obs = env.reset()
    steps = 0
    t0 = time.time()
    levels_at_start = 0

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
        if obs and obs.levels_completed > levels_at_start:
            levels_at_start = obs.levels_completed

    mode = compute_mode(freq)
    clusters = find_isolated_clusters(mode)
    n_zones = len(clusters)

    if n_zones < 2:
        return {
            'seed': seed, 'n_zones': n_zones,
            'n_orderings_tested': 0,
            'successes': 0, 'success_rate': 0.0,
            'success_orderings': [],
            'fail_orderings': [],
            'levels_reached': levels_at_start,
            'steps': steps, 'elapsed': round(time.time() - t0, 1)
        }

    # Phase 2: WHEN probe — try N_ORDERINGS random permutations
    successes = 0
    success_orderings = []
    fail_orderings = []
    orderings_tested = 0

    for oi in range(N_ORDERINGS):
        if time.time() - t0 > TIME_CAP: break

        # Reset to known state for fair comparison
        obs = env.reset()
        if obs is None: continue
        # Wait for valid frame
        reset_attempts = 0
        while (obs is None or obs.state == GameState.GAME_OVER or
               not obs.frame or len(obs.frame) == 0):
            obs = env.reset()
            reset_attempts += 1
            if reset_attempts > 5: break
        if obs is None or not obs.frame: continue

        levels_before = obs.levels_completed

        # Generate random permutation of zone indices
        perm = np.random.permutation(n_zones).tolist()

        # Execute: click each zone in order, BURST times
        level_gained = False
        for zi in perm:
            cl = clusters[zi]
            for _ in range(BURST):
                if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                    break
                if not obs.frame or len(obs.frame) == 0:
                    break
                obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
                steps += 1
                if obs and obs.levels_completed > levels_before:
                    level_gained = True
                    break
            if level_gained:
                break

        orderings_tested += 1
        if level_gained:
            successes += 1
            if len(success_orderings) < 5:
                success_orderings.append(perm)
        else:
            if len(fail_orderings) < 5:
                fail_orderings.append(perm)

    success_rate = successes / orderings_tested if orderings_tested > 0 else 0
    elapsed = time.time() - t0

    return {
        'seed': seed,
        'n_zones': n_zones,
        'n_orderings_tested': orderings_tested,
        'successes': successes,
        'success_rate': round(success_rate, 3),
        'success_orderings': success_orderings[:3],  # first 3 for inspection
        'fail_orderings': fail_orderings[:3],
        'levels_reached': max(levels_at_start, levels_before if 'levels_before' in dir() else 0),
        'steps': steps,
        'elapsed': round(elapsed, 1),
        'zones': [(cl['cx_int'], cl['cy_int'], cl['color'], cl['size']) for cl in clusters]
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

    print("=== Step 1028: Ordering Sensitivity (WHEN Discovery) ===")
    print(f"WHERE={MODE_WARMUP} steps, ORDERINGS={N_ORDERINGS}, BURST={BURST}")
    print(f"Testing: does click ordering affect level progression?\n")

    for game_name in ['VC33', 'FT09']:
        game_id = games.get(game_name)
        if not game_id:
            print(f"SKIP {game_name}: not found"); continue

        print(f"--- {game_name} ({game_id}) ---")
        all_results = []
        for seed in range(5):
            r = run_seed(arc, game_id, seed)
            print(f"  s{seed}: zones={r['n_zones']}  tested={r['n_orderings_tested']}  "
                  f"success={r['successes']}/{r['n_orderings_tested']}  "
                  f"rate={r['success_rate']}  {r['elapsed']}s")
            if r['success_orderings']:
                print(f"    example success order: {r['success_orderings'][0]}")
            if r['fail_orderings']:
                print(f"    example fail order:    {r['fail_orderings'][0]}")
            all_results.append(r)

        # Summary
        rates = [r['success_rate'] for r in all_results if r['n_orderings_tested'] > 0]
        if rates:
            avg_rate = np.mean(rates)
            std_rate = np.std(rates)
            print(f"\n  {game_name} summary: avg_success_rate={avg_rate:.3f} "
                  f"±{std_rate:.3f}")
            if avg_rate > 0.9:
                print(f"  ORDERING IRRELEVANT: >90% orderings succeed → WHEN is trivial")
            elif avg_rate < 0.1:
                print(f"  ORDERING CRITICAL: <10% orderings succeed → strong WHEN dependence")
            elif std_rate > 0.15:
                print(f"  SIGNAL: Variable success rate across seeds → ordering interacts with game state")
            else:
                print(f"  PARTIAL: {avg_rate:.0%} orderings succeed → moderate WHEN dependence")
        else:
            print(f"\n  {game_name}: no valid results")
        print()

    print("Step 1028 DONE")


if __name__ == "__main__":
    main()
