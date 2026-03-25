"""
Step 1026 — Response Classification (HOW Discovery)

D2-grounded experiment: WHERE → HOW pipeline.
R3 hypothesis: Frame-diff clustering after systematic zone probing reveals
distinct mechanical response classes. Zones with similar diff patterns share
the same game mechanic. This is HOW discovery — mapping WHERE targets to
WHAT HAPPENS when you click them.

Method:
  Phase 1 (WHERE, 5000 steps): Random clicking → mode map → isolated CC → zones.
  Phase 2 (HOW probe, 20 clicks per zone): Click each zone 20x. Record frame diff
    (current_frame - previous_frame) after each click. 20 diffs per zone.
  Phase 3 (Classify): Compute mean diff pattern per zone. Cluster zones by cosine
    similarity of their mean diff. Report: n_clusters, zone-to-cluster map, per-cluster
    diff magnitude.

Games: VC33 + FT09 (click games). LS20 excluded — directional actions, no click targets.
Seeds: 5 per game.

Kill: <2 distinct clusters on both games → zones are mechanically identical, HOW = trivial.
Signal: ≥3 distinct clusters → game has multiple mechanics discoverable from interaction.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import label as ndlabel
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
MODE_WARMUP = 5000
HOW_CLICKS = 20       # clicks per zone in HOW phase
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 90
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
COSINE_THRESHOLD = 0.3  # zones with cosine distance < this are same cluster

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

# ─── HOW discovery ───
def compute_frame_diff(frame_before, frame_after):
    """Compute per-pixel absolute diff between two frames → flat 64x64 array."""
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    return (b - a).flatten()  # 4096D, signed diff preserves direction

def cluster_by_cosine(mean_diffs, threshold=COSINE_THRESHOLD):
    """Single-linkage clustering of zone mean-diff vectors by cosine distance."""
    n = len(mean_diffs)
    if n <= 1:
        return list(range(n)), 1

    # Compute pairwise cosine distance
    norms = [np.linalg.norm(d) for d in mean_diffs]
    labels = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] < 1e-8 or norms[j] < 1e-8:
                # Zero-diff zones: group together
                if norms[i] < 1e-8 and norms[j] < 1e-8:
                    _merge(labels, i, j)
                continue
            cos_sim = np.dot(mean_diffs[i], mean_diffs[j]) / (norms[i] * norms[j])
            cos_dist = 1.0 - cos_sim
            if cos_dist < threshold:
                _merge(labels, i, j)

    # Renumber clusters
    unique = sorted(set(_root(labels, i) for i in range(n)))
    mapping = {u: idx for idx, u in enumerate(unique)}
    result = [mapping[_root(labels, i)] for i in range(n)]
    return result, len(unique)

def _root(labels, i):
    while labels[i] != i:
        labels[i] = labels[labels[i]]
        i = labels[i]
    return i

def _merge(labels, i, j):
    ri, rj = _root(labels, i), _root(labels, j)
    if ri != rj:
        labels[rj] = ri


# ─── Run ───
def run_seed(arc, game_id, seed):
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]  # ACTION6 = click action
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    obs = env.reset()
    steps = 0
    t0 = time.time()
    levels = 0

    # Phase 1: WHERE (mode map warmup)
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

    if not clusters:
        return {
            'seed': seed, 'n_zones': 0, 'n_clusters': 0,
            'zones': [], 'cluster_labels': [], 'cluster_magnitudes': [],
            'levels': levels, 'steps': steps,
            'elapsed': round(time.time() - t0, 1)
        }

    # Phase 2: HOW probe — click each zone 20x, record frame diffs
    zone_diffs = {i: [] for i in range(len(clusters))}

    for zi, cl in enumerate(clusters):
        for click_idx in range(HOW_CLICKS):
            if time.time() - t0 > TIME_CAP: break
            if obs is None or obs.state == GameState.GAME_OVER:
                obs = env.reset(); continue
            if obs.state == GameState.WIN: break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            frame_before = obs.frame
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            steps += 1

            if obs is not None and obs.frame and len(obs.frame) > 0:
                diff = compute_frame_diff(frame_before, obs.frame)
                zone_diffs[zi].append(diff)
                if obs.levels_completed > levels:
                    levels = obs.levels_completed

    # Phase 3: Classify — mean diff per zone, cluster by cosine similarity
    mean_diffs = []
    zone_magnitudes = []
    for zi in range(len(clusters)):
        if zone_diffs[zi]:
            md = np.mean(zone_diffs[zi], axis=0)
            mean_diffs.append(md)
            zone_magnitudes.append(float(np.linalg.norm(md)))
        else:
            mean_diffs.append(np.zeros(4096))
            zone_magnitudes.append(0.0)

    cluster_labels, n_clusters = cluster_by_cosine(mean_diffs)

    # Per-cluster aggregate magnitude
    cluster_mags = {}
    for zi, label in enumerate(cluster_labels):
        if label not in cluster_mags:
            cluster_mags[label] = []
        cluster_mags[label].append(zone_magnitudes[zi])
    cluster_magnitudes = {k: round(np.mean(v), 3) for k, v in sorted(cluster_mags.items())}

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'n_zones': len(clusters),
        'n_clusters': n_clusters,
        'zones': [(cl['cx_int'], cl['cy_int'], cl['color'], cl['size']) for cl in clusters],
        'cluster_labels': cluster_labels,
        'cluster_magnitudes': cluster_magnitudes,
        'zone_magnitudes': [round(m, 3) for m in zone_magnitudes],
        'levels': levels,
        'steps': steps,
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

    print("=== Step 1026: Response Classification (HOW Discovery) ===")
    print(f"WHERE={MODE_WARMUP} steps, HOW={HOW_CLICKS} clicks/zone")
    print(f"Cosine cluster threshold={COSINE_THRESHOLD}\n")

    for game_name in ['VC33', 'FT09']:
        game_id = games.get(game_name)
        if not game_id:
            print(f"SKIP {game_name}: not found"); continue

        print(f"--- {game_name} ({game_id}) ---")
        all_results = []
        for seed in range(5):
            r = run_seed(arc, game_id, seed)
            print(f"  s{seed}: zones={r['n_zones']}  mech_clusters={r['n_clusters']}  "
                  f"L={r['levels']}  {r['elapsed']}s")
            if r['zones']:
                for zi, (x, y, col, sz) in enumerate(r['zones']):
                    label = r['cluster_labels'][zi] if zi < len(r['cluster_labels']) else '?'
                    mag = r['zone_magnitudes'][zi] if zi < len(r['zone_magnitudes']) else 0
                    print(f"    zone{zi}: ({x},{y}) c={col} sz={sz}  "
                          f"cluster={label}  mag={mag:.3f}")
                print(f"    cluster_magnitudes: {r['cluster_magnitudes']}")
            all_results.append(r)

        # Summary
        avg_zones = np.mean([r['n_zones'] for r in all_results])
        avg_clusters = np.mean([r['n_clusters'] for r in all_results])
        max_clusters = max(r['n_clusters'] for r in all_results)
        print(f"  {game_name} summary: avg_zones={avg_zones:.1f}  "
              f"avg_mech_clusters={avg_clusters:.1f}  max_clusters={max_clusters}")
        if max_clusters >= 3:
            print(f"  SIGNAL: ≥3 distinct mechanical response classes discovered")
        elif max_clusters >= 2:
            print(f"  WEAK SIGNAL: 2 response classes (binary mechanic)")
        else:
            print(f"  NO SIGNAL: all zones respond identically")
        print()

    print("Step 1026 DONE")


if __name__ == "__main__":
    main()
