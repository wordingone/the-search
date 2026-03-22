"""
Tier 1b (d=256, 2 signal dims, 4 clusters) + Robustness (d=32, 5 seeds).
Both synthetic. Should complete in <5s.
"""
import sys, time
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal import TemporalPrediction


SIGNAL_DIMS = [3, 147]


def run_discrimination(d, n_clusters, warm_steps, signal_dims=None):
    """Cyclic cluster presentation. Returns (avg_dominance, n_distinct_actions)."""
    s = TemporalPrediction(d=d, n_actions=4)

    if signal_dims is not None:
        # Sparse signal: clusters separated only in signal_dims
        centers = []
        offsets = [(2.0, 2.0), (2.0, -2.0), (-2.0, 2.0), (-2.0, -2.0)]
        for sd0, sd1 in offsets[:n_clusters]:
            c = torch.zeros(d)
            c[signal_dims[0]] = sd0
            c[signal_dims[1]] = sd1
            centers.append(c)
    else:
        # Dense: random well-separated clusters
        centers = [torch.randn(d) * 3 for _ in range(n_clusters)]

    # Warmup: cyclic presentation
    k = 0
    for _ in range(warm_steps):
        c = centers[k % n_clusters]
        s.step(c + 0.3 * torch.randn(d))
        k += 1

    # Test: 100 rounds per cluster
    results = {i: [] for i in range(n_clusters)}
    for _ in range(100):
        for i, c in enumerate(centers):
            x = c + 0.3 * torch.randn(d)
            a = s.step(x)
            results[i].append(a)

    fracs = []
    dom_actions = []
    for i in range(n_clusters):
        counts = Counter(results[i])
        dom = counts.most_common(1)[0]
        fracs.append(dom[1] / 100)
        dom_actions.append(dom[0])

    avg = float(np.mean(fracs))
    n_distinct = len(set(dom_actions))
    return avg, n_distinct


def main():
    t0 = time.time()

    print("=" * 55)
    print("TemporalPrediction: Synthetic Benchmarks")
    print("=" * 55)

    # Tier 1b: d=256, 2 signal dims, 4 clusters
    print("\nTier 1b: d=256, 2 signal dims, 4 clusters")
    print("  (SelfRef baseline: ~94% avg dom, 4/4 distinct)")
    print("  (Prediction: 40-60%)")
    avg, n_distinct = run_discrimination(256, 4, warm_steps=800, signal_dims=SIGNAL_DIMS)
    status = "PASS" if avg > 0.6 and n_distinct >= 2 else ("DEGENERATE" if n_distinct <= 1 else "MARGINAL")
    print(f"  avg_dominance={avg*100:.1f}%  distinct_actions={n_distinct}/4  [{status}]")

    # Robustness: d=32, 5 seeds
    print("\nRobustness: d=32, dense, 4 clusters, 5 seeds")
    print("  (Prediction: dom 85-97%, distinct 2-4)")
    dom_vals = []
    dist_vals = []
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        avg_s, n_s = run_discrimination(32, 4, warm_steps=200)
        dom_vals.append(avg_s * 100)
        dist_vals.append(n_s)
        print(f"  seed={seed}: dom={avg_s*100:.1f}%  distinct={n_s}/4")

    print(f"  Summary: dom min={min(dom_vals):.1f}% max={max(dom_vals):.1f}% mean={np.mean(dom_vals):.1f}%")
    print(f"           distinct min={min(dist_vals)} max={max(dist_vals)} mean={np.mean(dist_vals):.1f}")

    elapsed = time.time() - t0
    print(f"\nelapsed={elapsed:.1f}s")


if __name__ == '__main__':
    main()
