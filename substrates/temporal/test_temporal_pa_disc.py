"""Disc test (d=32, 4 clusters, 5 seeds) on TemporalPerAction. Leo predicts ~60-70%."""
import sys, time
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal_pa import TemporalPerAction
from temporal import TemporalPrediction


def run_disc(substrate_cls, seed, d=32, n_clusters=4, warm_steps=200):
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = substrate_cls(d=d, n_actions=n_clusters)
    centers = [torch.randn(d) * 3 for _ in range(n_clusters)]

    k = 0
    for _ in range(warm_steps):
        c = centers[k % n_clusters]
        s.step(c + 0.3 * torch.randn(d))
        k += 1

    results = {i: [] for i in range(n_clusters)}
    for _ in range(100):
        for i, c in enumerate(centers):
            a = s.step(c + 0.3 * torch.randn(d))
            results[i].append(a)

    fracs = [Counter(results[i]).most_common(1)[0][1] / 100 for i in range(n_clusters)]
    dom_actions = [Counter(results[i]).most_common(1)[0][0] for i in range(n_clusters)]
    return float(np.mean(fracs)), len(set(dom_actions))


def main():
    t0 = time.time()
    print("Disc test: d=32, 4 clusters, 5 seeds")
    print("Leo predicts PerAction ~60-70% (worse than baseline ~93.5%)")
    print("Eli predicts PerAction ~55-65%\n")

    for name, cls in [("Baseline (1W)", TemporalPrediction), ("PerAction (4W)", TemporalPerAction)]:
        avgs, dists = [], []
        for seed in range(5):
            avg, nd = run_disc(cls, seed)
            avgs.append(avg * 100)
            dists.append(nd)
            print(f"  {name} seed={seed}: dom={avg*100:.1f}%  distinct={nd}/4")
        print(f"  {name} summary: dom mean={np.mean(avgs):.1f}%  distinct mean={np.mean(dists):.1f}\n")

    print(f"elapsed={time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
