"""
Tier 1 for ExprU20 (temporal_smoothness x coverage scoring).

Two variants:
  - random: obs from clusters presented randomly (no temporal correlation)
  - temporal: obs presented in runs of 8 from same cluster (correlated)

U20 hypothesis: temporal variant should discriminate better because
smoothness rewards stable policy within same-cluster runs.
"""

import time, torch, numpy as np
from collections import Counter
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from expr_u20 import ExprU20


def run_tier1(d, n_clusters, warm_steps, temporal, label):
    centers = [torch.randn(d) for _ in range(n_clusters)]
    for c in centers:
        c /= c.norm()

    s = ExprU20(n_dims=d, n_actions=4)
    run_len = 8 if temporal else 1

    # Warm up
    k = 0
    run_pos = 0
    for _ in range(warm_steps):
        x = centers[k] + 0.3 * torch.randn(d)
        s.step(x, n_actions=4)
        run_pos += 1
        if run_pos >= run_len:
            k = (k + 1) % n_clusters
            run_pos = 0

    # Measure discrimination
    results = {i: [] for i in range(n_clusters)}
    for _ in range(100):
        for i in range(n_clusters):
            x = centers[i] + 0.3 * torch.randn(d)
            results[i].append(s.step(x, n_actions=4))

    fracs = []
    dom_actions = []
    for i in range(n_clusters):
        counts = Counter(results[i])
        dom = counts.most_common(1)[0]
        frac = dom[1] / 100
        fracs.append(frac)
        dom_actions.append(dom[0])

    avg = float(np.mean(fracs))
    unique_doms = len(set(dom_actions))
    collapsed = unique_doms == 1

    # Also report final scores and smoothness
    recent = s.history[-s.window:]
    if len(recent) > 1:
        acts = [s.pop[s.best].__class__ and
                __import__('expr', fromlist=['evaluate']).evaluate(s.pop[s.best], obs) % 4
                for obs, _ in recent]
        from expr import evaluate
        acts = [evaluate(s.pop[s.best], obs) % 4 for obs, _ in recent]
        same = sum(a == b for a, b in zip(acts[:-1], acts[1:]))
        sm = same / (len(acts) - 1)
    else:
        sm = 0.0

    status = "DEGENERATE" if collapsed else ("PASS" if avg > 0.6 else "FAIL")
    print(f"  {label}: avg={avg*100:.1f}%  unique_doms={unique_doms}/{n_clusters}"
          f"  smoothness={sm:.2f}  scores={[f'{x:.2f}' for x in s.scores]}"
          f"  [{status}]")
    return avg, collapsed


if __name__ == '__main__':
    t0 = time.time()
    print("ExprU20 Tier 1 (temporal_smoothness x coverage)")
    print("=" * 55)

    print("\nTier 1a: D=32, 2 clusters")
    run_tier1(32, 2, 200, temporal=False, label="random mix")
    run_tier1(32, 2, 200, temporal=True,  label="temporal runs")

    print("\nTier 1b: D=256, 2 signal dims, 5 clusters")
    # Note: signal is in dims 3, 147 (see original test)
    # Here we just test with full random clusters (ExprU20 must discover structure)
    run_tier1(256, 5, 500, temporal=False, label="random mix")
    run_tier1(256, 5, 500, temporal=True,  label="temporal runs")

    print(f"\nelapsed={time.time()-t0:.1f}s")
