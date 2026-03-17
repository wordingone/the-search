"""Tier 1 discrimination: D=32 2-cluster and D=256 sparse signal."""
import time, torch, numpy as np
from collections import Counter
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from expr import ExprSubstrate


def tier1a():
    """D=32, 2 orthogonal clusters. Basic discrimination."""
    print("Tier 1a: D=32, 2 clusters...", flush=True)
    d = 32
    c0 = torch.randn(d); c0 = c0 / c0.norm()
    c1 = -c0

    s = ExprSubstrate(n_dims=d, n_actions=4)
    for _ in range(200):
        x = (c0 if np.random.random() < 0.5 else c1) + 0.3 * torch.randn(d)
        s.step(x, n_actions=4)

    results = {0: [], 1: []}
    for _ in range(100):
        results[0].append(s.step(c0 + 0.3 * torch.randn(d), n_actions=4))
        results[1].append(s.step(c1 + 0.3 * torch.randn(d), n_actions=4))

    fracs = []
    for i in range(2):
        counts = Counter(results[i])
        dom = counts.most_common(1)[0]
        frac = dom[1] / 100
        fracs.append(frac)
        print(f"  Cluster {i}: dom={dom[0]} ({frac*100:.0f}%)  {dict(counts)}")
    avg = float(np.mean(fracs))
    print(f"  Avg dominance: {avg*100:.1f}%  ({'PASS' if avg > 0.6 else 'FAIL'})")
    return avg


def tier1b():
    """D=256, 2 signal dims, 5 clusters. Tests feature selection."""
    print("Tier 1b: D=256, 2 signal dims, 5 clusters...", flush=True)
    d = 256
    sig_dims = [3, 147]

    centers = []
    for k in range(5):
        c = torch.zeros(d)
        c[sig_dims[0]] = (k % 3 - 1) * 2.0
        c[sig_dims[1]] = (k // 3 - 0.5) * 2.0
        centers.append(c)

    s = ExprSubstrate(n_dims=d, n_actions=4)
    for _ in range(500):
        k = np.random.randint(5)
        x = centers[k] + 0.3 * torch.randn(d)
        s.step(x, n_actions=4)

    results = {k: [] for k in range(5)}
    for _ in range(100):
        for k in range(5):
            x = centers[k] + 0.3 * torch.randn(d)
            results[k].append(s.step(x, n_actions=4))

    fracs = []
    for k in range(5):
        counts = Counter(results[k])
        dom = counts.most_common(1)[0]
        frac = dom[1] / 100
        fracs.append(frac)
        print(f"  Cluster {k}: dom={dom[0]} ({frac*100:.0f}%)")
    avg = float(np.mean(fracs))
    print(f"  Avg dominance: {avg*100:.1f}%  [SelfRef=11%, LVQ=100%]")
    return avg


if __name__ == '__main__':
    t0 = time.time()
    print("ExprSubstrate Tier 1")
    print("=" * 40)
    a = tier1a()
    print()
    b = tier1b()
    print(f"\n1a={a*100:.1f}%  1b={b*100:.1f}%  elapsed={time.time()-t0:.1f}s")
