"""
Test centering hypothesis on Tier 1b.

Two variables isolated:
1. Aliasing: dims [3,147] both → action 3. Dims [1,6] → actions 1,2 (no aliasing).
2. Centering: subtract running mean before update and action.

4 conditions:
  A. baseline [3,147]      — original test (confirm aliasing)
  B. baseline [1,6]        — aliasing fixed, no centering
  C. centered  [3,147]     — centering only
  D. centered  [1,6]       — both fixes

5 seeds each. Should complete in <30s.
"""
import sys, time
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal import TemporalPrediction


class TemporalCentered(TemporalPrediction):
    """TemporalPrediction with input centering (subtract running mean)."""
    def __init__(self, d, n_actions, alpha=0.99, device='cpu'):
        super().__init__(d, n_actions, device)
        self.mu = torch.zeros(d, device=device)
        self.alpha = alpha

    def step(self, x):
        x = x.to(self.device).float()
        self.mu = self.alpha * self.mu + (1 - self.alpha) * x
        x = x - self.mu  # center
        if self.prev is None:
            self.prev = x.clone()
            return 0
        pred = self.W @ self.prev
        err = x - pred
        self.pred_err = err.norm().item()
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W += torch.outer(err, self.prev) / denom
        p1 = self.W @ x
        p2 = self.W @ p1
        action = p2.abs().argmax().item() % self.n_actions
        self.prev = x.clone()
        return action


def run_tier1b(substrate_cls, signal_dims, seed, d=256, n_actions=4, warm_steps=800):
    torch.manual_seed(seed)
    np.random.seed(seed)

    s = substrate_cls(d=d, n_actions=n_actions)
    n_clusters = 4
    offsets = [(2.0, 2.0), (2.0, -2.0), (-2.0, 2.0), (-2.0, -2.0)]
    centers = []
    for sd0, sd1 in offsets:
        c = torch.zeros(d)
        c[signal_dims[0]] = sd0
        c[signal_dims[1]] = sd1
        centers.append(c)

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
            a = s.step(c + 0.3 * torch.randn(d))
            results[i].append(a)

    fracs = [Counter(results[i]).most_common(1)[0][1] / 100 for i in range(n_clusters)]
    dom_actions = [Counter(results[i]).most_common(1)[0][0] for i in range(n_clusters)]
    avg = float(np.mean(fracs))
    n_distinct = len(set(dom_actions))
    return avg, n_distinct


def run_condition(label, substrate_cls, signal_dims, n_seeds=5):
    alias_note = f"(dim{signal_dims[0]}%4={signal_dims[0]%4}, dim{signal_dims[1]}%4={signal_dims[1]%4})"
    avgs, dists = [], []
    for seed in range(n_seeds):
        avg, nd = run_tier1b(substrate_cls, signal_dims, seed)
        avgs.append(avg)
        dists.append(nd)
    status = "DEGENERATE" if max(dists) <= 1 else ("PASS" if np.mean(avgs) > 0.6 else "MARGINAL")
    print(f"  {label} {alias_note}: dom={np.mean(avgs)*100:.1f}%  distinct={np.mean(dists):.1f}/4  [{status}]")
    return avgs, dists


def main():
    t0 = time.time()
    print("=" * 65)
    print("Centering hypothesis: Tier 1b (d=256, 2 signal dims, 4 clusters)")
    print("Question: does centering suppress noise-scale, preserve signal-scale?")
    print("=" * 65)

    print("\nA. Baseline   dims=[3,147] (aliased: both→action 3)")
    run_condition("baseline ", TemporalPrediction, [3, 147])

    print("\nB. Baseline   dims=[1,6]   (no alias: →actions 1,2)")
    run_condition("baseline ", TemporalPrediction, [1, 6])

    print("\nC. Centered   dims=[3,147] (centering only)")
    run_condition("centered ", TemporalCentered,   [3, 147])

    print("\nD. Centered   dims=[1,6]   (centering + no alias)")
    run_condition("centered ", TemporalCentered,   [1, 6])

    elapsed = time.time() - t0
    print(f"\nelapsed={elapsed:.1f}s")
    print("\nDiagnosis:")
    print("  If A=DEGEN, B=DEGEN: noise domination (not aliasing)")
    print("  If A=DEGEN, B=PASS:  aliasing was the problem (not noise)")
    print("  If C or D > A or B:  centering helps")


if __name__ == '__main__':
    main()
