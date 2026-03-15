#!/usr/bin/env python3
"""
SeedGPU Fixed: Scale-invariant attention

Removes D normalization from attention mechanism to prevent
degradation at higher dimensions.

Change from original:
  OLD: dots = (xs @ xs.T) / (D * tau)
  NEW: dots = (xs @ xs.T) / tau

Run: python SeedGPU_fixed.py --quick
"""

import math
import random
import time
import argparse
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)

# Constants
NC = 6
BETA = 0.5
GAMMA = 0.9
EPS_COUP = 0.15
TAU = 0.3
MIX = 0.35
NOISE = 0.005
CLIP = 4.0
ETA = 0.0003
ALPHA_MIN = 0.3
ALPHA_MAX = 1.8
W = 72


class OrganismFast:
    """Fast organism with FIXED attention (D-invariant)."""

    def __init__(self, D, seed=42, alive=False, eta=ETA, device='cuda'):
        self.D = D
        self.alive = alive
        self.eta = eta
        self.device = device

        random.seed(seed)
        alpha_list = []
        for i in range(NC):
            row = [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            alpha_list.append(row)
        self.alpha = torch.tensor(alpha_list, dtype=torch.float32, device=device)

        self.kp = torch.arange(D, device=device).roll(-1)
        self.km = torch.arange(D, device=device).roll(1)
        self.eye = torch.eye(NC, device=device)

    @torch.no_grad()
    def step(self, xs, signal=None):
        D = self.D
        alpha = self.alpha
        kp, km = self.kp, self.km

        xs_kp = xs[:, kp]
        xs_km = xs[:, km]

        phi_bare = torch.tanh(alpha * xs + BETA * xs_kp * xs_km)

        if signal is not None:
            s_kp = signal[kp].unsqueeze(0)
            s_km = signal[km].unsqueeze(0)
            phi_sig = torch.tanh(
                alpha * xs
                + BETA * (xs_kp + GAMMA * s_kp) * (xs_km + GAMMA * s_km)
            )
        else:
            phi_sig = phi_bare

        if self.alive and signal is not None:
            response = (phi_sig - phi_bare).abs()
            r_mean = response.mean()
            r_std = response.std().clamp(min=1e-10)
            resp_z = (response - r_mean) / r_std

            col_mean = alpha.mean(dim=0, keepdim=True)
            dev = alpha - col_mean

            noise_push = self.eta * 0.3 * torch.randn_like(alpha)
            amplify_push = self.eta * 0.5 * torch.tanh(resp_z) * dev.sign()
            drift_push = self.eta * 0.1 * torch.randn_like(alpha)

            at_mean = dev.abs() < 0.01
            sensitive = resp_z > 0
            push = torch.where(at_mean, noise_push,
                   torch.where(sensitive, amplify_push, drift_push))
            self.alpha = (alpha + push).clamp(ALPHA_MIN, ALPHA_MAX)

        # ══ FIXED ATTENTION (D-invariant) ══
        dots = (xs @ xs.T) / TAU  # Removed D normalization
        dots = dots - self.eye * 1e10
        weights = F.softmax(dots, dim=1)

        weighted_avg = weights @ phi_bare
        pull = weighted_avg - phi_bare
        bare_diff = phi_bare - xs
        fp_d = bare_diff.norm(dim=1) / xs.norm(dim=1).clamp(min=1.0)
        plast = torch.exp(-fp_d.pow(2) / 0.0225).unsqueeze(1)
        p = phi_sig + plast * EPS_COUP * pull

        new_xs = (1 - MIX) * xs + MIX * p + NOISE * torch.randn_like(xs)
        return new_xs.clamp(-CLIP, CLIP)

    def centroid(self, xs):
        return xs.mean(dim=0)


def make_signals(K, D, seed):
    random.seed(seed)
    sigs = {}
    for i in range(K):
        s = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def gen_perms(K, n_perm, seed):
    random.seed(seed)
    base = list(range(K))
    perms = [tuple(base), tuple(reversed(base))]
    seen = set(perms)
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]
        random.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        att += 1
    return perms[:n_perm]


def run_sequence(org, order, signals, base_seed, trial,
                 n_org=200, n_per_sig=50, n_settle=20, n_final=40):
    D = org.D
    device = org.device

    random.seed(base_seed)
    xs_list = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    xs = torch.tensor(xs_list, dtype=torch.float32, device=device)

    for _ in range(n_org):
        xs = org.step(xs)

    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig_list = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        sig = torch.tensor(sig_list, dtype=torch.float32, device=device)

        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    for _ in range(n_final):
        xs = org.step(xs)

    c = org.centroid(xs)
    return c.cpu().tolist()


def vcosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def measure_gap(D, K, birth_seed, sig_seed, test_seed, alive,
                device='cuda', n_perm=3, n_trials=2):
    signals = make_signals(K, D, seed=sig_seed)
    perms = gen_perms(K, n_perm, seed=test_seed * 10 + K)

    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            org = OrganismFast(D, seed=birth_seed, alive=alive, device=device)
            c = run_sequence(org, perm, signals, test_seed, trial)
            trials.append(c)
        endpoints[pi] = trials

    within = []
    between = []
    pis = sorted(endpoints.keys())
    for pi in pis:
        cs = endpoints[pi]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                within.append(vcosine(cs[i], cs[j]))
    for i in range(len(pis)):
        for j in range(i + 1, len(pis)):
            for c1 in endpoints[pis[i]]:
                for c2 in endpoints[pis[j]]:
                    between.append(vcosine(c1, c2))

    avg_w = sum(within) / max(len(within), 1)
    avg_b = sum(between) / max(len(between), 1)
    return avg_w - avg_b


def test_dimension(D, n_births=3, n_worlds=8, K=6, device='cuda'):
    birth_seeds = [42, 77, 200][:n_births]
    alive_gaps = []
    still_gaps = []
    total = n_births * n_worlds
    done = 0

    for bs in birth_seeds:
        for wi in range(n_worlds):
            sig_seed = bs * 1000 + wi * 137 + D * 7
            test_seed = bs * 100 + wi * 31 + D

            ag = measure_gap(D, K, bs, sig_seed, test_seed,
                             alive=True, device=device)
            sg = measure_gap(D, K, bs, sig_seed, test_seed,
                             alive=False, device=device)
            alive_gaps.append(ag)
            still_gaps.append(sg)
            done += 1
            if done % max(1, total // 4) == 0 or done == total:
                print(f"    [{done}/{total}] "
                      f"A={sum(alive_gaps)/len(alive_gaps):+.4f} "
                      f"S={sum(still_gaps)/len(still_gaps):+.4f}",
                      flush=True)

    alive_avg = sum(alive_gaps) / len(alive_gaps)
    still_avg = sum(still_gaps) / len(still_gaps)
    delta = alive_avg - still_avg
    wins = sum(1 for i in range(len(alive_gaps))
               if alive_gaps[i] > still_gaps[i] + 0.01)
    losses = sum(1 for i in range(len(alive_gaps))
                 if still_gaps[i] > alive_gaps[i] + 0.01)
    ratio = delta / max(abs(still_avg), 0.01)

    return {
        'D': D, 'alive_avg': alive_avg, 'still_avg': still_avg,
        'delta': delta, 'ratio': ratio,
        'wins': wins, 'losses': losses,
        'alive_gaps': alive_gaps, 'still_gaps': still_gaps
    }


def main():
    parser = argparse.ArgumentParser(description='SeedGPU Fixed (D-invariant attention)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dims', nargs='+', type=int,
                        default=[12, 16, 24, 32, 48, 64])
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--births', type=int, default=3)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if args.quick:
        args.worlds = 4
        args.births = 2
        args.dims = [12, 24, 48]

    print("=" * W)
    print("  SEEDGPU FIXED: D-Invariant Attention")
    print(f"  Device: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Dimensions: {args.dims}")
    n_tests = args.births * args.worlds
    print(f"  {args.births} births x {args.worlds} worlds = {n_tests} tests/D")
    print(f"  CHANGE: dots = (xs @ xs.T) / tau  (removed D normalization)")
    print("=" * W)

    t_start = time.time()
    results = []

    print(f"\n{'-'*W}")
    print(f"  SCALING TEST: ALIVE vs STILL (FIXED)")
    print(f"{'-'*W}\n")

    for D in args.dims:
        print(f"  --- D={D} ({NC} cells, {NC*D} params) ---")
        t0 = time.time()
        r = test_dimension(D, n_births=args.births,
                           n_worlds=args.worlds, device=device)
        elapsed = time.time() - t0
        r['elapsed'] = elapsed
        results.append(r)
        print(f"  D={D:>3}: ALIVE={r['alive_avg']:+.4f} "
              f"STILL={r['still_avg']:+.4f} "
              f"delta={r['delta']:+.4f} ratio={r['ratio']:+.2f} "
              f"{r['wins']}W/{r['losses']}L [{elapsed:.0f}s]\n")
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Results
    print(f"{'='*W}")
    print(f"  RESULTS")
    print(f"{'='*W}\n")

    print(f"  {'D':>4} {'ALIVE':>8} {'STILL':>8} {'delta':>8} "
          f"{'ratio':>8} {'W/L':>8} {'time':>8}")
    for r in results:
        print(f"  {r['D']:>4} {r['alive_avg']:>+8.4f} "
              f"{r['still_avg']:>+8.4f} "
              f"{r['delta']:>+8.4f} {r['ratio']:>+8.2f} "
              f"{r['wins']:>3}/{r['losses']:<3} "
              f"{r.get('elapsed',0):>7.0f}s")

    deltas = [r['delta'] for r in results]
    ratios = [r['ratio'] for r in results]

    # STILL degradation
    still_vals = [r['still_avg'] for r in results]
    print(f"\n  STILL degradation: ", end="")
    print(" -> ".join(f"{s:+.3f}" for s in still_vals))

    # Per-seed detail
    if len(results) > 1:
        print(f"\n  Per-seed delta at D={results[0]['D']} and D={results[-1]['D']}:")
        birth_seeds = [42, 77, 200][:args.births]
        for bi, bs in enumerate(birth_seeds):
            s, e = bi * args.worlds, (bi + 1) * args.worlds
            d_first = (sum(results[0]['alive_gaps'][s:e]) -
                       sum(results[0]['still_gaps'][s:e])) / args.worlds
            d_last = (sum(results[-1]['alive_gaps'][s:e]) -
                      sum(results[-1]['still_gaps'][s:e])) / args.worlds
            print(f"    seed={bs}: D={results[0]['D']}:{d_first:+.4f} "
                  f"-> D={results[-1]['D']}:{d_last:+.4f} "
                  f"{'GROWS' if d_last > d_first else 'shrinks'}")

    # Verdict
    delta_pos = all(d > 0 for d in deltas)
    delta_mostly_pos = sum(1 for d in deltas if d > 0) >= len(deltas) * 0.7
    grows = deltas[-1] > deltas[0] if len(deltas) > 1 else False
    r_grows = ratios[-1] > ratios[0] if len(ratios) > 1 else False
    wins_all = all(r['wins'] > r['losses'] for r in results)

    print(f"\n  Delta all positive: {'YES' if delta_pos else 'NO'} "
          f"({sum(1 for d in deltas if d > 0)}/{len(deltas)})")
    print(f"  Delta mostly pos:   {'YES' if delta_mostly_pos else 'NO'}")
    print(f"  Delta grows:        {'YES' if grows else 'NO'} "
          f"({deltas[0]:+.4f} -> {deltas[-1]:+.4f})")
    print(f"  Ratio grows:        {'YES' if r_grows else 'NO'} "
          f"({ratios[0]:+.2f} -> {ratios[-1]:+.2f})")
    print(f"  Wins all D:         {'YES' if wins_all else 'NO'}")

    works = deltas[0] > 0 and delta_mostly_pos
    scales = grows or r_grows

    if works and scales:
        print(f"\n  FIX SUCCESSFUL. THE SEED SCALES.")
    elif works:
        print(f"\n  FIX PARTIAL. Seed works but scaling unclear.")
    else:
        print(f"\n  FIX FAILED. Honest result.")

    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print(f"  {'='*W}")


if __name__ == '__main__':
    main()
