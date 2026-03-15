#!/usr/bin/env python3
"""
MI COMPUTATION COST BENCHMARK

Measure computational cost of MI calculation at different system sizes.
Informs feasibility of online MI-driven adaptation.

Questions:
- Does MI scale polynomially or exponentially with NC (number of cells)?
- What's the actual cost for NC=6 (current system)?
- Is per-step MI evaluation feasible?
"""

import math
import random
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Modified Organism class that allows variable NC
D = 12  # Fixed dimension


class OrganismVariable:
    """Modified Organism with variable NC for benchmarking."""

    def __init__(self, nc, seed=42, beta=0.5, gamma=0.9):
        self.nc = nc
        self.beta = beta
        self.gamma = gamma
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(nc)
        ]

    def step(self, xs, signal=None):
        """Single step of dynamics."""
        beta, gamma, nc = self.beta, self.gamma, self.nc

        # Bare dynamics
        phi_bare = []
        for i in range(nc):
            row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D
                row.append(math.tanh(
                    self.alpha[i][k] * xs[i][k]
                    + beta * xs[i][kp] * xs[i][km]))
            phi_bare.append(row)

        # Signal-modulated dynamics
        if signal:
            phi_sig = []
            for i in range(nc):
                row = []
                for k in range(D):
                    kp = (k + 1) % D
                    km = (k - 1) % D
                    row.append(math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + beta * (xs[i][kp] + gamma * signal[kp])
                               * (xs[i][km] + gamma * signal[km])))
                phi_sig.append(row)
        else:
            phi_sig = phi_bare

        # Attention
        weights = []
        for i in range(nc):
            raw = []
            for j in range(nc):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][k] * xs[j][k] for k in range(D))
                    raw.append(d / (D * self.tau))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # State update
        new = []
        for i in range(nc):
            p = [v for v in phi_sig[i]]

            bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
            fp_d = math.sqrt(sum(bd * bd for bd in bare_diff) + 1e-15) / max(
                math.sqrt(sum(xs[i][k] * xs[i][k] for k in range(D)) + 1e-15), 1.0)
            plast = math.exp(-(fp_d * fp_d) / 0.0225)

            if plast > 0.01 and self.eps > 0:
                pull = [0.0] * D
                for j in range(nc):
                    if i == j or weights[i][j] < 1e-8:
                        continue
                    for k in range(D):
                        pull[k] += weights[i][j] * (phi_bare[j][k] - phi_bare[i][k])
                p = [p[k] + plast * self.eps * pull[k] for k in range(D)]

            nx = []
            for k in range(D):
                v = (1 - self.delta) * xs[i][k] + self.delta * p[k]
                v += random.gauss(0, self.noise)
                v = max(-self.clip, min(self.clip, v))
                nx.append(v)
            new.append(nx)
        return new

    def centroid(self, xs):
        """Compute centroid of states."""
        return [sum(xs[i][k] for i in range(self.nc)) / self.nc for k in range(D)]


def vcosine(a, b):
    """Vector cosine similarity."""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def make_signals(k, seed):
    """Generate k normalized signal vectors."""
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def gen_perms(k, n_perm, seed):
    """Generate permutations for sequence ordering."""
    random.seed(seed)
    base = list(range(k))
    perms = []
    seen = set()
    perms.append(tuple(base))
    seen.add(tuple(base))
    perms.append(tuple(reversed(base)))
    seen.add(tuple(reversed(base)))
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]
        random.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        att += 1
    return perms


def run_sequence(org, order, signals, base_seed, trial):
    """Run a sequence of signals through organism."""
    nc = org.nc
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(nc)]

    # Settle
    for _ in range(100):
        xs = org.step(xs)

    # Signal presentations
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        for _ in range(20):
            xs = org.step(xs, sig)
        for _ in range(10):
            xs = org.step(xs)

    # Final settle
    for _ in range(20):
        xs = org.step(xs)

    return org.centroid(xs)


def measure_mi_timed(org, signals, k, seed, n_perm=4, n_trials=3):
    """Measure MI with timing."""
    start = time.time()

    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c = run_sequence(org, perm, signals, seed, trial)
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
    mi = avg_w - avg_b

    elapsed = time.time() - start
    return mi, elapsed


def benchmark_nc_scaling():
    """Benchmark MI computation cost vs NC."""
    print("=" * 72)
    print("  MI COMPUTATION COST BENCHMARK")
    print("  Scaling analysis: NC = [3, 6, 12, 24]")
    print("=" * 72)
    print()

    nc_values = [3, 6, 12, 24]
    k = 6  # Signal count
    seed = 42
    n_reps = 3  # Repetitions for averaging

    results = []

    print(f"{'NC':>4} {'Time (s)':>12} {'Time/NC':>12} {'Time/NC²':>12} {'MI':>12}")
    print("-" * 72)

    for nc in nc_values:
        times = []
        mis = []

        for rep in range(n_reps):
            org = OrganismVariable(nc=nc, seed=seed + rep)
            sigs = make_signals(k, seed=seed + nc * 100 + rep)

            mi, elapsed = measure_mi_timed(org, sigs, k, seed + rep, n_perm=4, n_trials=3)
            times.append(elapsed)
            mis.append(mi)

        avg_time = sum(times) / len(times)
        avg_mi = sum(mis) / len(mis)
        time_per_nc = avg_time / nc
        time_per_nc2 = avg_time / (nc * nc)

        results.append({
            'nc': nc,
            'time': avg_time,
            'time_per_nc': time_per_nc,
            'time_per_nc2': time_per_nc2,
            'mi': avg_mi
        })

        print(f"{nc:4d} {avg_time:12.4f} {time_per_nc:12.4f} {time_per_nc2:12.6f} {avg_mi:+12.4f}")

    print()
    print("-" * 72)
    print("  SCALING ANALYSIS")
    print("-" * 72)
    print()

    # Check if time/NC or time/NC² is more constant
    time_per_nc_vals = [r['time_per_nc'] for r in results]
    time_per_nc2_vals = [r['time_per_nc2'] for r in results]

    var_per_nc = sum((t - sum(time_per_nc_vals) / len(time_per_nc_vals)) ** 2
                     for t in time_per_nc_vals) / len(time_per_nc_vals)
    var_per_nc2 = sum((t - sum(time_per_nc2_vals) / len(time_per_nc2_vals)) ** 2
                      for t in time_per_nc2_vals) / len(time_per_nc2_vals)

    print(f"Variance of Time/NC:  {var_per_nc:.6f}")
    print(f"Variance of Time/NC²: {var_per_nc2:.8f}")
    print()

    if var_per_nc2 < var_per_nc * 0.1:
        scaling = "O(NC²)"
        print(f"RESULT: Scaling appears {scaling} (quadratic)")
    elif var_per_nc < var_per_nc2 * 0.1:
        scaling = "O(NC)"
        print(f"RESULT: Scaling appears {scaling} (linear)")
    else:
        scaling = "between O(NC) and O(NC²)"
        print(f"RESULT: Scaling is {scaling}")

    print()
    print("-" * 72)
    print("  FEASIBILITY ANALYSIS")
    print("-" * 72)
    print()

    nc6_time = next(r['time'] for r in results if r['nc'] == 6)
    print(f"Current system (NC=6): {nc6_time:.4f} seconds per MI evaluation")
    print()

    # Per-step feasibility
    steps_per_sec = 1.0 / nc6_time if nc6_time > 0 else 0
    print(f"Maximum throughput: {steps_per_sec:.2f} MI evaluations/second")
    print()

    if nc6_time < 0.1:
        print("FEASIBILITY: Per-step MI evaluation is FEASIBLE")
        print("             Fast enough for online adaptation loops.")
    elif nc6_time < 1.0:
        print("FEASIBILITY: Per-step MI evaluation is MARGINAL")
        print("             Feasible for episodic adaptation (every N steps).")
    else:
        print("FEASIBILITY: Per-step MI evaluation is EXPENSIVE")
        print("             Requires batch optimization or approximations.")

    print()
    print("=" * 72)

    return results


if __name__ == '__main__':
    results = benchmark_nc_scaling()
