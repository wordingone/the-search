#!/usr/bin/env python3
"""
Finite-difference MI gradient experiment

Tests whether TRUE mutual information gradients can guide beta/gamma optimization.
Uses finite differences on actual MI measurements (not proxy loss) to compute gradients.

For each starting point, repeatedly:
1. Measure MI at current beta, gamma
2. Measure MI at beta±eps, gamma±eps (finite differences)
3. Compute gradient: dMI/dbeta, dMI/dgamma
4. Update: beta += lr * dMI/dbeta, gamma += lr * dMI/dgamma
5. Repeat for 50 steps

Starting points: (0.1,0.1), (0.5,0.9), (1.0,0.5), (0.3,1.5), (1.5,0.1)
Expected result: All trajectories should converge to the same high-MI region.
"""

import math
import random
import time


# ═══════════════════════════════════════════════════════════════
# Minimal organism for MI measurement
# ═══════════════════════════════════════════════════════════════

D = 12
NC = 6

class Organism:
    def __init__(self, beta, gamma, seed=42):
        self.beta = beta
        self.gamma = gamma
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

        if signal:
            phi = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D
                    km = (k - 1) % D
                    row.append(math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + beta * (xs[i][kp] + gamma * signal[kp])
                               * (xs[i][km] + gamma * signal[km])))
                phi.append(row)
        else:
            phi = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D
                    km = (k - 1) % D
                    row.append(math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + beta * xs[i][kp] * xs[i][km]))
                phi.append(row)

        new = []
        for i in range(NC):
            nx = []
            for k in range(D):
                v = (1 - self.delta) * xs[i][k] + self.delta * phi[i][k]
                v += random.gauss(0, self.noise)
                v = max(-self.clip, min(self.clip, v))
                nx.append(v)
            new.append(nx)
        return new

    def centroid(self, xs):
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]


# ═══════════════════════════════════════════════════════════════
# MI measurement via permutation test
# ═══════════════════════════════════════════════════════════════

def vcosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def make_signals(k, seed):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def gen_perms(k, n_perm, seed):
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


def run_sequence(org, order, signals, base_seed, trial,
                 n_org=150, n_per_sig=30, n_settle=15, n_final=30):
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    for _ in range(n_org):
        xs = org.step(xs)

    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    for _ in range(n_final):
        xs = org.step(xs)

    return org.centroid(xs)


def measure_mi(beta, gamma, seed=42, k=6, n_perm=4, n_trials=2):
    """Measure MI for given beta/gamma via permutation test."""
    org = Organism(beta, gamma, seed=seed)
    signals = make_signals(k, seed=seed + 1000)
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
    return avg_w - avg_b


# ═══════════════════════════════════════════════════════════════
# Finite-difference gradient ascent
# ═══════════════════════════════════════════════════════════════

def finite_diff_ascent(start_beta, start_gamma, n_steps=50,
                       eps=0.05, lr=0.01, seed=42):
    """
    Use finite differences on TRUE MI to optimize beta/gamma.

    Returns trajectory: [(beta, gamma, MI), ...]
    """
    beta = start_beta
    gamma = start_gamma
    trajectory = []

    for step in range(n_steps):
        # Measure MI at current point
        mi_base = measure_mi(beta, gamma, seed=seed)

        # Finite differences
        mi_bp = measure_mi(beta + eps, gamma, seed=seed)
        mi_bm = measure_mi(beta - eps, gamma, seed=seed)
        mi_gp = measure_mi(beta, gamma + eps, seed=seed)
        mi_gm = measure_mi(beta, gamma - eps, seed=seed)

        # Compute gradients
        dmi_dbeta = (mi_bp - mi_bm) / (2 * eps)
        dmi_dgamma = (mi_gp - mi_gm) / (2 * eps)

        trajectory.append((beta, gamma, mi_base, dmi_dbeta, dmi_dgamma))

        # Update
        beta += lr * dmi_dbeta
        gamma += lr * dmi_dgamma

        # Clip to reasonable range
        beta = max(0.05, min(2.0, beta))
        gamma = max(0.05, min(2.0, gamma))

        if step % 10 == 0:
            print(f"    step {step:2d}: beta={beta:.3f} gamma={gamma:.3f} "
                  f"MI={mi_base:+.4f} grad=({dmi_dbeta:+.4f},{dmi_dgamma:+.4f})",
                  flush=True)

    # Final measurement
    mi_final = measure_mi(beta, gamma, seed=seed)
    trajectory.append((beta, gamma, mi_final, 0.0, 0.0))

    return trajectory


# ═══════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * 72)
    print("  FINITE-DIFFERENCE MI GRADIENT EXPERIMENT")
    print("  Using TRUE MI measurements to guide beta/gamma optimization")
    print("=" * 72)

    t_start = time.time()

    starting_points = [
        (0.1, 0.1, "low-low"),
        (0.5, 0.9, "mid-high"),
        (1.0, 0.5, "high-mid"),
        (0.3, 1.5, "low-veryhigh"),
        (1.5, 0.1, "veryhigh-low"),
    ]

    results = []

    for idx, (b0, g0, label) in enumerate(starting_points, 1):
        print(f"\n{'-'*72}")
        print(f"  Test {idx}/5: {label} beta={b0:.2f} gamma={g0:.2f}")
        print(f"{'-'*72}")

        traj = finite_diff_ascent(b0, g0, n_steps=50, eps=0.05, lr=0.01, seed=42)

        b_start, g_start, mi_start = traj[0][0], traj[0][1], traj[0][2]
        b_end, g_end, mi_end = traj[-1][0], traj[-1][1], traj[-1][2]

        print(f"\n  Start: beta={b_start:.3f} gamma={g_start:.3f} MI={mi_start:+.4f}")
        print(f"  End:   beta={b_end:.3f} gamma={g_end:.3f} MI={mi_end:+.4f}")
        print(f"  Delta: MI {mi_end - mi_start:+.4f}")

        results.append({
            'label': label,
            'start': (b_start, g_start, mi_start),
            'end': (b_end, g_end, mi_end),
            'trajectory': traj
        })

    # ── Summary ──────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}\n")

    print(f"  {'Label':<16} {'Start (b,g,MI)':<28} {'End (b,g,MI)':<28} {'dMI':<8}")
    print(f"  {'-'*70}")

    for r in results:
        b0, g0, mi0 = r['start']
        b1, g1, mi1 = r['end']
        dmi = mi1 - mi0
        print(f"  {r['label']:<16} "
              f"({b0:.2f},{g0:.2f},{mi0:+.3f})        "
              f"({b1:.2f},{g1:.2f},{mi1:+.3f})        "
              f"{dmi:+.4f}")

    # Check convergence
    endpoints = [(r['end'][0], r['end'][1]) for r in results]
    avg_beta = sum(b for b, g in endpoints) / len(endpoints)
    avg_gamma = sum(g for b, g in endpoints) / len(endpoints)

    beta_spread = max(abs(b - avg_beta) for b, g in endpoints)
    gamma_spread = max(abs(g - avg_gamma) for b, g in endpoints)

    converged = beta_spread < 0.3 and gamma_spread < 0.3

    print(f"\n  Endpoint cluster: beta={avg_beta:.3f}±{beta_spread:.3f} "
          f"gamma={avg_gamma:.3f}±{gamma_spread:.3f}")
    print(f"  Convergence: {'YES' if converged else 'NO (spread too large)'}")

    avg_mi_start = sum(r['start'][2] for r in results) / len(results)
    avg_mi_end = sum(r['end'][2] for r in results) / len(results)

    print(f"\n  Average MI improvement: {avg_mi_end - avg_mi_start:+.4f}")
    print(f"  All improved: {all(r['end'][2] > r['start'][2] for r in results)}")

    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print(f"{'='*72}")

    return results


if __name__ == '__main__':
    run()
