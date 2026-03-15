#!/usr/bin/env python3
"""
Variance Reduction Analysis for measure_gap

Problem: measure_gap has 29.2% CV across seeds, drowning out parameter effects.

Tests three variance reduction approaches:
1. Within-seed paired design (compute differences per seed)
2. Longer signal exposure (more perms/trials)
3. More K values (broader averaging)

Goal: Find methodology that reduces variance enough to detect real effects.
"""

import math
import random
import time
from scipy import stats

D = 12
NC = 6
W = 72

SEEDS = [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
BIRTH_SEED = 42


def vcosine(a, b):
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na2 += ai * ai
        nb2 += bi * bi
    na = math.sqrt(na2 + 1e-15)
    nb = math.sqrt(nb2 + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


class Organism:
    def __init__(self, seed=42, alive=False, eta=0.0003, adaptive_eta=False):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed
        self.alive = alive
        self.adaptive_eta = adaptive_eta

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

        if adaptive_eta:
            self.eta = [[0.0001 + random.random() * 0.0009 for _ in range(D)]
                        for _ in range(NC)]
            self.prev_resp_z = [[0.0] * D for _ in range(NC)]
            self.has_prev = False
        else:
            self.eta_scalar = eta

        self.eta_lo = 0.00005
        self.eta_hi = 0.003

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

        phi_bare = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D
                row.append(math.tanh(
                    self.alpha[i][k] * xs[i][k]
                    + beta * xs[i][kp] * xs[i][km]))
            phi_bare.append(row)

        if signal:
            phi_sig = []
            for i in range(NC):
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

        if self.alive and signal:
            response = []
            for i in range(NC):
                response.append([abs(phi_sig[i][k] - phi_bare[i][k])
                                 for k in range(D)])

            all_resp = [response[i][k] for i in range(NC) for k in range(D)]
            overall_mean = sum(all_resp) / len(all_resp)
            overall_std = math.sqrt(
                sum((r - overall_mean) ** 2 for r in all_resp) / len(all_resp)
            ) + 1e-10

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std

                    if self.adaptive_eta:
                        eta_ik = self.eta[i][k]
                    else:
                        eta_ik = self.eta_scalar

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    if abs(dev) < 0.01:
                        push = eta_ik * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = eta_ik * math.tanh(resp_z) * direction * 0.5
                    else:
                        push = eta_ik * 0.1 * random.gauss(0, 1.0)

                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))

                    if self.adaptive_eta and self.has_prev:
                        delta_rz = resp_z - self.prev_resp_z[i][k]
                        push_e = 0.1 * eta_ik * math.tanh(delta_rz)
                        self.eta[i][k] += push_e
                        self.eta[i][k] = max(self.eta_lo, min(self.eta_hi, self.eta[i][k]))

                    if self.adaptive_eta:
                        self.prev_resp_z[i][k] = resp_z

            if self.adaptive_eta:
                self.has_prev = True

        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][kk] * xs[j][kk] for kk in range(D))
                    raw.append(d / (D * self.tau))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        new = []
        for i in range(NC):
            p = [v for v in phi_sig[i]]

            bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
            fp_d = vnorm(bare_diff) / max(vnorm(xs[i]), 1.0)
            plast = math.exp(-(fp_d * fp_d) / 0.0225)

            if plast > 0.01 and self.eps > 0:
                pull = [0.0] * D
                for j in range(NC):
                    if i == j or weights[i][j] < 1e-8:
                        continue
                    for kk in range(D):
                        pull[kk] += weights[i][j] * (phi_bare[j][kk] - phi_bare[i][kk])
                p = [p[kk] + plast * self.eps * pull[kk] for kk in range(D)]

            nx = []
            for k in range(D):
                v = (1 - self.delta) * xs[i][k] + self.delta * p[k]
                v += random.gauss(0, self.noise)
                v = max(-self.clip, min(self.clip, v))
                nx.append(v)
            new.append(nx)
        return new

    def centroid(self, xs):
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]


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
                 n_org=300, n_per_sig=60, n_settle=30, n_final=60):
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
    return org.centroid(xs), xs


def measure_gap(org, signals, k, seed, n_perm=4, n_trials=3):
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c, _ = run_sequence(org, perm, signals, seed, trial)
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


def compute_cv(values):
    """Compute coefficient of variation."""
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean)**2 for v in values) / (len(values) - 1))
    return std / abs(mean) if abs(mean) > 1e-10 else float('inf')


def main():
    print("=" * 80)
    print("  VARIANCE REDUCTION ANALYSIS")
    print("=" * 80)
    print(f"\nSeeds: {SEEDS}")
    print()

    t0 = time.time()

    # ── BASELINE: Raw gap CV ─────────────────────────────────────
    print("=" * 80)
    print("BASELINE: Raw gap CV (current protocol)")
    print("=" * 80)
    print("Protocol: K=[4,6,8,10], n_perm=4, n_trials=3")
    print()

    ks_baseline = [4, 6, 8, 10]
    canonical_gaps = []

    for i, seed in enumerate(SEEDS):
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed}", flush=True)
        gaps = []
        for k in ks_baseline:
            sig_seed = BIRTH_SEED + k * 200
            sigs = make_signals(k, seed=sig_seed)
            org = Organism(seed=BIRTH_SEED, alive=True, eta=0.0003, adaptive_eta=False)
            g = measure_gap(org, sigs, k, seed, n_perm=4, n_trials=3)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        canonical_gaps.append(avg)
        print(f"      gap={avg:+.4f}")

    baseline_cv = compute_cv(canonical_gaps)
    print(f"\nRaw gap CV: {baseline_cv:.1%}")
    print(f"Mean: {sum(canonical_gaps)/len(canonical_gaps):+.4f}")
    print(f"Std: {math.sqrt(sum((g - sum(canonical_gaps)/len(canonical_gaps))**2 for g in canonical_gaps)/(len(canonical_gaps)-1)):.4f}")

    # ── APPROACH 1: Within-seed paired design ────────────────────
    print("\n" + "=" * 80)
    print("APPROACH 1: Within-seed paired design")
    print("=" * 80)
    print("Compute canonical AND delta_rz for each seed, compare DIFFERENCES")
    print()

    paired_diffs = []

    for i, seed in enumerate(SEEDS):
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed}", flush=True)

        gaps_canonical = []
        gaps_adaptive = []

        for k in ks_baseline:
            sig_seed = BIRTH_SEED + k * 200
            sigs = make_signals(k, seed=sig_seed)

            org_can = Organism(seed=BIRTH_SEED, alive=True, eta=0.0003, adaptive_eta=False)
            g_can = measure_gap(org_can, sigs, k, seed, n_perm=4, n_trials=3)
            gaps_canonical.append(g_can)

            org_adp = Organism(seed=BIRTH_SEED, alive=True, eta=0.0003, adaptive_eta=True)
            g_adp = measure_gap(org_adp, sigs, k, seed, n_perm=4, n_trials=3)
            gaps_adaptive.append(g_adp)

        avg_can = sum(gaps_canonical) / len(gaps_canonical)
        avg_adp = sum(gaps_adaptive) / len(gaps_adaptive)
        diff = avg_adp - avg_can
        paired_diffs.append(diff)

        print(f"      canonical={avg_can:+.4f}, adaptive={avg_adp:+.4f}, diff={diff:+.4f}")

    paired_cv = compute_cv([abs(d) for d in paired_diffs])
    mean_diff = sum(paired_diffs) / len(paired_diffs)
    std_diff = math.sqrt(sum((d - mean_diff)**2 for d in paired_diffs) / (len(paired_diffs) - 1))

    print(f"\nPaired difference CV: {paired_cv:.1%}")
    print(f"Mean difference: {mean_diff:+.4f}")
    print(f"Std difference: {std_diff:.4f}")
    print(f"CV reduction: {baseline_cv/paired_cv:.2f}× (lower is better)")

    # ── APPROACH 2: Longer signal exposure ───────────────────────
    print("\n" + "=" * 80)
    print("APPROACH 2: Longer signal exposure (2× perms and trials)")
    print("=" * 80)
    print("Protocol: K=[4,6,8,10], n_perm=8, n_trials=6")
    print()

    longer_gaps = []

    # Test on subset of seeds (3 seeds) to save time
    test_seeds = [42, 137, 2024]

    for i, seed in enumerate(test_seeds):
        print(f"  [{i+1}/{len(test_seeds)}] seed={seed}", flush=True)
        gaps = []
        for k in ks_baseline:
            sig_seed = BIRTH_SEED + k * 200
            sigs = make_signals(k, seed=sig_seed)
            org = Organism(seed=BIRTH_SEED, alive=True, eta=0.0003, adaptive_eta=False)
            g = measure_gap(org, sigs, k, seed, n_perm=8, n_trials=6)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        longer_gaps.append(avg)
        print(f"      gap={avg:+.4f}")

    # Also get baseline for same 3 seeds for fair comparison
    baseline_3seeds = [canonical_gaps[SEEDS.index(s)] for s in test_seeds]
    baseline_cv_3 = compute_cv(baseline_3seeds)
    longer_cv = compute_cv(longer_gaps)

    print(f"\nBaseline CV (3 seeds): {baseline_cv_3:.1%}")
    print(f"Longer exposure CV: {longer_cv:.1%}")
    print(f"CV reduction: {baseline_cv_3/longer_cv:.2f}×")

    # ── APPROACH 3: More K values ────────────────────────────────
    print("\n" + "=" * 80)
    print("APPROACH 3: More K values")
    print("=" * 80)
    print("Protocol: K=[3,4,6,8,10,12], n_perm=4, n_trials=3")
    print()

    ks_extended = [3, 4, 6, 8, 10, 12]
    extended_gaps = []

    for i, seed in enumerate(test_seeds):
        print(f"  [{i+1}/{len(test_seeds)}] seed={seed}", flush=True)
        gaps = []
        for k in ks_extended:
            sig_seed = BIRTH_SEED + k * 200
            sigs = make_signals(k, seed=sig_seed)
            org = Organism(seed=BIRTH_SEED, alive=True, eta=0.0003, adaptive_eta=False)
            g = measure_gap(org, sigs, k, seed, n_perm=4, n_trials=3)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        extended_gaps.append(avg)
        print(f"      gap={avg:+.4f}")

    extended_cv = compute_cv(extended_gaps)

    print(f"\nBaseline CV (3 seeds): {baseline_cv_3:.1%}")
    print(f"Extended K CV: {extended_cv:.1%}")
    print(f"CV reduction: {baseline_cv_3/extended_cv:.2f}×")

    # ── SUMMARY ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Variance Reduction Effectiveness")
    print("=" * 80)

    print(f"\nBaseline raw gap CV: {baseline_cv:.1%}")
    print(f"\nApproach 1 (paired design):")
    print(f"  Difference CV: {paired_cv:.1%}")
    print(f"  Improvement: {baseline_cv/paired_cv:.2f}× reduction")
    print(f"  Status: {'EFFECTIVE' if baseline_cv/paired_cv > 1.5 else 'MARGINAL' if baseline_cv/paired_cv > 1.1 else 'NO IMPROVEMENT'}")

    print(f"\nApproach 2 (2× exposure):")
    print(f"  CV: {longer_cv:.1%}")
    print(f"  Improvement: {baseline_cv_3/longer_cv:.2f}× reduction")
    print(f"  Status: {'EFFECTIVE' if baseline_cv_3/longer_cv > 1.5 else 'MARGINAL' if baseline_cv_3/longer_cv > 1.1 else 'NO IMPROVEMENT'}")

    print(f"\nApproach 3 (more K values):")
    print(f"  CV: {extended_cv:.1%}")
    print(f"  Improvement: {baseline_cv_3/extended_cv:.2f}× reduction")
    print(f"  Status: {'EFFECTIVE' if baseline_cv_3/extended_cv > 1.5 else 'MARGINAL' if baseline_cv_3/extended_cv > 1.1 else 'NO IMPROVEMENT'}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)

    improvements = {
        'paired': baseline_cv / paired_cv if paired_cv > 0 else 0,
        'longer': baseline_cv_3 / longer_cv if longer_cv > 0 else 0,
        'extended_k': baseline_cv_3 / extended_cv if extended_cv > 0 else 0,
    }

    best = max(improvements, key=improvements.get)

    if best == 'paired':
        print("\nBest approach: WITHIN-SEED PAIRED DESIGN")
        print("Use paired t-tests comparing differences (variant - canonical) per seed.")
        print("This removes between-seed variance and should detect smaller effects.")
    elif best == 'longer':
        print("\nBest approach: LONGER SIGNAL EXPOSURE")
        print("Use n_perm=8, n_trials=6 (2× current protocol).")
        print("This reduces measurement noise within each condition.")
    else:
        print("\nBest approach: MORE K VALUES")
        print("Use K=[3,4,6,8,10,12] instead of K=[4,6,8,10].")
        print("This averages over more task conditions.")

    print(f"\nExpected CV with best approach: ~{min(paired_cv, longer_cv, extended_cv):.1%}")
    print(f"Expected improvement: {improvements[best]:.2f}× reduction in variance")

    print(f"\nRuntime: {time.time() - t0:.0f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
