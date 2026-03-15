#!/usr/bin/env python3
"""
Per-Cell Beta/Gamma Experiment

Tests whether making beta/gamma per-cell (like alpha) allows them to
self-optimize via the same resp_z mechanism.

Original: beta, gamma = scalars (frozen structural constants)
Modified: beta[i][k], gamma[i][k] = NC×D arrays (adaptive state)

Adaptation uses the SAME resp_z signal that works for alpha.
This tests the Stage 4 question: can the frozen frame shrink?
"""

import math
import random
import time


D = 12
NC = 6
W = 72


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

    def __init__(self, seed=42, alive=False, eta=0.0003, adapt_beta_gamma=False):
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed
        self.alive = alive
        self.eta = eta
        self.adapt_beta_gamma = adapt_beta_gamma
        self.total_alpha_shift = 0.0
        self.total_beta_shift = 0.0
        self.total_gamma_shift = 0.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

        # Per-cell beta and gamma
        if adapt_beta_gamma:
            self.beta = [
                [0.5 + 0.1 * (random.random() * 2 - 1) for _ in range(D)]
                for _ in range(NC)
            ]
            self.gamma = [
                [0.9 + 0.1 * (random.random() * 2 - 1) for _ in range(D)]
                for _ in range(NC)
            ]
        else:
            # Scalar mode for backwards compatibility
            self.beta = [[0.5 for _ in range(D)] for _ in range(NC)]
            self.gamma = [[0.9 for _ in range(D)] for _ in range(NC)]

    def step(self, xs, signal=None):
        # ── BARE DYNAMICS ────────────────────────────────────
        phi_bare = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D
                row.append(math.tanh(
                    self.alpha[i][k] * xs[i][k]
                    + self.beta[i][k] * xs[i][kp] * xs[i][km]))
            phi_bare.append(row)

        # ── SIGNAL-MODULATED DYNAMICS ────────────────────────
        if signal:
            phi_sig = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D
                    km = (k - 1) % D
                    row.append(math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + self.beta[i][k] * (xs[i][kp] + self.gamma[i][k] * signal[kp])
                                          * (xs[i][km] + self.gamma[i][k] * signal[km])))
                phi_sig.append(row)
        else:
            phi_sig = phi_bare

        # ── ONLINE PLASTICITY ────────────────────────────────
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

            # Adapt alpha (always)
            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    if abs(dev) < 0.01:
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * 0.5
                    else:
                        push = self.eta * 0.1 * random.gauss(0, 1.0)

                    old = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old)

            # Adapt beta and gamma (if enabled)
            if self.adapt_beta_gamma:
                for i in range(NC):
                    for k in range(D):
                        resp_z = (response[i][k] - overall_mean) / overall_std

                        # Beta adaptation
                        beta_col_mean = sum(self.beta[j][k] for j in range(NC)) / NC
                        beta_dev = self.beta[i][k] - beta_col_mean

                        if abs(beta_dev) < 0.01:
                            beta_push = self.eta * 0.3 * random.gauss(0, 1.0)
                        elif resp_z > 0:
                            beta_direction = 1.0 if beta_dev > 0 else -1.0
                            beta_push = self.eta * math.tanh(resp_z) * beta_direction * 0.5
                        else:
                            beta_push = self.eta * 0.1 * random.gauss(0, 1.0)

                        old_beta = self.beta[i][k]
                        self.beta[i][k] += beta_push
                        self.beta[i][k] = max(0.1, min(1.5, self.beta[i][k]))
                        self.total_beta_shift += abs(self.beta[i][k] - old_beta)

                        # Gamma adaptation
                        gamma_col_mean = sum(self.gamma[j][k] for j in range(NC)) / NC
                        gamma_dev = self.gamma[i][k] - gamma_col_mean

                        if abs(gamma_dev) < 0.01:
                            gamma_push = self.eta * 0.3 * random.gauss(0, 1.0)
                        elif resp_z > 0:
                            gamma_direction = 1.0 if gamma_dev > 0 else -1.0
                            gamma_push = self.eta * math.tanh(resp_z) * gamma_direction * 0.5
                        else:
                            gamma_push = self.eta * 0.1 * random.gauss(0, 1.0)

                        old_gamma = self.gamma[i][k]
                        self.gamma[i][k] += gamma_push
                        self.gamma[i][k] = max(0.1, min(1.5, self.gamma[i][k]))
                        self.total_gamma_shift += abs(self.gamma[i][k] - old_gamma)

        # ── ATTENTION ────────────────────────────────────────
        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][k] * xs[j][k] for k in range(D))
                    raw.append(d / (D * self.tau))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # ── STATE UPDATE ─────────────────────────────────────
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
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]

    def get_beta_stats(self):
        """Return mean and std of beta across all cells."""
        all_beta = [self.beta[i][k] for i in range(NC) for k in range(D)]
        mean_beta = sum(all_beta) / len(all_beta)
        std_beta = math.sqrt(sum((b - mean_beta) ** 2 for b in all_beta) / len(all_beta))
        return mean_beta, std_beta

    def get_gamma_stats(self):
        """Return mean and std of gamma across all cells."""
        all_gamma = [self.gamma[i][k] for i in range(NC) for k in range(D)]
        mean_gamma = sum(all_gamma) / len(all_gamma)
        std_gamma = math.sqrt(sum((g - mean_gamma) ** 2 for g in all_gamma) / len(all_gamma))
        return mean_gamma, std_gamma


# ═══════════════════════════════════════════════════════════════
# Signal generation and MI measurement
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * W)
    print("  PER-CELL BETA/GAMMA EXPERIMENT")
    print("  Can beta/gamma self-optimize when made per-cell like alpha?")
    print("=" * W)

    t_start = time.time()

    SEED = 42
    test_ks = [6, 8]
    test_seeds = [42, 77, 123]
    eta = 0.0003

    print(f"\n{'-'*W}")
    print(f"  THREE CONDITIONS")
    print(f"  STILL: all params fixed")
    print(f"  ALIVE-alpha-only: alpha adapts, beta/gamma fixed")
    print(f"  ALIVE-all: alpha, beta, gamma ALL adapt per-cell")
    print(f"{'-'*W}\n")

    # STILL baseline
    print("  --- STILL (all params fixed) ---")
    still_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            still = Organism(seed=s, alive=False, adapt_beta_gamma=False)
            g = measure_gap(still, sigs, k, s)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        still_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  K={k}: avg={avg:+.4f} [{min(gaps):+.3f}..{max(gaps):+.3f}]",
              flush=True)

    still_overall = sum(still_results[k]['avg'] for k in test_ks) / len(test_ks)
    print(f"  overall: {still_overall:+.4f}\n")

    # ALIVE-alpha-only (current behavior)
    print("  --- ALIVE-alpha-only (current) ---")
    alpha_only_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            alpha_only = Organism(seed=s, alive=True, eta=eta, adapt_beta_gamma=False)
            g = measure_gap(alpha_only, sigs, k, s)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        alpha_only_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  K={k}: avg={avg:+.4f} [{min(gaps):+.3f}..{max(gaps):+.3f}]",
              flush=True)

    alpha_only_overall = sum(alpha_only_results[k]['avg'] for k in test_ks) / len(test_ks)
    print(f"  overall: {alpha_only_overall:+.4f}\n")

    # ALIVE-all (new: beta/gamma also adapt)
    print("  --- ALIVE-all (alpha + beta + gamma adapt) ---")
    all_results = {}
    beta_diversity = []
    gamma_diversity = []
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            alive_all = Organism(seed=s, alive=True, eta=eta, adapt_beta_gamma=True)
            g = measure_gap(alive_all, sigs, k, s)
            gaps.append(g)
            b_mean, b_std = alive_all.get_beta_stats()
            g_mean, g_std = alive_all.get_gamma_stats()
            beta_diversity.append(b_std)
            gamma_diversity.append(g_std)
            print(f"    seed={s} K={k}: MI={g:+.4f} "
                  f"beta={b_mean:.3f}±{b_std:.3f} gamma={g_mean:.3f}±{g_std:.3f}",
                  flush=True)
        avg = sum(gaps) / len(gaps)
        all_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  K={k}: avg={avg:+.4f} [{min(gaps):+.3f}..{max(gaps):+.3f}]",
              flush=True)

    all_overall = sum(all_results[k]['avg'] for k in test_ks) / len(test_ks)
    avg_beta_div = sum(beta_diversity) / len(beta_diversity)
    avg_gamma_div = sum(gamma_diversity) / len(gamma_diversity)
    print(f"  overall: {all_overall:+.4f}")
    print(f"  avg beta diversity (std): {avg_beta_div:.4f}")
    print(f"  avg gamma diversity (std): {avg_gamma_div:.4f}\n")

    # ── COMPARISON ───────────────────────────────────────────

    print(f"\n{'='*W}")
    print(f"  RESULTS")
    print(f"{'='*W}\n")

    print(f"  STILL:           {still_overall:+.4f}")
    print(f"  ALIVE-alpha-only: {alpha_only_overall:+.4f} "
          f"(delta={alpha_only_overall - still_overall:+.4f})")
    print(f"  ALIVE-all:        {all_overall:+.4f} "
          f"(delta={all_overall - still_overall:+.4f})")

    improvement_over_alpha_only = all_overall - alpha_only_overall

    print(f"\n  ALIVE-all vs ALIVE-alpha-only: {improvement_over_alpha_only:+.4f}")

    if improvement_over_alpha_only > 0.02:
        print(f"\n  SUCCESS: Per-cell beta/gamma adaptation IMPROVES performance.")
        print(f"  Beta diversity: {avg_beta_div:.4f}")
        print(f"  Gamma diversity: {avg_gamma_div:.4f}")
        print(f"  The frozen frame CAN shrink.")
    elif improvement_over_alpha_only < -0.02:
        print(f"\n  NEGATIVE: Per-cell beta/gamma adaptation HURTS performance.")
        print(f"  Shared scalar values may be optimal by constraint.")
    else:
        print(f"\n  NEUTRAL: Per-cell beta/gamma provides no clear advantage.")
        print(f"  May be equivalent to scalar at equilibrium.")

    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print(f"{'='*W}")


if __name__ == '__main__':
    run()
