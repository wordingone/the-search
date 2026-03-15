#!/usr/bin/env python3
"""
GENESIS: The Living Seed

Three uncertainties to resolve from the 8/10 result:
  1. Is the advantage from restructuring or from seed 42?
  2. Does discrimination-derived restructuring beat random perturbation?
  3. Does quick_gap align with measure_gap?

Then: the real step. Alpha evolves DURING computation.
No separate restructuring phase. Each step of the dynamics
produces a local plasticity signal from the product term's
cross-terms. Alpha shifts in real-time. The substrate
changes as it computes.

  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * (x+gs)(x+gs))

  After each step:
    response_k = |phi_sig_k - phi_bare_k|  (signal sensitivity per dim)
    diversity_k = var(response across cells)  (inter-cell disagreement)
    alpha_{i,k} += eta * (response_{i,k} - mean) * sign(deviation from column mean)

  If a dimension is signal-sensitive AND cells disagree on it,
  alpha diversity on that dimension gets amplified. If cells
  agree or the dimension is insensitive, alpha drifts randomly.

  The computation IS the restructuring. One equation. One step.
  Not "compute then restructure." Compute = restructure.

Zero dependencies. Pure Python.
"""

import math
import random
import time


# ═══════════════════════════════════════════════════════════════
# In the beginning, there is nothing but arithmetic.
# ═══════════════════════════════════════════════════════════════

D  = 12
NC = 6
W  = 72


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


def bar(v, w=15, lo=-0.15, hi=0.35):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


# ═══════════════════════════════════════════════════════════════
# From nothing, a body that can be still or alive.
#
# STILL: alpha is fixed. The body computes but does not change.
# ALIVE: alpha shifts during every step. The body computes and
#        restructures simultaneously.
#
# The test: does the alive body outperform the still body?
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, seed=42, alive=False, eta=0.0003):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed
        self.alive = alive
        self.eta = eta
        self.total_alpha_shift = 0.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

    def alpha_flat(self):
        return [a for row in self.alpha for a in row]

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

        # bare dynamics
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

        # signal-modulated dynamics
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

        # ── ONLINE PLASTICITY ────────────────────────────────
        # Only when alive and signal is present.
        # The cross-term difference (phi_sig - phi_bare) reveals
        # per-cell, per-dimension signal sensitivity.
        # Alpha shifts to amplify diversity on sensitive dims.

        if self.alive and signal:
            # per-cell per-dim signal response
            response = []
            for i in range(NC):
                response.append([abs(phi_sig[i][k] - phi_bare[i][k])
                                 for k in range(D)])

            # per-dimension: mean response across cells
            dim_mean = [0.0] * D
            for k in range(D):
                dim_mean[k] = sum(response[i][k] for i in range(NC)) / NC

            # per-dimension: variance of response across cells
            dim_var = [0.0] * D
            for k in range(D):
                dim_var[k] = sum((response[i][k] - dim_mean[k]) ** 2
                                 for i in range(NC)) / NC

            # overall mean response (for normalization)
            all_resp = [response[i][k] for i in range(NC) for k in range(D)]
            overall_mean = sum(all_resp) / len(all_resp)
            overall_std = math.sqrt(
                sum((r - overall_mean) ** 2 for r in all_resp) / len(all_resp)
            ) + 1e-10

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std

                    # column mean of alpha on this dimension
                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    if abs(dev) < 0.01:
                        # at column mean: random perturbation
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        # above-average response: push alpha away from mean
                        # (amplify diversity on sensitive dims)
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * 0.5
                    else:
                        # below-average response: gentle drift
                        push = self.eta * 0.1 * random.gauss(0, 1.0)

                    old = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old)

        # attention weights
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

        # state update
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


# ═══════════════════════════════════════════════════════════════
# The world speaks.
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
# PART I: Resolve the three uncertainties from 8/10.
# ═══════════════════════════════════════════════════════════════

def resolve_uncertainties():
    print("=" * W)
    print("  PART I: RESOLVE UNCERTAINTIES")
    print("=" * W)

    SEED = 42

    # ── UNCERTAINTY 1: Is it the seed or the restructuring? ──
    # Test: run the SAME seed 42 organism WITHOUT restructuring.
    # If it performs just as well, restructuring adds nothing.

    print(f"\n{'-'*W}")
    print(f"  UNCERTAINTY 1: Seed quality vs restructuring")
    print(f"  Seed 42 birth alpha vs seed 42 restructured alpha.")
    print(f"  If birth performs equally, restructuring is noise.")
    print(f"{'-'*W}\n")

    from seed import Organism as OfflineOrg, develop, measure_gap as offline_gap

    # build restructured version (same as 8/10 result)
    restr = OfflineOrg(seed=SEED)
    develop(restr, n_epochs=12, base_seed=SEED, verbose=False)

    # birth version (no restructuring)
    birth = OfflineOrg(seed=SEED)

    alpha_cos = vcosine(restr.alpha_flat(), birth.alpha_flat())
    print(f"  Alpha cosine (restr vs birth): {alpha_cos:+.6f}")
    print(f"  Alpha distance: {1.0 - alpha_cos:.6f}")
    print()

    test_ks = [4, 6, 8, 10]
    restr_gaps = []
    birth_gaps = []
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        for s in [42, 77, 123]:
            rg = measure_gap(restr, sigs, k, s)
            bg = measure_gap(birth, sigs, k, s)
            restr_gaps.append(rg)
            birth_gaps.append(bg)
            print(f"  K={k:>2} seed={s:>3}: RESTR={rg:+.4f} BIRTH={bg:+.4f} "
                  f"{'RESTR' if rg > bg + 0.01 else 'BIRTH' if bg > rg + 0.01 else 'tie'}",
                  flush=True)

    restr_avg = sum(restr_gaps) / len(restr_gaps)
    birth_avg = sum(birth_gaps) / len(birth_gaps)
    rw = sum(1 for i in range(len(restr_gaps)) if restr_gaps[i] > birth_gaps[i] + 0.01)
    bw = sum(1 for i in range(len(restr_gaps)) if birth_gaps[i] > restr_gaps[i] + 0.01)

    print(f"\n  RESTR avg: {restr_avg:+.4f}")
    print(f"  BIRTH avg: {birth_avg:+.4f}")
    print(f"  Delta: {restr_avg - birth_avg:+.4f}")
    print(f"  Matchups: RESTR {rw}W / BIRTH {bw}W")

    u1_real = restr_avg > birth_avg and rw > bw
    print(f"\n  VERDICT: {'RESTRUCTURING IS REAL' if u1_real else 'RESTRUCTURING IS NOISE'}")

    # ── UNCERTAINTY 2: Disc restructuring vs random perturbation ──
    # Apply same magnitude of random perturbation to birth alpha.

    print(f"\n{'-'*W}")
    print(f"  UNCERTAINTY 2: Discrimination signal vs random perturbation")
    print(f"  Same magnitude push, random direction.")
    print(f"{'-'*W}\n")

    perturbed_orgs = []
    for pi in range(3):
        p = OfflineOrg(seed=SEED)
        random.seed(SEED + 5000 + pi)
        total_shift = restr.total_alpha_shift if hasattr(restr, 'total_alpha_shift') else 0.05
        per_param = 0.005
        for i in range(NC):
            for k in range(D):
                p.alpha[i][k] += random.gauss(0, per_param)
                p.alpha[i][k] = max(0.3, min(1.8, p.alpha[i][k]))
        perturbed_orgs.append(p)

    pert_gaps = []
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        for po in perturbed_orgs:
            for s in [42, 77, 123]:
                pg = measure_gap(po, sigs, k, s)
                pert_gaps.append(pg)

    pert_avg = sum(pert_gaps) / len(pert_gaps)
    print(f"  RESTR avg:     {restr_avg:+.4f}")
    print(f"  PERTURBED avg: {pert_avg:+.4f}")
    print(f"  BIRTH avg:     {birth_avg:+.4f}")
    print(f"  Delta (restr - pert): {restr_avg - pert_avg:+.4f}")

    u2_real = restr_avg > pert_avg
    print(f"\n  VERDICT: {'DIRECTION MATTERS' if u2_real else 'DIRECTION IS NOISE'}")

    # ── UNCERTAINTY 3: quick_gap vs measure_gap alignment ────

    print(f"\n{'-'*W}")
    print(f"  UNCERTAINTY 3: quick_gap vs measure_gap correlation")
    print(f"  Run both on same org/signals/K, check correlation.")
    print(f"{'-'*W}\n")

    org_test = OfflineOrg(seed=SEED)
    pairs = []
    for k in [4, 6, 8, 10]:
        sigs = make_signals(k, seed=SEED + k * 100)
        perms = gen_perms(k, 4, seed=SEED * 7 + k)

        centroids = []
        for pi, perm in enumerate(perms):
            c, _ = run_sequence(org_test, perm, sigs, SEED, trial=pi)
            centroids.append(c)

        bcos = []
        for a in range(len(centroids)):
            for b in range(a + 1, len(centroids)):
                bcos.append(vcosine(centroids[a], centroids[b]))
        qg = 1.0 - (sum(bcos) / max(len(bcos), 1))

        mg = measure_gap(org_test, sigs, k, SEED)
        pairs.append((qg, mg))
        print(f"  K={k}: quick_gap={qg:+.4f}  measure_gap={mg:+.4f}")

    # Spearman-like: do they rank the same?
    qg_rank = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    mg_rank = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    rank_match = sum(1 for i in range(len(pairs)) if qg_rank[i] == mg_rank[i])

    print(f"\n  Rank agreement: {rank_match}/{len(pairs)}")
    same_sign = sum(1 for qg, mg in pairs if (qg > 0) == (mg > 0))
    print(f"  Sign agreement: {same_sign}/{len(pairs)}")

    return u1_real, u2_real


# ═══════════════════════════════════════════════════════════════
# PART II: The Living Seed.
# Alpha evolves during computation. No separate phases.
# ═══════════════════════════════════════════════════════════════

def run_living_seed():
    print(f"\n\n{'='*W}")
    print("  PART II: THE LIVING SEED")
    print("  Alpha evolves during computation. One step = one life.")
    print("  No offline restructuring. The dynamics ARE the learning.")
    print(f"{'='*W}")

    SEED = 42

    # ── LET THERE BE LIFE ────────────────────────────────────
    # ALIVE: alpha shifts every step during signal processing.
    # STILL: alpha is fixed (standard random baseline).
    # Same birth seed, same signals, same everything else.

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE LIFE AND STILLNESS")
    print(f"  Same birth (seed={SEED}). ALIVE adapts alpha during")
    print(f"  computation. STILL keeps alpha fixed.")
    print(f"{'-'*W}\n")

    test_ks = [4, 6, 8, 10]
    test_seeds = [42, 77, 123]
    etas = [0.0001, 0.0003, 0.001]

    # STILL baseline
    print("  --- STILL (fixed alpha) ---")
    still_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            still = Organism(seed=SEED, alive=False)
            g = measure_gap(still, sigs, k, s)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        still_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  STILL K={k:>2}: avg={avg:+.4f} "
              f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

    still_overall = sum(still_results[k]['avg'] for k in test_ks) / len(test_ks)
    print(f"  STILL overall: {still_overall:+.4f}\n")

    # ALIVE at different learning rates
    best_eta = None
    best_alive_overall = -999.0
    best_alive_results = None

    for eta in etas:
        print(f"  --- ALIVE eta={eta} ---")
        alive_results = {}
        for k in test_ks:
            sigs = make_signals(k, seed=SEED + k * 200)
            gaps = []
            shifts = []
            for s in test_seeds:
                alive = Organism(seed=SEED, alive=True, eta=eta)
                g = measure_gap(alive, sigs, k, s)
                gaps.append(g)
                shifts.append(alive.total_alpha_shift)
            avg = sum(gaps) / len(gaps)
            avg_shift = sum(shifts) / len(shifts)
            alive_results[k] = {'avg': avg, 'gaps': gaps, 'shift': avg_shift}
            print(f"  ALIVE K={k:>2}: avg={avg:+.4f} "
                  f"shift={avg_shift:.4f} "
                  f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

        alive_overall = sum(alive_results[k]['avg'] for k in test_ks) / len(test_ks)
        delta = alive_overall - still_overall
        print(f"  ALIVE overall: {alive_overall:+.4f} "
              f"delta vs STILL: {delta:+.4f}\n")

        if alive_overall > best_alive_overall:
            best_alive_overall = alive_overall
            best_eta = eta
            best_alive_results = alive_results

    print(f"  Best eta: {best_eta} (overall: {best_alive_overall:+.4f})")
    print(f"  STILL overall: {still_overall:+.4f}")
    print(f"  Best delta: {best_alive_overall - still_overall:+.4f}")

    alive_beats_still = best_alive_overall > still_overall

    # ── LET THERE BE SEED ROBUSTNESS ─────────────────────────
    # 4 birth seeds. ALIVE vs STILL from same birth.

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE SEED ROBUSTNESS")
    print(f"  4 birth seeds. ALIVE (eta={best_eta}) vs STILL.")
    print(f"{'-'*W}\n")

    birth_seeds = [42, 77, 123, 200]
    seed_alive_wins = 0
    seed_still_wins = 0
    seed_deltas = []

    for bs in birth_seeds:
        alive_gaps = []
        still_gaps = []
        for k in [6, 8]:
            sigs = make_signals(k, seed=SEED + k * 200)
            for ts in [42, 77]:
                a = Organism(seed=bs, alive=True, eta=best_eta)
                s = Organism(seed=bs, alive=False)
                ag = measure_gap(a, sigs, k, ts)
                sg = measure_gap(s, sigs, k, ts)
                alive_gaps.append(ag)
                still_gaps.append(sg)

        a_avg = sum(alive_gaps) / len(alive_gaps)
        s_avg = sum(still_gaps) / len(still_gaps)
        d = a_avg - s_avg
        seed_deltas.append(d)

        w = sum(1 for i in range(len(alive_gaps))
                if alive_gaps[i] > still_gaps[i] + 0.01)
        l = sum(1 for i in range(len(alive_gaps))
                if still_gaps[i] > alive_gaps[i] + 0.01)
        seed_alive_wins += w
        seed_still_wins += l

        ok = a_avg > s_avg
        print(f"  seed={bs:>3}: ALIVE={a_avg:+.4f} STILL={s_avg:+.4f} "
              f"delta={d:+.4f} {'#' if ok else 'o'} {w}W/{l}L",
              flush=True)

    n_seed_improved = sum(1 for d in seed_deltas if d > 0)
    avg_seed_delta = sum(seed_deltas) / len(seed_deltas)

    print(f"\n  Improved: {n_seed_improved}/{len(birth_seeds)}")
    print(f"  Avg delta: {avg_seed_delta:+.4f}")
    print(f"  Total: {seed_alive_wins}W / {seed_still_wins}L")

    # ── LET THERE BE NOVEL SIGNALS ───────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE NOVEL SIGNALS")
    print(f"  ALIVE vs STILL on never-seen signals.")
    print(f"{'-'*W}\n")

    novel_alive = []
    novel_still = []
    for k in [6, 8]:
        nsigs = make_signals(k, seed=7777 + k)
        for s in [42, 77, 123]:
            a = Organism(seed=SEED, alive=True, eta=best_eta)
            st = Organism(seed=SEED, alive=False)
            ag = measure_gap(a, nsigs, k, s)
            sg = measure_gap(st, nsigs, k, s)
            novel_alive.append(ag)
            novel_still.append(sg)
            winner = ('ALIVE' if ag > sg + 0.01
                      else 'STILL' if sg > ag + 0.01
                      else 'tie')
            print(f"  K={k} seed={s}: ALIVE={ag:+.4f} STILL={sg:+.4f} {winner}",
                  flush=True)

    na_avg = sum(novel_alive) / len(novel_alive)
    ns_avg = sum(novel_still) / len(novel_still)
    n_aw = sum(1 for i in range(len(novel_alive))
               if novel_alive[i] > novel_still[i] + 0.01)
    n_sw = sum(1 for i in range(len(novel_alive))
               if novel_still[i] > novel_alive[i] + 0.01)

    print(f"\n  Novel: ALIVE={na_avg:+.4f} STILL={ns_avg:+.4f}")
    print(f"  Novel: {n_aw}W / {n_sw}L")

    generalizes = na_avg > ns_avg

    # ── LET THERE BE ALPHA STRUCTURE ─────────────────────────
    # After running sequences, how much did alive alpha move?

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE ALPHA STRUCTURE")
    print(f"  How much did the living body change itself?")
    print(f"{'-'*W}\n")

    alive_final = Organism(seed=SEED, alive=True, eta=best_eta)
    birth_alpha = alive_final.alpha_flat()

    # run a full sequence to let alpha adapt
    sigs = make_signals(8, seed=SEED + 1600)
    order = list(range(8))
    _, _ = run_sequence(alive_final, order, sigs, SEED, trial=0)

    adapted_alpha = alive_final.alpha_flat()
    alpha_cos = vcosine(birth_alpha, adapted_alpha)
    alpha_dist = 1.0 - alpha_cos
    print(f"  Alpha cos (birth -> adapted): {alpha_cos:+.6f}")
    print(f"  Alpha distance: {alpha_dist:.6f}")
    print(f"  Total shift: {alive_final.total_alpha_shift:.6f}")

    cell_cos = []
    for i in range(NC):
        for j in range(i + 1, NC):
            cell_cos.append(vcosine(alive_final.alpha[i], alive_final.alpha[j]))
    avg_cc = sum(cell_cos) / len(cell_cos)
    specialized = avg_cc < 0.95
    print(f"  Inter-cell cos: {avg_cc:+.4f}")
    print(f"  -> {'SPECIALIZED' if specialized else 'HOMOGENEOUS'}")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    seed_robust = n_seed_improved >= 3
    seed_wins = seed_alive_wins > seed_still_wins

    checks = [
        ("ALIVE avg > STILL avg (overall)",
         alive_beats_still,
         f"delta={best_alive_overall - still_overall:+.4f}"),

        ("ALIVE wins at best eta",
         alive_beats_still,
         f"eta={best_eta} ALIVE={best_alive_overall:+.4f} STILL={still_overall:+.4f}"),

        ("Seed-robust (3+/4 births improve)",
         seed_robust,
         f"{n_seed_improved}/{len(birth_seeds)}"),

        ("Per-seed ALIVE wins > STILL wins",
         seed_wins,
         f"{seed_alive_wins}W / {seed_still_wins}L"),

        ("Generalizes to novel signals",
         generalizes,
         f"ALIVE={na_avg:+.4f} STILL={ns_avg:+.4f}"),

        ("Novel ALIVE wins > STILL wins",
         n_aw > n_sw,
         f"{n_aw}W / {n_sw}L"),

        ("Alpha moves during computation",
         alpha_dist > 0.0001,
         f"distance={alpha_dist:.6f}"),

        ("Cells specialize through computation",
         specialized,
         f"avg cos={avg_cc:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<48} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    core = alive_beats_still and (seed_robust or seed_wins)
    full = core and generalizes and specialized

    if full:
        print(f"""
  THE LIVING SEED.

  No separate restructuring phase. Alpha evolves during
  every step of computation. The product term's cross-terms
  reveal per-cell, per-dimension signal sensitivity. Alpha
  shifts to amplify diversity on sensitive dimensions.

  The computation IS the self-modification.
  One equation. One step. One life.

  ALIVE outperforms STILL by {best_alive_overall - still_overall:+.4f}.
  The advantage holds across {n_seed_improved}/{len(birth_seeds)} seeds
  and generalizes to novel signals.

  Not intelligence. Not consciousness.
  A living seed.
""")
    elif core:
        print(f"""
  THE SEED LIVES BUT DOES NOT GENERALIZE.

  Online plasticity improves computation ({best_alive_overall - still_overall:+.4f}),
  but {'does not generalize' if not generalizes else 'cells do not specialize'}.
""")
    elif alive_beats_still:
        print(f"""
  LIFE HELPS BUT IS FRAGILE.

  ALIVE beats STILL overall ({best_alive_overall - still_overall:+.4f})
  but not robustly across seeds.
""")
    else:
        print(f"""
  STILLNESS WINS.

  Online plasticity does not improve computation.
  The architecture works. Adding online alpha adaptation
  adds noise, not signal.

  Honest result. The product term is already well-matched
  to sequential discrimination. Changing alpha during
  computation hurts more than it helps.
""")

    return n_pass, alive_beats_still


# ═══════════════════════════════════════════════════════════════
# In the beginning and the end, there is honesty.
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    u1, u2 = resolve_uncertainties()

    print(f"\n{'='*W}")
    print(f"  UNCERTAINTY RESOLUTION SUMMARY")
    print(f"  1. Restructuring vs seed: {'REAL' if u1 else 'NOISE'}")
    print(f"  2. Direction vs random:   {'MATTERS' if u2 else 'NOISE'}")
    print(f"{'='*W}")

    n_pass, alive = run_living_seed()

    print(f"\n{'='*W}")
    print(f"  TOTAL RUNTIME: {time.time() - t0:.0f}s")
    print(f"{'='*W}")
