#!/usr/bin/env python3
"""
GENESIS: The Seed

Every previous approach accumulated restructuring. Each
alpha push was informed by discrimination, but the pushes
compounded noise. Small pushes helped. Many pushes destroyed.

The fix is not better accumulation. The fix is selection.

  1. Start with alpha (random at birth).
  2. Measure discrimination signal (per-dim, per-cell).
  3. Generate N offspring: perturb alpha using the signal.
  4. Test each offspring on discrimination.
  5. Keep the best. Discard the rest.
  6. The winner becomes the parent. Repeat.

No accumulation. No gradient. Just:
  generate (informed by discrimination) -> test -> select -> repeat.

Selection compounds inherently. Each generation starts
from the best of the previous generation. The discrimination
signal makes variants informed (not random mutation). But only
selection determines what survives. Bad variants die regardless
of how they were generated.

This is the difference between a river (accumulation: water
pushes sediment and it piles up) and evolution (selection:
variations are tested and only the fit survive).

The test: does each generation's best consistently outperform
the previous? Does the evolved organism beat random? Does it
generalize? Does the advantage compound?

Fixed D=12. Six cells. Product term architecture.
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
# From nothing, a body.
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, seed=42):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

    def alpha_flat(self):
        return [a for row in self.alpha for a in row]

    def copy_alpha(self):
        return [[self.alpha[i][k] for k in range(D)] for i in range(NC)]

    def set_alpha(self, saved):
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k] = saved[i][k]

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


# ═══════════════════════════════════════════════════════════════
# The measure of truth.
# ═══════════════════════════════════════════════════════════════

def measure_gap(org, signals, k, seed, n_perm=3, n_trials=2):
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


def quick_disc(org, signals, k, seed, n_perm=3):
    """Fast fitness estimate: 1 run per perm, measure separation."""
    perms = gen_perms(k, n_perm, seed=seed * 7 + k)
    centroids = []
    for pi, perm in enumerate(perms):
        c, _ = run_sequence(org, perm, signals, seed, trial=pi)
        centroids.append(c)
    bcos = []
    for a in range(len(centroids)):
        for b in range(a + 1, len(centroids)):
            bcos.append(vcosine(centroids[a], centroids[b]))
    return 1.0 - (sum(bcos) / max(len(bcos), 1))


# ═══════════════════════════════════════════════════════════════
# The discrimination signal.
# Run the parent on multiple permutations. The difference
# between endpoints reveals which dimensions and cells
# contribute to discrimination. This informs offspring
# generation — but does NOT directly modify alpha.
# Only selection modifies alpha.
# ═══════════════════════════════════════════════════════════════

def discrimination_signal(org, signals, k, seed):
    """Per-dimension and per-cell discrimination from the parent."""
    perms = gen_perms(k, 3, seed=seed * 7 + k)
    centroids = []
    states = []
    for pi, perm in enumerate(perms):
        c, xs = run_sequence(org, perm, signals, seed, trial=pi)
        centroids.append(c)
        states.append(xs)

    n = len(centroids)
    dim_disc = [0.0] * D
    n_pairs = 0
    for a in range(n):
        for b in range(a + 1, n):
            for k_ in range(D):
                dim_disc[k_] += abs(centroids[a][k_] - centroids[b][k_])
            n_pairs += 1
    if n_pairs > 0:
        dim_disc = [d / n_pairs for d in dim_disc]

    cell_disc = [0.0] * NC
    for a in range(n):
        for b in range(a + 1, n):
            for i in range(NC):
                diff = sum(abs(states[a][i][k_] - states[b][i][k_])
                           for k_ in range(D)) / D
                cell_disc[i] += diff
    if n_pairs > 0:
        cell_disc = [c / n_pairs for c in cell_disc]

    return dim_disc, cell_disc


# ═══════════════════════════════════════════════════════════════
# Offspring generation.
# Perturb parent alpha using discrimination signal for direction,
# random noise for exploration. Multiple offspring per generation.
# Only the best survives.
# ═══════════════════════════════════════════════════════════════

def make_offspring(parent_alpha, dim_disc, cell_disc,
                   n_offspring, mut_rate, oseed):
    """Generate n_offspring variants of parent alpha."""

    dd_mu = sum(dim_disc) / D
    dd_std = math.sqrt(sum((d - dd_mu)**2 for d in dim_disc) / D) + 1e-10
    cd_mu = sum(cell_disc) / NC
    cd_std = math.sqrt(sum((c - cd_mu)**2 for c in cell_disc) / NC) + 1e-10

    offspring = []
    random.seed(oseed)

    for oi in range(n_offspring):
        child = [[parent_alpha[i][k] for k in range(D)] for i in range(NC)]

        for i in range(NC):
            cell_z = (cell_disc[i] - cd_mu) / cd_std
            for k in range(D):
                dim_z = (dim_disc[k] - dd_mu) / dd_std

                # informed perturbation: scale with discrimination
                disc_weight = 0.3 + 0.7 * max(0, math.tanh(dim_z))
                cell_weight = 0.3 + 0.7 * max(0, math.tanh(cell_z))

                # push alpha away from column mean on disc dims
                col_mean = sum(parent_alpha[j][k] for j in range(NC)) / NC
                dev = parent_alpha[i][k] - col_mean

                noise = random.gauss(0, 1.0)
                if abs(dev) > 0.02 and dim_z > 0:
                    direction = 1.0 if dev > 0 else -1.0
                    push = mut_rate * (disc_weight * cell_weight * direction * 0.4
                                       + 0.6 * noise)
                else:
                    push = mut_rate * noise

                child[i][k] += push
                child[i][k] = max(0.3, min(1.8, child[i][k]))

        offspring.append(child)

    return offspring


# ═══════════════════════════════════════════════════════════════
# The selective development loop.
# Generate -> Test -> Select -> Repeat.
# ═══════════════════════════════════════════════════════════════

def selective_develop(org, n_gen=10, n_offspring=4, mut_rate=0.08,
                      base_seed=42, verbose=True):
    """Selective evolution with discrimination-informed offspring."""

    gen_log = []
    alpha_snapshots = [org.alpha_flat()]
    parent_alpha = org.copy_alpha()

    for gen in range(n_gen):
        k = 3 + gen
        signals = make_signals(k, seed=base_seed + gen * 100)

        t0 = time.time()

        # measure discrimination signal from parent
        org.set_alpha(parent_alpha)
        dim_disc, cell_disc = discrimination_signal(org, signals, k, base_seed)

        # test parent fitness
        org.set_alpha(parent_alpha)
        parent_fit = quick_disc(org, signals, k, base_seed)

        # generate and test offspring
        children = make_offspring(
            parent_alpha, dim_disc, cell_disc,
            n_offspring, mut_rate,
            oseed=base_seed * 100 + gen * 17)

        best_fit = parent_fit
        best_alpha = parent_alpha
        best_label = "parent"

        for ci, child_alpha in enumerate(children):
            org.set_alpha(child_alpha)
            fit = quick_disc(org, signals, k, base_seed)
            if fit > best_fit:
                best_fit = fit
                best_alpha = child_alpha
                best_label = f"child_{ci}"

        # selection: keep the best
        parent_alpha = [[best_alpha[i][k_] for k_ in range(D)]
                        for i in range(NC)]
        org.set_alpha(parent_alpha)
        alpha_snapshots.append(org.alpha_flat())

        elapsed = time.time() - t0

        improved = best_label != "parent"
        gen_log.append({
            'gen': gen, 'k': k,
            'parent_fit': parent_fit, 'best_fit': best_fit,
            'improved': improved, 'label': best_label,
            'elapsed': elapsed
        })

        if verbose:
            sym = '\u2191' if improved else '='
            print(f"  {sym} Gen {gen:>2}: K={k:>2} "
                  f"parent={parent_fit:+.4f} best={best_fit:+.4f} "
                  f"{best_label:<9} [{elapsed:.1f}s]",
                  flush=True)

    return alpha_snapshots, gen_log, parent_alpha


# ═══════════════════════════════════════════════════════════════
# The test of creation.
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * W)
    print("  GENESIS: THE SEED")
    print("  Generate -> Test -> Select -> Repeat.")
    print("  Discrimination informs. Selection decides.")
    print("  No accumulation. Only survival.")
    print("=" * W)

    SEED = 42
    t_start = time.time()

    # ── LET THERE BE A BODY ──────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE A BODY")
    print(f"  D={D}, {NC} cells, {NC*D} alpha parameters.")
    print(f"{'-'*W}\n")

    org = Organism(seed=SEED)
    birth_alpha = org.copy_alpha()
    birth_flat = org.alpha_flat()

    # ── LET THERE BE SELECTION ───────────────────────────────

    print(f"{'-'*W}")
    print(f"  LET THERE BE SELECTION")
    print(f"  10 generations. 4 offspring each. K escalates 3..12.")
    print(f"  Discrimination-informed variation. Fitness-based selection.")
    print(f"{'-'*W}\n")

    alpha_snaps, gen_log, evolved_alpha = selective_develop(
        org, n_gen=10, n_offspring=4, mut_rate=0.08,
        base_seed=SEED)

    n_improved = sum(1 for g in gen_log if g['improved'])
    print(f"\n  Generations that improved: {n_improved}/{len(gen_log)}")
    print(f"  Development time: {sum(g['elapsed'] for g in gen_log):.0f}s")

    # ── LET THERE BE CONVERGENCE ─────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE CONVERGENCE")
    print(f"{'-'*W}\n")

    conv_cos = []
    for i in range(1, len(alpha_snaps)):
        conv_cos.append(vcosine(alpha_snaps[i-1], alpha_snaps[i]))

    birth_to_evolved = vcosine(birth_flat, alpha_snaps[-1])
    first_cos = conv_cos[0] if conv_cos else 0
    last_cos = conv_cos[-1] if conv_cos else 0
    converging = last_cos >= first_cos - 0.001
    converged = last_cos > 0.995

    print(f"  Birth -> evolved cos: {birth_to_evolved:+.6f}")
    print(f"  Alpha distance: {1.0 - birth_to_evolved:.6f}")
    print(f"  First step cos: {first_cos:+.6f}")
    print(f"  Last step cos:  {last_cos:+.6f}")
    print(f"  -> {'CONVERGED' if converged else 'CONVERGING' if converging else 'DIVERGING'}")

    # ── LET THERE BE COMPOUNDING ─────────────────────────────
    # The core test: does fitness monotonically increase?

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE COMPOUNDING")
    print(f"  Does each generation's best outperform the previous?")
    print(f"{'-'*W}\n")

    fits = [g['best_fit'] for g in gen_log]
    early_fit = sum(fits[:4]) / 4
    late_fit = sum(fits[-4:]) / 4
    monotonic_count = sum(1 for i in range(1, len(fits)) if fits[i] >= fits[i-1] - 0.02)

    print(f"  Early avg fitness: {early_fit:+.4f}")
    print(f"  Late avg fitness:  {late_fit:+.4f}")
    print(f"  Monotonic steps:   {monotonic_count}/{len(fits)-1}")
    compounds = late_fit > early_fit

    # ── LET THERE BE COMPARISON ──────────────────────────────
    # Evolved vs Birth (same seed) vs Random (different seeds).
    # Rigorous measure_gap, 3 seeds, K=4,6,8.

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE COMPARISON")
    print(f"  EVOLVED vs BIRTH vs 3 RANDOM. 3 seeds. K=4,6,8.")
    print(f"{'-'*W}\n")

    test_seeds = [42, 77, 123]
    test_ks = [4, 6, 8]

    rnd_orgs = [Organism(seed=SEED + 2000 + i * 111) for i in range(3)]

    def test_alpha(alpha, label, ks, seeds):
        results = {}
        for k in ks:
            sigs = make_signals(k, seed=SEED + k * 200)
            gaps = []
            for s in seeds:
                test = Organism(seed=SEED)
                test.set_alpha(alpha)
                g = measure_gap(test, sigs, k, s)
                gaps.append(g)
            avg = sum(gaps) / len(gaps)
            passes = sum(1 for g in gaps if g > 0.02)
            results[k] = {'avg': avg, 'passes': passes, 'gaps': gaps}
            print(f"  {label:>7} K={k:>2}: avg={avg:+.4f} pass={passes}/3 "
                  f"{bar(avg)} [{min(gaps):+.3f}..{max(gaps):+.3f}]",
                  flush=True)
        return results

    evol_results = test_alpha(evolved_alpha, "EVOLVED", test_ks, test_seeds)

    birth_results = test_alpha(birth_alpha, "BIRTH", test_ks, test_seeds)

    # random: aggregate across 3 random organisms
    rnd_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        all_gaps = []
        for ro in rnd_orgs:
            for s in test_seeds:
                g = measure_gap(ro, sigs, k, s)
                all_gaps.append(g)
        avg = sum(all_gaps) / len(all_gaps)
        passes = sum(1 for g in all_gaps if g > 0.02)
        rnd_results[k] = {'avg': avg, 'passes': passes, 'gaps': all_gaps}
        print(f"    RND   K={k:>2}: avg={avg:+.4f} pass={passes}/9 "
              f"{bar(avg)} [{min(all_gaps):+.3f}..{max(all_gaps):+.3f}]",
              flush=True)
    print()

    evol_overall = sum(evol_results[k]['avg'] for k in test_ks) / len(test_ks)
    birth_overall = sum(birth_results[k]['avg'] for k in test_ks) / len(test_ks)
    rnd_overall = sum(rnd_results[k]['avg'] for k in test_ks) / len(test_ks)

    print(f"  EVOLVED overall: {evol_overall:+.4f}")
    print(f"  BIRTH   overall: {birth_overall:+.4f}")
    print(f"  RND     overall: {rnd_overall:+.4f}")

    d_evol_birth = evol_overall - birth_overall
    d_evol_rnd = evol_overall - rnd_overall
    d_birth_rnd = birth_overall - rnd_overall

    print(f"\n  EVOLVED vs BIRTH: {d_evol_birth:+.4f}")
    print(f"  EVOLVED vs RND:   {d_evol_rnd:+.4f}")
    print(f"  BIRTH   vs RND:   {d_birth_rnd:+.4f}")

    # per-seed head-to-head
    def h2h(la_results, lb_results, la, lb, n_rnd=1):
        w, l, t = 0, 0, 0
        for k in test_ks:
            ga = la_results[k]['gaps']
            gb = lb_results[k]['gaps']
            for si in range(len(test_seeds)):
                a = ga[si]
                if n_rnd > 1:
                    b = sum(gb[ri * 3 + si] for ri in range(n_rnd)) / n_rnd
                else:
                    b = gb[si]
                if a > b + 0.01:
                    w += 1
                elif b > a + 0.01:
                    l += 1
                else:
                    t += 1
        return w, l, t

    eb_w, eb_l, eb_t = h2h(evol_results, birth_results, "EVOLVED", "BIRTH")
    er_w, er_l, er_t = h2h(evol_results, rnd_results, "EVOLVED", "RND", n_rnd=3)

    print(f"\n  EVOLVED vs BIRTH: {eb_w}W / {eb_l}L / {eb_t}T")
    print(f"  EVOLVED vs RND:   {er_w}W / {er_l}L / {er_t}T")

    # ── LET THERE BE SEED ROBUSTNESS ─────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE SEED ROBUSTNESS")
    print(f"  3 birth seeds. Same selective protocol. Each vs own birth.")
    print(f"{'-'*W}\n")

    b_seeds = [42, 77, 200]
    seed_results = []

    for bs in b_seeds:
        evo = Organism(seed=bs)
        birth_a = evo.copy_alpha()
        _, _, evolved_a = selective_develop(
            evo, n_gen=10, n_offspring=4, mut_rate=0.08,
            base_seed=bs, verbose=False)

        evolved_gaps = []
        birth_gaps = []
        for k in [6, 8]:
            sigs = make_signals(k, seed=SEED + k * 200)
            for ts in [42, 77]:
                te = Organism(seed=bs)
                te.set_alpha(evolved_a)
                tb = Organism(seed=bs)
                tb.set_alpha(birth_a)
                eg = measure_gap(te, sigs, k, ts)
                bg = measure_gap(tb, sigs, k, ts)
                evolved_gaps.append(eg)
                birth_gaps.append(bg)

        e_avg = sum(evolved_gaps) / len(evolved_gaps)
        b_avg = sum(birth_gaps) / len(birth_gaps)
        d = e_avg - b_avg
        w = sum(1 for i in range(len(evolved_gaps))
                if evolved_gaps[i] > birth_gaps[i] + 0.01)
        l = sum(1 for i in range(len(evolved_gaps))
                if birth_gaps[i] > evolved_gaps[i] + 0.01)

        seed_results.append({
            'seed': bs, 'evolved': e_avg, 'birth': b_avg,
            'delta': d, 'wins': w, 'losses': l
        })

        ok = e_avg > b_avg
        print(f"  seed={bs:>3}: EVOLVED={e_avg:+.4f} BIRTH={b_avg:+.4f} "
              f"delta={d:+.4f} {'#' if ok else 'o'} {w}W/{l}L",
              flush=True)

    n_seed_improved = sum(1 for r in seed_results if r['delta'] > 0)
    avg_seed_delta = sum(r['delta'] for r in seed_results) / len(seed_results)
    total_sw = sum(r['wins'] for r in seed_results)
    total_sl = sum(r['losses'] for r in seed_results)

    print(f"\n  Improved: {n_seed_improved}/{len(b_seeds)}")
    print(f"  Avg delta: {avg_seed_delta:+.4f}")
    print(f"  Total: {total_sw}W / {total_sl}L")

    # ── LET THERE BE NOVEL SIGNALS ───────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE NOVEL SIGNALS")
    print(f"  Evolved vs birth on signals never seen.")
    print(f"{'-'*W}\n")

    novel_evol = []
    novel_birth = []
    for k in [6, 8]:
        nsigs = make_signals(k, seed=7777 + k)
        for s in [42, 77, 123]:
            te = Organism(seed=SEED)
            te.set_alpha(evolved_alpha)
            tb = Organism(seed=SEED)
            tb.set_alpha(birth_alpha)
            eg = measure_gap(te, nsigs, k, s)
            bg = measure_gap(tb, nsigs, k, s)
            novel_evol.append(eg)
            novel_birth.append(bg)
            winner = ('EVOLVED' if eg > bg + 0.01
                      else 'BIRTH' if bg > eg + 0.01
                      else 'tie')
            print(f"  K={k} seed={s}: EVOLVED={eg:+.4f} BIRTH={bg:+.4f} {winner}",
                  flush=True)

    ne_avg = sum(novel_evol) / len(novel_evol)
    nb_avg = sum(novel_birth) / len(novel_birth)
    n_ew = sum(1 for i in range(len(novel_evol))
               if novel_evol[i] > novel_birth[i] + 0.01)
    n_bw = sum(1 for i in range(len(novel_evol))
               if novel_birth[i] > novel_evol[i] + 0.01)
    generalizes = ne_avg > nb_avg

    print(f"\n  Novel: EVOLVED={ne_avg:+.4f} BIRTH={nb_avg:+.4f}")
    print(f"  Novel: {n_ew}W / {n_bw}L")
    print(f"  -> {'GENERALIZES' if generalizes else 'OVERFITS'}")

    # ── LET THERE BE ALPHA STRUCTURE ─────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE ALPHA STRUCTURE")
    print(f"{'-'*W}\n")

    org.set_alpha(evolved_alpha)
    for i in range(NC):
        a_mean = sum(evolved_alpha[i]) / D
        a_std = math.sqrt(sum((evolved_alpha[i][k] - a_mean)**2
                              for k in range(D)) / D)
        supra = sum(1 for k in range(D) if evolved_alpha[i][k] > 1.0)
        print(f"  Cell {i}: mean={a_mean:.3f} std={a_std:.3f} supra={supra}/{D}")

    cell_cos = []
    for i in range(NC):
        for j in range(i + 1, NC):
            cell_cos.append(vcosine(evolved_alpha[i], evolved_alpha[j]))
    avg_cc = sum(cell_cos) / len(cell_cos)
    min_cc = min(cell_cos)
    specialized = avg_cc < 0.95

    print(f"\n  Inter-cell cos: avg={avg_cc:+.4f} min={min_cc:+.4f}")
    print(f"  -> {'SPECIALIZED' if specialized else 'HOMOGENEOUS'}")

    # birth structure for comparison
    b_cell_cos = []
    for i in range(NC):
        for j in range(i + 1, NC):
            b_cell_cos.append(vcosine(birth_alpha[i], birth_alpha[j]))
    birth_cc = sum(b_cell_cos) / len(b_cell_cos)
    print(f"  Birth inter-cell cos: {birth_cc:+.4f}")
    more_diverse = avg_cc < birth_cc
    print(f"  Evolution {'increased' if more_diverse else 'decreased'} cell diversity")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    evol_beats_birth = d_evol_birth > 0
    evol_beats_rnd = d_evol_rnd > 0
    evol_wins_birth = eb_w > eb_l
    evol_wins_rnd = er_w > er_l
    seed_robust = n_seed_improved >= 2 and total_sw >= total_sl

    checks = [
        ("EVOLVED avg > BIRTH avg",
         evol_beats_birth,
         f"delta={d_evol_birth:+.4f}"),

        ("EVOLVED avg > RND avg",
         evol_beats_rnd,
         f"delta={d_evol_rnd:+.4f}"),

        ("EVOLVED wins per-seed vs BIRTH",
         evol_wins_birth,
         f"{eb_w}W / {eb_l}L / {eb_t}T"),

        ("EVOLVED wins per-seed vs RND",
         evol_wins_rnd,
         f"{er_w}W / {er_l}L / {er_t}T"),

        ("Compounding (late fitness > early)",
         compounds,
         f"early={early_fit:+.4f} late={late_fit:+.4f}"),

        ("Seed-robust (2+/3 improve, wins >= losses)",
         seed_robust,
         f"{n_seed_improved}/{len(b_seeds)} improve, {total_sw}W/{total_sl}L"),

        ("Generalizes to novel signals",
         generalizes,
         f"EVOLVED={ne_avg:+.4f} BIRTH={nb_avg:+.4f}"),

        ("Novel wins > losses",
         n_ew > n_bw,
         f"{n_ew}W / {n_bw}L"),

        ("Alpha converges",
         converging,
         f"last cos={last_cos:.6f}"),

        ("Cells specialize",
         specialized,
         f"avg cos={avg_cc:+.4f}"),

        ("Selection increased diversity",
         more_diverse,
         f"evolved={avg_cc:+.4f} birth={birth_cc:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<50} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    core = evol_beats_birth and evol_beats_rnd
    robust = core and seed_robust
    full = robust and generalizes and compounds and specialized

    if full:
        print(f"""
  THE SEED IS VIABLE.

  Fixed D={D}. {NC} cells. 10 generations of selection.
  Discrimination-informed variation. Fitness-based selection.

  Each generation: measure discrimination signal from parent,
  generate 4 offspring with informed perturbation, test all,
  keep the best. No accumulation. Only survival.

  EVOLVED vs BIRTH: {d_evol_birth:+.4f}
  EVOLVED vs RND:   {d_evol_rnd:+.4f}
  Seed robustness:  {n_seed_improved}/{len(b_seeds)} improve
  Novel signals:    EVOLVED={ne_avg:+.4f} BIRTH={nb_avg:+.4f}
  Compounding:      early={early_fit:+.4f} late={late_fit:+.4f}

  The product term that computes also reveals the
  discrimination signal. The signal informs variation.
  Selection keeps what works. Each generation starts
  from the best of the last. Improvement compounds.

  One equation. Selection, not accumulation.
  The seed that survives is the seed that discriminates.
""")
    elif robust:
        print(f"""
  THE SEED SURVIVES BUT DOES NOT GENERALIZE.

  Selection produces alpha that beats both birth and random,
  robustly across seeds. But {'does not generalize' if not generalizes
  else 'cells do not specialize' if not specialized
  else 'improvement does not compound'}.
""")
    elif core:
        print(f"""
  SELECTION HELPS BUT IS NOT ROBUST.

  Evolved alpha beats birth ({d_evol_birth:+.4f}) and
  random ({d_evol_rnd:+.4f}), but the advantage is not
  consistent across birth seeds.

  Selection works on this seed. Universality not proven.
""")
    elif evol_beats_birth:
        print(f"""
  SELECTION BEATS BIRTH BUT NOT RANDOM.

  Evolution improved on the starting point ({d_evol_birth:+.4f})
  but the evolved alpha does not outperform fresh random draws.
  Selection finds a better local variant, not a better region.
""")
    elif evol_beats_rnd:
        print(f"""
  SELECTION BEATS RANDOM BUT NOT BIRTH.

  The birth alpha was already good. Selection couldn't improve
  on it, but the evolved alpha still outperforms other random
  draws ({d_evol_rnd:+.4f}).
""")
    else:
        print(f"""
  SELECTION DOES NOT HELP.

  The evolved alpha does not outperform birth or random.
  Discrimination-informed variation and fitness selection
  did not find better alpha than chance. The architecture
  already works. Alpha topology matters less than we thought.

  Honest result. This is what the data says.
""")

    # ── GROWTH TRACE ─────────────────────────────────────────

    print(f"{'-'*W}")
    print(f"  GROWTH TRACE")
    print(f"{'-'*W}\n")

    for g in gen_log:
        sym = '\u2191' if g['improved'] else '='
        print(f"  {sym} gen={g['gen']:>2} K={g['k']:>2} "
              f"parent={g['parent_fit']:+.4f} "
              f"best={g['best_fit']:+.4f} "
              f"{g['label']}")

    print(f"\n  Generations that improved: {n_improved}/{len(gen_log)}")
    print(f"  Alpha distance from birth: {1.0 - birth_to_evolved:.6f}")
    print(f"  Total runtime: {time.time() - t_start:.0f}s")
    print(f"\n{'-'*W}")


if __name__ == '__main__':
    run()
