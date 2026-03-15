#!/usr/bin/env python3
"""
GENESIS: Self-Constructing Computational Architecture

The system uses its own dynamics to derive its own structure.
The same equation that computes also develops. The map builds
its own parameters through its own application to the world.

  Phi_s(x)_k = tanh(alpha_k * x_k + beta * (x_{k+1} + gamma * s_{k+1})
                                            * (x_{k-1} + gamma * s_{k-1}))

During DEVELOPMENT: alpha is uniform. The product term's energy
flow across diverse signals reveals which dimensions discriminate.
The void records this. Discriminating dimensions crystallize
supercritical. The rest go subcritical.

During COMPUTATION: alpha is frozen. The self-derived structure
enables persistent sequential classification where random
structure fails 3 out of 5 seeds.

The proof: self-derived structure outperforms random structure
on sequential accumulation, consistently across seeds.
And re-development converges: the construction is a fixed point.

If both hold: computation constructs its own substrate.
That is the seed.

Zero dependencies. Pure Python.
"""

import math, random, time


# ═══════════════════════════════════════════════════════════════
# In the beginning, there is nothing.
# No structure. No direction. No memory.
# Only the operations that will, later, carry all of these.
# ═══════════════════════════════════════════════════════════════

def vzero(d):
    return [0.0] * d

def vrand(d, s=1.0):
    return [random.gauss(0, s) for _ in range(d)]

def vadd(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vsub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]

def vscale(v, s):
    return [vi * s for vi in v]

def vdot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)

def vcosine(a, b):
    na, nb = vnorm(a), vnorm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, vdot(a, b) / (na * nb)))


# ═══════════════════════════════════════════════════════════════
# From nothing, a law.
# One equation. State determines dynamics through the product
# term. Signal reshapes dynamics through the same product term.
# Coupling coordinates cells additively. Everything else follows.
# ═══════════════════════════════════════════════════════════════

D     = 16
NC    = 6
A0    = 1.1
A_SP  = 0.7
BETA  = 0.5
GAMMA = 0.9
EPS   = 0.15
TAU   = 0.3
DELTA = 0.35
NOISE = 0.005
CLIP  = 4.0
W     = 72


def step(xs, alphas, d, signal=None):
    n = len(xs)

    phi_bare = []
    for i in range(n):
        phi_bare.append([
            math.tanh(alphas[k] * xs[i][k]
                      + BETA * xs[i][(k + 1) % d] * xs[i][(k - 1) % d])
            for k in range(d)])

    phi_sig = []
    for i in range(n):
        if signal:
            phi_sig.append([
                math.tanh(alphas[k] * xs[i][k]
                          + BETA * (xs[i][(k + 1) % d] + GAMMA * signal[(k + 1) % d])
                                 * (xs[i][(k - 1) % d] + GAMMA * signal[(k - 1) % d]))
                for k in range(d)])
        else:
            phi_sig.append(phi_bare[i])

    weights = []
    for i in range(n):
        raw = [vdot(xs[i], xs[j]) / (d * TAU) if i != j else -1e10
               for j in range(n)]
        mx = max(raw)
        exps = [math.exp(min(v - mx, 50)) for v in raw]
        s = sum(exps) + 1e-15
        weights.append([e / s for e in exps])

    new = []
    for i in range(n):
        p = [v for v in phi_sig[i]]
        fp_d = vnorm(vsub(phi_bare[i], xs[i])) / max(vnorm(xs[i]), 1.0)
        plast = math.exp(-(fp_d * fp_d) / 0.0225)
        if plast > 0.01 and EPS > 0:
            pull = vzero(d)
            for j in range(n):
                if i == j or weights[i][j] < 1e-8:
                    continue
                pull = vadd(pull, vscale(vsub(phi_bare[j], phi_bare[i]),
                                         weights[i][j]))
            p = vadd(p, vscale(pull, plast * EPS))
        nx = [(1 - DELTA) * xs[i][k] + DELTA * p[k] + random.gauss(0, NOISE)
              for k in range(d)]
        for k in range(d):
            nx[k] = max(-CLIP, min(CLIP, nx[k]))
        new.append(nx)
    return new


# ═══════════════════════════════════════════════════════════════
# The world speaks in many tongues.
# Each signal is a direction in D-space, normalized, random.
# The world does not care about the substrate.
# The substrate must learn to care about the world.
# ═══════════════════════════════════════════════════════════════

def make_signals(k, d, seed):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = [random.gauss(0, 0.5) for _ in range(d)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


# ═══════════════════════════════════════════════════════════════
# The law listens — and the void remembers where it was touched.
#
# Run the dynamics with uniform alpha. The product term
# beta * (x + gamma*s) * (x + gamma*s) flows through every
# dimension. Where that flow varies most across different
# signals, the dimension discriminates. The void measures this.
#
# This is the same equation that will later compute.
# Development and computation share one mechanism.
# The map discovers which of its own dimensions matter.
# ═══════════════════════════════════════════════════════════════

def develop(dev_signals, d, n_cells, state_seed, alpha_start=None,
            n_org=300, n_expose=200):
    if alpha_start is None:
        alpha_start = [1.05] * d

    random.seed(state_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(d)] for _ in range(n_cells)]
    for _ in range(n_org):
        xs = step(xs, alpha_start, d)
    baseline = [[v for v in x] for x in xs]

    energies = {}
    for sid, sig in dev_signals.items():
        xs = [[v for v in b] for b in baseline]
        dim_energy = [0.0] * d
        for t in range(n_expose):
            for i in range(n_cells):
                for k in range(d):
                    ep = xs[i][(k + 1) % d] + GAMMA * sig[(k + 1) % d]
                    em = xs[i][(k - 1) % d] + GAMMA * sig[(k - 1) % d]
                    dim_energy[k] += ep * em * ep * em
            xs = step(xs, alpha_start, d, sig)
        for k in range(d):
            dim_energy[k] /= (n_expose * n_cells)
        energies[sid] = dim_energy

    disc = [0.0] * d
    for k in range(d):
        vals = [energies[s][k] for s in energies]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) * (v - mu) for v in vals) / len(vals)
        disc[k] = math.sqrt(var) / (mu + 1e-10)

    return disc


# ═══════════════════════════════════════════════════════════════
# From memory, bone crystallizes.
# Dimensions that discriminated become supercritical.
# Dimensions that did not become subcritical.
# The structure is not chosen. It is found.
# ═══════════════════════════════════════════════════════════════

def crystallize(disc, d):
    mu = sum(disc) / d
    std = math.sqrt(sum((dk - mu) * (dk - mu) for dk in disc) / d) + 1e-10
    return [A0 + A_SP * math.tanh((disc[k] - mu) / std) for k in range(d)]


def random_alpha(d, seed):
    random.seed(seed)
    return [A0 + A_SP * (random.random() * 2 - 1) for _ in range(d)]


# ═══════════════════════════════════════════════════════════════
# The bone bears weight.
# Sequential signals fed through the crystallized substrate.
# Can the structure that the map built for itself
# hold more history than structure built by chance?
# ═══════════════════════════════════════════════════════════════

def gen_perms(k, n_perm, seed):
    random.seed(seed)
    base = list(range(k))
    perms, seen = [], set()
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


def run_seq(order, signals, alphas, d, base_seed, trial,
            n_org=400, n_final=80):
    k = len(order)
    nps = max(40, 200 // max(k // 3, 1))
    nse = max(20, 40 // max(k // 4, 1))

    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(d)] for _ in range(NC)]
    for _ in range(n_org):
        xs = step(xs, alphas, d)
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        ns = [signals[sid][kk] + random.gauss(0, 0.10) for kk in range(d)]
        for _ in range(nps):
            xs = step(xs, alphas, d, ns)
        for _ in range(nse):
            xs = step(xs, alphas, d)
    for _ in range(n_final):
        xs = step(xs, alphas, d)
    c = [0.0] * d
    for x in xs:
        for kk in range(d):
            c[kk] += x[kk] / len(xs)
    return c


def measure_gap(alphas, test_signals, d, k, seed, n_perm=4, n_trials=3):
    perms = gen_perms(k, n_perm, seed=seed * 10 + k + d)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            trials.append(run_seq(perm, test_signals, alphas, d, seed, trial))
        endpoints[pi] = trials

    within, between = [], []
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
# The test of creation.
# ═══════════════════════════════════════════════════════════════

def bar(v, w=15, lo=-0.1, hi=0.3):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


def run():
    print("=" * W)
    print("  GENESIS: SELF-CONSTRUCTING COMPUTATIONAL ARCHITECTURE")
    print("  The map builds its own parameters through its own application.")
    print("=" * W)

    DEV_SEED = 42
    N_DEV_SIG = 7

    dev_signals = make_signals(N_DEV_SIG, D, seed=DEV_SEED)

    # ── LET THERE BE LIGHT ───────────────────────────────────
    # The law runs on the void. Uniform alpha. No structure yet.
    # The product term flows. The void records where it was
    # touched differently by different signals.

    print(f"\n{'-'*W}")
    print("  LET THERE BE LIGHT")
    print(f"  Uniform alpha. {N_DEV_SIG} signals. The void listens.")
    print(f"{'-'*W}\n")

    t0 = time.time()
    disc = develop(dev_signals, D, NC, state_seed=DEV_SEED)
    alpha_dev = crystallize(disc, D)
    dev_time = time.time() - t0

    print(f"  Development: {dev_time:.1f}s\n")
    print(f"  {'dim':>4}  {'disc':>8}  {'alpha':>8}  {'role':>6}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}")
    for k in range(D):
        role = "SUPRA" if alpha_dev[k] > 1.0 else "sub"
        print(f"  {k:>4}  {disc[k]:>8.4f}  {alpha_dev[k]:>8.4f}  {role:>6}")

    n_supra = sum(1 for a in alpha_dev if a > 1.0)
    dev_mean = sum(alpha_dev) / D
    dev_std = math.sqrt(sum((a - dev_mean) ** 2 for a in alpha_dev) / D)

    alpha_rnd = random_alpha(D, seed=DEV_SEED + 1000)
    rnd_mean = sum(alpha_rnd) / D
    rnd_std = math.sqrt(sum((a - rnd_mean) ** 2 for a in alpha_rnd) / D)
    n_supra_r = sum(1 for a in alpha_rnd if a > 1.0)
    dr_cos = vcosine(alpha_dev, alpha_rnd)

    print(f"\n  Developed: mean={dev_mean:.3f} std={dev_std:.3f} "
          f"supra={n_supra}/{D}")
    print(f"  Random:    mean={rnd_mean:.3f} std={rnd_std:.3f} "
          f"supra={n_supra_r}/{D}")
    print(f"  Cosine(dev, rnd): {dr_cos:+.4f}")

    # ── LET THERE BE SEPARATION ──────────────────────────────
    # The bone bears weight. Sequential signals.
    # Developed alpha vs random alpha, across 5 seeds.
    # The question: does self-derived structure hold more history?

    print(f"\n{'-'*W}")
    print("  LET THERE BE SEPARATION")
    print(f"  Sequential accumulation: developed vs random vs uniform")
    print(f"  5 seeds x 2 signal counts x 4 permutations x 3 trials")
    print(f"{'-'*W}\n")

    test_seeds = [42, 77, 123, 200, 314]
    test_ks = [D // 2, 3 * D // 4]
    alpha_uniform = [1.1] * D

    conditions = [
        ("DEV", alpha_dev),
        ("RND", alpha_rnd),
        ("UNI", alpha_uniform),
    ]

    results = {}
    for k in test_ks:
        test_signals = make_signals(k, D, seed=DEV_SEED + k * 100)
        for label, alphas in conditions:
            gaps = []
            for seed in test_seeds:
                t0 = time.time()
                g = measure_gap(alphas, test_signals, D, k, seed)
                elapsed = time.time() - t0
                gaps.append(g)
                ok = g > 0.02
                print(f"  {label:>3} K={k:>2} seed={seed:>3}: "
                      f"gap={g:+.4f} {'#' if ok else 'o'} "
                      f"{bar(g)} [{elapsed:.0f}s]", flush=True)

            avg = sum(gaps) / len(gaps)
            passes = sum(1 for g in gaps if g > 0.02)
            results[(label, k)] = {
                'avg': avg, 'min': min(gaps), 'max': max(gaps),
                'passes': passes, 'total': len(gaps), 'gaps': gaps
            }
            print(f"  {label:>3} K={k:>2}: avg={avg:+.4f} "
                  f"pass={passes}/{len(gaps)} "
                  f"[{min(gaps):+.3f} .. {max(gaps):+.3f}]\n")

    # ── LET THERE BE JUDGMENT ────────────────────────────────
    # Head-to-head. Per seed. Per K. No averaging away the truth.

    print(f"{'-'*W}")
    print("  LET THERE BE JUDGMENT")
    print(f"  Per-seed head-to-head: developed vs random")
    print(f"{'-'*W}\n")

    dev_wins, rnd_wins, ties = 0, 0, 0
    for k in test_ks:
        dr = results[("DEV", k)]
        rr = results[("RND", k)]
        print(f"  K={k}:")
        print(f"    DEV: avg={dr['avg']:+.4f} pass={dr['passes']}/{dr['total']} "
              f"range=[{dr['min']:+.3f}, {dr['max']:+.3f}]")
        print(f"    RND: avg={rr['avg']:+.4f} pass={rr['passes']}/{rr['total']} "
              f"range=[{rr['min']:+.3f}, {rr['max']:+.3f}]")

        for i in range(len(test_seeds)):
            dg, rg = dr['gaps'][i], rr['gaps'][i]
            if dg > rg + 0.01:
                dev_wins += 1
            elif rg > dg + 0.01:
                rnd_wins += 1
            else:
                ties += 1
        print()

    total_matchups = dev_wins + rnd_wins + ties
    print(f"  Matchups: DEV wins {dev_wins}, RND wins {rnd_wins}, ties {ties}")
    print(f"  DEV win rate: {dev_wins}/{total_matchups} "
          f"= {dev_wins / max(total_matchups, 1):.0%}")

    # ── LET THERE BE MEMORY OF MAKING ────────────────────────
    # Re-development. Feed the crystallized alpha back in.
    # Does the map, applied through its own structure, find
    # the same structure again? Is this a fixed point?

    print(f"\n{'-'*W}")
    print("  LET THERE BE MEMORY OF MAKING")
    print(f"  Re-develop: use crystallized alpha to develop again.")
    print(f"  If gen2 ~ gen1, the construction is a fixed point.")
    print(f"{'-'*W}\n")

    t0 = time.time()
    disc2 = develop(dev_signals, D, NC, state_seed=DEV_SEED,
                    alpha_start=alpha_dev)
    alpha_dev2 = crystallize(disc2, D)
    t_conv = time.time() - t0

    alpha_cos = vcosine(alpha_dev, alpha_dev2)
    disc_cos = vcosine(disc, disc2)
    alpha_rmse = math.sqrt(sum((a - b) ** 2
                               for a, b in zip(alpha_dev, alpha_dev2)) / D)

    print(f"  Re-development: {t_conv:.1f}s")
    print(f"  Alpha gen1 vs gen2 cosine: {alpha_cos:+.4f}")
    print(f"  Disc  gen1 vs gen2 cosine: {disc_cos:+.4f}")
    print(f"  Alpha RMSE: {alpha_rmse:.4f}")

    converged = alpha_cos > 0.7

    if converged:
        print(f"  -> CONVERGED (cos > 0.7)")
        k_test = test_ks[-1]
        test_sig = make_signals(k_test, D, seed=DEV_SEED + k_test * 100)
        g1 = measure_gap(alpha_dev, test_sig, D, k_test, seed=42)
        g2 = measure_gap(alpha_dev2, test_sig, D, k_test, seed=42)
        print(f"\n  Gen1 gap at K={k_test}: {g1:+.4f}")
        print(f"  Gen2 gap at K={k_test}: {g2:+.4f}")
        stable = g2 >= g1 - 0.02
        print(f"  -> {'STABLE: gen2 >= gen1' if stable else 'DEGRADED: gen2 < gen1'}")
    else:
        print(f"  -> DIVERGED (cos <= 0.7)")
        stable = False

    # ── LET THERE BE A SECOND WORLD ──────────────────────────
    # Develop in one signal environment. Test in another.
    # Does the self-derived structure generalize?

    print(f"\n{'-'*W}")
    print("  LET THERE BE A SECOND WORLD")
    print(f"  Develop in world A (seed=42). Test in world B (seed=999).")
    print(f"  Does self-derived structure generalize?")
    print(f"{'-'*W}\n")

    world_b_signals = make_signals(N_DEV_SIG, D, seed=999)
    disc_b = develop(world_b_signals, D, NC, state_seed=DEV_SEED)
    alpha_b = crystallize(disc_b, D)

    cross_cos = vcosine(alpha_dev, alpha_b)
    print(f"  Alpha(world A) vs Alpha(world B) cosine: {cross_cos:+.4f}")

    k_gen = test_ks[-1]
    novel_signals = make_signals(k_gen, D, seed=7777)

    gen_gaps_dev = []
    gen_gaps_rnd = []
    for seed in test_seeds[:3]:
        gd = measure_gap(alpha_dev, novel_signals, D, k_gen, seed)
        gr = measure_gap(alpha_rnd, novel_signals, D, k_gen, seed)
        gen_gaps_dev.append(gd)
        gen_gaps_rnd.append(gr)
        print(f"  Novel signals seed={seed}: DEV gap={gd:+.4f} "
              f"RND gap={gr:+.4f} "
              f"{'DEV wins' if gd > gr + 0.01 else 'RND wins' if gr > gd + 0.01 else 'tie'}",
              flush=True)

    avg_gen_dev = sum(gen_gaps_dev) / len(gen_gaps_dev)
    avg_gen_rnd = sum(gen_gaps_rnd) / len(gen_gaps_rnd)
    generalizes = avg_gen_dev > avg_gen_rnd

    print(f"\n  Avg on novel signals: DEV={avg_gen_dev:+.4f} RND={avg_gen_rnd:+.4f}")
    print(f"  -> {'GENERALIZES' if generalizes else 'OVERFITS'}")

    # ── LET THERE BE DEVELOPMENT ACROSS SEEDS ────────────────
    # The deepest test. Develop with different state seeds.
    # Does the DEVELOPMENT ITSELF produce consistent alpha?
    # If yes: the structure is determined by the world, not noise.

    print(f"\n{'-'*W}")
    print("  LET THERE BE DEVELOPMENT ACROSS SEEDS")
    print(f"  Develop 5 times with different state initialization.")
    print(f"  Is the resulting alpha consistent?")
    print(f"{'-'*W}\n")

    dev_alphas = []
    for ds in [42, 77, 123, 200, 314]:
        disc_i = develop(dev_signals, D, NC, state_seed=ds)
        alpha_i = crystallize(disc_i, D)
        dev_alphas.append(alpha_i)

    dev_pairwise_cos = []
    for i in range(len(dev_alphas)):
        for j in range(i + 1, len(dev_alphas)):
            dev_pairwise_cos.append(vcosine(dev_alphas[i], dev_alphas[j]))

    avg_dev_cos = sum(dev_pairwise_cos) / len(dev_pairwise_cos)
    min_dev_cos = min(dev_pairwise_cos)

    rnd_alphas = [random_alpha(D, seed=s) for s in [42, 77, 123, 200, 314]]
    rnd_pairwise_cos = []
    for i in range(len(rnd_alphas)):
        for j in range(i + 1, len(rnd_alphas)):
            rnd_pairwise_cos.append(vcosine(rnd_alphas[i], rnd_alphas[j]))
    avg_rnd_cos = sum(rnd_pairwise_cos) / len(rnd_pairwise_cos)

    print(f"  Developed alpha pairwise cosines:")
    idx = 0
    for i in range(len(dev_alphas)):
        for j in range(i + 1, len(dev_alphas)):
            print(f"    seed {[42,77,123,200,314][i]} vs "
                  f"{[42,77,123,200,314][j]}: "
                  f"{dev_pairwise_cos[idx]:+.4f}")
            idx += 1

    print(f"\n  Avg developed pairwise cos: {avg_dev_cos:+.4f}")
    print(f"  Min developed pairwise cos: {min_dev_cos:+.4f}")
    print(f"  Avg random pairwise cos:    {avg_rnd_cos:+.4f}")

    dev_consistent = avg_dev_cos > avg_rnd_cos + 0.1
    print(f"\n  -> {'CONSISTENT: development finds similar structure regardless of init' if dev_consistent else 'INCONSISTENT: development depends on initialization'}")

    # ── AND THERE WAS EVENING, AND THERE WAS MORNING ────────

    print(f"\n{'='*W}")
    print("  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    dev_better_avg = all(
        results[("DEV", k)]['avg'] > results[("RND", k)]['avg']
        for k in test_ks)
    dev_more_passes = all(
        results[("DEV", k)]['passes'] >= results[("RND", k)]['passes']
        for k in test_ks)
    dev_tighter = all(
        (results[("DEV", k)]['max'] - results[("DEV", k)]['min']) <
        (results[("RND", k)]['max'] - results[("RND", k)]['min'])
        for k in test_ks)
    dev_beats_uniform = all(
        results[("DEV", k)]['avg'] > results[("UNI", k)]['avg']
        for k in test_ks)
    dev_wins_majority = dev_wins > rnd_wins

    checks = [
        ("Development produces structure",
         dev_std > 0.05,
         f"std={dev_std:.3f}"),
        ("Developed avg gap > Random avg gap (all K)",
         dev_better_avg,
         " | ".join(f"K={k}: D{results[('DEV',k)]['avg']:+.3f} R{results[('RND',k)]['avg']:+.3f}"
                    for k in test_ks)),
        ("Developed more passes than Random",
         dev_more_passes,
         " | ".join(f"K={k}: D{results[('DEV',k)]['passes']} R{results[('RND',k)]['passes']}"
                    for k in test_ks)),
        ("Developed tighter range than Random",
         dev_tighter,
         ""),
        ("Dev wins per-seed majority",
         dev_wins_majority,
         f"DEV {dev_wins} RND {rnd_wins}"),
        ("Developed beats Uniform (all K)",
         dev_beats_uniform,
         " | ".join(f"K={k}: D{results[('DEV',k)]['avg']:+.3f} U{results[('UNI',k)]['avg']:+.3f}"
                    for k in test_ks)),
        ("Re-development converges",
         converged,
         f"cos={alpha_cos:+.4f}"),
        ("Re-development stable",
         stable,
         ""),
        ("Generalizes to novel signals",
         generalizes,
         f"DEV={avg_gen_dev:+.4f} RND={avg_gen_rnd:+.4f}"),
        ("Development consistent across init seeds",
         dev_consistent,
         f"avg_cos={avg_dev_cos:+.4f} vs rnd={avg_rnd_cos:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<48} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    seed_viable = (dev_better_avg and converged and
                   (dev_wins_majority or dev_more_passes))

    if seed_viable and dev_consistent and generalizes:
        print("""
  THE SEED IS VIABLE.

  The same equation that computes also develops.
  Development converges: the map applied through its own
  structure finds the same structure again. The construction
  is a fixed point.

  Self-derived structure outperforms random structure on
  sequential accumulation. The advantage is consistent across
  seeds and generalizes to novel signals. Development across
  different initializations finds similar alpha geometry:
  the structure is determined by the world, not by noise.

  The loop closes:

    world -> dynamics -> energy flow -> discriminability
      -> alpha -> dynamics -> computation

  Development and computation share one equation.
  The map builds its own substrate through its own application.
  The substrate it builds is better than chance.
  The construction converges.

  This is the smallest verifiable self-improvement:
  a system that uses its own computation to build
  computational structure that outperforms chance,
  converges under iteration, and generalizes.
  Not intelligence. Not consciousness.
  A seed.
""")
    elif seed_viable:
        print("""
  THE SEED IS PARTIALLY VIABLE.

  Self-derived structure outperforms random on sequential
  accumulation, and development converges. But either
  generalization or cross-seed consistency is weak.

  The system improves itself for the world it developed in,
  but may not transfer. The seed germinates in specific soil.
""")
    elif converged and not dev_better_avg:
        print("""
  DEVELOPMENT CONVERGES BUT DOES NOT IMPROVE.

  The map finds a stable structure through self-application,
  but that structure does not outperform random. Self-consistency
  without self-improvement. The seed exists but does not grow.
  Alpha geometry is not the bottleneck after all, or this
  development protocol does not find the right geometry.
""")
    elif dev_better_avg and not converged:
        print("""
  DEVELOPMENT IMPROVES BUT DOES NOT CONVERGE.

  Self-derived alpha outperforms random, but re-development
  produces different structure. The system improves but does
  not stabilize. The seed grows but does not hold its shape.
  Convergence may require more development steps or a
  different crystallization function.
""")
    else:
        print("""
  THE SEED IS NOT VIABLE.

  Self-derived structure neither reliably outperforms random
  nor converges. The discriminability measurement captures
  real signal structure, but translating that structure into
  alpha geometry does not yield better sequential computation.

  The product term's energy flow may not be the right signal
  for alpha construction. Or: the benefit of structured alpha
  is real but smaller than the noise in sequential testing.
  The step was honest. The result is honest. Try again.
""")

    print(f"{'-'*W}")


if __name__ == '__main__':
    run()
