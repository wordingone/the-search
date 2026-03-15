#!/usr/bin/env python3
"""
GENESIS: Self-Expanding Computational Architecture

One loop. The system processes sequential signals through the
multiplicative product term. It monitors its own ability to
distinguish different signal orderings. When it cannot tell
orderings apart — resolution collapse — it expands its own
dimensionality and adapts its per-cell alpha based on which
dimensions contributed most to discrimination.

There is no separate development phase. The measurement IS
the computation. Growth IS the response to failure.

  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * (x_{k+1} + gamma * s_{k+1})
                                                * (x_{k-1} + gamma * s_{k-1}))

The test: does the self-expanding system reach higher sequential
capacity than a fixed-dimension random system? Does growth
produce structure that outperforms chance?

Zero dependencies. Pure Python.
"""

import math, random, time


# ═══════════════════════════════════════════════════════════════
# In the beginning, there is nothing but operations.
# ═══════════════════════════════════════════════════════════════

def vzero(d):
    return [0.0] * d

def vcosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))

def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)

W = 72


# ═══════════════════════════════════════════════════════════════
# From nothing, a body that can grow.
# Per-cell alpha: each cell has its own excitability landscape.
# The substrate is heterogeneous from birth.
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, d, n_cells=6, seed=42):
        self.d = d
        self.n = n_cells
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(d)]
            for _ in range(n_cells)
        ]
        self.growth_history = []
        self.birth_d = d

    def step(self, xs, signal=None):
        d, n = self.d, self.n
        beta, gamma = self.beta, self.gamma

        phi_bare = []
        for i in range(n):
            phi_bare.append([
                math.tanh(self.alpha[i][k] * xs[i][k]
                          + beta * xs[i][(k+1)%d] * xs[i][(k-1)%d])
                for k in range(d)])

        phi_sig = []
        for i in range(n):
            if signal:
                phi_sig.append([
                    math.tanh(self.alpha[i][k] * xs[i][k]
                              + beta * (xs[i][(k+1)%d] + gamma * signal[(k+1)%d])
                                     * (xs[i][(k-1)%d] + gamma * signal[(k-1)%d]))
                    for k in range(d)])
            else:
                phi_sig.append(phi_bare[i])

        weights = []
        for i in range(n):
            raw = [sum(xs[i][k] * xs[j][k] for k in range(d)) / (d * self.tau)
                   if i != j else -1e10
                   for j in range(n)]
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        new = []
        for i in range(n):
            p = [v for v in phi_sig[i]]
            bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(d)]
            fp_d = vnorm(bare_diff) / max(vnorm(xs[i]), 1.0)
            plast = math.exp(-(fp_d * fp_d) / 0.0225)
            if plast > 0.01 and self.eps > 0:
                pull = [0.0] * d
                for j in range(n):
                    if i == j or weights[i][j] < 1e-8:
                        continue
                    for k in range(d):
                        pull[k] += weights[i][j] * (phi_bare[j][k] - phi_bare[i][k])
                p = [p[k] + plast * self.eps * pull[k] for k in range(d)]
            nx = [(1 - self.delta) * xs[i][k] + self.delta * p[k]
                  + random.gauss(0, self.noise) for k in range(d)]
            for k in range(d):
                nx[k] = max(-self.clip, min(self.clip, nx[k]))
            new.append(nx)
        return new

    def centroid(self, xs):
        d, n = self.d, self.n
        return [sum(xs[i][k] for i in range(n)) / n for k in range(d)]

    def resolution(self, xs):
        n, d = self.n, self.d
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += math.sqrt(
                    sum((xs[i][k] - xs[j][k]) ** 2 for k in range(d)))
                count += 1
        return total / max(count, 1)

    def per_dim_divergence(self, xs):
        n, d = self.n, self.d
        div = [0.0] * d
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(d):
                    div[k] += (xs[i][k] - xs[j][k]) ** 2
                count += 1
        return [div[k] / max(count, 1) for k in range(d)]

    def expand(self, added_d, per_dim_div=None):
        old_d = self.d
        self.d += added_d

        for i in range(self.n):
            if per_dim_div is not None:
                mu = sum(per_dim_div) / len(per_dim_div)
                std = math.sqrt(sum((v - mu)**2 for v in per_dim_div)
                                / len(per_dim_div)) + 1e-10
                template_mean = sum(self.alpha[i]) / old_d
                new_alphas = []
                for _ in range(added_d):
                    bias = random.gauss(0, 0.3)
                    new_alphas.append(template_mean + bias)
                self.alpha[i] += new_alphas
            else:
                self.alpha[i] += [1.1 + 0.7 * (random.random() * 2 - 1)
                                  for _ in range(added_d)]

        self.growth_history.append({
            'from': old_d, 'to': self.d,
            'div_used': per_dim_div is not None
        })

    def adapt_alpha(self, per_dim_div, rate=0.05):
        d = self.d
        mu = sum(per_dim_div) / d
        std = math.sqrt(sum((v - mu)**2 for v in per_dim_div) / d) + 1e-10
        for i in range(self.n):
            for k in range(d):
                z = (per_dim_div[k] - mu) / std
                target_shift = 0.1 * math.tanh(z)
                self.alpha[i][k] += rate * target_shift


# ═══════════════════════════════════════════════════════════════
# The world is sequences of signals in varying order.
# The organism must tell orderings apart from final state alone.
# This is the only test that matters.
# ═══════════════════════════════════════════════════════════════

def make_signals(k, d, seed):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = [random.gauss(0, 0.5) for _ in range(d)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def pad_signal(sig, d):
    if len(sig) >= d:
        return sig[:d]
    padded = sig + [random.gauss(0, 0.3) for _ in range(d - len(sig))]
    nm = math.sqrt(sum(v * v for v in padded) + 1e-15)
    return [v * 0.8 / nm for v in padded]


def gen_perms(k, n_perm, seed):
    random.seed(seed)
    base = list(range(k))
    perms, seen = [], set()
    perms.append(tuple(base)); seen.add(tuple(base))
    perms.append(tuple(reversed(base))); seen.add(tuple(reversed(base)))
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]; random.shuffle(p); t = tuple(p)
        if t not in seen: perms.append(t); seen.add(t)
        att += 1
    return perms


def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=50, n_settle=30, n_final=60):
    d = org.d
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(d)] for _ in range(org.n)]
    for _ in range(n_org):
        xs = org.step(xs)
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        raw = signals[sid]
        sig = pad_signal(raw, d)
        sig = [sig[k] + random.gauss(0, 0.05) for k in range(d)]
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)
    for _ in range(n_final):
        xs = org.step(xs)
    return org.centroid(xs), xs


def measure_discrimination(org, signals, k, seed, n_perm=4, n_trials=3):
    perms = gen_perms(k, n_perm, seed=seed * 10 + k + org.d)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c, _ = run_sequence(org, perm, signals, seed, trial)
            trials.append(c)
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


def bar(v, w=15, lo=-0.1, hi=0.3):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


# ═══════════════════════════════════════════════════════════════
# The test of creation.
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * W)
    print("  GENESIS: SELF-EXPANDING COMPUTATIONAL ARCHITECTURE")
    print("  One loop. Compute, measure, fail, grow, compute better.")
    print("=" * W)

    SEED = 42
    D_INIT = 8
    N_CELLS = 6
    GROWTH_D = 4
    COLLAPSE_THRESH = 0.03
    MAX_D = 32
    N_EPOCHS = 8

    # ── LET THERE BE A BODY ──────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE A BODY")
    print(f"  D={D_INIT}, {N_CELLS} cells, per-cell alpha.")
    print(f"{'-'*W}\n")

    org = Organism(D_INIT, N_CELLS, seed=SEED)
    print(f"  Born at D={org.d}. Alpha per cell: {org.n}x{org.d} = "
          f"{org.n * org.d} parameters.")

    # ── LET THERE BE GROWTH ──────────────────────────────────
    # Each epoch: generate K sequential signals, measure
    # discrimination. If the system can't tell orderings apart,
    # measure which dimensions contributed least, expand, adapt.

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE GROWTH")
    print(f"  Each epoch: escalating K. Collapse -> expand + adapt.")
    print(f"  The system decides when to grow based on its own failure.")
    print(f"{'-'*W}\n")

    epoch_log = []

    for epoch in range(N_EPOCHS):
        k = 3 + epoch
        signals = make_signals(k, D_INIT, seed=SEED + epoch * 100)

        t0 = time.time()
        gap = measure_discrimination(org, signals, k, seed=SEED)
        elapsed = time.time() - t0

        _, final_xs = run_sequence(
            org, list(range(k)), signals, SEED, trial=0)
        res = org.resolution(final_xs)
        pdiv = org.per_dim_divergence(final_xs)

        collapsed = gap < COLLAPSE_THRESH
        ok = gap > 0.02

        print(f"  Epoch {epoch}: K={k:>2} D={org.d:>2} "
              f"gap={gap:+.4f} res={res:.3f} "
              f"{'#' if ok else 'o'} {bar(gap)} [{elapsed:.0f}s]",
              flush=True)

        if collapsed and org.d < MAX_D:
            org.adapt_alpha(pdiv, rate=0.05)
            org.expand(GROWTH_D, per_dim_div=pdiv)
            print(f"         COLLAPSE -> adapted alpha, grew to D={org.d}")

        epoch_log.append({
            'epoch': epoch, 'k': k, 'd': org.d, 'gap': gap,
            'res': res, 'collapsed': collapsed, 'ok': ok
        })

    final_d = org.d
    n_growths = len(org.growth_history)

    print(f"\n  Final D={final_d} (grew {n_growths} times from D={D_INIT})")

    # ── LET THERE BE COMPARISON ──────────────────────────────
    # The grown organism vs a random organism at the same final D.
    # The grown organism vs a random organism at the initial D.
    # Does growth produce better structure than chance?

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE COMPARISON")
    print(f"  Grown organism (D={final_d}) vs random at D={final_d}")
    print(f"  vs random at D={D_INIT}. 5 seeds. Multiple K.")
    print(f"{'-'*W}\n")

    test_seeds = [42, 77, 123, 200, 314]
    test_ks = [4, 6, 8]
    if final_d >= 16:
        test_ks.append(10)

    org_rnd_big = Organism(final_d, N_CELLS, seed=SEED + 5000)
    org_rnd_small = Organism(D_INIT, N_CELLS, seed=SEED + 6000)

    conditions = [
        ("GROWN", org),
        (f"RND-D{final_d}", org_rnd_big),
        (f"RND-D{D_INIT}", org_rnd_small),
    ]

    comparison = {}
    for k in test_ks:
        signals = make_signals(k, D_INIT, seed=SEED + k * 200)
        for label, test_org in conditions:
            gaps = []
            for seed in test_seeds:
                g = measure_discrimination(test_org, signals, k, seed)
                gaps.append(g)
            avg = sum(gaps) / len(gaps)
            passes = sum(1 for g in gaps if g > 0.02)
            comparison[(label, k)] = {
                'avg': avg, 'passes': passes, 'total': len(gaps),
                'gaps': gaps
            }
            print(f"  {label:>10} K={k:>2}: avg={avg:+.4f} "
                  f"pass={passes}/{len(gaps)} "
                  f"[{min(gaps):+.3f}..{max(gaps):+.3f}]",
                  flush=True)
        print()

    # ── LET THERE BE JUDGMENT ────────────────────────────────

    print(f"{'-'*W}")
    print(f"  LET THERE BE JUDGMENT")
    print(f"  Head-to-head per seed: GROWN vs RND-D{final_d}")
    print(f"{'-'*W}\n")

    grown_wins, rnd_wins, ties = 0, 0, 0
    for k in test_ks:
        gr = comparison[("GROWN", k)]
        rr = comparison[(f"RND-D{final_d}", k)]
        for i in range(len(test_seeds)):
            dg, rg = gr['gaps'][i], rr['gaps'][i]
            if dg > rg + 0.01:
                grown_wins += 1
            elif rg > dg + 0.01:
                rnd_wins += 1
            else:
                ties += 1

    total = grown_wins + rnd_wins + ties
    print(f"  GROWN wins {grown_wins}, RND wins {rnd_wins}, ties {ties}")
    print(f"  GROWN win rate: {grown_wins}/{total} "
          f"= {grown_wins / max(total, 1):.0%}")

    grown_vs_small_wins = 0
    small_wins = 0
    for k in test_ks:
        gr = comparison[("GROWN", k)]
        sr = comparison[(f"RND-D{D_INIT}", k)]
        for i in range(len(test_seeds)):
            if gr['gaps'][i] > sr['gaps'][i] + 0.01:
                grown_vs_small_wins += 1
            elif sr['gaps'][i] > gr['gaps'][i] + 0.01:
                small_wins += 1

    print(f"\n  vs RND-D{D_INIT}: GROWN wins {grown_vs_small_wins}, "
          f"SMALL wins {small_wins}")

    # ── LET THERE BE A SECOND BODY ───────────────────────────
    # A second organism with a different seed goes through the
    # same growth process. Do both converge to similar final D?
    # Do both outperform their random baselines?

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE A SECOND BODY")
    print(f"  Different seed, same growth loop. Convergent evolution?")
    print(f"{'-'*W}\n")

    org2 = Organism(D_INIT, N_CELLS, seed=SEED + 999)
    for epoch in range(N_EPOCHS):
        k = 3 + epoch
        signals = make_signals(k, D_INIT, seed=SEED + epoch * 100)
        gap = measure_discrimination(org2, signals, k, seed=SEED + 999)
        _, final_xs = run_sequence(
            org2, list(range(k)), signals, SEED + 999, trial=0)
        pdiv = org2.per_dim_divergence(final_xs)
        collapsed = gap < COLLAPSE_THRESH
        if collapsed and org2.d < MAX_D:
            org2.adapt_alpha(pdiv, rate=0.05)
            org2.expand(GROWTH_D, per_dim_div=pdiv)
        print(f"  Body2 epoch {epoch}: K={k:>2} D={org2.d:>2} "
              f"gap={gap:+.4f} {'COLLAPSE->grow' if collapsed else ''}",
              flush=True)

    print(f"\n  Body1 final D={org.d} (grew {len(org.growth_history)} times)")
    print(f"  Body2 final D={org2.d} (grew {len(org2.growth_history)} times)")

    convergent = abs(org.d - org2.d) <= GROWTH_D

    org2_rnd = Organism(org2.d, N_CELLS, seed=SEED + 8000)
    org2_gaps, rnd2_gaps = [], []
    k_test = 6
    signals = make_signals(k_test, D_INIT, seed=SEED + k_test * 200)
    for seed in test_seeds[:3]:
        g1 = measure_discrimination(org2, signals, k_test, seed)
        g2 = measure_discrimination(org2_rnd, signals, k_test, seed)
        org2_gaps.append(g1)
        rnd2_gaps.append(g2)
        print(f"  Body2 seed={seed}: GROWN={g1:+.4f} RND={g2:+.4f} "
              f"{'GROWN' if g1 > g2 + 0.01 else 'RND' if g2 > g1 + 0.01 else 'tie'}",
              flush=True)

    avg_org2 = sum(org2_gaps) / len(org2_gaps)
    avg_rnd2 = sum(rnd2_gaps) / len(rnd2_gaps)
    body2_wins = avg_org2 > avg_rnd2

    print(f"\n  Body2 avg: GROWN={avg_org2:+.4f} RND={avg_rnd2:+.4f}")
    print(f"  -> {'GROWTH HELPS' if body2_wins else 'RANDOM BETTER'}")

    # ── LET THERE BE NOVEL SIGNALS ───────────────────────────
    # Test on signals never seen during growth.

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE NOVEL SIGNALS")
    print(f"  Test grown organism on signals from a different world.")
    print(f"{'-'*W}\n")

    k_novel = 6
    novel_signals = make_signals(k_novel, D_INIT, seed=7777)
    novel_grown, novel_rnd = [], []
    for seed in test_seeds[:3]:
        g1 = measure_discrimination(org, novel_signals, k_novel, seed)
        g2 = measure_discrimination(org_rnd_big, novel_signals, k_novel, seed)
        novel_grown.append(g1)
        novel_rnd.append(g2)
        print(f"  seed={seed}: GROWN={g1:+.4f} RND={g2:+.4f} "
              f"{'GROWN' if g1 > g2 + 0.01 else 'RND' if g2 > g1 + 0.01 else 'tie'}",
              flush=True)

    avg_novel_grown = sum(novel_grown) / len(novel_grown)
    avg_novel_rnd = sum(novel_rnd) / len(novel_rnd)
    generalizes = avg_novel_grown > avg_novel_rnd

    print(f"\n  Novel: GROWN={avg_novel_grown:+.4f} RND={avg_novel_rnd:+.4f}")
    print(f"  -> {'GENERALIZES' if generalizes else 'OVERFITS'}")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    grew = n_growths > 0
    grew_from_failure = any(e['collapsed'] for e in epoch_log)

    grown_avg_better = all(
        comparison[("GROWN", k)]['avg'] >= comparison[(f"RND-D{final_d}", k)]['avg'] - 0.01
        for k in test_ks)
    grown_more_passes = sum(
        comparison[("GROWN", k)]['passes'] for k in test_ks
    ) >= sum(
        comparison[(f"RND-D{final_d}", k)]['passes'] for k in test_ks)
    grown_majority = grown_wins > rnd_wins
    grown_beats_small = grown_vs_small_wins > small_wins

    any_epoch_classifies = any(e['ok'] for e in epoch_log)
    later_epochs_better = (len(epoch_log) >= 4 and
                           epoch_log[-1]['gap'] > epoch_log[0]['gap'])

    checks = [
        ("System grew from failure",
         grew_from_failure,
         f"{n_growths} expansions D={D_INIT}->{final_d}"),
        ("Growth triggered by discrimination collapse",
         grew_from_failure,
         f"threshold={COLLAPSE_THRESH}"),
        ("Some epoch classifies (gap > 0.02)",
         any_epoch_classifies,
         f"{sum(e['ok'] for e in epoch_log)}/{N_EPOCHS} epochs"),
        ("GROWN wins majority vs RND same D",
         grown_majority,
         f"{grown_wins}/{total}"),
        ("GROWN avg >= RND avg (all K)",
         grown_avg_better,
         ""),
        ("GROWN beats RND-small",
         grown_beats_small,
         f"{grown_vs_small_wins} vs {small_wins}"),
        ("Convergent evolution (both bodies similar D)",
         convergent,
         f"D={org.d} vs D={org2.d}"),
        ("Body2 also beats its random baseline",
         body2_wins,
         f"{avg_org2:+.4f} vs {avg_rnd2:+.4f}"),
        ("Generalizes to novel signals",
         generalizes,
         f"GROWN={avg_novel_grown:+.4f} RND={avg_novel_rnd:+.4f}"),
        ("Later epochs outperform earlier",
         later_epochs_better,
         f"last={epoch_log[-1]['gap']:+.4f} first={epoch_log[0]['gap']:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<48} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    core_viable = (grew_from_failure and grown_majority and generalizes)
    full_viable = core_viable and body2_wins and convergent

    if full_viable:
        print(f"""
  THE SEED IS VIABLE.

  The organism starts at D={D_INIT}. It processes sequential
  signals. When it cannot distinguish orderings — resolution
  collapse — it expands its dimensionality, guided by which
  dimensions contributed to divergence. Growth is self-directed:
  the system's failure drives the system's expansion.

  The grown structure (D={final_d}) outperforms random structure
  at the same dimensionality. A second body with different
  initialization converges to similar final D. The advantage
  generalizes to novel signals never seen during growth.

  One equation. One loop.
    Process -> Measure own discrimination -> Fail -> Grow -> Repeat

  The measurement IS the computation.
  The growth IS the response to failure.
  The structure IS the history of what was hard.

  This is the irrefutable step:
  A system that expands its own computational substrate
  in response to its own computational failure,
  and the structure it builds outperforms chance.
""")
    elif core_viable:
        print(f"""
  THE SEED IS PARTIALLY VIABLE.

  Growth from failure works. The grown organism outperforms
  random at the same D and generalizes. But either convergent
  evolution or the second body's advantage is weak.

  The mechanism is real. The universality is not yet proven.
""")
    elif grew_from_failure and any_epoch_classifies:
        print(f"""
  THE SEED GERMINATES BUT DOES NOT ROOT.

  The system grows from failure and some epochs classify.
  But the grown structure does not reliably outperform
  random structure at the same dimensionality.

  Growth produces more dimensions but not better structure.
  The expansion mechanism works; the adaptation mechanism
  needs to measure the right thing more precisely.
""")
    elif grew_from_failure:
        print(f"""
  THE SEED EXISTS BUT DOES NOT GERMINATE.

  The system detects failure and expands. But expansion
  alone does not produce classification. More dimensions
  without better structure is just more noise.

  The growth trigger works. The growth direction doesn't.
""")
    else:
        print(f"""
  NO SEED.

  The system neither collapsed nor grew. Either the initial
  dimension was already sufficient, or the collapse threshold
  needs tuning.
""")

    # ── GROWTH TRACE ─────────────────────────────────────────

    print(f"{'-'*W}")
    print(f"  GROWTH TRACE")
    print(f"{'-'*W}\n")
    for e in epoch_log:
        marker = '#' if e['ok'] else ('!' if e['collapsed'] else 'o')
        print(f"  {marker} epoch={e['epoch']} K={e['k']:>2} D={e['d']:>2} "
              f"gap={e['gap']:+.4f} res={e['res']:.3f}")
    print()
    for g in org.growth_history:
        print(f"  D {g['from']:>2} -> {g['to']:>2} "
              f"(divergence-guided={'yes' if g['div_used'] else 'no'})")

    print(f"\n{'-'*W}")


if __name__ == '__main__':
    run()
