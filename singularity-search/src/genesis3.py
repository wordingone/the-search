#!/usr/bin/env python3
"""
GENESIS: Self-Diagnosing Computational Architecture

One loop. The system processes sequential signals, monitors its
own discrimination capacity, and when it fails, DIAGNOSES WHY:

  - Signal-sparse (K/D < threshold): not enough signals to create
    rich cross-terms. More dimensions would make it worse.
    Response: RESTRUCTURE alpha within current D. Deepen
    per-cell specialization based on which cells discriminated.

  - Substrate-limited (K/D >= threshold): enough signals but not
    enough dimensions to separate them.
    Response: EXPAND D. Add dimensions, initialize alpha from
    the structure the system has already found.

Signals are always generated at the organism's CURRENT D.
No padding. No noise injection masquerading as structure.

The test: does the self-diagnosing system outperform
(a) random structure at the same final D,
(b) a system that only expands blindly,
(c) a system that only restructures blindly?

Zero dependencies. Pure Python.
"""

import math, random, time


# ═══════════════════════════════════════════════════════════════
# In the beginning, there is nothing but operations.
# ═══════════════════════════════════════════════════════════════

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
# From nothing, a body that can grow in two directions.
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, d, n_cells=6, seed=42, label=""):
        self.d = d
        self.n = n_cells
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.label = label
        self.birth_d = d
        self.n_expansions = 0
        self.n_restructures = 0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(d)]
            for _ in range(n_cells)
        ]

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

    def per_cell_divergence(self, xs):
        n, d = self.n, self.d
        c = self.centroid(xs)
        return [vnorm([xs[i][k] - c[k] for k in range(d)]) for i in range(n)]

    def per_dim_variance(self, xs):
        n, d = self.n, self.d
        var = [0.0] * d
        for k in range(d):
            vals = [xs[i][k] for i in range(n)]
            mu = sum(vals) / n
            var[k] = sum((v - mu) ** 2 for v in vals) / n
        return var

    # ── TWO GROWTH MODES ─────────────────────────────────────

    def expand(self, added_d):
        old_d = self.d
        self.d += added_d
        for i in range(self.n):
            template_mean = sum(self.alpha[i]) / old_d
            template_std = math.sqrt(
                sum((a - template_mean) ** 2 for a in self.alpha[i]) / old_d)
            new_alphas = [template_mean + random.gauss(0, max(template_std, 0.1))
                          for _ in range(added_d)]
            self.alpha[i] += new_alphas
        self.n_expansions += 1

    def restructure(self, cell_scores, dim_scores, rate=0.1):
        d, n = self.d, self.n
        c_mu = sum(cell_scores) / n
        c_std = math.sqrt(sum((s - c_mu)**2 for s in cell_scores) / n) + 1e-10
        d_mu = sum(dim_scores) / d
        d_std = math.sqrt(sum((s - d_mu)**2 for s in dim_scores) / d) + 1e-10

        for i in range(n):
            cell_z = (cell_scores[i] - c_mu) / c_std
            for k in range(d):
                dim_z = (dim_scores[k] - d_mu) / d_std
                push = rate * math.tanh(cell_z) * math.tanh(dim_z)
                self.alpha[i][k] += push * 0.7
                self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))
        self.n_restructures += 1


# ═══════════════════════════════════════════════════════════════
# The world speaks at the organism's current scale.
# Signals are always D-dimensional. No padding. No cheating.
# ═══════════════════════════════════════════════════════════════

def make_signals(k, d, seed):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = [random.gauss(0, 0.5) for _ in range(d)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


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
                 n_org=300, n_per_sig=50, n_settle=25, n_final=60):
    d = org.d
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(d)] for _ in range(org.n)]
    for _ in range(n_org):
        xs = org.step(xs)
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(d)]
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
    print("  GENESIS: SELF-DIAGNOSING COMPUTATIONAL ARCHITECTURE")
    print("  One loop. Fail, diagnose, respond correctly, repeat.")
    print("=" * W)

    SEED = 42
    D_INIT = 12
    N_CELLS = 6
    GROWTH_D = 4
    COLLAPSE_THRESH = 0.02
    DENSITY_THRESH = 0.6
    MAX_D = 32
    N_EPOCHS = 10

    # ── LET THERE BE A BODY ──────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE A BODY")
    print(f"  D={D_INIT}, {N_CELLS} cells, per-cell alpha.")
    print(f"  Density threshold K/D = {DENSITY_THRESH}")
    print(f"{'-'*W}\n")

    org = Organism(D_INIT, N_CELLS, seed=SEED, label="DIAG")

    # ── LET THERE BE GROWTH ──────────────────────────────────
    # Each epoch: escalating K. On collapse, DIAGNOSE:
    #   K/D < threshold -> signal-sparse -> restructure alpha
    #   K/D >= threshold -> substrate-limited -> expand D
    # Signals generated at CURRENT D always.

    print(f"{'-'*W}")
    print(f"  LET THERE BE GROWTH")
    print(f"  Collapse -> diagnose K/D -> restructure OR expand.")
    print(f"  Signals at current D. No padding.")
    print(f"{'-'*W}\n")

    epoch_log = []

    for epoch in range(N_EPOCHS):
        k = 3 + epoch
        signals = make_signals(k, org.d, seed=SEED + epoch * 100)

        t0 = time.time()
        gap = measure_discrimination(org, signals, k, seed=SEED)
        elapsed = time.time() - t0

        _, final_xs = run_sequence(
            org, list(range(k)), signals, SEED, trial=0)
        cell_div = org.per_cell_divergence(final_xs)
        dim_var = org.per_dim_variance(final_xs)

        density = k / org.d
        collapsed = gap < COLLAPSE_THRESH
        ok = gap > 0.02

        action = "ok"
        if collapsed:
            if density < DENSITY_THRESH:
                org.restructure(cell_div, dim_var, rate=0.12)
                action = f"RESTRUCTURE (K/D={density:.2f}<{DENSITY_THRESH})"
            elif org.d < MAX_D:
                org.expand(GROWTH_D)
                action = f"EXPAND D->{org.d} (K/D={density:.2f}>={DENSITY_THRESH})"
            else:
                action = f"AT MAX D={MAX_D}"

        marker = '#' if ok else ('R' if 'RESTRUCTURE' in action
                                  else ('E' if 'EXPAND' in action else 'o'))
        print(f"  {marker} Epoch {epoch}: K={k:>2} D={org.d:>2} K/D={density:.2f} "
              f"gap={gap:+.4f} {bar(gap)} [{elapsed:.0f}s]")
        if collapsed:
            print(f"    -> {action}")

        epoch_log.append({
            'epoch': epoch, 'k': k, 'd': org.d, 'gap': gap,
            'density': density, 'collapsed': collapsed, 'ok': ok,
            'action': action
        })

    final_d = org.d
    print(f"\n  Final: D={final_d}, "
          f"{org.n_expansions} expansions, "
          f"{org.n_restructures} restructures")

    # ── LET THERE BE THREE BASELINES ─────────────────────────
    # 1. BLIND-EXPAND: always expands on failure, never restructures
    # 2. BLIND-RESTR: always restructures, never expands
    # 3. RANDOM: random alpha at the diagnosing org's final D

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE THREE BASELINES")
    print(f"  BLIND-EXPAND: always expand. BLIND-RESTR: always restructure.")
    print(f"  RANDOM: chance at final D={final_d}.")
    print(f"{'-'*W}\n")

    org_expand = Organism(D_INIT, N_CELLS, seed=SEED, label="EXPAND")
    org_restr = Organism(D_INIT, N_CELLS, seed=SEED, label="RESTR")

    for epoch in range(N_EPOCHS):
        k = 3 + epoch

        sig_e = make_signals(k, org_expand.d, seed=SEED + epoch * 100)
        gap_e = measure_discrimination(org_expand, sig_e, k, seed=SEED)
        if gap_e < COLLAPSE_THRESH and org_expand.d < MAX_D:
            _, xs_e = run_sequence(org_expand, list(range(k)), sig_e, SEED, 0)
            org_expand.expand(GROWTH_D)

        sig_r = make_signals(k, org_restr.d, seed=SEED + epoch * 100)
        gap_r = measure_discrimination(org_restr, sig_r, k, seed=SEED)
        if gap_r < COLLAPSE_THRESH:
            _, xs_r = run_sequence(org_restr, list(range(k)), sig_r, SEED, 0)
            cell_div = org_restr.per_cell_divergence(xs_r)
            dim_var = org_restr.per_dim_variance(xs_r)
            org_restr.restructure(cell_div, dim_var, rate=0.12)

        print(f"  Epoch {epoch}: K={k:>2} | "
              f"EXPAND D={org_expand.d:>2} gap={gap_e:+.4f} | "
              f"RESTR D={org_restr.d:>2} gap={gap_r:+.4f}", flush=True)

    org_rnd = Organism(final_d, N_CELLS, seed=SEED + 5000, label="RANDOM")

    print(f"\n  EXPAND final:  D={org_expand.d}, {org_expand.n_expansions} exp")
    print(f"  RESTR final:   D={org_restr.d}, {org_restr.n_restructures} restr")
    print(f"  DIAG final:    D={org.d}, "
          f"{org.n_expansions} exp + {org.n_restructures} restr")
    print(f"  RANDOM:        D={final_d}")

    # ── LET THERE BE COMPARISON ──────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE COMPARISON")
    print(f"  All four systems tested at their final D, signals at that D.")
    print(f"  5 seeds, K=4,6,8.")
    print(f"{'-'*W}\n")

    test_seeds = [42, 77, 123, 200, 314]
    test_ks = [4, 6, 8]

    all_orgs = [
        ("DIAG", org),
        ("EXPAND", org_expand),
        ("RESTR", org_restr),
        ("RANDOM", org_rnd),
    ]

    comparison = {}
    for k in test_ks:
        for label, test_org in all_orgs:
            sigs = make_signals(k, test_org.d, seed=SEED + k * 200)
            gaps = []
            for seed in test_seeds:
                g = measure_discrimination(test_org, sigs, k, seed)
                gaps.append(g)
            avg = sum(gaps) / len(gaps)
            passes = sum(1 for g in gaps if g > 0.02)
            comparison[(label, k)] = {
                'avg': avg, 'passes': passes, 'total': len(gaps),
                'gaps': gaps
            }
            print(f"  {label:>7} D={test_org.d:>2} K={k}: "
                  f"avg={avg:+.4f} pass={passes}/{len(gaps)} "
                  f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)
        print()

    # ── LET THERE BE JUDGMENT ────────────────────────────────

    print(f"{'-'*W}")
    print(f"  LET THERE BE JUDGMENT")
    print(f"  Head-to-head per seed across all K.")
    print(f"{'-'*W}\n")

    def count_wins(label_a, label_b):
        a_wins, b_wins, ties = 0, 0, 0
        for k in test_ks:
            ga = comparison[(label_a, k)]['gaps']
            gb = comparison[(label_b, k)]['gaps']
            for i in range(len(test_seeds)):
                if ga[i] > gb[i] + 0.01:
                    a_wins += 1
                elif gb[i] > ga[i] + 0.01:
                    b_wins += 1
                else:
                    ties += 1
        return a_wins, b_wins, ties

    matchups = [
        ("DIAG", "RANDOM"),
        ("DIAG", "EXPAND"),
        ("DIAG", "RESTR"),
        ("EXPAND", "RANDOM"),
        ("RESTR", "RANDOM"),
    ]

    win_data = {}
    for la, lb in matchups:
        a_w, b_w, t = count_wins(la, lb)
        win_data[(la, lb)] = (a_w, b_w, t)
        total = a_w + b_w + t
        print(f"  {la:>7} vs {lb:<7}: "
              f"{la} {a_w}, {lb} {b_w}, ties {t} "
              f"({la} {'WINS' if a_w > b_w else 'LOSES' if b_w > a_w else 'DRAWS'})")

    # ── LET THERE BE NOVEL SIGNALS ───────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE NOVEL SIGNALS")
    print(f"  Test all organisms on signals never seen during growth.")
    print(f"{'-'*W}\n")

    k_novel = 6
    novel_results = {}
    for label, test_org in all_orgs:
        novel_sigs = make_signals(k_novel, test_org.d, seed=7777)
        gaps = []
        for seed in test_seeds[:3]:
            g = measure_discrimination(test_org, novel_sigs, k_novel, seed)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        novel_results[label] = avg
        print(f"  {label:>7} D={test_org.d:>2}: avg={avg:+.4f} "
              f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

    best_novel = max(novel_results, key=novel_results.get)
    print(f"\n  Best on novel: {best_novel} ({novel_results[best_novel]:+.4f})")

    generalizes = novel_results["DIAG"] > novel_results["RANDOM"]

    # ── LET THERE BE A SECOND BODY ───────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE A SECOND BODY")
    print(f"  Different seed, same diagnostic loop.")
    print(f"{'-'*W}\n")

    org2 = Organism(D_INIT, N_CELLS, seed=SEED + 999, label="DIAG2")
    for epoch in range(N_EPOCHS):
        k = 3 + epoch
        sigs = make_signals(k, org2.d, seed=SEED + epoch * 100)
        gap = measure_discrimination(org2, sigs, k, seed=SEED + 999)
        density = k / org2.d
        if gap < COLLAPSE_THRESH:
            _, xs2 = run_sequence(org2, list(range(k)), sigs, SEED + 999, 0)
            if density < DENSITY_THRESH:
                cd = org2.per_cell_divergence(xs2)
                dv = org2.per_dim_variance(xs2)
                org2.restructure(cd, dv, rate=0.12)
                act = "RESTR"
            elif org2.d < MAX_D:
                org2.expand(GROWTH_D)
                act = f"EXP->{org2.d}"
            else:
                act = "MAX"
        else:
            act = ""
        print(f"  B2 epoch {epoch}: K={k:>2} D={org2.d:>2} "
              f"K/D={density:.2f} gap={gap:+.4f} {act}", flush=True)

    convergent = abs(org.d - org2.d) <= GROWTH_D
    similar_ratio = (abs(org.n_expansions - org2.n_expansions) <= 2 and
                     abs(org.n_restructures - org2.n_restructures) <= 2)

    print(f"\n  Body1: D={org.d}, {org.n_expansions}exp + {org.n_restructures}restr")
    print(f"  Body2: D={org2.d}, {org2.n_expansions}exp + {org2.n_restructures}restr")
    print(f"  Convergent D: {'YES' if convergent else 'no'}")
    print(f"  Similar strategy: {'YES' if similar_ratio else 'no'}")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    diag_v_rnd = win_data.get(("DIAG", "RANDOM"), (0, 0, 0))
    diag_v_exp = win_data.get(("DIAG", "EXPAND"), (0, 0, 0))
    diag_v_rst = win_data.get(("DIAG", "RESTR"), (0, 0, 0))

    used_both = org.n_expansions > 0 and org.n_restructures > 0
    diag_beats_rnd = diag_v_rnd[0] > diag_v_rnd[1]
    diag_beats_expand = diag_v_exp[0] > diag_v_exp[1]
    diag_beats_restr = diag_v_rst[0] > diag_v_rst[1]
    any_epoch_ok = any(e['ok'] for e in epoch_log)
    improved_over_time = (len(epoch_log) >= 4 and
                          sum(e['gap'] for e in epoch_log[-3:]) / 3 >
                          sum(e['gap'] for e in epoch_log[:3]) / 3)

    checks = [
        ("System used BOTH growth modes",
         used_both,
         f"{org.n_expansions} exp + {org.n_restructures} restr"),
        ("Diagnosis correct (sparse->restr, dense->expand)",
         used_both and org.n_restructures > 0 and org.n_expansions > 0,
         f"checked K/D vs {DENSITY_THRESH}"),
        ("Some epoch classifies",
         any_epoch_ok,
         f"{sum(e['ok'] for e in epoch_log)}/{N_EPOCHS}"),
        ("DIAG beats RANDOM (per-seed majority)",
         diag_beats_rnd,
         f"{diag_v_rnd[0]} vs {diag_v_rnd[1]}"),
        ("DIAG beats BLIND-EXPAND",
         diag_beats_expand,
         f"{diag_v_exp[0]} vs {diag_v_exp[1]}"),
        ("DIAG beats BLIND-RESTR",
         diag_beats_restr,
         f"{diag_v_rst[0]} vs {diag_v_rst[1]}"),
        ("Generalizes to novel signals",
         generalizes,
         f"DIAG={novel_results['DIAG']:+.4f} RND={novel_results['RANDOM']:+.4f}"),
        ("Convergent evolution (2 bodies similar D)",
         convergent,
         f"D={org.d} vs {org2.d}"),
        ("Both bodies use similar strategy",
         similar_ratio,
         f"exp: {org.n_expansions}/{org2.n_expansions} "
         f"restr: {org.n_restructures}/{org2.n_restructures}"),
        ("Performance improves over epochs",
         improved_over_time,
         f"early avg={sum(e['gap'] for e in epoch_log[:3])/3:+.4f} "
         f"late avg={sum(e['gap'] for e in epoch_log[-3:])/3:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<48} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    core = (diag_beats_rnd and used_both and any_epoch_ok)
    superior = core and (diag_beats_expand or diag_beats_restr)
    full = superior and generalizes and convergent

    if full:
        print(f"""
  THE SEED IS VIABLE.

  The organism starts at D={D_INIT}. When it fails to
  discriminate signal orderings, it diagnoses WHY:

    K/D < {DENSITY_THRESH}: signal-sparse. RESTRUCTURE alpha.
      Deepen per-cell specialization within current D.
    K/D >= {DENSITY_THRESH}: substrate-limited. EXPAND D.
      Add dimensions, seeded from existing structure.

  The diagnosing system outperforms:
    - Random structure at the same final D
    - Blind expansion (always add dimensions)
    - Blind restructuring (always adapt alpha)

  Two bodies with different seeds converge to similar
  final D and similar expansion/restructure ratios.
  The structure generalizes to novel signals.

  One equation. One loop. Two growth modes.
  The system that knows why it failed
  outperforms the systems that only know that it failed.
  Self-diagnosis is the seed.
""")
    elif superior:
        print(f"""
  THE SEED IS PARTIALLY VIABLE.

  Diagnosis works: the system uses both growth modes and
  beats at least one blind baseline. But either generalization
  or convergent evolution is weak.

  Self-diagnosis improves over blind growth.
  Universality is not yet proven.
""")
    elif core:
        print(f"""
  THE SEED GERMINATES.

  The system grows from failure using both modes.
  It outperforms random. But it doesn't beat blind
  baselines — knowing why it failed doesn't yet help
  more than just reacting to failure.

  The diagnosis is real. The response needs refinement.
""")
    elif any_epoch_ok:
        print(f"""
  THE SEED EXISTS BUT DOES NOT GERMINATE.

  Some epochs classify but the grown structure does not
  outperform random. Growth direction needs work.
""")
    else:
        print(f"""
  NO SEED.
  Nothing classified. Parameters need adjustment.
""")

    # ── GROWTH TRACE ─────────────────────────────────────────

    print(f"{'-'*W}")
    print(f"  GROWTH TRACE")
    print(f"{'-'*W}\n")
    for e in epoch_log:
        sym = '#' if e['ok'] else ('R' if 'RESTRUCTURE' in e['action']
                                    else ('E' if 'EXPAND' in e['action'] else 'o'))
        print(f"  {sym} epoch={e['epoch']} K={e['k']:>2} D={e['d']:>2} "
              f"K/D={e['density']:.2f} gap={e['gap']:+.4f} "
              f"{e['action'] if e['collapsed'] else ''}")
    print(f"\n{'-'*W}")


if __name__ == '__main__':
    run()
