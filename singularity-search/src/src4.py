#!/usr/bin/env python3
"""
Self-Referential Computation via Multiplicative Input Geometry

  Phi(x_i)_k = tanh(alpha_k * x_{i,k} + beta * (x_{k+1} + gamma*s_{k+1}) * (x_{k-1} + gamma*s_{k-1}))
  coupling: additive pull toward neighbors weighted by state similarity

Signal enters multiplicatively (reshapes the map).
Coupling enters additively (coordinates cells).
These serve structurally different roles verified by ablation.

Core claim: multiplicative input creates persistent basin separation.
Scaling test: sequential multi-signal classification. Can the system
retain signal A after exposure to signal B? Does it accumulate history
or does each signal overwrite the last?

Zero dependencies. Pure Python.
"""

import math, random

def vzero(d):        return [0.0] * d
def vrand(d, s=1.0): return [random.gauss(0, s) for _ in range(d)]
def vadd(a, b):      return [ai + bi for ai, bi in zip(a, b)]
def vsub(a, b):      return [ai - bi for ai, bi in zip(a, b)]
def vscale(v, s):    return [vi * s for vi in v]
def vdot(a, b):      return sum(ai * bi for ai, bi in zip(a, b))
def vnorm(v):        return math.sqrt(sum(vi * vi for vi in v) + 1e-15)

def vcosine(a, b):
    na, nb = vnorm(a), vnorm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return max(-1.0, min(1.0, vdot(a, b) / (na * nb)))

N     = 6
D     = 16
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


def make_alphas(seed):
    random.seed(seed)
    return [A0 + A_SP * (random.random() * 2 - 1) for _ in range(D)]


def init_xs(seed=None):
    if seed is not None:
        random.seed(seed)
    return [vrand(D, 0.5) for _ in range(N)]


def phi_mult(x, alphas, signal=None):
    d = len(x)
    if signal:
        return [math.tanh(alphas[k] * x[k] +
                BETA * (x[(k+1)%d] + GAMMA * signal[(k+1)%d]) *
                       (x[(k-1)%d] + GAMMA * signal[(k-1)%d]))
                for k in range(d)]
    return [math.tanh(alphas[k] * x[k] + BETA * x[(k+1)%d] * x[(k-1)%d])
            for k in range(d)]


def phi_add(x, alphas, signal=None):
    d = len(x)
    if signal:
        return [math.tanh(alphas[k] * x[k] + BETA * x[(k+1)%d] * x[(k-1)%d]
                + GAMMA * signal[k])
                for k in range(d)]
    return [math.tanh(alphas[k] * x[k] + BETA * x[(k+1)%d] * x[(k-1)%d])
            for k in range(d)]


def step(xs, alphas, signal=None, mode='mult'):
    n, d = len(xs), D
    phi = phi_mult if mode == 'mult' else phi_add

    phi0 = [phi(x, alphas) for x in xs]
    phis = [phi(x, alphas, signal) for x in xs]

    raw_w = []
    for i in range(n):
        r = [vdot(xs[i], xs[j]) / (d * TAU) if i != j else -1e10
             for j in range(n)]
        mx = max(r)
        exps = [math.exp(min(v - mx, 50)) for v in r]
        s = sum(exps) + 1e-15
        raw_w.append([e / s for e in exps])

    new = []
    for i in range(n):
        p = phis[i]

        fp_d = vnorm(vsub(phi0[i], xs[i])) / max(vnorm(xs[i]), 1.0)
        plast = math.exp(-(fp_d * fp_d) / 0.0225)

        if plast > 0.01 and EPS > 0:
            pull = vzero(d)
            for j in range(n):
                if i == j or raw_w[i][j] < 1e-8:
                    continue
                pull = vadd(pull, vscale(vsub(phi0[j], phi0[i]), raw_w[i][j]))
            p = vadd(p, vscale(pull, plast * EPS))

        nx = [(1 - DELTA) * xs[i][k] + DELTA * p[k] + random.gauss(0, NOISE)
              for k in range(d)]
        for k in range(d):
            nx[k] = max(-CLIP, min(CLIP, nx[k]))
        new.append(nx)
    return new


def centroid(xs):
    c = vzero(D)
    for x in xs:
        c = vadd(c, vscale(x, 1.0 / len(xs)))
    return c


def mean_fp(xs, alphas):
    total = 0.0
    for x in xs:
        p = [math.tanh(alphas[k] * x[k] + BETA * x[(k+1)%D] * x[(k-1)%D])
             for k in range(D)]
        total += vnorm(vsub(p, x)) / max(vnorm(x), 1.0)
    return total / len(xs)


def mean_norm(xs):
    return sum(vnorm(x) for x in xs) / len(xs)


def disp(xs):
    c = centroid(xs)
    return sum(vnorm(vsub(x, c)) ** 2 for x in xs) / len(xs)


def make_signals():
    a = [0.0] * D
    for k in range(D // 3): a[k] = 0.8
    b = [0.0] * D
    for k in range(2 * D // 3, D): b[k] = 0.8
    c = [0.8 * (1 if k % 2 == 0 else -1) for k in range(D)]
    return {'A': a, 'B': b, 'C': c}


def gap(within, between):
    min_w = min(within.values())
    max_b = max(abs(v) for v in between.values())
    return min_w - max_b


def run_classify(mode, n_trials=8, n_org=500, n_sig=300, n_settle=120, seed=42):
    signals = make_signals()
    alphas = make_alphas(seed)
    res_dur, res_aft = {}, {}
    for label, base in sorted(signals.items()):
        dur, aft = [], []
        for trial in range(n_trials):
            random.seed(seed)
            xs = init_xs()
            for _ in range(n_org):
                xs = step(xs, alphas, mode=mode)
            random.seed(seed * 1000 + ord(label[0]) * 100 + trial)
            sig = [base[k] + random.gauss(0, 0.12) for k in range(D)]
            for _ in range(n_sig):
                xs = step(xs, alphas, sig, mode=mode)
            dur.append(centroid(xs))
            for _ in range(n_settle):
                xs = step(xs, alphas, mode=mode)
            aft.append(centroid(xs))
        res_dur[label] = dur
        res_aft[label] = aft

    def metrics(res):
        within, between = {}, {}
        labels = sorted(res.keys())
        for l in labels:
            sims = [vcosine(res[l][i], res[l][j])
                    for i in range(len(res[l]))
                    for j in range(i + 1, len(res[l]))]
            within[l] = sum(sims) / max(len(sims), 1)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                sims = [vcosine(c1, c2) for c1 in res[l1] for c2 in res[l2]]
                between[(l1, l2)] = sum(sims) / max(len(sims), 1)
        return within, between
    return metrics(res_dur), metrics(res_aft)


def bar(v, w=12, lo=-1.0, hi=1.0):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


def print_class(wc, bc):
    g = gap(wc, bc)
    for cl in sorted(wc):
        print(f"    within {cl}: {wc[cl]:+.4f} {bar(wc[cl])}")
    for k in sorted(bc):
        print(f"    betw {k[0]}v{k[1]}: {bc[k]:+.4f} {bar(bc[k])}")
    print(f"    gap={g:+.4f} -> {'CLASSIFIES' if g > 0.02 else 'FAILS'}")
    return g


def run():
    print("=" * W)
    print("  Phi_s(x)_k = tanh(a_k*x_k + b*(x_{k+1}+g*s_{k+1})*(x_{k-1}+g*s_{k-1}))")
    print("  Signal: multiplicative. Coupling: additive. Separate channels.")
    print("=" * W)

    SEED = 42
    alphas = make_alphas(SEED)

    # ── S1 SELF-ORGANIZATION ─────────────────────────────────

    print(f"\n{'-'*W}")
    print("  S1  SELF-ORGANIZATION")
    print(f"{'-'*W}\n")

    supra = sum(1 for a in alphas if a > 1.0)
    print(f"  alpha_k = [{min(alphas):.2f} .. {max(alphas):.2f}], "
          f"{supra}/{D} supercritical\n")

    random.seed(SEED)
    xs = init_xs()
    print(f"  {'step':>5}  {'fp':>7}  {'|x|':>7}  {'disp':>7}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")
    for epoch in range(10):
        for _ in range(60):
            xs = step(xs, alphas)
        t = (epoch + 1) * 60
        print(f"  {t:>5}  {mean_fp(xs, alphas):>7.4f}  "
              f"{mean_norm(xs):>7.3f}  {disp(xs):>7.3f}")
    so_fp = mean_fp(xs, alphas)

    # ── S2 SINGLE-SIGNAL PERSISTENCE ─────────────────────────

    print(f"\n{'-'*W}")
    print("  S2  SINGLE-SIGNAL PERSISTENCE (MULT vs ADD)")
    print(f"{'-'*W}")

    results = {}
    for mode in ['mult', 'add']:
        tag = "MULT" if mode == 'mult' else "ADD"
        (w_dur, b_dur), (w_aft, b_aft) = run_classify(mode, seed=SEED)
        for phase, wc, bc in [("during", w_dur, b_dur),
                               ("after removal", w_aft, b_aft)]:
            print(f"\n  {tag} -- {phase}:")
            g = print_class(wc, bc)
            results[(mode, phase)] = g

    m_dur = results[('mult', 'during')]
    m_aft = results[('mult', 'after removal')]
    a_dur = results[('add', 'during')]
    a_aft = results[('add', 'after removal')]
    m_ret = m_aft / max(m_dur, 0.01)
    a_ret = a_aft / max(a_dur, 0.01)
    persist_delta = m_aft - a_aft

    print(f"\n  MULT: retained {m_ret:.0%} of gap after removal")
    print(f"  ADD:  retained {a_ret:.0%} of gap after removal")
    print(f"  Persistence delta = {persist_delta:+.4f}")

    mult_persists = m_aft > 0.02 and m_ret > 0.6
    add_collapses = a_ret < 0.3
    persist_ok = persist_delta > 0.1
    print(f"  -> {'PERSISTENCE VERIFIED' if mult_persists and add_collapses and persist_ok else 'NOT YET VERIFIED'}")

    # ── S3 SEQUENTIAL MULTI-SIGNAL ───────────────────────────

    print(f"\n{'-'*W}")
    print("  S3  SEQUENTIAL MULTI-SIGNAL CLASSIFICATION")
    print(f"      Feed A then B. Does the system remember A?")
    print(f"      Feed B then A. Does the system remember B?")
    print(f"      Is (A then B) different from (B then A)?")
    print(f"{'-'*W}\n")

    signals = make_signals()
    pairs = [('A', 'B'), ('B', 'A'), ('A', 'C'), ('C', 'A'),
             ('B', 'C'), ('C', 'B')]

    n_org, n_sig1, n_settle1, n_sig2, n_settle2 = 500, 250, 80, 250, 120

    seq_centroids = {}
    for first, second in pairs:
        trials = []
        for trial in range(6):
            random.seed(SEED)
            xs = init_xs()
            for _ in range(n_org):
                xs = step(xs, alphas)

            random.seed(SEED * 1000 + ord(first[0]) * 100 + trial)
            sig1 = [signals[first][k] + random.gauss(0, 0.12) for k in range(D)]
            for _ in range(n_sig1):
                xs = step(xs, alphas, sig1)
            for _ in range(n_settle1):
                xs = step(xs, alphas)

            random.seed(SEED * 2000 + ord(second[0]) * 100 + trial)
            sig2 = [signals[second][k] + random.gauss(0, 0.12) for k in range(D)]
            for _ in range(n_sig2):
                xs = step(xs, alphas, sig2)
            for _ in range(n_settle2):
                xs = step(xs, alphas)

            trials.append(centroid(xs))
        seq_centroids[(first, second)] = trials

    print(f"  Pairwise cosine between sequence endpoints:\n")
    all_pairs_cos = {}
    keys = sorted(seq_centroids.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            sims = [vcosine(c1, c2) for c1 in seq_centroids[k1]
                    for c2 in seq_centroids[k2]]
            avg = sum(sims) / len(sims)
            all_pairs_cos[(k1, k2)] = avg

    within_same_second = []
    within_same_first = []
    between_reversed = []
    between_different = []

    for (k1, k2), cos in all_pairs_cos.items():
        if k1[1] == k2[1] and k1[0] != k2[0]:
            within_same_second.append((k1, k2, cos))
        elif k1[0] == k2[0] and k1[1] != k2[1]:
            within_same_first.append((k1, k2, cos))
        elif k1 == (k2[1], k2[0]):
            between_reversed.append((k1, k2, cos))
        else:
            between_different.append((k1, k2, cos))

    print(f"  Same second signal (does first signal differentiate?):")
    first_diff_cos = []
    for k1, k2, cos in sorted(within_same_second):
        tag = f"    {k1[0]}->{k1[1]} vs {k2[0]}->{k2[1]}"
        print(f"  {tag:<22} cos={cos:+.4f} {bar(cos)}")
        first_diff_cos.append(cos)

    print(f"\n  Same first signal (does second signal differentiate?):")
    second_diff_cos = []
    for k1, k2, cos in sorted(within_same_first):
        tag = f"    {k1[0]}->{k1[1]} vs {k2[0]}->{k2[1]}"
        print(f"  {tag:<22} cos={cos:+.4f} {bar(cos)}")
        second_diff_cos.append(cos)

    print(f"\n  Reversed order (is A->B different from B->A?):")
    reversed_cos = []
    for k1, k2, cos in sorted(between_reversed):
        tag = f"    {k1[0]}->{k1[1]} vs {k2[0]}->{k2[1]}"
        print(f"  {tag:<22} cos={cos:+.4f} {bar(cos)}")
        reversed_cos.append(cos)

    within_seq = []
    for key in keys:
        sims = [vcosine(seq_centroids[key][i], seq_centroids[key][j])
                for i in range(len(seq_centroids[key]))
                for j in range(i + 1, len(seq_centroids[key]))]
        within_seq.append(sum(sims) / max(len(sims), 1))

    avg_within = sum(within_seq) / len(within_seq)
    avg_first_diff = sum(first_diff_cos) / max(len(first_diff_cos), 1)
    avg_second_diff = sum(second_diff_cos) / max(len(second_diff_cos), 1)
    avg_reversed = sum(reversed_cos) / max(len(reversed_cos), 1)

    print(f"\n  Summary:")
    print(f"    Avg within-sequence consistency:   {avg_within:+.4f}")
    print(f"    Avg same-second, diff-first:       {avg_first_diff:+.4f}")
    print(f"    Avg same-first, diff-second:       {avg_second_diff:+.4f}")
    print(f"    Avg reversed-order:                {avg_reversed:+.4f}")

    second_differentiates = avg_second_diff < avg_within - 0.05
    first_remembered = avg_first_diff < avg_within - 0.05
    order_matters = avg_reversed < avg_within - 0.05

    print(f"\n    Second signal differentiates: {'YES' if second_differentiates else 'NO'} "
          f"(delta={avg_within - avg_second_diff:+.4f})")
    print(f"    First signal remembered:     {'YES' if first_remembered else 'NO'} "
          f"(delta={avg_within - avg_first_diff:+.4f})")
    print(f"    Order matters:               {'YES' if order_matters else 'NO'} "
          f"(delta={avg_within - avg_reversed:+.4f})")

    seq_classifies = second_differentiates and first_remembered

    if seq_classifies:
        print(f"\n  -> SEQUENTIAL CLASSIFICATION: the system accumulates history.")
        print(f"     Both signals leave persistent traces. Order matters.")
    elif second_differentiates and not first_remembered:
        print(f"\n  -> OVERWRITE: second signal erases first. No accumulation.")
        print(f"     Multiplicative persistence is single-shot, not sequential.")
    elif first_remembered and not second_differentiates:
        print(f"\n  -> SATURATION: first signal dominates. Second has no effect.")
    else:
        print(f"\n  -> NEITHER: signals do not differentiate endpoints.")

    # ── S4 SEQUENTIAL ABLATION (MULT vs ADD) ─────────────────

    print(f"\n{'-'*W}")
    print("  S4  SEQUENTIAL ABLATION: MULT vs ADD ORDER SENSITIVITY")
    print(f"{'-'*W}\n")

    def run_seq_ablation(mode):
        sigs = make_signals()
        al = make_alphas(SEED)
        endpoints = {}
        for first, second in [('A', 'B'), ('B', 'A')]:
            trials = []
            for trial in range(6):
                random.seed(SEED)
                xs = init_xs()
                for _ in range(n_org):
                    xs = step(xs, al, mode=mode)
                random.seed(SEED * 1000 + ord(first[0]) * 100 + trial)
                s1 = [sigs[first][k] + random.gauss(0, 0.12) for k in range(D)]
                for _ in range(n_sig1):
                    xs = step(xs, al, s1, mode=mode)
                for _ in range(n_settle1):
                    xs = step(xs, al, mode=mode)
                random.seed(SEED * 2000 + ord(second[0]) * 100 + trial)
                s2 = [sigs[second][k] + random.gauss(0, 0.12) for k in range(D)]
                for _ in range(n_sig2):
                    xs = step(xs, al, s2, mode=mode)
                for _ in range(n_settle2):
                    xs = step(xs, al, mode=mode)
                trials.append(centroid(xs))
            endpoints[(first, second)] = trials

        sims = [vcosine(c1, c2) for c1 in endpoints[('A', 'B')]
                for c2 in endpoints[('B', 'A')]]
        return sum(sims) / len(sims)

    mult_order_cos = run_seq_ablation('mult')
    add_order_cos = run_seq_ablation('add')

    print(f"  MULT: cos(A->B, B->A) = {mult_order_cos:+.4f}")
    print(f"  ADD:  cos(A->B, B->A) = {add_order_cos:+.4f}")
    print(f"  Lower = more order-sensitive.\n")

    mult_order_sensitive = mult_order_cos < 0.7
    add_order_sensitive = add_order_cos < 0.7
    order_ablation = mult_order_cos < add_order_cos - 0.05

    print(f"  MULT order-sensitive: {'YES' if mult_order_sensitive else 'NO'}")
    print(f"  ADD order-sensitive:  {'YES' if add_order_sensitive else 'NO'}")
    print(f"  -> {'MULT MORE ORDER-SENSITIVE' if order_ablation else 'COMPARABLE' if abs(mult_order_cos - add_order_cos) < 0.05 else 'ADD MORE ORDER-SENSITIVE'}")

    # ── S5 THREE-SIGNAL SEQUENCE ─────────────────────────────

    print(f"\n{'-'*W}")
    print("  S5  THREE-SIGNAL SEQUENCE: A->B->C vs C->B->A")
    print(f"      Does the system retain traces of all three?")
    print(f"{'-'*W}\n")

    def run_three_seq(order, seed_base):
        sigs = make_signals()
        al = make_alphas(SEED)
        trials = []
        for trial in range(6):
            random.seed(SEED)
            xs = init_xs()
            for _ in range(n_org):
                xs = step(xs, al)
            for idx, label in enumerate(order):
                random.seed(seed_base + ord(label) * 100 + idx * 10 + trial)
                sig = [sigs[label][k] + random.gauss(0, 0.12) for k in range(D)]
                for _ in range(200):
                    xs = step(xs, al, sig)
                for _ in range(60):
                    xs = step(xs, al)
            for _ in range(120):
                xs = step(xs, al)
            trials.append(centroid(xs))
        return trials

    orders = [
        ('A', 'B', 'C'),
        ('C', 'B', 'A'),
        ('A', 'C', 'B'),
        ('B', 'A', 'C'),
    ]

    three_centroids = {}
    for order in orders:
        label = '->'.join(order)
        three_centroids[label] = run_three_seq(order, SEED * 3000)

    labels_3 = sorted(three_centroids.keys())
    print(f"  Pairwise cosine between 3-signal sequence endpoints:\n")
    three_within = {}
    for l in labels_3:
        sims = [vcosine(three_centroids[l][i], three_centroids[l][j])
                for i in range(len(three_centroids[l]))
                for j in range(i + 1, len(three_centroids[l]))]
        three_within[l] = sum(sims) / max(len(sims), 1)
        print(f"    {l:<12} within={three_within[l]:+.4f}")

    three_between = {}
    for i in range(len(labels_3)):
        for j in range(i + 1, len(labels_3)):
            l1, l2 = labels_3[i], labels_3[j]
            sims = [vcosine(c1, c2) for c1 in three_centroids[l1]
                    for c2 in three_centroids[l2]]
            three_between[(l1, l2)] = sum(sims) / len(sims)

    print()
    for k in sorted(three_between):
        print(f"    {k[0]:<12} vs {k[1]:<12} cos={three_between[k]:+.4f} {bar(three_between[k])}")

    avg_3_within = sum(three_within.values()) / len(three_within)
    avg_3_between = sum(three_between.values()) / len(three_between)
    three_gap = avg_3_within - avg_3_between

    abc_cba_cos = None
    for k, v in three_between.items():
        if 'A->B->C' in k and 'C->B->A' in k:
            abc_cba_cos = v
            break

    print(f"\n    Avg within:  {avg_3_within:+.4f}")
    print(f"    Avg between: {avg_3_between:+.4f}")
    print(f"    Gap:         {three_gap:+.4f}")
    if abc_cba_cos is not None:
        print(f"    A->B->C vs C->B->A: {abc_cba_cos:+.4f}")

    three_classifies = three_gap > 0.02
    reversal_different = abc_cba_cos is not None and abc_cba_cos < 0.7

    print(f"\n    Three-signal classification: {'YES' if three_classifies else 'NO'}")
    print(f"    Forward vs reverse different: {'YES' if reversal_different else 'NO'}")

    if three_classifies:
        print(f"    -> ALIVE: system accumulates ordered signal history.")
    else:
        print(f"    -> CEILING: sequential signals converge. History collapses.")

    # ── S6 FUNCTIONAL RECOVERY ───────────────────────────────

    print(f"\n{'-'*W}")
    print("  S6  FUNCTIONAL RECOVERY")
    print(f"{'-'*W}\n")

    random.seed(SEED)
    xs = init_xs()
    for _ in range(500):
        xs = step(xs, alphas)
    pre_fp = mean_fp(xs, alphas)

    sig = vzero(D)
    sig[0] = 2.5; sig[3] = -2.0; sig[8] = 1.5; sig[12] = -1.5
    for _ in range(160):
        xs = step(xs, alphas, sig)
    for _ in range(240):
        xs = step(xs, alphas)
    post_fp = mean_fp(xs, alphas)
    recovery_ok = post_fp < 0.12
    print(f"  Pre: fp={pre_fp:.4f}  Post: fp={post_fp:.4f}")
    print(f"  -> {'FUNCTIONAL RECOVERY' if recovery_ok else 'DEGRADED'}")

    # ── S7 HISTORY DEPENDENCE ────────────────────────────────

    print(f"\n{'-'*W}")
    print("  S7  HISTORY DEPENDENCE")
    print(f"{'-'*W}\n")

    def conditioned(sig, n_o=400, n_c=300, n_s=200):
        random.seed(100)
        xs = init_xs()
        for _ in range(n_o):
            xs = step(xs, alphas)
        for _ in range(n_c):
            xs = step(xs, alphas, sig)
        for _ in range(n_s):
            xs = step(xs, alphas)
        return xs

    cA, cB = vzero(D), vzero(D)
    for k in range(5): cA[k] = 1.5
    for k in range(10, 15): cB[k] = 1.5
    xsA, xsB = conditioned(cA), conditioned(cB)
    h_cos = vcosine(centroid(xsA), centroid(xsB))
    hist_ok = abs(h_cos) < 0.7
    print(f"  Centroid cosine: {h_cos:+.4f}")
    print(f"  -> {'HISTORY SHAPES COMPUTATION' if hist_ok else 'Minimal effect'}")

    # ── S8 SELF-REFERENCE ABLATION ───────────────────────────

    print(f"\n{'-'*W}")
    print("  S8  SELF-REFERENCE ABLATION (beta=0.5 vs beta=0)")
    print(f"{'-'*W}\n")

    global BETA
    orig = BETA

    random.seed(50)
    xs_f = init_xs()
    random.seed(50)
    xs_a = init_xs()
    al_sr = make_alphas(50)

    BETA = 0.5
    for _ in range(500): xs_f = step(xs_f, al_sr)
    df = disp(xs_f)

    BETA = 0.0
    for _ in range(500): xs_a = step(xs_a, al_sr)
    da = disp(xs_a)

    BETA = orig
    sr_diff = abs(df - da) / max(df, da, 0.001)
    sr_ok = sr_diff > 0.1
    print(f"  beta=0.5: disp={df:.4f}  beta=0: disp={da:.4f}")
    print(f"  Relative diff: {sr_diff:.4f}")
    print(f"  -> {'SELF-REFERENCE MATTERS' if sr_ok else 'Minimal effect'}")

    # ── S9 GAMMA SWEEP ───────────────────────────────────────

    print(f"\n{'-'*W}")
    print("  S9  GAMMA SWEEP")
    print(f"{'-'*W}\n")

    g_gaps = []
    for gv in [0.0, 0.3, 0.6, 0.9, 1.2]:
        global GAMMA
        old_g = GAMMA
        GAMMA = gv
        (_, _), (wg, bg) = run_classify('mult', seed=SEED)
        g = gap(wg, bg)
        g_gaps.append((gv, g))
        print(f"  gamma={gv:.1f}: after-removal gap={g:+.4f} "
              f"{'CLASSIFIES' if g > 0.02 else 'FAILS'}")
        GAMMA = old_g

    g0 = g_gaps[0][1]
    gmax = g_gaps[-1][1]
    gamma_helps = gmax > g0 + 0.05
    print(f"\n  gamma=0.0: {g0:+.4f}  gamma=1.2: {gmax:+.4f}")
    print(f"  -> {'GAMMA ENABLES PERSISTENCE' if gamma_helps else 'gamma effect unclear'}")

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print("  RESULTS")
    print(f"{'='*W}\n")

    core = mult_persists and add_collapses and persist_ok

    checks = [
        ("Self-organization",
         so_fp < 0.15, f"fp={so_fp:.4f}"),
        ("MULT persists (>60% retained)",
         mult_persists, f"{m_ret:.0%}"),
        ("ADD collapses (>70% loss)",
         add_collapses, f"{a_ret:.0%}"),
        ("Persistence delta significant",
         persist_ok, f"{persist_delta:+.4f}"),
        ("Second signal differentiates",
         second_differentiates, f"delta={avg_within - avg_second_diff:+.4f}"),
        ("First signal remembered",
         first_remembered, f"delta={avg_within - avg_first_diff:+.4f}"),
        ("Order matters (MULT vs ADD)",
         order_ablation, f"M={mult_order_cos:+.3f} A={add_order_cos:+.3f}"),
        ("Three-signal sequence classifies",
         three_classifies, f"gap={three_gap:+.4f}"),
        ("Functional recovery",
         recovery_ok, f"fp={post_fp:.4f}"),
        ("History dependence",
         hist_ok, f"cos={h_cos:+.4f}"),
        ("Self-reference matters",
         sr_ok, f"diff={sr_diff:.3f}"),
        ("Gamma enables persistence",
         gamma_helps, f"d={gmax-g0:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<42} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")
    print(f"  Core (single-signal persistence): "
          f"{'VERIFIED' if core else 'NOT YET VERIFIED'}")
    print(f"  Sequential scaling: "
          f"{'ACCUMULATES' if seq_classifies else 'OVERWRITES' if second_differentiates else 'FAILS'}")
    print(f"  Three-signal: "
          f"{'CLASSIFIES' if three_classifies else 'CEILING'}")

    if core and seq_classifies and three_classifies:
        print("""
  The system accumulates ordered signal history through
  multiplicative input geometry. Each signal reshapes the
  attractor landscape, leaving persistent traces in the
  collective cell configuration. Sequential signals compose
  rather than overwrite. Order matters.

  This is computation: input-dependent, history-sensitive,
  persistent state transformation on a self-organized substrate.
  The frozen layer (alpha_k) provides the reference frame.
  The dynamics provide the computation. The collective
  provides the memory.
""")
    elif core and seq_classifies:
        print("""
  Two-signal sequences work. Three-signal sequences hit
  a capacity ceiling. The system has finite memory depth
  determined by attractor basin structure, not parameter count.
""")
    elif core:
        print("""
  Single-signal persistence verified. Sequential signals
  overwrite rather than accumulate. The system computes
  but does not compose computations. Each signal reshapes
  the landscape, but the previous landscape is not retained
  in the new configuration.
""")

    print(f"{'-'*W}")


if __name__ == '__main__':
    run()
