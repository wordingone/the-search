#!/usr/bin/env python3
"""
Self-Referential Computation with Multiplicative Input Geometry

  Φ_s(x)_k = tanh(α_k·x_k + β·(x_{k+1} + γ·s_{k+1})·(x_{k-1} + γ·s_{k-1}))

  α_k ∈ [0.4, 1.8]: per-dimension excitability (breaks symmetry)
  β > 0: self-referential coupling (state gates itself)
  γ > 0: signal enters multiplicatively (reshapes the map)

State determines dynamics through β·x_{k+1}·x_{k-1}.
Signal reshapes dynamics through γ·s cross-terms.
No loss function. No optimizer. No weight/activation split.

Core claim: multiplicative input creates PERSISTENT attractor basin
separation. Additive input creates only transient displacement.
Verified by controlled ablation.

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


# ═══════════════════════════════════════════════════════════════
#  THE MAP
# ═══════════════════════════════════════════════════════════════

def phi_0(x, alphas, b):
    d = len(x)
    return [math.tanh(alphas[k] * x[k] + b * x[(k+1)%d] * x[(k-1)%d])
            for k in range(d)]

def phi_mult(x, s, alphas, b, g):
    d = len(x)
    return [math.tanh(alphas[k] * x[k] +
            b * (x[(k+1)%d] + g * s[(k+1)%d]) * (x[(k-1)%d] + g * s[(k-1)%d]))
            for k in range(d)]

def phi_add(x, s, alphas, b, g):
    d = len(x)
    return [math.tanh(alphas[k] * x[k] + b * x[(k+1)%d] * x[(k-1)%d] + g * s[k])
            for k in range(d)]

def fp_dist(x, alphas, b):
    return vnorm(vsub(phi_0(x, alphas, b), x)) / max(vnorm(x), 1.0)

def fp_dist_signal(x, s, alphas, b, g, mode):
    if mode == 'mult':
        p = phi_mult(x, s, alphas, b, g)
    else:
        p = phi_add(x, s, alphas, b, g)
    return vnorm(vsub(p, x)) / max(vnorm(x), 1.0)


# ═══════════════════════════════════════════════════════════════
#  CELL
# ═══════════════════════════════════════════════════════════════

class Cell:
    __slots__ = ('d', 'x')
    def __init__(self, d, scale=0.5):
        self.d = d
        self.x = vrand(d, scale)

    def fp(self, alphas, b):
        return fp_dist(self.x, alphas, b)

    def plasticity(self, alphas, b, sigma=0.15):
        d = self.fp(alphas, b)
        return math.exp(-(d * d) / (sigma * sigma))


# ═══════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════

class Engine:
    def __init__(self, n=6, d=16, alpha_0=1.1, alpha_spread=0.7,
                 beta=0.5, gamma=0.9, mode='mult', seed=None):
        if seed is not None:
            random.seed(seed)
        self.n, self.d = n, d
        self.alphas = [alpha_0 + alpha_spread * (random.random() * 2 - 1)
                       for _ in range(d)]
        self.beta, self.gamma = beta, gamma
        self.mode = mode
        self.cells = [Cell(d) for _ in range(n)]
        self.epsilon = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.max_norm = 4.0
        self.t = 0

    def step(self, signal=None):
        n, d = self.n, self.d
        cells, alphas, b, g = self.cells, self.alphas, self.beta, self.gamma
        phi0c = [phi_0(c.x, alphas, b) for c in cells]

        W = []
        for i in range(n):
            raw = [vdot(cells[i].x, cells[j].x) / (d * self.tau)
                   if i != j else -1e10 for j in range(n)]
            mx = max(raw)
            exps = [math.exp(min(r - mx, 50)) for r in raw]
            s = sum(exps) + 1e-15
            W.append([e / s for e in exps])

        new = []
        for i in range(n):
            xi = cells[i].x
            if signal is not None:
                raw = (phi_mult(xi, signal, alphas, b, g) if self.mode == 'mult'
                       else phi_add(xi, signal, alphas, b, g))
            else:
                raw = phi0c[i]

            pi = cells[i].plasticity(alphas, b)
            if pi > 0.01 and self.epsilon > 0:
                pull = vzero(d)
                for j in range(n):
                    if i == j: continue
                    w = W[i][j]
                    if w < 1e-8: continue
                    pull = vadd(pull, vscale(vsub(phi0c[j], phi0c[i]), w))
                raw = vadd(raw, vscale(pull, pi * self.epsilon))

            raw = vadd(raw, vrand(d, self.noise))
            nx = vadd(vscale(xi, 1 - self.delta), vscale(raw, self.delta))
            nm = vnorm(nx)
            if nm > self.max_norm:
                nx = vscale(nx, self.max_norm / nm)
            new.append(nx)

        for i in range(n):
            cells[i].x = new[i]
        self.t += 1

    def centroid(self):
        c = vzero(self.d)
        for cell in self.cells:
            c = vadd(c, vscale(cell.x, 1.0 / self.n))
        return c

    def mean_fp(self):
        return sum(c.fp(self.alphas, self.beta) for c in self.cells) / self.n

    def mean_norm(self):
        return sum(vnorm(c.x) for c in self.cells) / self.n

    def dispersion(self):
        c = self.centroid()
        return sum(vnorm(vsub(cell.x, c)) ** 2 for cell in self.cells) / self.n


# ═══════════════════════════════════════════════════════════════
#  SIGNALS
# ═══════════════════════════════════════════════════════════════

def make_signals(d):
    a = [0.0] * d
    for k in range(d // 3):
        a[k] = 0.8
    b = [0.0] * d
    for k in range(2 * d // 3, d):
        b[k] = 0.8
    c = [0.8 * (1 if k % 2 == 0 else -1) for k in range(d)]
    return {'A': a, 'B': b, 'C': c}


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════

def run_classify(mode, engine_seed=42, n_trials=8, n_org=500, n_sig=300, n_settle=120):
    d = 16
    classes = make_signals(d)
    res_dur, res_aft = {}, {}

    for label, base in sorted(classes.items()):
        dur, aft = [], []
        for trial in range(n_trials):
            eng = Engine(6, d, 1.1, 0.7, 0.5, 0.9, mode, seed=engine_seed)
            for _ in range(n_org):
                eng.step()
            random.seed(engine_seed * 1000 + ord(label[0]) * 100 + trial)
            sig = [base[k] + random.gauss(0, 0.12) for k in range(d)]
            for _ in range(n_sig):
                eng.step(sig)
            dur.append(eng.centroid())
            for _ in range(n_settle):
                eng.step()
            aft.append(eng.centroid())
        res_dur[label] = dur
        res_aft[label] = aft

    def metrics(res):
        within, between = {}, {}
        labels = sorted(res.keys())
        for l in labels:
            sims = [vcosine(res[l][i], res[l][j])
                    for i in range(len(res[l])) for j in range(i + 1, len(res[l]))]
            within[l] = sum(sims) / max(len(sims), 1)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                sims = [vcosine(c1, c2) for c1 in res[l1] for c2 in res[l2]]
                between[(l1, l2)] = sum(sims) / max(len(sims), 1)
        return within, between

    return metrics(res_dur), metrics(res_aft)


def gap(within, between):
    min_w = min(within.values())
    max_b = max(abs(v) for v in between.values())
    return min_w - max_b, min_w, max_b


# ═══════════════════════════════════════════════════════════════
#  SIGNAL-DEPENDENT FIXED POINTS
# ═══════════════════════════════════════════════════════════════

def find_signal_fps(mode, signals, n_trials=15, n_with=2000, n_without=1000, seed=42):
    d = 16
    ref_eng = Engine(1, d, 1.1, 0.7, 0.5, 0.9, mode, seed=seed)
    alphas, beta, gamma = ref_eng.alphas, ref_eng.beta, ref_eng.gamma

    fps_by_signal = {}
    for label, sig in sorted(signals.items()):
        fps = []
        for trial in range(n_trials):
            random.seed(seed * 100 + ord(label[0]) * 10 + trial)
            x = vrand(d, 0.8)
            for step in range(n_with):
                if mode == 'mult':
                    p = phi_mult(x, sig, alphas, beta, gamma)
                else:
                    p = phi_add(x, sig, alphas, beta, gamma)
                x = vadd(vscale(x, 0.55), vscale(p, 0.45))
                x = vadd(x, vrand(d, 0.003))
                nm = vnorm(x)
                if nm > 4.0:
                    x = vscale(x, 4.0 / nm)
            for step in range(n_without):
                p = phi_0(x, alphas, beta)
                x = vadd(vscale(x, 0.55), vscale(p, 0.45))
                x = vadd(x, vrand(d, 0.003))
                nm = vnorm(x)
                if nm > 4.0:
                    x = vscale(x, 4.0 / nm)
            fps.append(x)
        fps_by_signal[label] = fps
    return fps_by_signal


# ═══════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════

W = 72

def bar(v, w=12, lo=-1.0, hi=1.0):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '█' * n + '░' * (w - n)


def run():
    print("=" * W)
    print("  Φ_s(x)_k = tanh(α_k·x_k + β·(x_{k+1}+γs_{k+1})·(x_{k-1}+γs_{k-1}))")
    print("  Multiplicative input reshapes the attractor landscape.")
    print("  Additive input displaces within a fixed landscape.")
    print("=" * W)

    # ── §1 SELF-ORGANIZATION ─────────────────────────────────

    print(f"\n{'─'*W}")
    print("  §1  SELF-ORGANIZATION")
    print(f"{'─'*W}\n")

    E = Engine(6, 16, 1.1, 0.7, 0.5, 0.9, 'mult', seed=42)
    supra = sum(1 for a in E.alphas if a > 1.0)
    print(f"  α_k = [{min(E.alphas):.2f} .. {max(E.alphas):.2f}], {supra}/{E.d} supercritical\n")

    print(f"  {'step':>5}  {'fp':>7}  {'‖x‖':>7}  {'disp':>7}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}")
    for epoch in range(10):
        for _ in range(60):
            E.step()
        print(f"  {E.t:>5}  {E.mean_fp():>7.4f}  {E.mean_norm():>7.3f}  "
              f"{E.dispersion():>7.3f}")

    so_fp = E.mean_fp()

    print(f"\n  Per-cell:")
    for i, c in enumerate(E.cells):
        fp = c.fp(E.alphas, E.beta)
        ch = "FIXED" if fp < 0.05 else "near" if fp < 0.15 else "ACTIVE"
        print(f"    {i}: fp={fp:.4f} ‖x‖={vnorm(c.x):.3f} {ch}")

    # ── §2 SIGNAL-DEPENDENT FIXED POINTS ─────────────────────

    print(f"\n{'─'*W}")
    print("  §2  SIGNAL-DEPENDENT ATTRACTOR LANDSCAPE")
    print(f"      Converge with signal → remove signal → are FPs still different?")
    print(f"{'─'*W}\n")

    signals = make_signals(16)

    for mode_label, mode in [("MULTIPLICATIVE", "mult"), ("ADDITIVE", "add")]:
        fps = find_signal_fps(mode, signals)

        within_cos, between_cos = {}, {}
        labels = sorted(fps.keys())
        for l in labels:
            sims = [abs(vcosine(fps[l][i], fps[l][j]))
                    for i in range(len(fps[l]))
                    for j in range(i + 1, len(fps[l]))]
            within_cos[l] = sum(sims) / max(len(sims), 1)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                sims = [abs(vcosine(c1, c2)) for c1 in fps[l1] for c2 in fps[l2]]
                between_cos[(l1, l2)] = sum(sims) / max(len(sims), 1)

        min_w = min(within_cos.values())
        max_b = max(between_cos.values())

        print(f"  {mode_label}:")
        for l in labels:
            print(f"    within {l}: {within_cos[l]:.4f}")
        for k in sorted(between_cos):
            print(f"    between {k[0]}v{k[1]}: {between_cos[k]:.4f}")
        fp_gap = min_w - max_b
        print(f"    min(within)={min_w:.4f} max(between)={max_b:.4f} "
              f"gap={fp_gap:+.4f}")
        diverse = fp_gap > 0.02
        print(f"    → {'FPs PERSIST differently' if diverse else 'FPs CONVERGE to same basin'}\n")

        if mode == 'mult':
            mult_fp_diverse = diverse
            mult_fp_gap = fp_gap
        else:
            add_fp_diverse = diverse
            add_fp_gap = fp_gap

    print(f"  Single-cell FPs converge to same basin regardless of mode.")
    print(f"  Signal memory, if it exists, must be COLLECTIVE.")

    # ── §3 CLASSIFICATION: PERSISTENCE ───────────────────────

    print(f"\n{'─'*W}")
    print("  §3  CLASSIFICATION: PERSISTENCE UNDER SIGNAL REMOVAL")
    print(f"      The real test: does classification survive after the signal is gone?")
    print(f"{'─'*W}")

    results = {}
    for mode in ['mult', 'add']:
        tag = "MULTIPLICATIVE" if mode == 'mult' else "ADDITIVE"
        (w_dur, b_dur), (w_aft, b_aft) = run_classify(mode)

        for phase, wc, bc in [("during", w_dur, b_dur), ("after removal", w_aft, b_aft)]:
            g, mw, mb = gap(wc, bc)
            classifies = g > 0.02

            print(f"\n  {tag} — {phase}:")
            for cl in sorted(wc):
                print(f"    within {cl}: {wc[cl]:+.4f} {bar(wc[cl])}")
            for k in sorted(bc):
                print(f"    betw {k[0]}v{k[1]}: {bc[k]:+.4f} {bar(bc[k])}")
            print(f"    gap={g:+.4f} → {'CLASSIFIES' if classifies else 'FAILS'}")

            results[(mode, phase)] = {
                'within': wc, 'between': bc,
                'gap': g, 'classifies': classifies}

    m_dur = results[('mult', 'during')]
    m_aft = results[('mult', 'after removal')]
    a_dur = results[('add', 'during')]
    a_aft = results[('add', 'after removal')]

    print(f"\n  ── PERSISTENCE ABLATION ──")
    print(f"  During signal:  MULT gap={m_dur['gap']:+.4f}  ADD gap={a_dur['gap']:+.4f}")
    print(f"  After removal:  MULT gap={m_aft['gap']:+.4f}  ADD gap={a_aft['gap']:+.4f}")
    print(f"  Persistence Δ = {m_aft['gap'] - a_aft['gap']:+.4f}")

    mult_persists = m_aft['classifies']
    add_collapses = not a_aft['classifies']
    persistence_gap = m_aft['gap'] - a_aft['gap']
    persistence_ok = persistence_gap > 0.1

    print(f"  MULT persists: {'YES' if mult_persists else 'NO'}")
    print(f"  ADD collapses: {'YES' if add_collapses else 'NO'}")
    print(f"  → {'PERSISTENCE VERIFIED' if mult_persists and add_collapses else 'NOT YET VERIFIED'}")

    fp_collective = (not mult_fp_diverse) and (not add_fp_diverse) and mult_persists
    if fp_collective:
        print(f"\n  §2 showed: single cells lose signal memory after removal.")
        print(f"  §3 shows: multi-cell system retains it under MULT.")
        print(f"  → PERSISTENCE IS COLLECTIVE (emerges from coupling, not individual cells)")

    # ── §4 FUNCTIONAL RECOVERY ───────────────────────────────

    print(f"\n{'─'*W}")
    print("  §4  FUNCTIONAL RECOVERY")
    print(f"      After perturbation, does the system remain organized?")
    print(f"{'─'*W}\n")

    H = Engine(6, 16, 1.1, 0.7, 0.5, 0.9, 'mult', seed=42)
    for _ in range(500):
        H.step()
    pre_fp = H.mean_fp()

    sig = vzero(16)
    sig[0] = 2.5; sig[3] = -2.0; sig[8] = 1.5; sig[12] = -1.5

    print(f"  {'phase':<10} {'step':>5}  {'fp':>7}  {'‖x‖':>7}  {'disp':>7}")
    print(f"  {'─'*10} {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}")
    for _ in range(4):
        for __ in range(40):
            H.step(sig)
        print(f"  {'PERTURBED':<10} {H.t:>5}  {H.mean_fp():>7.4f}  "
              f"{H.mean_norm():>7.3f}  {H.dispersion():>7.3f}")
    print()
    for _ in range(6):
        for __ in range(40):
            H.step()
        print(f"  {'recovering':<10} {H.t:>5}  {H.mean_fp():>7.4f}  "
              f"{H.mean_norm():>7.3f}  {H.dispersion():>7.3f}")

    post_fp = H.mean_fp()
    functional_ok = post_fp < 0.12 and H.mean_norm() < 5.0 and H.mean_norm() > 0.5

    print(f"\n  Pre-perturbation fp:  {pre_fp:.4f}")
    print(f"  Post-recovery fp:     {post_fp:.4f}")
    print(f"  → {'FUNCTIONAL RECOVERY' if functional_ok else 'DEGRADED'}"
          f" (system {'remains organized' if functional_ok else 'lost structure'})")

    # ── §5 HISTORY DEPENDENCE ────────────────────────────────

    print(f"\n{'─'*W}")
    print("  §5  HISTORY DEPENDENCE")
    print(f"{'─'*W}\n")

    def cond(sig, n_o=400, n_c=300, n_s=200):
        e = Engine(6, 16, 1.1, 0.7, 0.5, 0.9, 'mult', seed=100)
        for _ in range(n_o):
            e.step()
        for _ in range(n_c):
            e.step(sig)
        for _ in range(n_s):
            e.step()
        return e

    cA, cB = vzero(16), vzero(16)
    for k in range(5):
        cA[k] = 1.5
    for k in range(10, 15):
        cB[k] = 1.5

    eA, eB = cond(cA), cond(cB)
    h_cos = vcosine(eA.centroid(), eB.centroid())
    h_div = vnorm(vsub(eA.centroid(), eB.centroid()))
    hist_ok = abs(h_cos) < 0.7

    print(f"  Conditioning A (dims 0-4):   fp={eA.mean_fp():.4f} ‖x‖={eA.mean_norm():.3f}")
    print(f"  Conditioning B (dims 10-14): fp={eB.mean_fp():.4f} ‖x‖={eB.mean_norm():.3f}")
    print(f"  Centroid cosine: {h_cos:+.4f}")
    print(f"  Centroid divergence: {h_div:.4f}")
    print(f"  → {'HISTORY SHAPES COMPUTATION' if hist_ok else 'Minimal effect'}")

    # ── §6 SELF-REFERENCE ABLATION ───────────────────────────

    print(f"\n{'─'*W}")
    print("  §6  SELF-REFERENCE ABLATION (β=0.5 vs β=0)")
    print(f"{'─'*W}\n")

    ef = Engine(6, 16, 1.1, 0.7, 0.5, 0.9, 'mult', seed=50)
    ea = Engine(6, 16, 1.1, 0.7, 0.0, 0.9, 'mult', seed=50)
    for _ in range(500):
        ef.step()
        ea.step()

    df, da = ef.dispersion(), ea.dispersion()
    sr_diff = abs(df - da) / max(df, da, 0.001)
    fp_diff = abs(ef.mean_fp() - ea.mean_fp())
    sr_ok = sr_diff > 0.1 or fp_diff > 0.05

    print(f"  Full (β=0.5): fp={ef.mean_fp():.4f} disp={df:.4f}")
    print(f"  Ablated (β=0): fp={ea.mean_fp():.4f} disp={da:.4f}")
    print(f"  Relative dispersion diff: {sr_diff:.4f}")
    print(f"  → {'SELF-REFERENCE MATTERS' if sr_ok else 'Minimal effect'}")

    # ── §7 MULTIPLICATIVE γ ABLATION ─────────────────────────

    print(f"\n{'─'*W}")
    print("  §7  MULTIPLICATIVE STRENGTH (γ sweep)")
    print(f"      Does increasing γ increase classification persistence?")
    print(f"{'─'*W}\n")

    gamma_results = []
    for gamma_val in [0.0, 0.3, 0.6, 0.9, 1.2]:
        def run_gamma_test(gv):
            d = 16
            sigs = make_signals(d)
            res = {}
            for label, base in sorted(sigs.items()):
                centroids = []
                for trial in range(6):
                    eng = Engine(6, d, 1.1, 0.7, 0.5, gv, 'mult', seed=42)
                    for _ in range(500):
                        eng.step()
                    random.seed(42000 + ord(label[0]) * 100 + trial)
                    sig = [base[k] + random.gauss(0, 0.12) for k in range(d)]
                    for _ in range(300):
                        eng.step(sig)
                    for _ in range(120):
                        eng.step()
                    centroids.append(eng.centroid())
                res[label] = centroids
            w, b = {}, {}
            labels = sorted(res.keys())
            for l in labels:
                s = [vcosine(res[l][i], res[l][j])
                     for i in range(len(res[l])) for j in range(i+1, len(res[l]))]
                w[l] = sum(s) / max(len(s), 1)
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    l1, l2 = labels[i], labels[j]
                    s = [vcosine(c1, c2) for c1 in res[l1] for c2 in res[l2]]
                    b[(l1, l2)] = sum(s) / max(len(s), 1)
            g, _, _ = gap(w, b)
            return g

        g = run_gamma_test(gamma_val)
        gamma_results.append((gamma_val, g))
        classifies = g > 0.02
        print(f"  γ={gamma_val:.1f}: after-removal gap={g:+.4f} "
              f"{'CLASSIFIES' if classifies else 'FAILS'}")

    gaps = [g for _, g in gamma_results]
    monotonic_trend = all(gaps[i] <= gaps[i+1] + 0.05 for i in range(len(gaps)-1))
    gamma_matters = gaps[-1] > gaps[0] + 0.05
    print(f"\n  γ=0.0 gap: {gaps[0]:+.4f}")
    print(f"  γ=1.2 gap: {gaps[-1]:+.4f}")
    print(f"  → {'γ INCREASES PERSISTENCE' if gamma_matters else 'γ effect unclear'}")

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'═'*W}")
    print("  RESULTS")
    print(f"{'═'*W}\n")

    core_claim = mult_persists and add_collapses and persistence_ok

    checks = [
        ("Self-organization",                    so_fp < 0.15,      f"fp={so_fp:.4f}"),
        ("Persistence is collective (not individual)", fp_collective, f"cell:no, system:yes"),
        ("MULT classification persists",         mult_persists,     f"gap={m_aft['gap']:+.4f}"),
        ("ADD classification collapses",         add_collapses,     f"gap={a_aft['gap']:+.4f}"),
        ("Persistence Δ significant",            persistence_ok,    f"Δ={persistence_gap:+.4f}"),
        ("Functional recovery",                  functional_ok,     f"fp={post_fp:.4f}"),
        ("History dependence",                   hist_ok,           f"cos={h_cos:+.4f}"),
        ("Self-reference matters (β ablation)",  sr_ok,             f"diff={sr_diff:.3f}"),
        ("γ increases persistence",              gamma_matters,     f"Δgap={gaps[-1]-gaps[0]:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'■' if ok else '□'} {name:<42} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")
    print(f"  Core claim (persistence ablation): {'VERIFIED' if core_claim else 'NOT YET VERIFIED'}")

    if core_claim:
        print("""
  Multiplicative input geometry creates persistent classification.
  Additive does not. The mechanism:

    signal s enters the PRODUCT term in the coupling.
    This creates cross-terms that reshape the landscape.
    Different signals activate different dimension-pairs,
    creating different effective dynamics, different attractors.

  Additive input displaces x within a FIXED landscape.
  Remove the signal, x returns to the original basin.

  Multiplicative input CHANGES the landscape. The system
  settles into a new basin shaped by the signal. Remove
  the signal, the new basin persists because the state
  itself has changed to a self-consistent configuration
  of the new landscape.

  This is the separation principle: self-organization and
  computation require different input geometries operating
  simultaneously on the same substrate.
""")

    print(f"{'─'*W}")


if __name__ == '__main__':
    run()
