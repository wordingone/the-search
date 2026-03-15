#!/usr/bin/env python3
"""
Self-Referential Computation with Adaptive Excitability

  Phi_s(x)_k = tanh(a_k * x_k + b * (x_{k+1} + g*s_{k+1}) * (x_{k-1} + g*s_{k-1}))
  a_k = a0 + sigma * tanh(rho_k)
  rho_k <- (1 - lam) * rho_k + lam * x_k^2

State x determines dynamics through the product term.
Signal s reshapes dynamics multiplicatively.
Excitability a_k self-organizes through rho.
No frozen computational structure. Scalar hyperparameters only.
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


A0    = 0.85
SIGMA = 0.8
BETA  = 0.5
GAMMA = 0.9
DELTA = 0.4
LAM   = 0.008
NOISE = 0.008
D     = 48
CLIP  = 3.5


def get_alphas(rho, a0=A0, sigma=SIGMA):
    return [a0 + sigma * math.tanh(r) for r in rho]

def phi(x, rho, s, a0=A0, sigma=SIGMA, beta=BETA, gamma=GAMMA, mode='mult'):
    D = len(x)
    a = get_alphas(rho, a0, sigma)
    if mode == 'mult' and s:
        return [math.tanh(a[k] * x[k] + beta *
                (x[(k+1)%D] + gamma * s[(k+1)%D]) *
                (x[(k-1)%D] + gamma * s[(k-1)%D]))
                for k in range(D)]
    elif mode == 'add' and s:
        return [math.tanh(a[k] * x[k] + beta * x[(k+1)%D] * x[(k-1)%D]
                + gamma * s[k])
                for k in range(D)]
    else:
        return [math.tanh(a[k] * x[k] + beta * x[(k+1)%D] * x[(k-1)%D])
                for k in range(D)]

def step(x, rho, s=None, a0=A0, sigma=SIGMA, beta=BETA, gamma=GAMMA,
         delta=DELTA, lam=LAM, noise=NOISE, mode='mult'):
    p = phi(x, rho, s, a0, sigma, beta, gamma, mode)
    new_x = [(1 - delta) * xi + delta * pi + random.gauss(0, noise)
             for xi, pi in zip(x, p)]
    for k in range(len(new_x)):
        if new_x[k] > CLIP: new_x[k] = CLIP
        elif new_x[k] < -CLIP: new_x[k] = -CLIP
    new_rho = [(1 - lam) * ri + lam * xi * xi for ri, xi in zip(rho, new_x)]
    return new_x, new_rho

def fp_dist(x, rho, a0=A0, sigma=SIGMA, beta=BETA):
    p = phi(x, rho, None, a0, sigma, beta, 0.0, 'mult')
    return vnorm(vsub(p, x)) / max(vnorm(x), 1.0)


def make_signals(D):
    a = [0.0] * D
    for k in range(D // 3): a[k] = 0.8
    b = [0.0] * D
    for k in range(2 * D // 3, D): b[k] = 0.8
    c = [0.8 * (1 if k % 2 == 0 else -1) for k in range(D)]
    return {'A': a, 'B': b, 'C': c}


def run_classify(mode, rho_init='zero', n_trials=6, n_org=400, n_sig=250, n_settle=120,
                 seed=42, lam_override=None):
    signals = make_signals(D)
    lam = LAM if lam_override is None else lam_override
    x_dur, x_aft, r_dur, r_aft = {}, {}, {}, {}

    for label, base_sig in sorted(signals.items()):
        xd, xa, rd, ra = [], [], [], []
        for trial in range(n_trials):
            random.seed(seed)
            x = vrand(D, 0.3)
            if rho_init == 'zero':
                rho = [0.0] * D
            elif rho_init == 'random':
                rho = [random.uniform(-1.5, 1.5) for _ in range(D)]
            else:
                rho = [0.0] * D

            for _ in range(n_org):
                x, rho = step(x, rho, None, lam=lam, mode=mode)

            random.seed(seed * 1000 + ord(label[0]) * 100 + trial)
            sig = [base_sig[k] + random.gauss(0, 0.12) for k in range(D)]
            for _ in range(n_sig):
                x, rho = step(x, rho, sig, lam=lam, mode=mode)
            xd.append(x[:])
            rd.append(rho[:])

            for _ in range(n_settle):
                x, rho = step(x, rho, None, lam=lam, mode=mode)
            xa.append(x[:])
            ra.append(rho[:])

        x_dur[label] = xd; x_aft[label] = xa
        r_dur[label] = rd; r_aft[label] = ra

    return x_dur, x_aft, r_dur, r_aft


def gap_metric(results):
    within, between = {}, {}
    labels = sorted(results.keys())
    for l in labels:
        sims = [vcosine(results[l][i], results[l][j])
                for i in range(len(results[l]))
                for j in range(i + 1, len(results[l]))]
        within[l] = sum(sims) / max(len(sims), 1)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            sims = [vcosine(c1, c2) for c1 in results[l1] for c2 in results[l2]]
            between[(l1, l2)] = sum(sims) / max(len(sims), 1)
    min_w = min(within.values())
    max_b = max(abs(v) for v in between.values())
    return min_w - max_b, within, between


W = 72

def bar(v, w=12, lo=-1.0, hi=1.0):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '█' * n + '░' * (w - n)

def show_gap(within, between, label=""):
    for cl in sorted(within):
        print(f"    within {cl}: {within[cl]:+.4f} {bar(within[cl])}")
    for k in sorted(between):
        print(f"    betw {k[0]}v{k[1]}: {between[k]:+.4f} {bar(between[k])}")
    g = min(within.values()) - max(abs(v) for v in between.values())
    ok = g > 0.02
    print(f"    gap={g:+.4f} -> {'CLASSIFIES' if ok else 'FAILS'}")
    return g, ok


def run():
    print("=" * W)
    print("  Self-Referential Computation with Adaptive Excitability")
    print()
    print("  a_k = a0 + sigma * tanh(rho_k)")
    print("  rho_k <- (1-lam)*rho_k + lam*x_k^2")
    print("  No frozen computational structure. Excitability self-organizes.")
    print("=" * W)

    # ── §1 SELF-ORGANIZATION OF EXCITABILITY ─────────────────

    print(f"\n{'─'*W}")
    print("  S1  SELF-ORGANIZATION OF EXCITABILITY")
    print(f"{'─'*W}\n")

    random.seed(42)
    x = vrand(D, 0.3)
    rho = [0.0] * D

    print(f"  {'step':>5}  {'fp':>7}  {'|x|':>7}  {'<rho>':>7}  "
          f"{'max_rho':>8}  {'supra':>5}  {'sub':>5}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*5}  {'─'*5}")

    for epoch in range(12):
        for _ in range(50):
            x, rho = step(x, rho)
        alphas = get_alphas(rho)
        n_supra = sum(1 for a in alphas if a > 1.05)
        n_sub = sum(1 for a in alphas if a < 0.95)
        mean_rho = sum(rho) / D
        max_rho = max(rho)
        fp = fp_dist(x, rho)
        print(f"  {(epoch+1)*50:>5}  {fp:>7.4f}  {vnorm(x):>7.3f}  {mean_rho:>7.4f}  "
              f"{max_rho:>8.4f}  {n_supra:>5}  {n_sub:>5}")

    so_fp = fp_dist(x, rho)
    alphas_final = get_alphas(rho)
    spread = max(alphas_final) - min(alphas_final)
    n_supra_final = sum(1 for a in alphas_final if a > 1.05)

    print(f"\n  alpha range: [{min(alphas_final):.3f} .. {max(alphas_final):.3f}]")
    print(f"  alpha spread: {spread:.4f}")
    print(f"  Supercritical (>1.05): {n_supra_final}/{D}")

    rho_pattern = rho[:]
    print(f"\n  rho profile (first 24 dims):")
    print(f"  ", end="")
    for k in range(24):
        level = min(8, int(rho[k] / max(max(rho), 0.01) * 8))
        print("▁▂▃▄▅▆▇█"[level], end="")
    print()

    so_ok = spread > 0.05 and n_supra_final > 3

    # ── §2 CLASSIFICATION PERSISTENCE: MULT vs ADD ───────────

    print(f"\n{'─'*W}")
    print("  S2  CLASSIFICATION PERSISTENCE: MULTIPLICATIVE vs ADDITIVE")
    print(f"{'─'*W}")

    results = {}
    for mode in ['mult', 'add']:
        tag = "MULT" if mode == 'mult' else "ADD"
        xd, xa, rd, ra = run_classify(mode, rho_init='zero')

        for phase, data, rdata, plabel in [
            ("during", xd, rd, "x"), ("after", xa, ra, "x"),
            ("after(rho)", ra, ra, "rho")]:
            src = rdata if plabel == "rho" else data
            g, w, b = gap_metric(src)
            ok = g > 0.02
            results[(mode, phase)] = {'gap': g, 'ok': ok, 'w': w, 'b': b}
            if phase != "during":
                print(f"\n  {tag} — {phase}:")
                show_gap(w, b)

    m_aft = results[('mult', 'after')]
    a_aft = results[('add', 'after')]
    m_rho = results[('mult', 'after(rho)')]
    a_rho = results[('add', 'after(rho)')]

    persistence_gap = m_aft['gap'] - a_aft['gap']
    rho_persistence_gap = m_rho['gap'] - a_rho['gap']

    print(f"\n  ── PERSISTENCE SUMMARY ──")
    print(f"  State x after removal:  MULT gap={m_aft['gap']:+.4f}  ADD gap={a_aft['gap']:+.4f}  delta={persistence_gap:+.4f}")
    print(f"  Rho after removal:      MULT gap={m_rho['gap']:+.4f}  ADD gap={a_rho['gap']:+.4f}  delta={rho_persistence_gap:+.4f}")

    mult_persists = m_aft['ok']
    add_collapses = not a_aft['ok']
    rho_classifies = m_rho['ok']

    print(f"  MULT persists (x): {'YES' if mult_persists else 'NO'}")
    print(f"  ADD collapses (x): {'YES' if add_collapses else 'NO'}")
    print(f"  MULT persists (rho): {'YES' if rho_classifies else 'NO'}")

    # ── §3 DYNAMIC vs FIXED EXCITABILITY ─────────────────────

    print(f"\n{'─'*W}")
    print("  S3  DYNAMIC vs FIXED EXCITABILITY (mult mode, after removal)")
    print(f"{'─'*W}\n")

    configs = [
        ("dynamic",       'zero',   LAM,  "rho adapts from zero"),
        ("fixed-random",  'random', 0.0,  "rho fixed at random init"),
        ("fixed-uniform", 'zero',   0.0,  "rho=0, alpha=a0 everywhere"),
    ]

    ablation_gaps = {}
    for name, rinit, lam_val, desc in configs:
        _, xa, _, ra = run_classify('mult', rho_init=rinit, lam_override=lam_val)
        g_x, w_x, b_x = gap_metric(xa)
        g_r, w_r, b_r = gap_metric(ra)
        ablation_gaps[name] = {'x': g_x, 'rho': g_r}
        ok_x = g_x > 0.02
        print(f"  {name:<16} ({desc})")
        print(f"    x gap={g_x:+.4f} {'CLASSIFIES' if ok_x else 'FAILS'}  "
              f"  rho gap={g_r:+.4f}")

    dyn_x = ablation_gaps['dynamic']['x']
    fix_r_x = ablation_gaps['fixed-random']['x']
    fix_u_x = ablation_gaps['fixed-uniform']['x']

    variation_matters = fix_r_x > fix_u_x + 0.05
    dynamic_matches = dyn_x > fix_r_x - 0.1
    dynamic_ok = dyn_x > 0.02
    uniform_fails = fix_u_x < 0.02

    print(f"\n  Variation matters (random > uniform + 0.05): "
          f"{'YES' if variation_matters else 'NO'} ({fix_r_x:+.4f} vs {fix_u_x:+.4f})")
    print(f"  Dynamic matches random (within 0.1): "
          f"{'YES' if dynamic_matches else 'NO'} ({dyn_x:+.4f} vs {fix_r_x:+.4f})")
    print(f"  Dynamic classifies: {'YES' if dynamic_ok else 'NO'}")

    # ── §4 EXCITABILITY IMPRINTING ───────────────────────────

    print(f"\n{'─'*W}")
    print("  S4  EXCITABILITY IMPRINTING")
    print(f"      Does rho capture the spatial structure of the signal?")
    print(f"{'─'*W}\n")

    signals = make_signals(D)
    imprints = {}

    for label, sig in sorted(signals.items()):
        random.seed(42)
        x = vrand(D, 0.3)
        rho = [0.0] * D
        for _ in range(400):
            x, rho = step(x, rho)
        rho_before = rho[:]
        for _ in range(300):
            x, rho = step(x, rho, sig)
        rho_after = rho[:]
        delta_rho = vsub(rho_after, rho_before)
        imprints[label] = delta_rho

    print("  Cosine(delta_rho, signal):")
    imprint_sims = {}
    for l1 in sorted(signals):
        for l2 in sorted(signals):
            c = vcosine(imprints[l1], list(signals[l2]))
            imprint_sims[(l1, l2)] = c
            match = " <-- SELF" if l1 == l2 else ""
            print(f"    rho change from {l1} vs signal {l2}: {c:+.4f}{match}")

    self_match = min(imprint_sims[(l, l)] for l in signals)
    cross_match = max(abs(imprint_sims[(l1, l2)])
                      for l1 in signals for l2 in signals if l1 != l2)
    imprint_ok = self_match > cross_match
    print(f"\n  Min self-match: {self_match:+.4f}")
    print(f"  Max cross-match: {cross_match:.4f}")
    print(f"  -> {'RHO IMPRINTS SIGNAL STRUCTURE' if imprint_ok else 'IMPRINT UNCLEAR'}")

    # ── §5 FUNCTIONAL RECOVERY ───────────────────────────────

    print(f"\n{'─'*W}")
    print("  S5  FUNCTIONAL RECOVERY")
    print(f"{'─'*W}\n")

    random.seed(42)
    x = vrand(D, 0.3)
    rho = [0.0] * D
    for _ in range(500):
        x, rho = step(x, rho)

    pre_fp = fp_dist(x, rho)
    pre_rho = rho[:]

    sig = vzero(D)
    sig[0] = 2.5; sig[5] = -2.0; sig[20] = 1.5; sig[35] = -1.5

    print(f"  {'phase':<10} {'step':>5}  {'fp':>7}  {'|x|':>7}  {'<rho>':>7}")
    print(f"  {'─'*10} {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}")
    t = 500
    for _ in range(4):
        for __ in range(40):
            x, rho = step(x, rho, sig)
            t += 1
        print(f"  {'PERTURBED':<10} {t:>5}  {fp_dist(x,rho):>7.4f}  "
              f"{vnorm(x):>7.3f}  {sum(rho)/D:>7.4f}")
    print()
    for _ in range(5):
        for __ in range(40):
            x, rho = step(x, rho)
            t += 1
        print(f"  {'recovering':<10} {t:>5}  {fp_dist(x,rho):>7.4f}  "
              f"{vnorm(x):>7.3f}  {sum(rho)/D:>7.4f}")

    post_fp = fp_dist(x, rho)
    rho_recovery = vcosine(pre_rho, rho)
    recovery_ok = post_fp < 0.15 and rho_recovery > 0.3

    print(f"\n  Pre fp: {pre_fp:.4f}  Post fp: {post_fp:.4f}")
    print(f"  Rho recovery (cosine with pre): {rho_recovery:.4f}")
    print(f"  -> {'FUNCTIONAL RECOVERY' if recovery_ok else 'DEGRADED'}")

    # ── §6 HISTORY THROUGH EXCITABILITY ──────────────────────

    print(f"\n{'─'*W}")
    print("  S6  HISTORY DEPENDENCE THROUGH EXCITABILITY")
    print(f"{'─'*W}\n")

    def conditioned(sig, n_o=400, n_c=300, n_s=200):
        random.seed(100)
        x = vrand(D, 0.3)
        rho = [0.0] * D
        for _ in range(n_o):
            x, rho = step(x, rho)
        for _ in range(n_c):
            x, rho = step(x, rho, sig)
        for _ in range(n_s):
            x, rho = step(x, rho)
        return x, rho

    cA, cB = vzero(D), vzero(D)
    for k in range(8): cA[k] = 1.5
    for k in range(24, 32): cB[k] = 1.5

    xA, rA = conditioned(cA)
    xB, rB = conditioned(cB)

    h_cos_x = vcosine(xA, xB)
    h_cos_rho = vcosine(rA, rB)
    hist_ok = abs(h_cos_x) < 0.7 or abs(h_cos_rho) < 0.7

    print(f"  Conditioning A (dims 0-7):   |x|={vnorm(xA):.3f}")
    print(f"  Conditioning B (dims 24-31): |x|={vnorm(xB):.3f}")
    print(f"  State cosine: {h_cos_x:+.4f}")
    print(f"  Rho cosine:   {h_cos_rho:+.4f}")

    alphas_A = get_alphas(rA)
    alphas_B = get_alphas(rB)
    a_cos = vcosine(alphas_A, alphas_B)
    print(f"  Alpha cosine: {a_cos:+.4f}")
    print(f"  -> {'HISTORY SHAPES EXCITABILITY' if hist_ok else 'Minimal effect'}")

    # ── §7 GAMMA SWEEP ───────────────────────────────────────

    print(f"\n{'─'*W}")
    print("  S7  MULTIPLICATIVE STRENGTH SWEEP")
    print(f"{'─'*W}\n")

    gamma_gaps = []
    for gv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        sigs = make_signals(D)
        res = {}
        for label, base_sig in sorted(sigs.items()):
            centroids = []
            for trial in range(5):
                random.seed(42)
                x = vrand(D, 0.3)
                rho = [0.0] * D
                for _ in range(400):
                    x, rho = step(x, rho, gamma=gv)
                random.seed(42000 + ord(label[0]) * 100 + trial)
                sig = [base_sig[k] + random.gauss(0, 0.12) for k in range(D)]
                for _ in range(250):
                    x, rho = step(x, rho, sig, gamma=gv)
                for _ in range(120):
                    x, rho = step(x, rho, gamma=gv)
                centroids.append(x[:])
            res[label] = centroids
        g, _, _ = gap_metric(res)
        gamma_gaps.append((gv, g))
        print(f"  gamma={gv:.1f}: after-removal gap={g:+.4f} "
              f"{'CLASSIFIES' if g > 0.02 else 'FAILS'}")

    g0 = gamma_gaps[0][1]
    gmax = gamma_gaps[-1][1]
    gamma_ok = gmax > g0 + 0.05
    print(f"\n  gamma=0.0 gap: {g0:+.4f}")
    print(f"  gamma=1.0 gap: {gmax:+.4f}")
    print(f"  -> {'GAMMA ENABLES PERSISTENCE' if gamma_ok else 'Gamma effect unclear'}")

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'═'*W}")
    print("  RESULTS")
    print(f"{'═'*W}\n")

    checks = [
        ("Excitability self-organizes",
         so_ok, f"spread={spread:.3f}, supra={n_supra_final}"),
        ("MULT classification persists (x)",
         mult_persists, f"gap={m_aft['gap']:+.4f}"),
        ("ADD classification collapses (x)",
         add_collapses, f"gap={a_aft['gap']:+.4f}"),
        ("MULT classification persists (rho)",
         rho_classifies, f"gap={m_rho['gap']:+.4f}"),
        ("Persistence delta significant",
         persistence_gap > 0.1, f"delta={persistence_gap:+.4f}"),
        ("Dynamic alpha classifies",
         dynamic_ok, f"gap={dyn_x:+.4f}"),
        ("Dynamic matches fixed-random",
         dynamic_matches, f"{dyn_x:+.4f} vs {fix_r_x:+.4f}"),
        ("Rho imprints signal structure",
         imprint_ok, f"self={self_match:+.3f} cross={cross_match:.3f}"),
        ("Functional recovery",
         recovery_ok, f"fp={post_fp:.4f} rho_cos={rho_recovery:.3f}"),
        ("History shapes excitability",
         hist_ok, f"rho_cos={h_cos_rho:+.4f}"),
        ("Gamma enables persistence",
         gamma_ok, f"delta={gmax-g0:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'■' if ok else '□'} {name:<42} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)

    core_new = so_ok and dynamic_ok and dynamic_matches and imprint_ok
    core_inherited = mult_persists and add_collapses

    print(f"\n  {n_pass}/{len(checks)} criteria met.")
    print(f"  Core new claim (adaptive excitability):   {'VERIFIED' if core_new else 'NOT YET'}")
    print(f"  Core inherited claim (mult persistence):  {'VERIFIED' if core_inherited else 'NOT YET'}")

    if core_new and core_inherited:
        print("""
  The last frozen layer is eliminated. Excitability self-organizes
  through rho, which tracks local activity: dimensions that are
  active become more excitable, reinforcing their activity.

  Classification persists after signal removal under multiplicative
  input. The rho landscape carries signal structure -- excitability
  itself becomes the memory.

  Dynamic alpha matches fixed-random alpha for classification,
  proving the system does not need pre-set computational structure.
  Seven scalar hyperparameters define the physics. Everything else
  emerges from dynamics.

  One equation. No classes, no attention, no optimizer.
  The code is the math. The math is the computation.
""")
    elif n_pass >= 7:
        print("\n  Partial success. Most claims hold, see failures above.")
    else:
        print("\n  Insufficient. Back to the mathematics.")

    print(f"{'─'*W}")


if __name__ == '__main__':
    run()
