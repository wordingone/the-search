#!/usr/bin/env python3
"""
Self-Referential Computation with Adaptive Excitability

  eff_{i,k} = x_{i,k} + gamma*s_k + eps*sum_j(w_ij * x_{j,k})
  Phi(x_i)_k = tanh(a_{i,k} * x_{i,k} + beta * eff_{k+1} * eff_{k-1})
  a_{i,k} = a0 + sigma * tanh(rho_{i,k})
  rho_{i,k} <- (1-lam)*rho_{i,k} + lam*x_{i,k}^2

Signal and coupling enter the SAME multiplicative channel.
Excitability self-organizes per cell. No frozen structure.
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
A0    = 0.85
SIGMA = 0.7
BETA  = 0.5
GAMMA = 0.9
EPS   = 0.25
TAU   = 0.3
DELTA = 0.35
LAM   = 0.01
NOISE = 0.005
CLIP  = 4.0


def get_alphas(rho):
    return [A0 + SIGMA * math.tanh(r) for r in rho]

def coupling_weights(xs):
    n, d = len(xs), len(xs[0])
    W = []
    for i in range(n):
        raw = [vdot(xs[i], xs[j]) / (d * TAU) if i != j else -1e10
               for j in range(n)]
        mx = max(raw)
        exps = [math.exp(min(r - mx, 50)) for r in raw]
        s = sum(exps) + 1e-15
        W.append([e / s for e in exps])
    return W

def step(xs, rhos, signal=None, mode='mult', gamma=GAMMA, eps=EPS, lam=LAM):
    n, d = len(xs), len(xs[0])
    W = coupling_weights(xs)
    new_xs, new_rhos = [], []

    for i in range(n):
        al = get_alphas(rhos[i])
        eff = xs[i][:]
        for j in range(n):
            if i == j or W[i][j] < 1e-8: continue
            for k in range(d):
                eff[k] += eps * W[i][j] * xs[j][k]
        if signal and mode == 'mult':
            for k in range(d):
                eff[k] += gamma * signal[k]

        if mode == 'add' and signal:
            p = [math.tanh(al[k] * xs[i][k] +
                 BETA * eff[(k+1)%d] * eff[(k-1)%d] +
                 gamma * signal[k])
                 for k in range(d)]
        else:
            p = [math.tanh(al[k] * xs[i][k] +
                 BETA * eff[(k+1)%d] * eff[(k-1)%d])
                 for k in range(d)]

        nx = [(1-DELTA)*xs[i][k] + DELTA*p[k] + random.gauss(0, NOISE)
              for k in range(d)]
        for k in range(d):
            nx[k] = max(-CLIP, min(CLIP, nx[k]))
        nr = [(1-lam)*rhos[i][k] + lam*nx[k]*nx[k] for k in range(d)]
        new_xs.append(nx)
        new_rhos.append(nr)

    return new_xs, new_rhos

def init_state(n=N, d=D, x_scale=0.5, rho_val=0.0):
    xs = [vrand(d, x_scale) for _ in range(n)]
    rhos = [[rho_val]*d for _ in range(n)]
    return xs, rhos

def centroid(xs):
    n, d = len(xs), len(xs[0])
    c = vzero(d)
    for x in xs:
        c = vadd(c, vscale(x, 1.0/n))
    return c

def mean_fp(xs, rhos):
    total = 0.0
    for i in range(len(xs)):
        al = get_alphas(rhos[i])
        d = len(xs[i])
        p = [math.tanh(al[k]*xs[i][k] + BETA*xs[i][(k+1)%d]*xs[i][(k-1)%d])
             for k in range(d)]
        total += vnorm(vsub(p, xs[i])) / max(vnorm(xs[i]), 1.0)
    return total / len(xs)

def mean_norm(xs):
    return sum(vnorm(x) for x in xs) / len(xs)

def dispersion(xs):
    c = centroid(xs)
    return sum(vnorm(vsub(x, c))**2 for x in xs) / len(xs)

def mean_rho(rhos):
    total = 0.0
    for r in rhos:
        total += sum(r) / len(r)
    return total / len(rhos)

def rho_spread(rhos):
    all_a = []
    for r in rhos:
        all_a.extend(get_alphas(r))
    return max(all_a) - min(all_a)

def count_supra(rhos, threshold=1.05):
    total = 0
    for r in rhos:
        total += sum(1 for a in get_alphas(r) if a > threshold)
    return total


def make_signals(d):
    a = [0.0]*d
    for k in range(d//3): a[k] = 0.8
    b = [0.0]*d
    for k in range(2*d//3, d): b[k] = 0.8
    c = [0.8*(1 if k%2==0 else -1) for k in range(d)]
    return {'A': a, 'B': b, 'C': c}


def run_classify(mode, n_trials=8, n_org=500, n_sig=300, n_settle=120,
                 seed=42, eps=EPS, lam=LAM, gamma=GAMMA, rho_init=0.0):
    signals = make_signals(D)
    x_dur, x_aft, r_dur, r_aft = {}, {}, {}, {}

    for label, base in sorted(signals.items()):
        xd, xa, rd, ra = [], [], [], []
        for trial in range(n_trials):
            random.seed(seed)
            xs, rhos = init_state(rho_val=rho_init)
            for _ in range(n_org):
                xs, rhos = step(xs, rhos, eps=eps, lam=lam, gamma=gamma, mode=mode)
            random.seed(seed*1000 + ord(label[0])*100 + trial)
            sig = [base[k] + random.gauss(0, 0.12) for k in range(D)]
            for _ in range(n_sig):
                xs, rhos = step(xs, rhos, sig, mode=mode, eps=eps, lam=lam, gamma=gamma)
            xd.append(centroid(xs))
            rd.append([r[:] for r in rhos])
            for _ in range(n_settle):
                xs, rhos = step(xs, rhos, eps=eps, lam=lam, gamma=gamma, mode=mode)
            xa.append(centroid(xs))
            ra.append([r[:] for r in rhos])
        x_dur[label] = xd; x_aft[label] = xa
        r_dur[label] = rd; r_aft[label] = ra

    return x_dur, x_aft, r_dur, r_aft


def gap_metric(results):
    within, between = {}, {}
    labels = sorted(results.keys())
    for l in labels:
        sims = [vcosine(results[l][i], results[l][j])
                for i in range(len(results[l]))
                for j in range(i+1, len(results[l]))]
        within[l] = sum(sims)/max(len(sims), 1)
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            l1, l2 = labels[i], labels[j]
            sims = [vcosine(c1, c2) for c1 in results[l1] for c2 in results[l2]]
            between[(l1,l2)] = sum(sims)/max(len(sims), 1)
    min_w = min(within.values())
    max_b = max(abs(v) for v in between.values())
    return min_w - max_b, within, between


def rho_gap(r_results):
    flat = {}
    for label, trials in r_results.items():
        vecs = []
        for trial_rhos in trials:
            flat_rho = []
            for r in trial_rhos:
                flat_rho.extend(r)
            vecs.append(flat_rho)
        flat[label] = vecs
    return gap_metric(flat)


W = 72

def bar(v, w=12, lo=-1.0, hi=1.0):
    f = max(0.0, min(1.0, (v - lo)/(hi - lo + 1e-10)))
    n = int(f * w)
    return '|'*n + '.'*(w-n)


def run():
    print("=" * W)
    print("  Self-Referential Computation with Adaptive Excitability")
    print("  Signal + coupling enter ONE multiplicative channel.")
    print("  a_{i,k} = 0.85 + 0.7*tanh(rho_{i,k}).  No frozen structure.")
    print("=" * W)

    print(f"\n{'~'*W}")
    print("  S1  SELF-ORGANIZATION OF EXCITABILITY")
    print(f"{'~'*W}\n")

    random.seed(42)
    xs, rhos = init_state()

    print(f"  {'step':>5}  {'fp':>7}  {'|x|':>7}  {'<rho>':>7}  "
          f"{'spread':>7}  {'supra':>5}/{N*D}")
    print(f"  {'~'*5}  {'~'*7}  {'~'*7}  {'~'*7}  {'~'*7}  {'~'*8}")

    for epoch in range(10):
        for _ in range(60):
            xs, rhos = step(xs, rhos)
        sp = rho_spread(rhos)
        ns = count_supra(rhos)
        print(f"  {(epoch+1)*60:>5}  {mean_fp(xs,rhos):>7.4f}  {mean_norm(xs):>7.3f}  "
              f"{mean_rho(rhos):>7.4f}  {sp:>7.4f}  {ns:>5}/{N*D}")

    so_fp = mean_fp(xs, rhos)
    so_spread = rho_spread(rhos)
    so_supra = count_supra(rhos)
    so_total = N * D

    print(f"\n  Per-cell rho profiles:")
    for i in range(N):
        al = get_alphas(rhos[i])
        n_s = sum(1 for a in al if a > 1.05)
        mx = max(rhos[i])
        print(f"    cell {i}: supra={n_s:>2}/{D}  max_rho={mx:.3f}  "
              f"alpha=[{min(al):.2f}..{max(al):.2f}]")

    so_ok = so_spread > 0.05 and so_supra > 5 and so_supra < so_total - 5

    print(f"\n{'~'*W}")
    print("  S2  CLASSIFICATION PERSISTENCE: MULT vs ADD")
    print(f"{'~'*W}")

    results = {}
    for mode in ['mult', 'add']:
        tag = "MULT" if mode == 'mult' else "ADD"
        xd, xa, rd, ra = run_classify(mode)

        for phase, data in [("during", xd), ("after", xa)]:
            g, w, b = gap_metric(data)
            ok = g > 0.02
            results[(mode, phase)] = {'gap': g, 'ok': ok, 'w': w, 'b': b}

            print(f"\n  {tag} -- {phase}:")
            for cl in sorted(w):
                print(f"    within {cl}: {w[cl]:+.4f} {bar(w[cl])}")
            for k in sorted(b):
                print(f"    betw {k[0]}v{k[1]}: {b[k]:+.4f} {bar(b[k])}")
            print(f"    gap={g:+.4f} -> {'CLASSIFIES' if ok else 'FAILS'}")

    m_aft = results[('mult','after')]
    a_aft = results[('add','after')]
    persistence_gap = m_aft['gap'] - a_aft['gap']

    mult_persists = m_aft['ok']
    add_collapses = not a_aft['ok']

    print(f"\n  PERSISTENCE: mult={m_aft['gap']:+.4f}  add={a_aft['gap']:+.4f}  "
          f"delta={persistence_gap:+.4f}")
    print(f"  MULT persists: {'YES' if mult_persists else 'NO'}  "
          f"ADD collapses: {'YES' if add_collapses else 'NO'}")

    print(f"\n{'~'*W}")
    print("  S3  RHO-BASED CLASSIFICATION")
    print(f"      Does excitability itself carry the classification?")
    print(f"{'~'*W}\n")

    _, _, _, ra_mult = run_classify('mult')
    _, _, _, ra_add = run_classify('add')

    rg_m, rw_m, rb_m = rho_gap(ra_mult)
    rg_a, rw_a, rb_a = rho_gap(ra_add)
    rho_m_ok = rg_m > 0.02
    rho_a_ok = rg_a > 0.02

    print(f"  MULT rho gap: {rg_m:+.4f} -> {'CLASSIFIES' if rho_m_ok else 'FAILS'}")
    print(f"  ADD  rho gap: {rg_a:+.4f} -> {'CLASSIFIES' if rho_a_ok else 'FAILS'}")

    print(f"\n{'~'*W}")
    print("  S4  DYNAMIC vs FIXED EXCITABILITY")
    print(f"{'~'*W}\n")

    ablation = {}
    for name, lv, ri, desc in [
        ("dynamic",  LAM, 0.0,  "rho adapts"),
        ("fix-rand", 0.0, None, "rho frozen random"),
        ("fix-uni",  0.0, 0.0,  "rho=0, alpha=a0")]:

        if ri is None:
            random.seed(42)
            ri_val = random.uniform(-1.5, 1.5)
        else:
            ri_val = ri

        _, xa, _, _ = run_classify('mult', lam=lv, rho_init=ri_val)
        g, _, _ = gap_metric(xa)
        ablation[name] = g
        print(f"  {name:<12} ({desc}): gap={g:+.4f} "
              f"{'CLASSIFIES' if g > 0.02 else 'FAILS'}")

    dyn_ok = ablation['dynamic'] > 0.02
    dyn_matches = ablation['dynamic'] > ablation['fix-rand'] - 0.15
    var_matters = ablation['fix-rand'] > ablation['fix-uni'] + 0.03

    print(f"\n  Dynamic classifies: {'YES' if dyn_ok else 'NO'}")
    print(f"  Variation matters:  {'YES' if var_matters else 'NO'}")

    print(f"\n{'~'*W}")
    print("  S5  COUPLING ABLATION (eps=0 vs eps=0.25)")
    print(f"{'~'*W}\n")

    _, xa_coupled, _, _ = run_classify('mult', eps=EPS)
    _, xa_uncoupled, _, _ = run_classify('mult', eps=0.0)

    g_c, _, _ = gap_metric(xa_coupled)
    g_u, _, _ = gap_metric(xa_uncoupled)

    coupling_helps = g_c > g_u + 0.03

    print(f"  Coupled (eps=0.25):   gap={g_c:+.4f}")
    print(f"  Uncoupled (eps=0):    gap={g_u:+.4f}")
    print(f"  -> {'COUPLING HELPS' if coupling_helps else 'COUPLING NEUTRAL'}")

    print(f"\n{'~'*W}")
    print("  S6  FUNCTIONAL RECOVERY")
    print(f"{'~'*W}\n")

    random.seed(42)
    xs, rhos = init_state()
    for _ in range(500):
        xs, rhos = step(xs, rhos)
    pre_fp = mean_fp(xs, rhos)
    pre_rhos_flat = []
    for r in rhos: pre_rhos_flat.extend(r)

    sig = vzero(D)
    sig[0]=2.5; sig[3]=-2.0; sig[8]=1.5; sig[12]=-1.5

    print(f"  {'phase':<10} {'step':>5}  {'fp':>7}  {'|x|':>7}  {'spread':>7}")
    print(f"  {'~'*10} {'~'*5}  {'~'*7}  {'~'*7}  {'~'*7}")
    t = 500
    for _ in range(4):
        for __ in range(40):
            xs, rhos = step(xs, rhos, sig)
            t += 1
        print(f"  {'PERTURBED':<10} {t:>5}  {mean_fp(xs,rhos):>7.4f}  "
              f"{mean_norm(xs):>7.3f}  {rho_spread(rhos):>7.4f}")
    print()
    for _ in range(5):
        for __ in range(40):
            xs, rhos = step(xs, rhos)
            t += 1
        print(f"  {'recovering':<10} {t:>5}  {mean_fp(xs,rhos):>7.4f}  "
              f"{mean_norm(xs):>7.3f}  {rho_spread(rhos):>7.4f}")

    post_fp = mean_fp(xs, rhos)
    post_rhos_flat = []
    for r in rhos: post_rhos_flat.extend(r)
    rho_recov = vcosine(pre_rhos_flat, post_rhos_flat)
    recovery_ok = post_fp < 0.15 and rho_recov > 0.3

    print(f"\n  Pre fp: {pre_fp:.4f}  Post fp: {post_fp:.4f}")
    print(f"  Rho recovery cosine: {rho_recov:.4f}")
    print(f"  -> {'FUNCTIONAL RECOVERY' if recovery_ok else 'DEGRADED'}")

    print(f"\n{'~'*W}")
    print("  S7  HISTORY DEPENDENCE THROUGH EXCITABILITY")
    print(f"{'~'*W}\n")

    def conditioned(sig, n_o=400, n_c=300, n_s=200):
        random.seed(100)
        xs, rhos = init_state()
        for _ in range(n_o):
            xs, rhos = step(xs, rhos)
        for _ in range(n_c):
            xs, rhos = step(xs, rhos, sig)
        for _ in range(n_s):
            xs, rhos = step(xs, rhos)
        return xs, rhos

    cA, cB = vzero(D), vzero(D)
    for k in range(5): cA[k] = 1.5
    for k in range(10, 15): cB[k] = 1.5

    xsA, rA = conditioned(cA)
    xsB, rB = conditioned(cB)

    h_cos_x = vcosine(centroid(xsA), centroid(xsB))
    rA_flat = []; rB_flat = []
    for r in rA: rA_flat.extend(r)
    for r in rB: rB_flat.extend(r)
    h_cos_rho = vcosine(rA_flat, rB_flat)

    hist_ok = abs(h_cos_x) < 0.7

    print(f"  Conditioning A (dims 0-4):   centroid |x|={vnorm(centroid(xsA)):.3f}")
    print(f"  Conditioning B (dims 10-14): centroid |x|={vnorm(centroid(xsB)):.3f}")
    print(f"  Centroid cosine: {h_cos_x:+.4f}")
    print(f"  Rho cosine:      {h_cos_rho:+.4f}")
    print(f"  -> {'HISTORY SHAPES COMPUTATION' if hist_ok else 'Minimal effect'}")

    print(f"\n{'~'*W}")
    print("  S8  MULTIPLICATIVE STRENGTH SWEEP")
    print(f"{'~'*W}\n")

    gamma_gaps = []
    for gv in [0.0, 0.3, 0.6, 0.9, 1.2]:
        _, xa, _, _ = run_classify('mult', gamma=gv)
        g, _, _ = gap_metric(xa)
        gamma_gaps.append((gv, g))
        print(f"  gamma={gv:.1f}: after-removal gap={g:+.4f} "
              f"{'CLASSIFIES' if g > 0.02 else 'FAILS'}")

    g0 = gamma_gaps[0][1]
    gmax = max(g for _, g in gamma_gaps)
    gamma_ok = gmax > g0 + 0.05

    print(f"\n  gamma=0 gap: {g0:+.4f}  max gap: {gmax:+.4f}")
    print(f"  -> {'GAMMA ENABLES PERSISTENCE' if gamma_ok else 'Gamma effect unclear'}")

    print(f"\n{'~'*W}")
    print("  S9  SELF-REFERENCE ABLATION (beta=0.5 vs beta=0)")
    print(f"{'~'*W}\n")

    def step_nobeta(xs, rhos, signal=None, mode='mult', gamma=GAMMA, eps=EPS, lam=LAM):
        n, d = len(xs), len(xs[0])
        W2 = coupling_weights(xs)
        new_xs, new_rhos = [], []
        for i in range(n):
            al = get_alphas(rhos[i])
            p = [math.tanh(al[k] * xs[i][k]) for k in range(d)]
            nx = [(1-DELTA)*xs[i][k] + DELTA*p[k] + random.gauss(0, NOISE)
                  for k in range(d)]
            for k in range(d):
                nx[k] = max(-CLIP, min(CLIP, nx[k]))
            nr = [(1-lam)*rhos[i][k] + lam*nx[k]*nx[k] for k in range(d)]
            new_xs.append(nx)
            new_rhos.append(nr)
        return new_xs, new_rhos

    random.seed(50)
    xsF, rF = init_state()
    random.seed(50)
    xsA2, rA2 = init_state()

    for _ in range(500):
        xsF, rF = step(xsF, rF)
        xsA2, rA2 = step_nobeta(xsA2, rA2)

    df, da = dispersion(xsF), dispersion(xsA2)
    sr_diff = abs(df - da) / max(df, da, 0.001)
    fp_diff = abs(mean_fp(xsF, rF) - mean_fp(xsA2, rA2))
    sr_ok = sr_diff > 0.1 or fp_diff > 0.05

    print(f"  Full (beta=0.5): fp={mean_fp(xsF,rF):.4f} disp={df:.4f}")
    print(f"  beta=0:          fp={mean_fp(xsA2,rA2):.4f} disp={da:.4f}")
    print(f"  Dispersion diff: {sr_diff:.4f}")
    print(f"  -> {'SELF-REFERENCE MATTERS' if sr_ok else 'Minimal effect'}")

    print(f"\n{'='*W}")
    print("  RESULTS")
    print(f"{'='*W}\n")

    checks = [
        ("Excitability self-organizes",
         so_ok, f"spread={so_spread:.3f} supra={so_supra}/{so_total}"),
        ("MULT persists after removal (x)",
         mult_persists, f"gap={m_aft['gap']:+.4f}"),
        ("ADD collapses after removal (x)",
         add_collapses, f"gap={a_aft['gap']:+.4f}"),
        ("Persistence delta significant",
         persistence_gap > 0.1, f"delta={persistence_gap:+.4f}"),
        ("Rho classifies under MULT",
         rho_m_ok, f"gap={rg_m:+.4f}"),
        ("Dynamic alpha classifies",
         dyn_ok, f"gap={ablation['dynamic']:+.4f}"),
        ("Functional recovery",
         recovery_ok, f"fp={post_fp:.4f} rho_cos={rho_recov:.3f}"),
        ("History shapes computation",
         hist_ok, f"cos={h_cos_x:+.4f}"),
        ("Gamma enables persistence",
         gamma_ok, f"max_gap={gmax:+.4f}"),
        ("Self-reference matters",
         sr_ok, f"disp_diff={sr_diff:.3f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'*' if ok else ' '} {name:<42} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)

    core_inherited = mult_persists and add_collapses and persistence_gap > 0.1
    core_new = so_ok and dyn_ok and rho_m_ok

    print(f"\n  {n_pass}/{len(checks)} criteria met.")
    print(f"  Inherited (mult persistence):  {'VERIFIED' if core_inherited else 'NOT YET'}")
    print(f"  New (adaptive excitability):    {'VERIFIED' if core_new else 'NOT YET'}")

    if core_inherited and core_new:
        print("""
  The last frozen layer is eliminated. Each cell develops its own
  excitability profile through rho, which tracks local activity.
  Signal and coupling enter the SAME multiplicative channel:
  eff = x + gamma*s + eps*sum(w*x_j). Both reshape the product
  term, both reshape the attractor landscape.

  Classification persists in BOTH x (state) and rho (excitability)
  after signal removal under multiplicative input. Additive fails.
  The system writes its computational structure on the same
  substrate that executes it.
""")
    elif n_pass >= 7:
        print("\n  Partial success. See failures above.")
    else:
        print("\n  Insufficient. Back to the mathematics.")

    print(f"{'~'*W}")


if __name__ == '__main__':
    run()
