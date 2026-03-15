#!/usr/bin/env python3
"""
Capacity Ceiling: Sequential Signal Accumulation

Proven architecture (multiplicative signal, additive coupling).
Scale signal count K from 3 to 16. At each K, run multiple random
permutations and measure whether the system distinguishes different
signal orderings from the final state alone.

The capacity ceiling is the K at which permutation discrimination
collapses. If no ceiling exists, this is an architecture.

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


def step(xs, alphas, signal=None):
    n, d = len(xs), D
    phi0 = [phi_mult(x, alphas) for x in xs]
    phis = [phi_mult(x, alphas, signal) for x in xs]
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


def make_k_signals(k, d=D, seed=0):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = vrand(d, 0.5)
        nm = vnorm(s)
        if nm > 1e-10:
            s = vscale(s, 0.8 / nm)
        sigs[i] = s
    return sigs


def sig_steps(k):
    return max(40, 200 // max(k // 3, 1))


def settle_steps(k):
    return max(20, 60 // max(k // 4, 1))


def run_sequence(order, signals, alphas, n_org=400, n_settle_final=80,
                 base_seed=42, trial=0):
    k = len(order)
    nps = sig_steps(k)
    nse = settle_steps(k)
    random.seed(base_seed)
    xs = init_xs()
    for _ in range(n_org):
        xs = step(xs, alphas)
    for idx, sig_id in enumerate(order):
        random.seed(base_seed * 1000 + sig_id * 100 + idx * 10 + trial)
        noise_sig = [signals[sig_id][kk] + random.gauss(0, 0.10)
                     for kk in range(D)]
        for _ in range(nps):
            xs = step(xs, alphas, noise_sig)
        for _ in range(nse):
            xs = step(xs, alphas)
    for _ in range(n_settle_final):
        xs = step(xs, alphas)
    return centroid(xs)


def run_sequence_trace(order, signals, alphas, n_org=400,
                       base_seed=42, trial=0):
    k = len(order)
    nps = sig_steps(k)
    nse = settle_steps(k)
    random.seed(base_seed)
    xs = init_xs()
    for _ in range(n_org):
        xs = step(xs, alphas)
    trace = [centroid(xs)]
    for idx, sig_id in enumerate(order):
        random.seed(base_seed * 1000 + sig_id * 100 + idx * 10 + trial)
        noise_sig = [signals[sig_id][kk] + random.gauss(0, 0.10)
                     for kk in range(D)]
        for _ in range(nps):
            xs = step(xs, alphas, noise_sig)
        for _ in range(nse):
            xs = step(xs, alphas)
        trace.append(centroid(xs))
    return trace


def generate_permutations(k, n_perm, seed=99):
    random.seed(seed)
    base = list(range(k))
    perms = []
    seen = set()
    perms.append(tuple(base))
    seen.add(tuple(base))
    perms.append(tuple(reversed(base)))
    seen.add(tuple(reversed(base)))
    attempts = 0
    while len(perms) < n_perm and attempts < n_perm * 50:
        p = base[:]
        random.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        attempts += 1
    return perms


def bar(v, w=20, lo=-1.0, hi=1.0):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


def run():
    print("=" * W)
    print("  CAPACITY CEILING: SEQUENTIAL SIGNAL ACCUMULATION")
    print("  How many sequential signals before history saturates?")
    print("=" * W)

    SEED = 42
    alphas = make_alphas(SEED)
    N_TRIALS = 4
    N_PERM = 6

    # ── PHASE 1: CAPACITY CURVE ──────────────────────────────

    print(f"\n{'-'*W}")
    print("  PHASE 1: PERMUTATION DISCRIMINATION vs SIGNAL COUNT")
    print(f"  {N_PERM} permutations x {N_TRIALS} trials each, random signals")
    print(f"{'-'*W}\n")

    ks = [3, 4, 6, 8, 10, 12, 16]
    capacity_data = []

    for k in ks:
        signals = make_k_signals(k, seed=SEED + k)
        perms = generate_permutations(k, N_PERM, seed=SEED * 10 + k)

        endpoints = {}
        for pi, perm in enumerate(perms):
            trials = []
            for trial in range(N_TRIALS):
                c = run_sequence(perm, signals, alphas,
                                 base_seed=SEED, trial=trial)
                trials.append(c)
            endpoints[pi] = trials

        within_sims = []
        for pi in endpoints:
            cs = endpoints[pi]
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    within_sims.append(vcosine(cs[i], cs[j]))

        between_sims = []
        pis = sorted(endpoints.keys())
        for i in range(len(pis)):
            for j in range(i + 1, len(pis)):
                for c1 in endpoints[pis[i]]:
                    for c2 in endpoints[pis[j]]:
                        between_sims.append(vcosine(c1, c2))

        avg_w = sum(within_sims) / max(len(within_sims), 1)
        avg_b = sum(between_sims) / max(len(between_sims), 1)
        g = avg_w - avg_b

        fwd_rev_cos = None
        if len(perms) >= 2:
            sims = [vcosine(c1, c2)
                    for c1 in endpoints[0] for c2 in endpoints[1]]
            fwd_rev_cos = sum(sims) / len(sims)

        capacity_data.append({
            'k': k, 'within': avg_w, 'between': avg_b,
            'gap': g, 'fwd_rev': fwd_rev_cos, 'n_perm': len(perms)
        })

        ok = g > 0.02
        print(f"  K={k:>2}: within={avg_w:+.4f} between={avg_b:+.4f} "
              f"gap={g:+.4f} {'CLASSIFIES' if ok else 'SATURATED'} "
              f"{bar(g, 15, -0.05, 0.2)}")
        if fwd_rev_cos is not None:
            print(f"        fwd vs rev: {fwd_rev_cos:+.4f}")

    # ── CAPACITY CURVE ───────────────────────────────────────

    print(f"\n{'-'*W}")
    print("  CAPACITY CURVE")
    print(f"{'-'*W}\n")

    last_classifying = 0
    for d_entry in capacity_data:
        marker = '#' if d_entry['gap'] > 0.02 else 'o'
        fill = int(max(0, min(25, (d_entry['gap'] + 0.05) / 0.25 * 25)))
        vbar = '\u2588' * fill + '\u2591' * (25 - fill)
        print(f"  {marker} K={d_entry['k']:>2}  {vbar}  gap={d_entry['gap']:+.4f}")
        if d_entry['gap'] > 0.02:
            last_classifying = d_entry['k']

    classifying_ks = [d_entry['k'] for d_entry in capacity_data if d_entry['gap'] > 0.02]
    failing_ks = [d_entry['k'] for d_entry in capacity_data if d_entry['gap'] <= 0.02]

    no_ceiling = last_classifying == ks[-1]
    inverted = (len(classifying_ks) > 1 and
                capacity_data[-1]['gap'] > capacity_data[0]['gap'] + 0.01)

    if no_ceiling:
        print(f"\n  NO CEILING up to K={ks[-1]}.")
    else:
        print(f"\n  Last classifying: K={last_classifying}")

    if inverted:
        print(f"  INVERTED: K={ks[-1]} gap ({capacity_data[-1]['gap']:+.4f}) > "
              f"K={ks[0]} gap ({capacity_data[0]['gap']:+.4f})")

    # ── PHASE 2: ACCUMULATION TRACE ──────────────────────────

    print(f"\n{'-'*W}")
    print("  PHASE 2: ACCUMULATION TRACE")
    print(f"{'-'*W}\n")

    trace_ks = [6, 12]
    trace_decays = {}

    for k in trace_ks:
        signals = make_k_signals(k, seed=SEED + k)
        order = list(range(k))
        rev_order = list(reversed(range(k)))

        fwd_traces, rev_traces = [], []
        for trial in range(4):
            fwd_traces.append(run_sequence_trace(order, signals, alphas,
                                                  base_seed=SEED, trial=trial))
            rev_traces.append(run_sequence_trace(rev_order, signals, alphas,
                                                  base_seed=SEED, trial=trial))

        print(f"  K={k}, forward [0..{k-1}]:")
        print(f"  {'pos':>4}  {'shift':>8}  {'cum_dist':>10}  {'origin_cos':>10}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*10}")

        shifts = []
        for pos in range(k):
            sh = [vnorm(vsub(tr[pos + 1], tr[pos])) for tr in fwd_traces]
            cum = [vnorm(vsub(tr[pos + 1], tr[0])) for tr in fwd_traces]
            oc = [vcosine(tr[pos + 1], tr[0]) for tr in fwd_traces]
            shifts.append(sum(sh) / len(sh))
            print(f"  {pos:>4}  {sum(sh)/len(sh):>8.4f}  "
                  f"{sum(cum)/len(cum):>10.4f}  {sum(oc)/len(oc):+10.4f}")

        fwd_final = [tr[-1] for tr in fwd_traces]
        rev_final = [tr[-1] for tr in rev_traces]
        fwd_rev = sum(vcosine(f, r) for f in fwd_final
                      for r in rev_final) / (len(fwd_final) * len(rev_final))
        decay = shifts[-1] / max(shifts[0], 1e-6) if len(shifts) >= 2 else 1.0
        trace_decays[k] = decay

        print(f"\n  Fwd vs rev final: {fwd_rev:+.4f}")
        print(f"  Shift decay: {decay:.4f} "
              f"({'sustains' if decay > 0.3 else 'decays'})\n")

    # ── PHASE 3: POSITION DECODABILITY ───────────────────────

    print(f"{'-'*W}")
    print("  PHASE 3: POSITION DECODABILITY")
    print(f"{'-'*W}\n")

    decode_results = {}
    for k in trace_ks:
        signals = make_k_signals(k, seed=SEED + k)
        base_order = list(range(k))

        base_endpoints = []
        for trial in range(N_TRIALS):
            base_endpoints.append(run_sequence(base_order, signals, alphas,
                                               base_seed=SEED, trial=trial))

        noise_sims = [vcosine(base_endpoints[i], base_endpoints[j])
                      for i in range(len(base_endpoints))
                      for j in range(i + 1, len(base_endpoints))]
        noise_cos = sum(noise_sims) / max(len(noise_sims), 1)

        print(f"  K={k}, noise baseline={noise_cos:+.4f}")
        print(f"  {'pos':>4}  {'swap_cos':>10}  {'delta':>8}  {'dec':>4}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*4}")

        decodable = []
        for pos in range(k):
            swap_val = (pos + k // 2) % k
            swapped = base_order[:]
            swapped[pos] = swap_val

            swap_eps = []
            for trial in range(N_TRIALS):
                swap_eps.append(run_sequence(swapped, signals, alphas,
                                             base_seed=SEED, trial=trial))

            swap_sims = [vcosine(b, s) for b in base_endpoints for s in swap_eps]
            swap_cos = sum(swap_sims) / len(swap_sims)
            delta = noise_cos - swap_cos
            ok = delta > 0.02
            decodable.append(ok)
            print(f"  {pos:>4}  {swap_cos:+10.4f}  {delta:+8.4f}  "
                  f"{'YES' if ok else 'no'}")

        n_dec = sum(decodable)
        decode_results[k] = n_dec
        print(f"\n  Decodable: {n_dec}/{k}")

        if n_dec == 0 and any(d_entry['k'] == k and d_entry['gap'] > 0.02
                              for d_entry in capacity_data):
            print(f"  HOLOGRAPHIC: permutations distinguishable, positions not.")
        elif n_dec > k // 2:
            print(f"  PARTIAL DEPTH: majority readable.")
        elif n_dec > 0:
            print(f"  SHALLOW: some positions readable.")
        print()

    # ── PHASE 4: HOLOGRAPHIC VERIFICATION ────────────────────

    print(f"{'-'*W}")
    print("  PHASE 4: HOLOGRAPHIC VERIFICATION (K=12)")
    print(f"  Adjacent swap (changes interaction pattern) vs")
    print(f"  single replacement (changes element identity)")
    print(f"{'-'*W}\n")

    k = 12
    signals = make_k_signals(k, seed=SEED + k)
    base_order = list(range(k))

    base_eps = []
    for trial in range(N_TRIALS):
        base_eps.append(run_sequence(base_order, signals, alphas,
                                     base_seed=SEED, trial=trial))

    single_deltas, adj_deltas = [], []
    test_positions = list(range(0, k - 1, 2))
    for pos in test_positions:
        single_swap = base_order[:]
        single_swap[pos] = (pos + k // 2) % k

        adj_swap = base_order[:]
        adj_swap[pos], adj_swap[pos + 1] = adj_swap[pos + 1], adj_swap[pos]

        s_eps, a_eps = [], []
        for trial in range(N_TRIALS):
            s_eps.append(run_sequence(single_swap, signals, alphas,
                                      base_seed=SEED, trial=trial))
            a_eps.append(run_sequence(adj_swap, signals, alphas,
                                      base_seed=SEED, trial=trial))

        s_sims = [vcosine(b, s) for b in base_eps for s in s_eps]
        a_sims = [vcosine(b, a) for b in base_eps for a in a_eps]
        single_deltas.append(1.0 - sum(s_sims) / len(s_sims))
        adj_deltas.append(1.0 - sum(a_sims) / len(a_sims))

    avg_single = sum(single_deltas) / len(single_deltas)
    avg_adj = sum(adj_deltas) / len(adj_deltas)

    print(f"  Avg displacement from element replacement: {avg_single:.4f}")
    print(f"  Avg displacement from adjacent transposition: {avg_adj:.4f}")
    ratio = avg_adj / max(avg_single, 1e-6)
    print(f"  Ratio: {ratio:.2f}x")

    holographic = avg_adj > 0.01
    print(f"  -> {'INTERACTION PATTERNS ENCODED' if holographic else 'ELEMENT IDENTITY DOMINATES'}")

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print("  RESULTS")
    print(f"{'='*W}\n")

    shifts_sustain = any(trace_decays.get(k, 0) > 0.3 for k in trace_ks)
    k12_holographic = decode_results.get(12, 0) == 0 and \
                      any(d_entry['k'] == 12 and d_entry['gap'] > 0.02
                          for d_entry in capacity_data)

    checks = [
        ("K=6 permutations distinguishable",
         any(d_entry['k'] == 6 and d_entry['gap'] > 0.02 for d_entry in capacity_data),
         f"gap={next((d_entry['gap'] for d_entry in capacity_data if d_entry['k']==6), 0):+.4f}"),
        ("K=12 permutations distinguishable",
         any(d_entry['k'] == 12 and d_entry['gap'] > 0.02 for d_entry in capacity_data),
         f"gap={next((d_entry['gap'] for d_entry in capacity_data if d_entry['k']==12), 0):+.4f}"),
        ("K=16 permutations distinguishable",
         any(d_entry['k'] == 16 and d_entry['gap'] > 0.02 for d_entry in capacity_data),
         f"gap={next((d_entry['gap'] for d_entry in capacity_data if d_entry['k']==16), 0):+.4f}"),
        ("No ceiling up to K=D",
         no_ceiling, f"last={last_classifying}"),
        ("Inverted capacity curve",
         inverted, f"more signals = better"),
        ("Signal shifts sustain",
         shifts_sustain, f"decay={min(trace_decays.values()):.3f}"),
        ("Holographic memory at K=12",
         k12_holographic, f"0/{decode_results.get(12, '?')} decodable, perm gap>0"),
        ("Interaction patterns encoded",
         holographic, f"adj_disp={avg_adj:.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<44} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    if no_ceiling and inverted and k12_holographic:
        print("""
  NO CAPACITY CEILING. INVERTED CURVE. HOLOGRAPHIC MEMORY.

  The system discriminates K=16 permutations BETTER than K=3.
  More signals create richer multiplicative cross-terms in the
  product coupling, giving the system more material to hash.

  At K=12: 0 individual positions decodable, yet permutations
  ARE distinguishable. Memory encodes the PATTERN of how
  signals interacted through the product term, not the
  identity of signals at specific positions.

  The product term b*(x_{k+1}+g*s_{k+1})*(x_{k-1}+g*s_{k-1})
  acts as a natural hash function over signal sequences.
  Each signal modifies the state, and the NEXT signal's
  effect depends on the current state (which encodes all
  previous signals). This creates order-dependent, non-
  invertible state trajectories.

  This is not addressable memory. It is not a register file.
  It is a dynamical hash: the system remembers the pattern
  of interactions without remembering the individual elements.
  Capacity scales with dimension, not parameter count.
""")
    elif no_ceiling:
        print(f"\n  No ceiling found up to K={ks[-1]} = D.")
    else:
        print(f"\n  Ceiling: K={last_classifying}. "
              f"Ratio: {last_classifying}/{D} = {last_classifying/D:.2f}")

    print(f"{'-'*W}")


if __name__ == '__main__':
    run()
