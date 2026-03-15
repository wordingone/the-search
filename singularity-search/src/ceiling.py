#!/usr/bin/env python3
"""
Capacity Ceiling: Sequential Signal Accumulation

Same proven architecture (multiplicative signal, additive coupling).
Scale signal count K from 3 to 12. At each K, run multiple random
permutations and measure whether the system can still distinguish
different signal orderings from the final state.

The capacity ceiling is the K at which permutation discrimination
collapses. That number determines if this is a toy or an architecture.

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


def mean_fp(xs, alphas):
    total = 0.0
    for x in xs:
        p = [math.tanh(alphas[k] * x[k] + BETA * x[(k+1)%D] * x[(k-1)%D])
             for k in range(D)]
        total += vnorm(vsub(p, x)) / max(vnorm(x), 1.0)
    return total / len(xs)


def make_k_signals(k, d=D, seed=0):
    random.seed(seed)
    sigs = {}
    width = max(1, d // k)
    for i in range(k):
        s = [0.0] * d
        start = (i * d) // k
        end = min(start + max(width, 2), d)
        for j in range(start, end):
            s[j] = 0.8
        if i % 2 == 1:
            s = [-v for v in s]
        sigs[i] = s
    return sigs


def run_sequence(order, signals, alphas, n_org=500, n_per_sig=200,
                 n_settle_each=60, n_settle_final=100, base_seed=42, trial=0):
    random.seed(base_seed)
    xs = init_xs()
    for _ in range(n_org):
        xs = step(xs, alphas)

    for idx, sig_id in enumerate(order):
        random.seed(base_seed * 1000 + sig_id * 100 + idx * 10 + trial)
        noise_sig = [signals[sig_id][k] + random.gauss(0, 0.10)
                     for k in range(D)]
        for _ in range(n_per_sig):
            xs = step(xs, alphas, noise_sig)
        for _ in range(n_settle_each):
            xs = step(xs, alphas)

    for _ in range(n_settle_final):
        xs = step(xs, alphas)
    return centroid(xs)


def run_sequence_trace(order, signals, alphas, n_org=500, n_per_sig=200,
                       n_settle_each=60, base_seed=42, trial=0):
    random.seed(base_seed)
    xs = init_xs()
    for _ in range(n_org):
        xs = step(xs, alphas)

    trace = [centroid(xs)]
    for idx, sig_id in enumerate(order):
        random.seed(base_seed * 1000 + sig_id * 100 + idx * 10 + trial)
        noise_sig = [signals[sig_id][k] + random.gauss(0, 0.10)
                     for k in range(D)]
        for _ in range(n_per_sig):
            xs = step(xs, alphas, noise_sig)
        for _ in range(n_settle_each):
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
    N_TRIALS = 5
    N_PERM = 8

    # ── PHASE 1: CAPACITY CURVE ──────────────────────────────

    print(f"\n{'-'*W}")
    print("  PHASE 1: PERMUTATION DISCRIMINATION vs SIGNAL COUNT")
    print(f"  {N_PERM} permutations x {N_TRIALS} trials per permutation")
    print(f"{'-'*W}\n")

    ks = [3, 4, 5, 6, 8, 10, 12]
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
            'gap': g, 'fwd_rev': fwd_rev_cos,
            'n_perm': len(perms)
        })

        ok = g > 0.02
        print(f"  K={k:>2}: within={avg_w:+.4f} between={avg_b:+.4f} "
              f"gap={g:+.4f} {'CLASSIFIES' if ok else 'SATURATED'} "
              f"{bar(g, 15, -0.1, 0.3)}")
        if fwd_rev_cos is not None:
            print(f"        fwd vs rev: {fwd_rev_cos:+.4f}")

    # ── CAPACITY CEILING ─────────────────────────────────────

    print(f"\n{'-'*W}")
    print("  CAPACITY CURVE")
    print(f"{'-'*W}\n")

    ceiling = None
    last_classifying = 0
    for d in capacity_data:
        marker = '#' if d['gap'] > 0.02 else 'o'
        fill = int(max(0, min(25, (d['gap'] + 0.1) / 0.4 * 25)))
        vbar = '\u2588' * fill + '\u2591' * (25 - fill)
        print(f"  {marker} K={d['k']:>2}  {vbar}  gap={d['gap']:+.4f}")
        if d['gap'] > 0.02:
            last_classifying = d['k']
        elif ceiling is None:
            ceiling = d['k']

    if ceiling:
        print(f"\n  Capacity ceiling: K={ceiling} "
              f"(last classifying: K={last_classifying})")
    else:
        print(f"\n  No ceiling found up to K={ks[-1]}. "
              f"All signal counts classify.")

    # ── PHASE 2: ACCUMULATION TRACE ──────────────────────────

    print(f"\n{'-'*W}")
    print("  PHASE 2: ACCUMULATION TRACE (per-signal centroid shift)")
    print(f"  How much does each additional signal move the state?")
    print(f"{'-'*W}\n")

    for k in [6, 12]:
        signals = make_k_signals(k, seed=SEED + k)
        order = list(range(k))

        traces = []
        for trial in range(4):
            tr = run_sequence_trace(order, signals, alphas,
                                    base_seed=SEED, trial=trial)
            traces.append(tr)

        print(f"  K={k}, order=[0,1,...,{k-1}]:")
        print(f"  {'sig':>4}  {'shift':>8}  {'cum_shift':>10}  {'fp_cos':>8}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}")

        for pos in range(k):
            shifts = []
            fp_cos = []
            for tr in traces:
                before = tr[pos]
                after = tr[pos + 1]
                shifts.append(vnorm(vsub(after, before)))
                fp_cos.append(vcosine(after, tr[0]))
            avg_shift = sum(shifts) / len(shifts)
            avg_fp = sum(fp_cos) / len(fp_cos)
            cum = sum(vnorm(vsub(tr[pos + 1], tr[0])) for tr in traces) / len(traces)
            print(f"  {pos:>4}  {avg_shift:>8.4f}  {cum:>10.4f}  {avg_fp:+8.4f}")

        rev_traces = []
        rev_order = list(reversed(range(k)))
        for trial in range(4):
            tr = run_sequence_trace(rev_order, signals, alphas,
                                    base_seed=SEED, trial=trial)
            rev_traces.append(tr)

        fwd_final = [tr[-1] for tr in traces]
        rev_final = [tr[-1] for tr in rev_traces]
        sims = [vcosine(f, r) for f in fwd_final for r in rev_final]
        fwd_rev = sum(sims) / len(sims)
        print(f"\n  Forward vs reverse final cos: {fwd_rev:+.4f}")
        shift_0 = sum(vnorm(vsub(tr[1], tr[0])) for tr in traces) / len(traces)
        shift_last = sum(vnorm(vsub(tr[-1], tr[-2])) for tr in traces) / len(traces)
        decay = shift_last / max(shift_0, 1e-6)
        print(f"  First signal shift: {shift_0:.4f}")
        print(f"  Last signal shift:  {shift_last:.4f}")
        print(f"  Shift decay ratio:  {decay:.4f}")
        print()

    # ── PHASE 3: INFORMATION RETENTION ───────────────────────

    print(f"{'-'*W}")
    print("  PHASE 3: CAN WE DECODE SIGNAL IDENTITY FROM FINAL STATE?")
    print(f"  For each position in the sequence, does varying that signal")
    print(f"  change the final state more than noise?")
    print(f"{'-'*W}\n")

    for k in [6, 12]:
        signals = make_k_signals(k, seed=SEED + k)
        base_order = list(range(k))

        base_endpoints = []
        for trial in range(N_TRIALS):
            c = run_sequence(base_order, signals, alphas,
                             base_seed=SEED, trial=trial)
            base_endpoints.append(c)

        print(f"  K={k}, base order=[0..{k-1}]")
        print(f"  {'pos':>4}  {'swap_cos':>10}  {'noise_cos':>10}  {'delta':>8}  {'retained':>8}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

        position_retained = []
        for pos in range(k):
            swap_val = (pos + k // 2) % k
            swapped_order = base_order[:]
            swapped_order[pos] = swap_val

            swap_endpoints = []
            for trial in range(N_TRIALS):
                c = run_sequence(swapped_order, signals, alphas,
                                 base_seed=SEED, trial=trial)
                swap_endpoints.append(c)

            swap_sims = [vcosine(b, s) for b in base_endpoints
                         for s in swap_endpoints]
            swap_cos = sum(swap_sims) / len(swap_sims)

            noise_sims = [vcosine(base_endpoints[i], base_endpoints[j])
                          for i in range(len(base_endpoints))
                          for j in range(i + 1, len(base_endpoints))]
            noise_cos = sum(noise_sims) / max(len(noise_sims), 1)

            delta = noise_cos - swap_cos
            retained = delta > 0.02
            position_retained.append(retained)

            print(f"  {pos:>4}  {swap_cos:+10.4f}  {noise_cos:+10.4f}  "
                  f"{delta:+8.4f}  {'YES' if retained else 'no'}")

        n_retained = sum(position_retained)
        print(f"\n  Positions retained: {n_retained}/{k}")
        early = sum(position_retained[:k//2])
        late = sum(position_retained[k//2:])
        print(f"  Early half (0..{k//2-1}): {early}/{k//2}")
        print(f"  Late half ({k//2}..{k-1}): {late}/{k-k//2}")

        if n_retained == k:
            print(f"  -> ALL positions decodable. Full depth retention.")
        elif n_retained > k // 2:
            print(f"  -> Majority retained. Partial depth.")
        elif n_retained > 0:
            print(f"  -> Only recent signals retained. Shallow memory.")
        else:
            print(f"  -> No position decodable. System has no signal memory.")
        print()

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"{'='*W}")
    print("  RESULTS")
    print(f"{'='*W}\n")

    classifying_ks = [d['k'] for d in capacity_data if d['gap'] > 0.02]
    saturated_ks = [d['k'] for d in capacity_data if d['gap'] <= 0.02]

    print(f"  Classifying: K = {classifying_ks}")
    print(f"  Saturated:   K = {saturated_ks if saturated_ks else 'none up to K=12'}")

    if ceiling:
        print(f"\n  CAPACITY CEILING: K = {ceiling}")
        print(f"  System accumulates up to {last_classifying} sequential signals")
        print(f"  before permutation discrimination collapses.")
        ratio = last_classifying / D
        print(f"  Capacity/dimension ratio: {last_classifying}/{D} = {ratio:.2f}")
        if ratio >= 0.5:
            print(f"  -> ARCHITECTURE-SCALE capacity (scales with dimension)")
        else:
            print(f"  -> TOY-SCALE capacity (sub-linear in dimension)")
    else:
        max_k = ks[-1]
        ratio = max_k / D
        print(f"\n  NO CEILING FOUND up to K={max_k}")
        print(f"  Capacity/dimension ratio: >= {max_k}/{D} = {ratio:.2f}")
        if ratio >= 0.5:
            print(f"  -> ARCHITECTURE-SCALE: capacity grows with dimension")
        else:
            print(f"  -> Further testing needed at higher K")

        final_gap = capacity_data[-1]['gap']
        first_gap = capacity_data[0]['gap']
        degradation = 1.0 - final_gap / max(first_gap, 0.001)
        print(f"\n  Gap degradation K=3 to K={max_k}: {degradation:.0%}")
        if degradation < 0.5:
            print(f"  -> Graceful: system retains >{100-int(degradation*100)}% "
                  f"of discrimination at max K")
        elif degradation < 0.8:
            print(f"  -> Declining but functional")
        else:
            print(f"  -> Near collapse at K={max_k}")

    print(f"\n{'-'*W}")


if __name__ == '__main__':
    run()
