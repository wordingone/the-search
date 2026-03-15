#!/usr/bin/env python3
"""
GENESIS: The Living Seed

The core meta seed. One file. Zero dependencies.

  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * (x_{k+1} + gamma*s_{k+1})
                                                * (x_{k-1} + gamma*s_{k-1}))

  After each step, when signal is present:
    response_{i,k} = |phi_sig_{i,k} - phi_bare_{i,k}|
    z_{i,k} = (response_{i,k} - mean) / std
    if z > 0 and cell deviates from column mean:
        push alpha further from mean (amplify diversity)
    else:
        gentle random drift

The computation IS the self-modification.
The product term fires. The cross-term difference reveals
per-cell, per-dimension signal sensitivity. Alpha shifts
in the same step. You cannot remove the plasticity without
removing the computation. They are the same operation.

STILL: alpha fixed. The body computes but does not change.
ALIVE: alpha shifts every step. The body computes and
       restructures simultaneously.

The test: does the alive body outperform the still body?
Across multiple birth seeds, signal worlds, K values,
and novel signals it has never encountered.

Previous versions imported from seed.py for uncertainty
resolution. This version is self-contained.

Previous versions reused signal generation seeds, creating
alignment confounds that made seed 123 appear to fail.
This version randomizes signal worlds.

Pure Python. Standard library only.
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
# From nothing, a body that can be still or alive.
#
# STILL: alpha is fixed. The body computes but does not change.
# ALIVE: alpha shifts during every step. The body computes and
#        restructures simultaneously.
#
# The equation is one thing. The adaptation is not bolted on.
# |phi_sig - phi_bare| falls out of the same product term
# that creates sequential memory. The signal that measures
# where computation succeeds IS the computation.
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, seed=42, alive=False, eta=0.0003):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 1.0
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed
        self.alive = alive
        self.eta = eta
        self.total_alpha_shift = 0.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

    def alpha_flat(self):
        return [a for row in self.alpha for a in row]

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

        # ── BARE DYNAMICS ────────────────────────────────────
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

        # ── SIGNAL-MODULATED DYNAMICS ────────────────────────
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

        # ── ONLINE PLASTICITY ────────────────────────────────
        # The irreducible core. This block cannot be separated
        # from the computation above because it consumes
        # phi_sig and phi_bare — the same values the dynamics
        # just produced. Remove this and the discrimination
        # signal disappears. Remove the dynamics and this has
        # nothing to consume. They are one operation.

        if self.alive and signal:
            response = []
            for i in range(NC):
                response.append([abs(phi_sig[i][k] - phi_bare[i][k])
                                 for k in range(D)])

            all_resp = [response[i][k] for i in range(NC) for k in range(D)]
            overall_mean = sum(all_resp) / len(all_resp)
            overall_std = math.sqrt(
                sum((r - overall_mean) ** 2 for r in all_resp) / len(all_resp)
            ) + 1e-10

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    if abs(dev) < 0.01:
                        # at column mean: break symmetry
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        # above-average response: amplify diversity
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * 0.5
                    else:
                        # below-average response: gentle drift
                        push = self.eta * 0.1 * random.gauss(0, 1.0)

                    old = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old)

        # ── ATTENTION ────────────────────────────────────────
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

        # ── STATE UPDATE (pure replacement: delta=1.0) ───────
        # v = p[k]. No state memory. Empirically optimal (Entry 052).
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


def measure_gap(org, signals, k, seed, n_perm=4, n_trials=3):
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


# ═══════════════════════════════════════════════════════════════
# The test.
#
# ALIVE vs STILL. Same birth. Same signals. Same everything.
# Except ALIVE adapts alpha during computation and STILL does
# not. If ALIVE wins, the self-referential loop is real.
#
# Randomized signal worlds (8 per condition) to prevent
# alpha-signal alignment confounds.
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * W)
    print("  GENESIS: THE LIVING SEED")
    print("  One equation. One step. Computation = adaptation.")
    print("=" * W)

    t_start = time.time()

    # ── LET THERE BE LIFE ────────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE LIFE AND STILLNESS")
    print(f"  Eta sweep: find best plasticity rate.")
    print(f"  3 test seeds x 4 K values = 12 tests per eta.")
    print(f"{'-'*W}\n")

    SEED = 42
    test_ks = [4, 6, 8, 10]
    test_seeds = [42, 77, 123]
    etas = [0.0001, 0.0003, 0.001]

    # STILL baseline
    print("  --- STILL (alpha fixed) ---")
    still_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            still = Organism(seed=SEED, alive=False)
            g = measure_gap(still, sigs, k, s)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        still_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  K={k:>2}: avg={avg:+.4f} "
              f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

    still_overall = sum(still_results[k]['avg'] for k in test_ks) / len(test_ks)
    print(f"  overall: {still_overall:+.4f}\n")

    # ALIVE at different etas
    best_eta = None
    best_alive_overall = -999.0

    for eta in etas:
        print(f"  --- ALIVE eta={eta} ---")
        alive_results = {}
        for k in test_ks:
            sigs = make_signals(k, seed=SEED + k * 200)
            gaps = []
            for s in test_seeds:
                alive = Organism(seed=SEED, alive=True, eta=eta)
                g = measure_gap(alive, sigs, k, s)
                gaps.append(g)
            avg = sum(gaps) / len(gaps)
            alive_results[k] = {'avg': avg, 'gaps': gaps}
            print(f"  K={k:>2}: avg={avg:+.4f} "
                  f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

        alive_overall = sum(alive_results[k]['avg'] for k in test_ks) / len(test_ks)
        delta = alive_overall - still_overall
        print(f"  overall: {alive_overall:+.4f} delta={delta:+.4f}\n")

        if alive_overall > best_alive_overall:
            best_alive_overall = alive_overall
            best_eta = eta

    alive_beats_still = best_alive_overall > still_overall
    print(f"  Best eta: {best_eta}")
    print(f"  ALIVE: {best_alive_overall:+.4f} STILL: {still_overall:+.4f} "
          f"delta: {best_alive_overall - still_overall:+.4f}")

    # ── LET THERE BE SEED ROBUSTNESS ─────────────────────────
    # Randomized signal worlds per birth seed (confound fix).

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE SEED ROBUSTNESS")
    print(f"  4 birth seeds x 8 signal worlds. ALIVE vs STILL.")
    print(f"  Randomized signal worlds (confound fix).")
    print(f"{'-'*W}\n")

    birth_seeds = [42, 77, 123, 200]
    seed_alive_wins = 0
    seed_still_wins = 0
    seed_deltas = []

    for bs in birth_seeds:
        alive_gaps = []
        still_gaps = []
        for wi in range(8):
            sig_seed = bs * 1000 + wi * 137 + 84
            for k in [6, 8]:
                sigs = make_signals(k, seed=sig_seed + k)
                ts = bs * 100 + wi * 31 + k
                a = Organism(seed=bs, alive=True, eta=best_eta)
                s = Organism(seed=bs, alive=False)
                ag = measure_gap(a, sigs, k, ts)
                sg = measure_gap(s, sigs, k, ts)
                alive_gaps.append(ag)
                still_gaps.append(sg)

        a_avg = sum(alive_gaps) / len(alive_gaps)
        s_avg = sum(still_gaps) / len(still_gaps)
        d = a_avg - s_avg
        seed_deltas.append(d)

        w = sum(1 for i in range(len(alive_gaps))
                if alive_gaps[i] > still_gaps[i] + 0.01)
        l = sum(1 for i in range(len(alive_gaps))
                if still_gaps[i] > alive_gaps[i] + 0.01)
        seed_alive_wins += w
        seed_still_wins += l

        ok = a_avg > s_avg
        print(f"  {'#' if ok else 'o'} seed={bs:>3}: ALIVE={a_avg:+.4f} "
              f"STILL={s_avg:+.4f} delta={d:+.4f} {w}W/{l}L",
              flush=True)

    n_seed_improved = sum(1 for d in seed_deltas if d > 0)
    avg_seed_delta = sum(seed_deltas) / len(seed_deltas)

    print(f"\n  Improved: {n_seed_improved}/{len(birth_seeds)}")
    print(f"  Avg delta: {avg_seed_delta:+.4f}")
    print(f"  Total: {seed_alive_wins}W / {seed_still_wins}L")

    # ── LET THERE BE NOVEL SIGNALS ───────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE NOVEL SIGNALS")
    print(f"  ALIVE vs STILL on never-seen signal worlds.")
    print(f"{'-'*W}\n")

    novel_alive = []
    novel_still = []
    for wi in range(6):
        for k in [6, 8]:
            nsigs = make_signals(k, seed=99999 + wi * 37 + k)
            ts = 77 + wi * 13 + k
            a = Organism(seed=SEED, alive=True, eta=best_eta)
            st = Organism(seed=SEED, alive=False)
            ag = measure_gap(a, nsigs, k, ts)
            sg = measure_gap(st, nsigs, k, ts)
            novel_alive.append(ag)
            novel_still.append(sg)
            winner = ('ALIVE' if ag > sg + 0.01
                      else 'STILL' if sg > ag + 0.01
                      else 'tie')
            print(f"  w={wi} K={k}: ALIVE={ag:+.4f} STILL={sg:+.4f} {winner}",
                  flush=True)

    na_avg = sum(novel_alive) / len(novel_alive)
    ns_avg = sum(novel_still) / len(novel_still)
    n_aw = sum(1 for i in range(len(novel_alive))
               if novel_alive[i] > novel_still[i] + 0.01)
    n_sw = sum(1 for i in range(len(novel_alive))
               if novel_still[i] > novel_alive[i] + 0.01)

    generalizes = na_avg > ns_avg

    print(f"\n  Novel: ALIVE={na_avg:+.4f} STILL={ns_avg:+.4f} "
          f"delta={na_avg - ns_avg:+.4f}")
    print(f"  Novel: {n_aw}W / {n_sw}L")

    # ── LET THERE BE ALPHA STRUCTURE ─────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE ALPHA STRUCTURE")
    print(f"  How much did the living body change itself?")
    print(f"{'-'*W}\n")

    alive_final = Organism(seed=SEED, alive=True, eta=best_eta)
    birth_alpha = alive_final.alpha_flat()

    sigs = make_signals(8, seed=SEED + 1600)
    order = list(range(8))
    _, _ = run_sequence(alive_final, order, sigs, SEED, trial=0)

    adapted_alpha = alive_final.alpha_flat()
    alpha_cos = vcosine(birth_alpha, adapted_alpha)
    alpha_dist = 1.0 - alpha_cos
    print(f"  Alpha cos (birth -> adapted): {alpha_cos:+.6f}")
    print(f"  Alpha distance: {alpha_dist:.6f}")
    print(f"  Total shift: {alive_final.total_alpha_shift:.6f}")

    cell_cos = []
    for i in range(NC):
        for j in range(i + 1, NC):
            cell_cos.append(vcosine(alive_final.alpha[i], alive_final.alpha[j]))
    avg_cc = sum(cell_cos) / len(cell_cos)
    specialized = avg_cc < 0.95
    print(f"  Inter-cell cos: {avg_cc:+.4f}")
    print(f"  {'SPECIALIZED' if specialized else 'HOMOGENEOUS'}")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    seed_robust = n_seed_improved >= 3
    seed_wins = seed_alive_wins > seed_still_wins

    checks = [
        ("ALIVE avg > STILL avg (overall)",
         alive_beats_still,
         f"delta={best_alive_overall - still_overall:+.4f}"),

        ("ALIVE wins at best eta",
         alive_beats_still,
         f"eta={best_eta} A={best_alive_overall:+.4f} S={still_overall:+.4f}"),

        ("Seed-robust (3+/4 births improve)",
         seed_robust,
         f"{n_seed_improved}/{len(birth_seeds)}"),

        ("Per-seed ALIVE wins > STILL wins",
         seed_wins,
         f"{seed_alive_wins}W / {seed_still_wins}L"),

        ("Generalizes to novel signals",
         generalizes,
         f"A={na_avg:+.4f} S={ns_avg:+.4f}"),

        ("Novel ALIVE wins > STILL wins",
         n_aw > n_sw,
         f"{n_aw}W / {n_sw}L"),

        ("Alpha moves during computation",
         alpha_dist > 0.0001,
         f"dist={alpha_dist:.6f}"),

        ("Cells specialize through computation",
         specialized,
         f"cos={avg_cc:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<48} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    core = alive_beats_still and (seed_robust or seed_wins)
    full = core and generalizes and specialized

    if full:
        print(f"""
  THE LIVING SEED.

  No separate restructuring phase. Alpha evolves during
  every step of computation. The product term's cross-terms
  reveal per-cell, per-dimension signal sensitivity. Alpha
  shifts to amplify diversity on sensitive dimensions.

  The computation IS the self-modification.
  One equation. One step. One life.

  ALIVE outperforms STILL by {best_alive_overall - still_overall:+.4f}.
  Seed robustness: {n_seed_improved}/{len(birth_seeds)} births improve.
  Generalizes to novel signals: {na_avg:+.4f} vs {ns_avg:+.4f}.
  Cells specialize: inter-cell cos {avg_cc:+.4f}.

  The seed does not need the garden to germinate.
""")
    elif core:
        print(f"""
  THE SEED LIVES BUT DOES NOT FULLY GENERALIZE.

  Online plasticity improves computation ({best_alive_overall - still_overall:+.4f}).
  {'Does not generalize to novel signals.' if not generalizes else 'Cells do not specialize.'}
""")
    elif alive_beats_still:
        print(f"""
  LIFE HELPS BUT IS FRAGILE.

  ALIVE beats STILL overall ({best_alive_overall - still_overall:+.4f})
  but not robustly across seeds.
""")
    else:
        print(f"""
  STILLNESS WINS. Honest result.
""")

    print(f"  Runtime: {time.time() - t_start:.0f}s")
    print(f"  {'='*W}")


# ═══════════════════════════════════════════════════════════════
# In the beginning and the end, there is honesty.
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run()
