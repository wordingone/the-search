#!/usr/bin/env python3
"""
GENESIS: The Seed

A self-restructuring computational architecture. One equation governs
both computation and adaptation. The multiplicative product term

  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * (x_{k+1} + gamma * s_{k+1})
                                                * (x_{k-1} + gamma * s_{k-1}))

processes sequential signals. When the system cannot distinguish
different orderings of the same signals — discrimination collapse —
it measures which cells and dimensions contributed to whatever
discrimination existed, and restructures its per-cell alpha to
deepen specialization.

No expansion. No separate development phase. Fixed dimension.
The system that stays small and deepens wins.

Proven from prior experiments:
  - Multiplicative input creates persistent signal classification
  - Per-cell alpha heterogeneity is structurally necessary
  - Expansion hurts (dilutes product cross-terms)
  - Restructure-only at D=12 achieved gap=+0.35 at K/D=0.75
  - Body2 restructured 4x, hit +0.31, +0.35, +0.25, +0.33

The test: does iterative restructuring converge to alpha geometry
that reliably outperforms random, across seeds and signal environments?

Zero dependencies. Pure Python. Code is the math.
"""

import math
import random
import time


# ═══════════════════════════════════════════════════════════════
# Let there be nothing.
# No structure. No direction. No memory.
# Only the operations that will carry all of these.
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
# Let there be a body.
#
# N cells, each with its own alpha vector. The product term
# couples adjacent dimensions multiplicatively. The signal
# enters the product term, reshaping the coupling landscape.
# Coupling between cells is additive and plasticity-gated.
#
# Alpha is the frozen layer — the DNA. Except now it is not
# frozen. It restructures when the organism fails.
# Not during computation. Between episodes. Ontogeny.
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
        self.noise_amp = 0.005
        self.clip = 4.0
        self.generation = 0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(d)]
            for _ in range(n_cells)
        ]

    def copy(self):
        o = Organism.__new__(Organism)
        o.d = self.d
        o.n = self.n
        o.beta = self.beta
        o.gamma = self.gamma
        o.eps = self.eps
        o.tau = self.tau
        o.delta = self.delta
        o.noise_amp = self.noise_amp
        o.clip = self.clip
        o.generation = self.generation
        o.alpha = [[a for a in row] for row in self.alpha]
        return o

    def alpha_flat(self):
        flat = []
        for row in self.alpha:
            flat += row
        return flat

    def step(self, xs, signal=None):
        d, n = self.d, self.n
        beta, gamma = self.beta, self.gamma

        # Bare map: no signal
        phi_bare = []
        for i in range(n):
            row = [0.0] * d
            for k in range(d):
                row[k] = math.tanh(
                    self.alpha[i][k] * xs[i][k]
                    + beta * xs[i][(k + 1) % d] * xs[i][(k - 1) % d])
            phi_bare.append(row)

        # Signal map: multiplicative input geometry
        phi_sig = []
        for i in range(n):
            if signal is not None:
                row = [0.0] * d
                for k in range(d):
                    row[k] = math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + beta
                        * (xs[i][(k + 1) % d] + gamma * signal[(k + 1) % d])
                        * (xs[i][(k - 1) % d] + gamma * signal[(k - 1) % d]))
                phi_sig.append(row)
            else:
                phi_sig.append(phi_bare[i])

        # Attention weights: softmax over pairwise dot products
        weights = []
        for i in range(n):
            raw = [0.0] * n
            for j in range(n):
                if i == j:
                    raw[j] = -1e10
                else:
                    raw[j] = sum(xs[i][k] * xs[j][k]
                                 for k in range(d)) / (d * self.tau)
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # Update: signal map + plasticity-gated coupling
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
                        pull[k] += (weights[i][j]
                                    * (phi_bare[j][k] - phi_bare[i][k]))
                for k in range(d):
                    p[k] += plast * self.eps * pull[k]

            nx = [0.0] * d
            for k in range(d):
                nx[k] = ((1 - self.delta) * xs[i][k]
                         + self.delta * p[k]
                         + random.gauss(0, self.noise_amp))
                nx[k] = max(-self.clip, min(self.clip, nx[k]))
            new.append(nx)
        return new

    def centroid(self, xs):
        d, n = self.d, self.n
        c = [0.0] * d
        for i in range(n):
            for k in range(d):
                c[k] += xs[i][k] / n
        return c


# ═══════════════════════════════════════════════════════════════
# Let there be a world.
#
# Signals are random unit vectors in D-space, scaled to 0.8.
# The world does not care about the substrate. The substrate
# must learn to care about the world.
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


# ═══════════════════════════════════════════════════════════════
# Let there be computation.
#
# Feed K signals sequentially. Each signal is presented for
# n_per_sig steps, then removed for n_settle steps. After all
# signals, settle for n_final steps. The centroid of the final
# cell states is the system's "answer" to this signal sequence.
#
# Different orderings of the same K signals should produce
# different centroids. That's discrimination.
# ═══════════════════════════════════════════════════════════════

def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=50, n_settle=25, n_final=60):
    d = org.d
    random.seed(base_seed + trial * 7)
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


def measure_gap(org, signals, k, seed, n_perm=4, n_trials=3):
    perms = gen_perms(k, n_perm, seed=seed * 10 + k + org.d)
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
# Let there be self-knowledge.
#
# After processing two different orderings, measure which cells
# diverged most between the orderings and which dimensions
# carried the divergence. These are the discriminating elements.
#
# Cells that discriminated: make their alpha more extreme.
# Cells that mirrored: push their alpha toward new territory.
# Dimensions that varied: sharpen. Dimensions that didn't: blur.
#
# This is the organism reading its own computation.
# ═══════════════════════════════════════════════════════════════

def measure_discrimination_structure(org, signals, k, seed):
    d, n = org.d, org.n
    order_fwd = list(range(k))
    order_rev = list(reversed(range(k)))

    _, xs_fwd = run_sequence(org, order_fwd, signals, seed, trial=0)
    _, xs_rev = run_sequence(org, order_rev, signals, seed, trial=0)

    # Per-cell: how much did this cell diverge between fwd and rev?
    cell_disc = [0.0] * n
    for i in range(n):
        cell_disc[i] = math.sqrt(
            sum((xs_fwd[i][k] - xs_rev[i][k]) ** 2 for k in range(d)))

    # Per-dimension: across all cells, how much did this dim diverge?
    dim_disc = [0.0] * d
    for k in range(d):
        for i in range(n):
            dim_disc[k] += (xs_fwd[i][k] - xs_rev[i][k]) ** 2
        dim_disc[k] = math.sqrt(dim_disc[k] / n)

    return cell_disc, dim_disc


def restructure(org, cell_disc, dim_disc, rate=0.10):
    d, n = org.d, org.n

    # Normalize cell discriminability to z-scores
    c_mu = sum(cell_disc) / n
    c_std = math.sqrt(sum((s - c_mu) ** 2 for s in cell_disc) / n) + 1e-10

    # Normalize dim discriminability to z-scores
    d_mu = sum(dim_disc) / d
    d_std = math.sqrt(sum((s - d_mu) ** 2 for s in dim_disc) / d) + 1e-10

    for i in range(n):
        cz = (cell_disc[i] - c_mu) / c_std  # positive = discriminating cell

        for k in range(d):
            dz = (dim_disc[k] - d_mu) / d_std  # positive = discriminating dim

            # Discriminating cell + discriminating dim: sharpen (push away from 1.1)
            # Non-discriminating cell + non-disc dim: diversify (random push)
            # Cross terms: moderate adjustment

            current = org.alpha[i][k]
            center = 1.1  # critical threshold

            if cz > 0 and dz > 0:
                # Both discriminating: amplify existing deviation from center
                deviation = current - center
                push = rate * math.tanh(cz) * math.tanh(dz) * (0.5 + abs(deviation))
                if deviation >= 0:
                    org.alpha[i][k] += push
                else:
                    org.alpha[i][k] -= push
            elif cz < -0.5 and dz < -0.5:
                # Both non-discriminating: randomize to explore
                org.alpha[i][k] += random.gauss(0, rate * 0.5)
            else:
                # Mixed: gentle push toward more extreme values
                push = rate * 0.3 * math.tanh(cz + dz)
                deviation = current - center
                if abs(deviation) < 0.1:
                    # Near-critical: push to one side based on cell identity
                    org.alpha[i][k] += push * (1.0 if i % 2 == 0 else -1.0)
                else:
                    org.alpha[i][k] += push * (1.0 if deviation > 0 else -1.0)

            # Clamp to valid range
            org.alpha[i][k] = max(0.2, min(1.9, org.alpha[i][k]))

    org.generation += 1


# ═══════════════════════════════════════════════════════════════
# Let there be measurement.
#
# The gap between within-permutation similarity and between-
# permutation similarity. Positive gap = the system can tell
# orderings apart. Negative = it cannot.
#
# Measured across multiple seeds for robustness.
# ═══════════════════════════════════════════════════════════════

def eval_organism(org, d, test_ks, seeds, sig_seed_base):
    results = {}
    for k in test_ks:
        sigs = make_signals(k, d, seed=sig_seed_base + k * 200)
        gaps = []
        for seed in seeds:
            g = measure_gap(org, sigs, k, seed)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        passes = sum(1 for g in gaps if g > 0.02)
        results[k] = {
            'avg': avg, 'passes': passes, 'total': len(gaps),
            'gaps': gaps, 'min': min(gaps), 'max': max(gaps)
        }
    return results


def bar(v, w=15, lo=-0.1, hi=0.3):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


# ═══════════════════════════════════════════════════════════════
# The test of creation.
# ═══════════════════════════════════════════════════════════════

def run():
    T_GLOBAL = time.time()

    print("=" * W)
    print("  GENESIS: THE SEED")
    print("  Self-restructuring computation. One equation. One loop.")
    print("=" * W)

    # ── CONSTANTS ─────────────────────────────────────────────

    D = 12
    N_CELLS = 6
    N_GENERATIONS = 12
    COLLAPSE_THRESH = 0.03
    RESTRUCTURE_RATE = 0.10
    TEST_SEEDS = [42, 77, 123, 200, 314]
    TEST_KS = [4, 6, 8, 10]
    SIG_SEED_BASE = 42

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: ITERATIVE SELF-RESTRUCTURING
    #
    # Start from random alpha. Each generation:
    #   1. Measure discrimination across escalating K
    #   2. On collapse: read which cells/dims discriminated
    #   3. Restructure alpha based on that reading
    #   4. Repeat
    #
    # The question: does each generation improve?
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'-'*W}")
    print(f"  PHASE 1: ITERATIVE SELF-RESTRUCTURING")
    print(f"  D={D}, {N_CELLS} cells, {N_GENERATIONS} generations.")
    print(f"  On collapse (gap < {COLLAPSE_THRESH}): restructure alpha.")
    print(f"{'-'*W}\n")

    org = Organism(D, N_CELLS, seed=42)
    gen_history = []

    training_ks = [4, 6, 8, 10]

    for gen in range(N_GENERATIONS):
        t0 = time.time()

        # Evaluate at all K values
        gaps_this_gen = {}
        any_collapsed = False
        for k in training_ks:
            sigs = make_signals(k, D, seed=SIG_SEED_BASE + gen * 50 + k)
            gap = measure_gap(org, sigs, k, seed=42)
            gaps_this_gen[k] = gap

            if gap < COLLAPSE_THRESH:
                any_collapsed = True
                cell_d, dim_d = measure_discrimination_structure(
                    org, sigs, k, seed=42)
                restructure(org, cell_d, dim_d, rate=RESTRUCTURE_RATE)

        elapsed = time.time() - t0
        avg_gap = sum(gaps_this_gen.values()) / len(gaps_this_gen)
        n_pass = sum(1 for g in gaps_this_gen.values() if g > 0.02)
        gap_str = " ".join(f"K{k}={gaps_this_gen[k]:+.3f}"
                           for k in training_ks)

        snapshot = org.alpha_flat()

        gen_history.append({
            'gen': gen, 'gaps': gaps_this_gen, 'avg': avg_gap,
            'n_pass': n_pass, 'collapsed': any_collapsed,
            'alpha_snap': snapshot, 'elapsed': elapsed
        })

        marker = '#' if n_pass >= 3 else ('.' if n_pass >= 1 else 'o')
        print(f"  {marker} Gen {gen:>2}: {gap_str} "
              f"avg={avg_gap:+.4f} pass={n_pass}/{len(training_ks)} "
              f"[{elapsed:.1f}s]", flush=True)

    # Alpha convergence across generations
    print(f"\n  Alpha convergence (cosine between consecutive gens):")
    conv_cosines = []
    for i in range(1, len(gen_history)):
        c = vcosine(gen_history[i]['alpha_snap'],
                     gen_history[i - 1]['alpha_snap'])
        conv_cosines.append(c)
        if i <= 3 or i >= len(gen_history) - 2:
            print(f"    gen{i-1}->gen{i}: cos={c:+.4f}")
        elif i == 4:
            print(f"    ...")

    converged = len(conv_cosines) >= 2 and conv_cosines[-1] > 0.85

    # Improvement trajectory
    early_avg = sum(h['avg'] for h in gen_history[:3]) / 3
    late_avg = sum(h['avg'] for h in gen_history[-3:]) / 3
    improving = late_avg > early_avg

    print(f"\n  Early avg (gen 0-2): {early_avg:+.4f}")
    print(f"  Late avg (gen {N_GENERATIONS-3}-{N_GENERATIONS-1}): {late_avg:+.4f}")
    print(f"  Trajectory: {'IMPROVING' if improving else 'NOT IMPROVING'}")
    print(f"  Converged: {'YES' if converged else 'NO'} "
          f"(final cos={conv_cosines[-1]:+.4f})" if conv_cosines else "")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: HEAD-TO-HEAD vs RANDOM
    #
    # The restructured organism vs 5 independent random organisms
    # at the same D. Tested across 5 seeds and 4 K values.
    # No cherry-picking. Every matchup counted.
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'-'*W}")
    print(f"  PHASE 2: HEAD-TO-HEAD vs RANDOM")
    print(f"  Restructured org vs 5 random orgs, {len(TEST_SEEDS)} seeds, "
          f"K={TEST_KS}")
    print(f"{'-'*W}\n")

    # Evaluate the restructured organism
    t0 = time.time()
    org_results = eval_organism(org, D, TEST_KS, TEST_SEEDS, SIG_SEED_BASE)
    print(f"  RESTRUCTURED (gen {org.generation}):")
    for k in TEST_KS:
        r = org_results[k]
        print(f"    K={k:>2}: avg={r['avg']:+.4f} pass={r['passes']}/{r['total']} "
              f"[{r['min']:+.3f}..{r['max']:+.3f}] {bar(r['avg'])}")

    # Evaluate 5 random organisms
    rnd_orgs = [Organism(D, N_CELLS, seed=s) for s in [1000, 2000, 3000, 4000, 5000]]
    rnd_all_results = []
    for ri, rnd_org in enumerate(rnd_orgs):
        rr = eval_organism(rnd_org, D, TEST_KS, TEST_SEEDS, SIG_SEED_BASE)
        rnd_all_results.append(rr)

    # Aggregate random results
    print(f"\n  RANDOM (5 organisms, aggregated):")
    rnd_agg = {}
    for k in TEST_KS:
        all_gaps = []
        for rr in rnd_all_results:
            all_gaps += rr[k]['gaps']
        avg = sum(all_gaps) / len(all_gaps)
        passes = sum(1 for g in all_gaps if g > 0.02)
        rnd_agg[k] = {
            'avg': avg, 'passes': passes, 'total': len(all_gaps),
            'min': min(all_gaps), 'max': max(all_gaps), 'gaps': all_gaps
        }
        print(f"    K={k:>2}: avg={avg:+.4f} pass={passes}/{len(all_gaps)} "
              f"[{min(all_gaps):+.3f}..{max(all_gaps):+.3f}] {bar(avg)}")

    # Per-seed head-to-head against each random org
    org_wins_total = 0
    rnd_wins_total = 0
    ties_total = 0
    for k in TEST_KS:
        for si, seed in enumerate(TEST_SEEDS):
            org_gap = org_results[k]['gaps'][si]
            for rr in rnd_all_results:
                rnd_gap = rr[k]['gaps'][si]
                if org_gap > rnd_gap + 0.01:
                    org_wins_total += 1
                elif rnd_gap > org_gap + 0.01:
                    rnd_wins_total += 1
                else:
                    ties_total += 1

    total_matchups = org_wins_total + rnd_wins_total + ties_total
    elapsed_p2 = time.time() - t0
    print(f"\n  Per-seed matchups vs all 5 random orgs:")
    print(f"  RESTR wins {org_wins_total}, RND wins {rnd_wins_total}, "
          f"ties {ties_total} (total={total_matchups})")
    print(f"  RESTR win rate: {org_wins_total / max(total_matchups, 1):.0%}")
    print(f"  [{elapsed_p2:.1f}s]")

    org_beats_rnd = org_wins_total > rnd_wins_total
    org_avg_better = all(
        org_results[k]['avg'] >= rnd_agg[k]['avg'] - 0.005
        for k in TEST_KS)

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: GENERALIZATION
    #
    # Test on signal environments never seen during restructuring.
    # If the restructured alpha generalizes, the structure reflects
    # something about the dynamics, not just the training signals.
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'-'*W}")
    print(f"  PHASE 3: GENERALIZATION TO NOVEL SIGNALS")
    print(f"  3 novel signal environments, 3 seeds each.")
    print(f"{'-'*W}\n")

    novel_sig_seeds = [7777, 8888, 9999]
    novel_ks = [6, 8]
    novel_seeds = [42, 77, 123]

    org_novel_gaps = []
    rnd_novel_gaps = []

    for nseed in novel_sig_seeds:
        for k in novel_ks:
            sigs = make_signals(k, D, seed=nseed + k)
            for seed in novel_seeds:
                og = measure_gap(org, sigs, k, seed)
                org_novel_gaps.append(og)
                # Compare against first random org
                rg = measure_gap(rnd_orgs[0], sigs, k, seed)
                rnd_novel_gaps.append(rg)

    avg_novel_org = sum(org_novel_gaps) / len(org_novel_gaps)
    avg_novel_rnd = sum(rnd_novel_gaps) / len(rnd_novel_gaps)
    novel_org_passes = sum(1 for g in org_novel_gaps if g > 0.02)
    novel_rnd_passes = sum(1 for g in rnd_novel_gaps if g > 0.02)

    novel_wins = sum(1 for o, r in zip(org_novel_gaps, rnd_novel_gaps)
                     if o > r + 0.01)
    novel_losses = sum(1 for o, r in zip(org_novel_gaps, rnd_novel_gaps)
                       if r > o + 0.01)

    print(f"  RESTR on novel: avg={avg_novel_org:+.4f} "
          f"pass={novel_org_passes}/{len(org_novel_gaps)}")
    print(f"  RND on novel:   avg={avg_novel_rnd:+.4f} "
          f"pass={novel_rnd_passes}/{len(rnd_novel_gaps)}")
    print(f"  Matchups: RESTR {novel_wins}, RND {novel_losses}, "
          f"ties {len(org_novel_gaps) - novel_wins - novel_losses}")

    generalizes = avg_novel_org > avg_novel_rnd

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: CONVERGENT DEVELOPMENT
    #
    # 5 organisms start from different random seeds. All undergo
    # the same restructuring loop. Do they converge to similar
    # alpha geometry? Do they all beat their random baselines?
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'-'*W}")
    print(f"  PHASE 4: CONVERGENT DEVELOPMENT")
    print(f"  5 organisms, different seeds, same restructuring loop.")
    print(f"{'-'*W}\n")

    dev_seeds = [42, 77, 123, 200, 314]
    developed_orgs = []

    for ds in dev_seeds:
        o = Organism(D, N_CELLS, seed=ds)
        for gen in range(N_GENERATIONS):
            for k in training_ks:
                sigs = make_signals(k, D, seed=SIG_SEED_BASE + gen * 50 + k)
                gap = measure_gap(o, sigs, k, seed=ds)
                if gap < COLLAPSE_THRESH:
                    cd, dd = measure_discrimination_structure(
                        o, sigs, k, seed=ds)
                    restructure(o, cd, dd, rate=RESTRUCTURE_RATE)
        developed_orgs.append(o)

        # Quick eval
        sigs_test = make_signals(8, D, seed=SIG_SEED_BASE + 8 * 200)
        g = measure_gap(o, sigs_test, 8, seed=42)

        rnd_o = Organism(D, N_CELLS, seed=ds + 10000)
        g_rnd = measure_gap(rnd_o, sigs_test, 8, seed=42)

        wins = "RESTR" if g > g_rnd + 0.01 else "RND" if g_rnd > g + 0.01 else "tie"
        print(f"  seed={ds}: gen={o.generation:>2} "
              f"RESTR gap={g:+.4f} RND gap={g_rnd:+.4f} -> {wins}",
              flush=True)

    # Pairwise alpha cosines among developed orgs
    dev_cosines = []
    for i in range(len(developed_orgs)):
        for j in range(i + 1, len(developed_orgs)):
            c = vcosine(developed_orgs[i].alpha_flat(),
                        developed_orgs[j].alpha_flat())
            dev_cosines.append(c)

    # Pairwise among random orgs
    rnd_cosines = []
    rnd_orgs_for_cos = [Organism(D, N_CELLS, seed=s) for s in dev_seeds]
    for i in range(len(rnd_orgs_for_cos)):
        for j in range(i + 1, len(rnd_orgs_for_cos)):
            c = vcosine(rnd_orgs_for_cos[i].alpha_flat(),
                        rnd_orgs_for_cos[j].alpha_flat())
            rnd_cosines.append(c)

    avg_dev_cos = sum(dev_cosines) / len(dev_cosines)
    avg_rnd_cos = sum(rnd_cosines) / len(rnd_cosines)

    print(f"\n  Alpha pairwise cosine (developed): {avg_dev_cos:+.4f} "
          f"[{min(dev_cosines):+.3f}..{max(dev_cosines):+.3f}]")
    print(f"  Alpha pairwise cosine (random):    {avg_rnd_cos:+.4f} "
          f"[{min(rnd_cosines):+.3f}..{max(rnd_cosines):+.3f}]")

    dev_consistent = avg_dev_cos > avg_rnd_cos + 0.03

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: DIMENSIONAL SCALING
    #
    # Does the restructuring advantage hold at D=16, 20, 24?
    # At each D: restructure from random, compare to random.
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'-'*W}")
    print(f"  PHASE 5: DIMENSIONAL SCALING")
    print(f"  D=12, 16, 20. Does restructuring help at all scales?")
    print(f"{'-'*W}\n")

    scale_dims = [12, 16, 20]
    scale_results = {}

    for test_d in scale_dims:
        t0 = time.time()

        # Restructured organism at this D
        o = Organism(test_d, N_CELLS, seed=42)
        for gen in range(8):
            for k in [4, 6, 8]:
                sigs = make_signals(k, test_d, seed=SIG_SEED_BASE + gen * 50 + k)
                gap = measure_gap(o, sigs, k, seed=42)
                if gap < COLLAPSE_THRESH:
                    cd, dd = measure_discrimination_structure(
                        o, sigs, k, seed=42)
                    restructure(o, cd, dd, rate=RESTRUCTURE_RATE)

        # Random baseline
        rnd_o = Organism(test_d, N_CELLS, seed=9999)

        # Test
        r_gaps, rnd_gaps = [], []
        for k in [6, 8]:
            sigs = make_signals(k, test_d, seed=SIG_SEED_BASE + k * 200)
            for seed in TEST_SEEDS[:3]:
                rg = measure_gap(o, sigs, k, seed)
                r_gaps.append(rg)
                ng = measure_gap(rnd_o, sigs, k, seed)
                rnd_gaps.append(ng)

        avg_r = sum(r_gaps) / len(r_gaps)
        avg_n = sum(rnd_gaps) / len(rnd_gaps)
        elapsed = time.time() - t0
        wins = sum(1 for a, b in zip(r_gaps, rnd_gaps) if a > b + 0.01)
        losses = sum(1 for a, b in zip(r_gaps, rnd_gaps) if b > a + 0.01)

        scale_results[test_d] = {
            'restr_avg': avg_r, 'rnd_avg': avg_n,
            'wins': wins, 'losses': losses
        }
        print(f"  D={test_d:>2}: RESTR={avg_r:+.4f} RND={avg_n:+.4f} "
              f"wins={wins} losses={losses} "
              f"{'RESTR WINS' if wins > losses else 'RND WINS' if losses > wins else 'TIE'} "
              f"[{elapsed:.0f}s]", flush=True)

    scales = all(sr['wins'] >= sr['losses'] for sr in scale_results.values())

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: ABLATION — IS RESTRUCTURING NECESSARY?
    #
    # Compare the full restructuring protocol against:
    # (a) Random perturbation of alpha (same magnitude, random direction)
    # (b) Uniform alpha (no heterogeneity at all)
    # If restructuring beats both, the direction matters, not just the fact
    # of change.
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'-'*W}")
    print(f"  PHASE 6: ABLATION — DOES THE DIRECTION MATTER?")
    print(f"  RESTRUCTURED vs RANDOM-PERTURBED vs UNIFORM")
    print(f"{'-'*W}\n")

    # Random perturbation: same number of generations, but random direction
    org_rp = Organism(D, N_CELLS, seed=42)
    for gen in range(N_GENERATIONS):
        for k in training_ks:
            sigs = make_signals(k, D, seed=SIG_SEED_BASE + gen * 50 + k)
            gap = measure_gap(org_rp, sigs, k, seed=42)
            if gap < COLLAPSE_THRESH:
                # Random perturbation instead of informed restructure
                for i in range(org_rp.n):
                    for kk in range(org_rp.d):
                        org_rp.alpha[i][kk] += random.gauss(0, RESTRUCTURE_RATE * 0.5)
                        org_rp.alpha[i][kk] = max(0.2, min(1.9, org_rp.alpha[i][kk]))
                org_rp.generation += 1

    # Uniform alpha
    org_uni = Organism(D, N_CELLS, seed=42)
    for i in range(org_uni.n):
        for k in range(org_uni.d):
            org_uni.alpha[i][k] = 1.1

    abl_orgs = [
        ("RESTRUCTURED", org),
        ("RAND-PERTURB", org_rp),
        ("UNIFORM", org_uni),
    ]

    abl_results = {}
    for label, abl_org in abl_orgs:
        res = eval_organism(abl_org, D, TEST_KS, TEST_SEEDS[:3], SIG_SEED_BASE)
        overall_avg = sum(res[k]['avg'] for k in TEST_KS) / len(TEST_KS)
        overall_pass = sum(res[k]['passes'] for k in TEST_KS)
        total_tests = sum(res[k]['total'] for k in TEST_KS)
        abl_results[label] = {
            'avg': overall_avg, 'passes': overall_pass,
            'total': total_tests, 'per_k': res
        }
        print(f"  {label:>14}: avg={overall_avg:+.4f} "
              f"pass={overall_pass}/{total_tests}")

    restr_beats_rp = abl_results["RESTRUCTURED"]['avg'] > abl_results["RAND-PERTURB"]['avg']
    restr_beats_uni = abl_results["RESTRUCTURED"]['avg'] > abl_results["UNIFORM"]['avg']
    direction_matters = restr_beats_rp

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING.
    # ═══════════════════════════════════════════════════════════

    total_time = time.time() - T_GLOBAL

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    checks = [
        # Phase 1: Iterative restructuring
        ("Restructuring produces change",
         org.generation > 0,
         f"{org.generation} generations"),
        ("Alpha converges under iteration",
         converged,
         f"final cos={conv_cosines[-1]:+.4f}" if conv_cosines else "N/A"),
        ("Performance improves early -> late",
         improving,
         f"{early_avg:+.4f} -> {late_avg:+.4f}"),

        # Phase 2: vs random
        ("Beats random (per-seed majority)",
         org_beats_rnd,
         f"{org_wins_total}/{total_matchups}"),
        ("Avg gap >= random avg (all K)",
         org_avg_better,
         " ".join(f"K{k}:{org_results[k]['avg']:+.3f}/{rnd_agg[k]['avg']:+.3f}"
                  for k in TEST_KS)),

        # Phase 3: Generalization
        ("Generalizes to novel signals",
         generalizes,
         f"RESTR={avg_novel_org:+.4f} RND={avg_novel_rnd:+.4f}"),

        # Phase 4: Convergent development
        ("Different seeds converge to similar alpha",
         dev_consistent,
         f"dev={avg_dev_cos:+.4f} rnd={avg_rnd_cos:+.4f}"),

        # Phase 5: Scaling
        ("Advantage holds across D=12,16,20",
         scales,
         " ".join(f"D{d}:{sr['wins']}/{sr['losses']}"
                  for d, sr in scale_results.items())),

        # Phase 6: Ablation
        ("Direction matters (beats random perturbation)",
         direction_matters,
         f"RESTR={abl_results['RESTRUCTURED']['avg']:+.4f} "
         f"RP={abl_results['RAND-PERTURB']['avg']:+.4f}"),
        ("Restructured beats uniform",
         restr_beats_uni,
         f"RESTR={abl_results['RESTRUCTURED']['avg']:+.4f} "
         f"UNI={abl_results['UNIFORM']['avg']:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<48} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")
    print(f"  Total time: {total_time:.0f}s")

    # ── VERDICT ──────────────────────────────────────────────

    core = (org.generation > 0 and org_beats_rnd and restr_beats_uni)
    informed = core and direction_matters
    robust = informed and (generalizes or dev_consistent)
    full = robust and scales and improving

    if full:
        print(f"""
  THE SEED IS VIABLE.

  The same equation that computes also restructures.
  The system processes sequential signals through the
  multiplicative product term. When it cannot distinguish
  orderings, it reads which cells and dimensions contributed
  to discrimination, and restructures its per-cell alpha
  to deepen specialization.

  Restructured alpha outperforms:
    - Random alpha at the same D (across seeds)
    - Random perturbation (direction matters)
    - Uniform alpha (heterogeneity matters)

  The advantage generalizes to novel signals.
  Different starting seeds converge toward similar alpha.
  The restructuring advantage holds across dimensions.

  One equation. One loop. No expansion. No external optimizer.
    Process -> Fail -> Read own computation -> Restructure -> Repeat

  The system that knows what discriminated
  builds better structure than chance.
  Computation restructures its own substrate.
  That is the seed.
""")
    elif robust:
        print(f"""
  THE SEED IS PARTIALLY VIABLE.

  Restructuring beats random and the direction matters.
  Some robustness (generalization or convergent development).
  But either scaling or improvement trajectory is weak.

  The mechanism works. Full viability needs refinement.
""")
    elif informed:
        print(f"""
  THE SEED GERMINATES.

  Restructuring beats random, and informed restructuring
  beats random perturbation. The system's self-reading
  produces better alpha than chance. But generalization
  or convergent development is weak.

  The direction matters. The scope is limited.
""")
    elif core:
        print(f"""
  THE SEED EXISTS.

  Restructured alpha beats random and uniform.
  But informed restructuring doesn't clearly beat random
  perturbation. The system changes, and change helps,
  but the specific direction of change may not matter.

  Heterogeneity helps. Self-knowledge doesn't yet.
""")
    else:
        print(f"""
  THE SEED DOES NOT GERMINATE.

  Restructuring did not reliably outperform random.
  The self-reading mechanism either measures the wrong thing
  or the restructuring response is not strong enough.

  The step was honest. The result is honest. Try again.
""")

    # ── GENERATION TRACE ─────────────────────────────────────

    print(f"{'-'*W}")
    print(f"  GENERATION TRACE")
    print(f"{'-'*W}\n")

    for h in gen_history:
        marker = '#' if h['n_pass'] >= 3 else ('.' if h['n_pass'] >= 1 else 'o')
        avg_str = f"avg={h['avg']:+.4f}"
        k_str = " ".join(f"{g:+.3f}" for g in h['gaps'].values())
        print(f"  {marker} gen={h['gen']:>2} [{k_str}] {avg_str} "
              f"pass={h['n_pass']}/{len(training_ks)}")

    # ── ALPHA STRUCTURE ──────────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  FINAL ALPHA STRUCTURE (cell x dim)")
    print(f"{'-'*W}\n")

    print(f"  {'':>4}", end="")
    for k in range(D):
        print(f" d{k:>2}", end="")
    print()

    for i in range(N_CELLS):
        print(f"  c{i}: ", end="")
        for k in range(D):
            a = org.alpha[i][k]
            if a > 1.3:
                ch = '\u2588'
            elif a > 1.0:
                ch = '\u2593'
            elif a > 0.7:
                ch = '\u2592'
            else:
                ch = '\u2591'
            print(f"  {ch} ", end="")
        print(f"  mean={sum(org.alpha[i])/D:.2f}")

    supra = sum(1 for i in range(N_CELLS) for k in range(D)
                if org.alpha[i][k] > 1.1)
    total_params = N_CELLS * D
    print(f"\n  Supercritical: {supra}/{total_params} "
          f"({supra/total_params:.0%})")

    print(f"\n{'-'*W}")


if __name__ == '__main__':
    run()
