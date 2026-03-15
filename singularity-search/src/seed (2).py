#!/usr/bin/env python3
"""
GENESIS: The Seed

The system processes sequential signals through a product-term
map with per-cell excitability (alpha). When it fails to
distinguish signal orderings, it runs multiple orderings and
measures the DIFFERENCE in endpoints — per dimension, per cell.

Dimensions where endpoints differ: discrimination is happening.
  -> Amplify alpha diversity across cells on these dimensions.
Cells whose states differ between orderings: order-sensitive.
  -> Strengthen their discriminating dimensions.
Dimensions where endpoints agree: no discrimination.
  -> Perturb alpha randomly to break dead symmetry.
Cells that are order-blind:
  -> Perturb to create new sensitivity.

The restructuring signal IS the discrimination. Not a proxy
measurement of state geometry. Not cell divergence from a
centroid. Not dimension variance across cells. The actual
per-dimension, per-cell contribution to telling orderings apart.

  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * (x_{k+1} + gamma * s_{k+1})
                                                * (x_{k-1} + gamma * s_{k-1}))

Fixed D=12. Six cells. One equation. One loop.

The proof:
  1. Discrimination-derived restructuring outperforms random alpha
  2. The advantage holds across birth seeds
  3. The advantage generalizes to novel signals
  4. Alpha converges under iteration
  5. Cells specialize

Zero dependencies. Pure Python.
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
    v = dot / (na * nb)
    return max(-1.0, min(1.0, v))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


def bar(v, w=15, lo=-0.15, hi=0.35):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


# ═══════════════════════════════════════════════════════════════
# From nothing, a body with six cells.
# Each cell has its own alpha. The dimension is fixed forever.
# Only the excitability landscape changes.
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, seed=42):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.n_restructures = 0
        self.seed = seed

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

    def alpha_flat(self):
        return [a for row in self.alpha for a in row]

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

        # bare dynamics: no signal
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

        # signal-modulated dynamics
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

        # attention weights: softmax of pairwise dot products
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

        # update: phi + plasticity-gated neighbor pull
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


    # ── The body learns to see its own discrimination ────────
    # Not state geometry. The actual per-dimension, per-cell
    # contribution to telling orderings apart.

    def compute_discrimination(self, centroids_list, states_list):
        n_runs = len(centroids_list)

        # per-dimension: average |centroid_diff| across all pairs
        dim_disc = [0.0] * D
        n_pairs = 0
        for a in range(n_runs):
            for b in range(a + 1, n_runs):
                for k in range(D):
                    dim_disc[k] += abs(centroids_list[a][k] - centroids_list[b][k])
                n_pairs += 1
        if n_pairs > 0:
            dim_disc = [d / n_pairs for d in dim_disc]

        # per-cell: average |cell state diff| across all pairs
        cell_disc = [0.0] * NC
        for a in range(n_runs):
            for b in range(a + 1, n_runs):
                for i in range(NC):
                    diff = 0.0
                    for k in range(D):
                        diff += abs(states_list[a][i][k] - states_list[b][i][k])
                    cell_disc[i] += diff / D
        if n_pairs > 0:
            cell_disc = [c / n_pairs for c in cell_disc]

        return dim_disc, cell_disc


    # ── The body restructures from what it sees ──────────────
    # Discriminating dimensions get amplified diversity.
    # Discriminating cells get their disc dims sharpened.
    # Everything else gets perturbed to break symmetry.

    def restructure(self, dim_disc, cell_disc, rate=0.04):
        # z-score normalize dimension discrimination
        dd_mu = sum(dim_disc) / D
        dd_var = sum((d - dd_mu) ** 2 for d in dim_disc) / D
        dd_std = math.sqrt(dd_var) + 1e-10

        # z-score normalize cell discrimination
        cd_mu = sum(cell_disc) / NC
        cd_var = sum((c - cd_mu) ** 2 for c in cell_disc) / NC
        cd_std = math.sqrt(cd_var) + 1e-10

        for i in range(NC):
            cell_z = (cell_disc[i] - cd_mu) / cd_std

            for k in range(D):
                dim_z = (dim_disc[k] - dd_mu) / dd_std

                if dim_z > 0:
                    # discriminating dimension
                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean
                    if abs(dev) < 0.02:
                        dev = random.gauss(0, 0.1)
                    direction = 1.0 if dev > 0 else -1.0

                    if cell_z > 0:
                        # disc cell x disc dim: amplify diversity
                        push = rate * math.tanh(dim_z) * math.tanh(cell_z) * direction * 0.5
                    else:
                        # non-disc cell x disc dim: perturb
                        push = rate * math.tanh(dim_z) * 0.2 * random.gauss(0, 1.0)
                else:
                    # non-discriminating dim: gentle perturbation
                    push = rate * 0.12 * random.gauss(0, 1.0)

                self.alpha[i][k] += push
                self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))

        self.n_restructures += 1


# ═══════════════════════════════════════════════════════════════
# The world speaks. Signals at full dimension, always.
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


# ═══════════════════════════════════════════════════════════════
# The measure of truth.
# Within-permutation consistency minus between-permutation
# similarity. Positive = the system distinguishes orderings.
# ═══════════════════════════════════════════════════════════════

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
# The development loop.
# Curriculum: K escalates. Every epoch, the system runs
# multiple orderings, extracts the discrimination signal,
# and restructures alpha. One loop. No separate phases.
# ═══════════════════════════════════════════════════════════════

def develop(org, n_epochs=12, base_seed=42, verbose=True):
    alpha_snapshots = [org.alpha_flat()]
    epoch_log = []

    for epoch in range(n_epochs):
        k = 3 + epoch
        signals = make_signals(k, seed=base_seed + epoch * 100)
        perms = gen_perms(k, 4, seed=base_seed * 7 + epoch)

        t0 = time.time()

        # run 4 permutations, collect centroids and cell states
        centroids = []
        states = []
        for pi, perm in enumerate(perms):
            c, xs = run_sequence(org, perm, signals, base_seed, trial=pi)
            centroids.append(c)
            states.append(xs)

        # discrimination signal from these runs
        dim_disc, cell_disc = org.compute_discrimination(centroids, states)

        # quick gap: 1 - avg between-perm cosine
        bcos = []
        for a in range(len(centroids)):
            for b in range(a + 1, len(centroids)):
                bcos.append(vcosine(centroids[a], centroids[b]))
        quick_gap = 1.0 - (sum(bcos) / max(len(bcos), 1))

        # restructure
        org.restructure(dim_disc, cell_disc, rate=0.04)

        alpha_snapshots.append(org.alpha_flat())
        elapsed = time.time() - t0

        disc_strength = sum(dim_disc) / D

        epoch_log.append({
            'epoch': epoch, 'k': k, 'quick_gap': quick_gap,
            'disc_strength': disc_strength, 'elapsed': elapsed
        })

        if verbose:
            print(f"  {'#' if quick_gap > 0.05 else 'o'} "
                  f"Epoch {epoch:>2}: K={k:>2} "
                  f"qgap={quick_gap:+.4f} "
                  f"disc={disc_strength:.4f} "
                  f"[{elapsed:.1f}s]  "
                  f"R#{org.n_restructures}",
                  flush=True)

    return alpha_snapshots, epoch_log


# ═══════════════════════════════════════════════════════════════
# The test of creation.
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * W)
    print("  GENESIS: THE SEED")
    print("  Discrimination-derived self-restructuring.")
    print("  Fixed D=12. Six cells. The signal IS the computation.")
    print("=" * W)

    SEED = 42

    # ── LET THERE BE A BODY ──────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE A BODY")
    print(f"  D={D}, {NC} cells, {NC*D} alpha parameters.")
    print(f"{'-'*W}\n")

    org = Organism(seed=SEED)

    # ── LET THERE BE GROWTH ──────────────────────────────────

    print(f"{'-'*W}")
    print(f"  LET THERE BE GROWTH")
    print(f"  K=3..14. Every epoch: discriminate, measure, restructure.")
    print(f"{'-'*W}\n")

    t_start = time.time()
    alpha_snapshots, epoch_log = develop(org, n_epochs=12, base_seed=SEED)
    t_dev = time.time() - t_start

    print(f"\n  Development: {org.n_restructures} restructures in {t_dev:.0f}s")

    # ── LET THERE BE CONVERGENCE ─────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE CONVERGENCE")
    print(f"{'-'*W}\n")

    conv_cos = []
    for i in range(1, len(alpha_snapshots)):
        c = vcosine(alpha_snapshots[i - 1], alpha_snapshots[i])
        conv_cos.append(c)
        if i <= 4 or i >= len(alpha_snapshots) - 2:
            print(f"  step {i-1:>2}->{i:>2}: cos={c:+.6f}")
        elif i == 5:
            print(f"  ...")

    first_cos = conv_cos[0] if conv_cos else 0
    last_cos = conv_cos[-1] if conv_cos else 0
    birth_to_final = vcosine(alpha_snapshots[0], alpha_snapshots[-1])
    converging = last_cos > first_cos
    converged = last_cos > 0.995

    print(f"\n  Birth -> final cos: {birth_to_final:+.6f}")
    print(f"  First step: {first_cos:+.6f}, Last step: {last_cos:+.6f}")
    print(f"  -> {'CONVERGED' if converged else 'CONVERGING' if converging else 'DIVERGING'}")

    # ── LET THERE BE COMPARISON ──────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE COMPARISON")
    print(f"  Restructured vs 3 random organisms. 3 seeds. K=4,6,8,10.")
    print(f"{'-'*W}\n")

    test_seeds = [42, 77, 123]
    test_ks = [4, 6, 8, 10]
    rnd_orgs = [Organism(seed=SEED + 2000 + i * 111) for i in range(3)]

    restr_results = {}
    for k in test_ks:
        signals = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for seed in test_seeds:
            g = measure_gap(org, signals, k, seed)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        passes = sum(1 for g in gaps if g > 0.02)
        restr_results[k] = {'avg': avg, 'passes': passes, 'gaps': gaps}
        print(f"  DISC  K={k:>2}: avg={avg:+.4f} pass={passes}/3 "
              f"{bar(avg)} [{min(gaps):+.3f}..{max(gaps):+.3f}]",
              flush=True)

    rnd_results = {}
    for k in test_ks:
        signals = make_signals(k, seed=SEED + k * 200)
        all_gaps = []
        for ro in rnd_orgs:
            for seed in test_seeds:
                g = measure_gap(ro, signals, k, seed)
                all_gaps.append(g)
        avg = sum(all_gaps) / len(all_gaps)
        passes = sum(1 for g in all_gaps if g > 0.02)
        rnd_results[k] = {'avg': avg, 'passes': passes, 'gaps': all_gaps}
        print(f"  RND   K={k:>2}: avg={avg:+.4f} pass={passes}/9 "
              f"{bar(avg)} [{min(all_gaps):+.3f}..{max(all_gaps):+.3f}]",
              flush=True)
    print()

    disc_overall = sum(restr_results[k]['avg'] for k in test_ks) / len(test_ks)
    rnd_overall = sum(rnd_results[k]['avg'] for k in test_ks) / len(test_ks)
    delta = disc_overall - rnd_overall

    print(f"  DISC overall: {disc_overall:+.4f}")
    print(f"  RND  overall: {rnd_overall:+.4f}")
    print(f"  Delta: {delta:+.4f} {'(DISC WINS)' if delta > 0 else '(RND WINS)'}")

    # per-seed head-to-head
    disc_wins = 0
    rnd_wins = 0
    ties = 0
    for k in test_ks:
        for si, seed in enumerate(test_seeds):
            dg = restr_results[k]['gaps'][si]
            rg = sum(rnd_results[k]['gaps'][ri * 3 + si] for ri in range(3)) / 3
            if dg > rg + 0.01:
                disc_wins += 1
            elif rg > dg + 0.01:
                rnd_wins += 1
            else:
                ties += 1

    print(f"  Head-to-head: DISC {disc_wins}W / RND {rnd_wins}W / {ties}T")

    # ── LET THERE BE SEED ROBUSTNESS ─────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE SEED ROBUSTNESS")
    print(f"  4 birth seeds. Same development. Each vs own baseline.")
    print(f"{'-'*W}\n")

    birth_seeds = [42, 77, 123, 200]
    seed_results = []

    for bs in birth_seeds:
        grown = Organism(seed=bs)
        develop(grown, n_epochs=12, base_seed=bs, verbose=False)

        baseline = Organism(seed=bs)

        sigs_6 = make_signals(6, seed=SEED + 1200)
        sigs_8 = make_signals(8, seed=SEED + 1600)

        grown_gaps = []
        base_gaps = []
        for sigs, k in [(sigs_6, 6), (sigs_8, 8)]:
            for ts in [42, 77]:
                gg = measure_gap(grown, sigs, k, ts)
                bg = measure_gap(baseline, sigs, k, ts)
                grown_gaps.append(gg)
                base_gaps.append(bg)

        g_avg = sum(grown_gaps) / len(grown_gaps)
        b_avg = sum(base_gaps) / len(base_gaps)
        wins = sum(1 for i in range(len(grown_gaps))
                   if grown_gaps[i] > base_gaps[i] + 0.01)
        losses = sum(1 for i in range(len(grown_gaps))
                     if base_gaps[i] > grown_gaps[i] + 0.01)

        seed_results.append({
            'seed': bs, 'grown': g_avg, 'birth': b_avg,
            'delta': g_avg - b_avg, 'wins': wins, 'losses': losses,
            'n_restr': grown.n_restructures
        })

        ok = g_avg > b_avg
        print(f"  seed={bs:>3}: grown={g_avg:+.4f} birth={b_avg:+.4f} "
              f"delta={g_avg - b_avg:+.4f} restr={grown.n_restructures:>2} "
              f"{'#' if ok else 'o'} {wins}W/{losses}L",
              flush=True)

    n_improved = sum(1 for r in seed_results if r['delta'] > 0)
    avg_seed_delta = sum(r['delta'] for r in seed_results) / len(seed_results)
    total_seed_wins = sum(r['wins'] for r in seed_results)
    total_seed_losses = sum(r['losses'] for r in seed_results)

    print(f"\n  Improved: {n_improved}/{len(birth_seeds)}")
    print(f"  Avg delta: {avg_seed_delta:+.4f}")
    print(f"  Total: {total_seed_wins}W / {total_seed_losses}L")

    # ── LET THERE BE NOVEL SIGNALS ───────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE NOVEL SIGNALS")
    print(f"  Signals never seen during development.")
    print(f"{'-'*W}\n")

    novel_ks = [6, 8]
    novel_disc = []
    novel_rnd = []

    for k in novel_ks:
        novel_sigs = make_signals(k, seed=7777 + k)
        for seed in [42, 77, 123]:
            gd = measure_gap(org, novel_sigs, k, seed)
            gr = measure_gap(rnd_orgs[0], novel_sigs, k, seed)
            novel_disc.append(gd)
            novel_rnd.append(gr)
            winner = ('DISC' if gd > gr + 0.01
                      else 'RND' if gr > gd + 0.01
                      else 'tie')
            print(f"  K={k} seed={seed}: DISC={gd:+.4f} RND={gr:+.4f} {winner}",
                  flush=True)

    avg_nd = sum(novel_disc) / len(novel_disc)
    avg_nr = sum(novel_rnd) / len(novel_rnd)
    nw = sum(1 for i in range(len(novel_disc))
             if novel_disc[i] > novel_rnd[i] + 0.01)
    nl = sum(1 for i in range(len(novel_disc))
             if novel_rnd[i] > novel_disc[i] + 0.01)
    generalizes = avg_nd > avg_nr

    print(f"\n  Novel avg: DISC={avg_nd:+.4f} RND={avg_nr:+.4f}")
    print(f"  Novel: {nw}W / {nl}L")
    print(f"  -> {'GENERALIZES' if generalizes else 'OVERFITS'}")

    # ── LET THERE BE A SECOND BODY ───────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE A SECOND BODY")
    print(f"  Different birth seed, same development.")
    print(f"{'-'*W}\n")

    org2 = Organism(seed=SEED + 999)
    develop(org2, n_epochs=12, base_seed=SEED + 999, verbose=False)

    alpha_cos = vcosine(org.alpha_flat(), org2.alpha_flat())
    print(f"  Body1: seed={SEED}, {org.n_restructures} restructures")
    print(f"  Body2: seed={SEED+999}, {org2.n_restructures} restructures")
    print(f"  Alpha cosine between bodies: {alpha_cos:+.4f}")

    b2_gaps = []
    for k in [6, 8]:
        sigs = make_signals(k, seed=SEED + k * 200)
        for seed in test_seeds[:2]:
            b2_gaps.append(measure_gap(org2, sigs, k, seed))
    b2_avg = sum(b2_gaps) / len(b2_gaps)
    print(f"  Body2 avg gap: {b2_avg:+.4f}")

    # ── LET THERE BE ALPHA STRUCTURE ─────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE ALPHA STRUCTURE")
    print(f"{'-'*W}\n")

    for i in range(NC):
        a_mean = sum(org.alpha[i]) / D
        a_std = math.sqrt(sum((org.alpha[i][k] - a_mean) ** 2
                              for k in range(D)) / D)
        supra = sum(1 for k in range(D) if org.alpha[i][k] > 1.0)
        print(f"  Cell {i}: mean={a_mean:.3f} std={a_std:.3f} supra={supra}/{D}")

    cell_cos = []
    for i in range(NC):
        for j in range(i + 1, NC):
            cell_cos.append(vcosine(org.alpha[i], org.alpha[j]))
    avg_cell_cos = sum(cell_cos) / len(cell_cos)
    min_cell_cos = min(cell_cos)
    specialized = avg_cell_cos < 0.95

    print(f"\n  Inter-cell cos: avg={avg_cell_cos:+.4f} min={min_cell_cos:+.4f}")
    print(f"  -> {'SPECIALIZED' if specialized else 'HOMOGENEOUS'}")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    disc_beats_rnd = delta > 0
    disc_wins_majority = disc_wins > rnd_wins
    seed_robust = n_improved >= 3
    seed_wins_more = total_seed_wins > total_seed_losses

    monotonic = False
    if len(epoch_log) >= 6:
        early = [e['quick_gap'] for e in epoch_log[:4]]
        late = [e['quick_gap'] for e in epoch_log[-4:]]
        monotonic = sum(late) / len(late) > sum(early) / len(early)

    checks = [
        ("DISC avg > RND avg (overall)",
         disc_beats_rnd,
         f"delta={delta:+.4f}"),

        ("DISC wins per-seed majority",
         disc_wins_majority,
         f"{disc_wins}W / {rnd_wins}W / {ties}T"),

        ("Seed-robust (3+/4 births improve)",
         seed_robust,
         f"{n_improved}/{len(birth_seeds)} improve"),

        ("Per-seed wins > losses",
         seed_wins_more,
         f"{total_seed_wins}W / {total_seed_losses}L"),

        ("Generalizes to novel signals",
         generalizes,
         f"DISC={avg_nd:+.4f} RND={avg_nr:+.4f}"),

        ("Novel wins > losses",
         nw > nl,
         f"{nw}W / {nl}L"),

        ("Alpha converges",
         converging,
         f"first={first_cos:.4f} last={last_cos:.4f}"),

        ("Alpha stable (>0.995)",
         converged,
         f"last={last_cos:.6f}"),

        ("Cells specialize (cos < 0.95)",
         specialized,
         f"avg cos={avg_cell_cos:+.4f}"),

        ("Performance improves over epochs",
         monotonic,
         f"early={sum(e['quick_gap'] for e in epoch_log[:4])/4:.4f} "
         f"late={sum(e['quick_gap'] for e in epoch_log[-4:])/4:.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<48} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    core_beats = disc_beats_rnd
    core_seeds = seed_robust or seed_wins_more
    core_novel = generalizes
    core_conv = converging
    core_spec = specialized

    full = core_beats and core_seeds and core_novel and core_conv and core_spec
    partial = core_beats and core_conv and (core_seeds or core_novel)

    if full:
        print(f"""
  THE SEED IS VIABLE.

  Fixed D={D}. {NC} cells. {org.n_restructures} restructures.

  Discrimination-derived restructuring: the system runs
  multiple orderings, measures which dimensions and cells
  tell them apart, and amplifies diversity where discrimination
  happens. The restructuring signal IS the computation.

  Delta vs random: {delta:+.4f}
  Seed robustness: {n_improved}/{len(birth_seeds)} births improve
  Novel signals: DISC={avg_nd:+.4f} RND={avg_nr:+.4f}
  Alpha convergence: {last_cos:.6f}
  Specialization: avg cos={avg_cell_cos:+.4f}

  The product term that computes also reveals where
  computation succeeds. The success drives restructuring.
  The restructuring improves computation. The loop closes.
""")
    elif partial:
        print(f"""
  THE SEED IS PARTIALLY VIABLE.

  Discrimination-derived restructuring beats random
  ({delta:+.4f}) and alpha converges.
  {'Seed robustness holds.' if core_seeds else 'Seed robustness weak.'}
  {'Generalization holds.' if core_novel else 'Generalization weak (overfits).'}
  {'Cells specialized.' if core_spec else 'Cells homogeneous.'}
""")
    elif core_beats:
        print(f"""
  RESTRUCTURING HELPS BUT IS NOT A SEED.

  DISC beats random ({delta:+.4f}), but the advantage
  is not robust across seeds or does not generalize.
""")
    elif converging:
        print(f"""
  ALPHA CONVERGES BUT DOES NOT IMPROVE.

  Convergence without improvement. The discrimination
  signal does not translate to better alpha.
""")
    else:
        print(f"""
  THE SEED IS NOT VIABLE.

  Discrimination-derived restructuring does not outperform
  random. Honest result.
""")

    # ── GROWTH TRACE ─────────────────────────────────────────

    print(f"{'-'*W}")
    print(f"  GROWTH TRACE")
    print(f"{'-'*W}\n")

    for e in epoch_log:
        sym = '#' if e['quick_gap'] > 0.05 else 'o'
        print(f"  {sym} K={e['k']:>2} qgap={e['quick_gap']:+.4f} "
              f"disc={e['disc_strength']:.4f} {bar(e['quick_gap'])}")

    print(f"\n  Total restructures: {org.n_restructures}")
    print(f"  Alpha distance from birth: {1.0 - birth_to_final:.6f}")
    print(f"  Total runtime: {time.time() - t_start:.0f}s")
    print(f"\n{'-'*W}")


if __name__ == '__main__':
    run()
