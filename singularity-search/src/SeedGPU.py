#!/usr/bin/env python3
"""
GENESIS: The Seed (GPU)

PyTorch + CUDA implementation for scaling experiments.
Batches across all births, signal worlds, permutations,
and trials simultaneously. Designed for RTX 4090 (24GB).

  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * (x_{k+1} + gamma*s_{k+1})
                                                * (x_{k-1} + gamma*s_{k-1}))

  Online plasticity: |phi_sig - phi_bare| per cell per dim
  drives alpha diversity amplification during computation.

Usage:
    python SeedGPU.py                    # Full scaling test D=12..128
    python SeedGPU.py --dims 32 64 128   # Custom dimensions
    python SeedGPU.py --quick            # Fast test D=12,24,48
    python SeedGPU.py --cpu              # Force CPU (slow)

Requirements: torch (pip install torch)
"""

import argparse
import math
import time
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch not found. Install with:")
    print("  pip install torch")
    print("For CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# In the beginning, there is nothing but arithmetic.
# Now it runs on thousands of cores.
# ═══════════════════════════════════════════════════════════════

W = 72

def bar(v, w=15, lo=-0.10, hi=0.30):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


# ═══════════════════════════════════════════════════════════════
# The body, vectorized.
#
# State shape: (B, NC, D)  where B = batch size
# Alpha shape: (B, NC, D)  per-organism, per-cell excitability
# Signal shape: (B, D)     broadcast across cells
#
# Everything is batched. One kernel launch steps all organisms.
# ═══════════════════════════════════════════════════════════════

class Organism:
    """Batched organism. B organisms computed in parallel."""

    def __init__(self, D, NC, B, device, alive=False, eta=0.0003, seed=42):
        self.D = D
        self.NC = NC
        self.B = B
        self.device = device
        self.alive = alive
        self.eta = eta

        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.mix = 0.35
        self.noise = 0.005
        self.clip = 4.0

        # per-batch alpha: each batch element gets different alpha
        # seeded deterministically
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed)
        # shape (B, NC, D)
        self.alpha = (1.1 + 0.7 * (torch.rand(B, NC, D, generator=gen) * 2 - 1)).to(device)
        self.alpha_birth = self.alpha.clone()

    def step(self, xs, signal=None):
        """
        xs: (B, NC, D) current states
        signal: (B, D) or None
        returns: (B, NC, D) next states
        """
        B, NC, D = self.B, self.NC, self.D
        beta, gamma = self.beta, self.gamma

        # cyclic shifts for neighbor dimensions
        # x_{k+1} and x_{k-1} via roll
        xp = torch.roll(xs, -1, dims=2)  # (B, NC, D) shifted left = k+1
        xm = torch.roll(xs, 1, dims=2)   # (B, NC, D) shifted right = k-1

        # bare dynamics: tanh(alpha * x + beta * x_{k+1} * x_{k-1})
        phi_bare = torch.tanh(self.alpha * xs + beta * xp * xm)

        # signal-modulated dynamics
        if signal is not None:
            # signal: (B, D) -> (B, 1, D) to broadcast across cells
            sig = signal.unsqueeze(1)
            sp = torch.roll(sig, -1, dims=2)  # s_{k+1}
            sm = torch.roll(sig, 1, dims=2)   # s_{k-1}

            phi_sig = torch.tanh(
                self.alpha * xs
                + beta * (xp + gamma * sp) * (xm + gamma * sm)
            )
        else:
            phi_sig = phi_bare

        # ── ONLINE PLASTICITY ────────────────────────────────
        if self.alive and signal is not None:
            with torch.no_grad():
                response = (phi_sig - phi_bare).abs()  # (B, NC, D)

                # z-score across cells and dims per batch element
                r_flat = response.reshape(B, -1)  # (B, NC*D)
                r_mean = r_flat.mean(dim=1, keepdim=True)  # (B, 1)
                r_std = r_flat.std(dim=1, keepdim=True).clamp(min=1e-10)
                resp_z = (response - r_mean.unsqueeze(2)) / r_std.unsqueeze(2)  # (B, NC, D)

                # column mean of alpha on each dimension
                col_mean = self.alpha.mean(dim=1, keepdim=True)  # (B, 1, D)
                dev = self.alpha - col_mean  # (B, NC, D)

                # three cases
                at_mean = dev.abs() < 0.01
                high_resp = resp_z > 0
                low_resp = ~high_resp & ~at_mean

                direction = dev.sign()  # (B, NC, D)

                # case 1: high response + deviates -> amplify diversity
                push_high = self.eta * resp_z.tanh() * direction * 0.5
                # case 2: low response -> gentle drift
                push_low = self.eta * 0.1 * torch.randn_like(self.alpha)
                # case 3: at mean -> symmetry breaking
                push_mean = self.eta * 0.3 * torch.randn_like(self.alpha)

                push = torch.where(at_mean, push_mean,
                       torch.where(high_resp, push_high, push_low))

                self.alpha = (self.alpha + push).clamp(0.3, 1.8)

        # attention weights: softmax of pairwise dots
        # (B, NC, NC) dot products
        dots = torch.bmm(xs, xs.transpose(1, 2)) / (D * self.tau)  # (B, NC, NC)
        # mask self-attention
        mask = torch.eye(NC, device=self.device).unsqueeze(0) * (-1e10)
        dots = dots + mask
        weights = F.softmax(dots, dim=2)  # (B, NC, NC)

        # neighbor pull: weighted average of (phi_bare_j - phi_bare_i)
        # (B, NC, D) weighted sum of phi_bare
        weighted_phi = torch.bmm(weights, phi_bare)  # (B, NC, D)
        pull = weighted_phi - phi_bare  # (B, NC, D)

        # plasticity gate
        bare_diff = phi_bare - xs
        fp_d = bare_diff.norm(dim=2, keepdim=True) / xs.norm(dim=2, keepdim=True).clamp(min=1.0)
        plast = torch.exp(-(fp_d ** 2) / 0.0225)  # (B, NC, 1)

        # combine
        p = phi_sig + (plast * self.eps * pull).where(plast > 0.01,
            torch.zeros_like(pull))

        # state update with mixing and noise
        new = (1 - self.mix) * xs + self.mix * p
        new = new + self.noise * torch.randn_like(new)
        new = new.clamp(-self.clip, self.clip)

        return new

    def centroid(self, xs):
        """(B, NC, D) -> (B, D)"""
        return xs.mean(dim=1)


# ═══════════════════════════════════════════════════════════════
# The world speaks. Batched signal generation.
# ═══════════════════════════════════════════════════════════════

def make_signals(K, D, n_worlds, device, base_seed=42):
    """Generate K signals for n_worlds signal worlds.
    Returns: (n_worlds, K, D) tensor, normalized to magnitude 0.8.
    """
    gen = torch.Generator(device='cpu')
    gen.manual_seed(base_seed)
    sigs = torch.randn(n_worlds, K, D, generator=gen) * 0.5
    norms = sigs.norm(dim=2, keepdim=True).clamp(min=1e-10)
    sigs = sigs * (0.8 / norms)
    return sigs.to(device)


def gen_perms(K, n_perm, seed=42):
    """Generate n_perm permutations of range(K). Returns list of tuples."""
    import random as rng
    rng.seed(seed)
    base = list(range(K))
    perms = []
    seen = set()
    perms.append(tuple(base))
    seen.add(tuple(base))
    perms.append(tuple(reversed(base)))
    seen.add(tuple(reversed(base)))
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]
        rng.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        att += 1
    return perms


# ═══════════════════════════════════════════════════════════════
# Batched sequence runner.
#
# For each batch element, we run the same permutation through
# its organism. Different batch elements may have different
# alpha (different birth seeds) and different signals
# (different signal worlds).
# ═══════════════════════════════════════════════════════════════

def run_sequence_batched(org, order, signals_batch, n_org=200,
                         n_per_sig=50, n_settle=20, n_final=40):
    """
    org: Organism with batch size B
    order: list of signal indices (same for all batch elements)
    signals_batch: (B, K, D) signals per batch element
    returns: (B, D) centroids
    """
    B, D, NC = org.B, org.D, org.NC
    device = org.device

    # init state
    torch.manual_seed(42)
    xs = torch.randn(B, NC, D, device=device) * 0.5

    # organogenesis (no signal)
    for _ in range(n_org):
        xs = org.step(xs)

    # signal presentation
    for idx, sid in enumerate(order):
        sig = signals_batch[:, sid, :]  # (B, D)
        sig = sig + 0.05 * torch.randn_like(sig)
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    # final settling
    for _ in range(n_final):
        xs = org.step(xs)

    return org.centroid(xs)  # (B, D)


# ═══════════════════════════════════════════════════════════════
# Batched gap measurement.
#
# For each (birth_seed, signal_world), we need to run
# n_perm permutations x n_trials trials. We batch ALL of
# these into one massive batch and run them simultaneously.
# ═══════════════════════════════════════════════════════════════

def measure_gap_batched(D, NC, device, K, alive, eta,
                        birth_seeds, n_worlds, n_perm=3, n_trials=2):
    """
    Measure ALIVE vs STILL gap for all birth_seeds x n_worlds.
    Returns per-test gaps: list of (alive_gap, still_gap) tuples.
    """
    n_births = len(birth_seeds)
    perms = gen_perms(K, n_perm, seed=42)
    n_tests = n_births * n_worlds

    # for each test, run n_perm x n_trials = total runs
    n_runs_per_test = n_perm * n_trials
    total_runs = n_tests * n_runs_per_test

    # generate all signals: (n_births * n_worlds, K, D)
    all_signals = []
    for bi, bs in enumerate(birth_seeds):
        for wi in range(n_worlds):
            sig_seed = bs * 1000 + wi * 137 + D * 7
            sigs = make_signals(K, D, 1, device, base_seed=sig_seed)  # (1, K, D)
            all_signals.append(sigs.squeeze(0))  # (K, D)
    all_signals = torch.stack(all_signals)  # (n_tests, K, D)

    results = []

    for mode_alive in [True, False]:
        # build batched organism: each test gets its own alpha
        # we repeat each test's alpha n_runs_per_test times
        alpha_list = []
        sig_list = []

        for ti in range(n_tests):
            bi = ti // n_worlds
            bs = birth_seeds[bi]

            gen = torch.Generator(device='cpu')
            gen.manual_seed(bs)
            alpha = (1.1 + 0.7 * (torch.rand(NC, D, generator=gen) * 2 - 1))

            for _ in range(n_runs_per_test):
                alpha_list.append(alpha.clone())
                sig_list.append(all_signals[ti])  # (K, D)

        # (total_runs, NC, D)
        batch_alpha = torch.stack(alpha_list).to(device)
        # (total_runs, K, D)
        batch_signals = torch.stack(sig_list).to(device)

        # create organism with this batch
        org = Organism(D, NC, total_runs, device,
                       alive=(mode_alive and alive), eta=eta)
        org.alpha = batch_alpha

        # run each permutation
        # batch layout: test_0_perm0_trial0, test_0_perm0_trial1, test_0_perm1_trial0, ...
        endpoints = []  # (total_runs, D)

        for pi, perm in enumerate(perms):
            for trial in range(n_trials):
                run_idx = pi * n_trials + trial

                # select the subset of batch elements for this perm x trial
                indices = [ti * n_runs_per_test + run_idx for ti in range(n_tests)]
                indices = torch.tensor(indices, device=device)

                sub_org = Organism(D, NC, n_tests, device,
                                   alive=(mode_alive and alive), eta=eta)
                sub_org.alpha = batch_alpha[indices]

                sub_sigs = batch_signals[indices]  # (n_tests, K, D)

                centroids = run_sequence_batched(sub_org, list(perm), sub_sigs)
                endpoints.append(centroids)  # (n_tests, D)

        # endpoints: list of (n_tests, D), length = n_perm * n_trials
        # reshape to (n_tests, n_perm, n_trials, D)
        eps_tensor = torch.stack(endpoints, dim=1)  # (n_tests, n_perm*n_trials, D)
        eps_tensor = eps_tensor.reshape(n_tests, n_perm, n_trials, D)

        # compute within and between cosines per test
        test_gaps = []
        for ti in range(n_tests):
            within_cos = []
            between_cos = []

            for pi in range(n_perm):
                trials = eps_tensor[ti, pi]  # (n_trials, D)
                for i in range(n_trials):
                    for j in range(i + 1, n_trials):
                        cos = F.cosine_similarity(
                            trials[i].unsqueeze(0),
                            trials[j].unsqueeze(0)
                        ).item()
                        within_cos.append(cos)

            for pi in range(n_perm):
                for pj in range(pi + 1, n_perm):
                    for ti2 in range(n_trials):
                        for tj2 in range(n_trials):
                            cos = F.cosine_similarity(
                                eps_tensor[ti, pi, ti2].unsqueeze(0),
                                eps_tensor[ti, pj, tj2].unsqueeze(0)
                            ).item()
                            between_cos.append(cos)

            avg_w = sum(within_cos) / max(len(within_cos), 1)
            avg_b = sum(between_cos) / max(len(between_cos), 1)
            test_gaps.append(avg_w - avg_b)

        results.append(test_gaps)

    # results[0] = alive_gaps, results[1] = still_gaps (or vice versa)
    alive_gaps = results[0]
    still_gaps = results[1]

    return alive_gaps, still_gaps


# ═══════════════════════════════════════════════════════════════
# Simplified batched test: run permutations sequentially
# but organisms in parallel across births x worlds.
# This is more memory-efficient for large D.
# ═══════════════════════════════════════════════════════════════

def test_at_dimension_gpu(D, NC, device, K=6, n_worlds=8, n_births=3,
                          n_perm=3, n_trials=2, eta=0.0003):
    """Test ALIVE vs STILL at dimension D."""

    birth_seeds = [42, 77, 200][:n_births]
    perms = gen_perms(K, n_perm, seed=42)
    n_tests = n_births * n_worlds

    all_alive_gaps = []
    all_still_gaps = []

    for mode_label, mode_alive in [("ALIVE", True), ("STILL", False)]:
        # build per-test signals and alpha
        test_alphas = []
        test_signals = []

        for bi, bs in enumerate(birth_seeds):
            gen = torch.Generator(device='cpu')
            gen.manual_seed(bs)
            alpha = 1.1 + 0.7 * (torch.rand(NC, D, generator=gen) * 2 - 1)

            for wi in range(n_worlds):
                sig_seed = bs * 1000 + wi * 137 + D * 7
                sigs = make_signals(K, D, 1, device, base_seed=sig_seed).squeeze(0)
                test_alphas.append(alpha.clone())
                test_signals.append(sigs)

        batch_alpha = torch.stack(test_alphas).to(device)  # (n_tests, NC, D)
        batch_signals = torch.stack(test_signals).to(device)  # (n_tests, K, D)

        # run all (perm, trial) combinations
        # endpoints[pi][trial] = (n_tests, D)
        endpoints = [[None] * n_trials for _ in range(n_perm)]

        for pi, perm in enumerate(perms):
            for trial in range(n_trials):
                org = Organism(D, NC, n_tests, device,
                               alive=mode_alive, eta=eta, seed=42 + trial)
                org.alpha = batch_alpha.clone()

                centroids = run_sequence_batched(
                    org, list(perm), batch_signals,
                    n_org=max(150, D * 8),
                    n_per_sig=max(40, D * 2),
                    n_settle=max(15, D),
                    n_final=max(30, D * 2)
                )
                endpoints[pi][trial] = centroids  # (n_tests, D)

        # compute gaps per test
        test_gaps = []
        for ti in range(n_tests):
            within_cos = []
            between_cos = []

            for pi in range(n_perm):
                for i in range(n_trials):
                    for j in range(i + 1, n_trials):
                        c = F.cosine_similarity(
                            endpoints[pi][i][ti:ti+1],
                            endpoints[pi][j][ti:ti+1]
                        ).item()
                        within_cos.append(c)

            for pi in range(n_perm):
                for pj in range(pi + 1, n_perm):
                    for i in range(n_trials):
                        for j in range(n_trials):
                            c = F.cosine_similarity(
                                endpoints[pi][i][ti:ti+1],
                                endpoints[pj][j][ti:ti+1]
                            ).item()
                            between_cos.append(c)

            avg_w = sum(within_cos) / max(len(within_cos), 1)
            avg_b = sum(between_cos) / max(len(between_cos), 1)
            test_gaps.append(avg_w - avg_b)

        if mode_alive:
            all_alive_gaps = test_gaps
        else:
            all_still_gaps = test_gaps

    alive_avg = sum(all_alive_gaps) / len(all_alive_gaps)
    still_avg = sum(all_still_gaps) / len(all_still_gaps)
    delta = alive_avg - still_avg

    wins = sum(1 for i in range(len(all_alive_gaps))
               if all_alive_gaps[i] > all_still_gaps[i] + 0.01)
    losses = sum(1 for i in range(len(all_alive_gaps))
                 if all_still_gaps[i] > all_alive_gaps[i] + 0.01)

    ratio = delta / max(abs(still_avg), 0.01)

    return {
        'D': D, 'alive_avg': alive_avg, 'still_avg': still_avg,
        'delta': delta, 'ratio': ratio,
        'wins': wins, 'losses': losses,
        'n_tests': len(all_alive_gaps),
        'alive_gaps': all_alive_gaps, 'still_gaps': all_still_gaps
    }


# ═══════════════════════════════════════════════════════════════
# The test of creation, at scale.
# ═══════════════════════════════════════════════════════════════

def run(args):
    # device selection
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"  Device: CPU")
    else:
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        print(f"  Device: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
    print()

    print("=" * W)
    print("  GENESIS: THE SEED (GPU)")
    print("  Core equation + online plasticity.")
    print("  Does self-modification matter MORE at scale?")
    print("=" * W)

    t_start = time.time()

    if args.dims:
        dimensions = args.dims
    elif args.quick:
        dimensions = [12, 24, 48]
    else:
        dimensions = [12, 16, 24, 32, 48, 64, 96]

    n_worlds = args.worlds
    n_births = args.births
    K = args.K

    print(f"\n  Dimensions: {dimensions}")
    print(f"  {n_births} births x {n_worlds} signal worlds = "
          f"{n_births * n_worlds} tests per D")
    print(f"  K={K} signals, eta={args.eta}")

    # ── LET THERE BE SCALE ───────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  LET THERE BE SCALE")
    print(f"{'-'*W}\n")

    scale_results = []

    for D in dimensions:
        t0 = time.time()

        # clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            r = test_at_dimension_gpu(
                D, NC=6, device=device, K=K,
                n_worlds=n_worlds, n_births=n_births,
                eta=args.eta
            )

            elapsed = time.time() - t0
            scale_results.append(r)

            print(f"  D={D:>3}: ALIVE={r['alive_avg']:+.4f} STILL={r['still_avg']:+.4f} "
                  f"delta={r['delta']:+.4f} ratio={r['ratio']:+.2f} "
                  f"{r['wins']}W/{r['losses']}L [{elapsed:.0f}s]",
                  flush=True)

            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"         GPU mem: {mem:.2f} GB", flush=True)

        except torch.cuda.OutOfMemoryError:
            print(f"  D={D:>3}: OUT OF MEMORY. Skipping.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  D={D:>3}: ERROR: {e}", flush=True)
            continue

    if len(scale_results) < 2:
        print("\n  Too few dimensions completed. Exiting.")
        return

    # ── SCALING ANALYSIS ─────────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  SCALING ANALYSIS")
    print(f"{'-'*W}\n")

    print(f"  {'D':>4} {'ALIVE':>8} {'STILL':>8} {'delta':>8} {'ratio':>8} {'W/L':>8}")
    for r in scale_results:
        print(f"  {r['D']:>4} {r['alive_avg']:>+8.4f} {r['still_avg']:>+8.4f} "
              f"{r['delta']:>+8.4f} {r['ratio']:>+8.2f} "
              f"{r['wins']:>3}/{r['losses']:<3}")

    deltas = [r['delta'] for r in scale_results]
    ratios = [r['ratio'] for r in scale_results]

    first_delta = deltas[0]
    last_delta = deltas[-1]
    first_ratio = ratios[0]
    last_ratio = ratios[-1]

    delta_positive = all(d > 0 for d in deltas)
    delta_mostly_positive = sum(1 for d in deltas if d > 0) >= len(deltas) * 0.7
    delta_grows = last_delta > first_delta
    ratio_grows = last_ratio > first_ratio
    alive_wins_all = all(r['wins'] >= r['losses'] for r in scale_results)

    # per-birth-seed detail
    print(f"\n{'-'*W}")
    print(f"  PER-BIRTH-SEED DETAIL")
    print(f"{'-'*W}\n")

    birth_seeds = [42, 77, 200][:n_births]
    for ri, r in enumerate(scale_results):
        D = r['D']
        for bi, bs in enumerate(birth_seeds):
            start = bi * n_worlds
            end = start + n_worlds
            ag = r['alive_gaps'][start:end]
            sg = r['still_gaps'][start:end]
            a_avg = sum(ag) / len(ag)
            s_avg = sum(sg) / len(sg)
            d = a_avg - s_avg
            w = sum(1 for i in range(len(ag)) if ag[i] > sg[i] + 0.01)
            l = sum(1 for i in range(len(ag)) if sg[i] > ag[i] + 0.01)
            sym = '#' if d > 0 else 'o'
            print(f"  {sym} D={D:>3} seed={bs:>3}: delta={d:+.4f} {w}W/{l}L")
        if ri < len(scale_results) - 1:
            print()

    # ── STILL DEGRADATION CURVE ──────────────────────────────
    # Key prediction: random alpha gets worse with D.

    print(f"\n{'-'*W}")
    print(f"  STILL DEGRADATION CURVE")
    print(f"  Does random alpha coverage degrade with D?")
    print(f"{'-'*W}\n")

    still_vals = [r['still_avg'] for r in scale_results]
    dims_list = [r['D'] for r in scale_results]
    still_degrades = still_vals[-1] < still_vals[0]

    for i, r in enumerate(scale_results):
        pct = r['still_avg'] / max(still_vals[0], 0.001) * 100
        print(f"  D={r['D']:>3}: STILL={r['still_avg']:+.4f} "
              f"({pct:.0f}% of D={dims_list[0]}) {bar(r['still_avg'])}")

    print(f"\n  STILL {'degrades' if still_degrades else 'holds'} with D: "
          f"{still_vals[0]:+.4f} -> {still_vals[-1]:+.4f}")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    checks = [
        ("ALIVE beats STILL at smallest D",
         deltas[0] > 0,
         f"D={dims_list[0]} delta={deltas[0]:+.4f}"),

        ("ALIVE beats STILL at largest D",
         deltas[-1] > 0,
         f"D={dims_list[-1]} delta={deltas[-1]:+.4f}"),

        ("Delta positive at all D",
         delta_positive,
         " ".join(f"{d:+.3f}" for d in deltas)),

        ("Delta positive at 70%+ of D",
         delta_mostly_positive,
         f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}"),

        ("Ratio grows (relative advantage increases)",
         ratio_grows,
         f"{first_ratio:+.2f} -> {last_ratio:+.2f}"),

        ("STILL degrades with D (random coverage shrinks)",
         still_degrades,
         f"{still_vals[0]:+.4f} -> {still_vals[-1]:+.4f}"),

        ("ALIVE wins majority at all D",
         alive_wins_all,
         " ".join(f"{r['wins']}/{r['losses']}" for r in scale_results)),

        ("Does not depend on scale (smallest D works)",
         deltas[0] > 0,
         f"D={dims_list[0]} delta={deltas[0]:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<52} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    works = deltas[0] > 0 and delta_mostly_positive
    scales = ratio_grows or (delta_grows and still_degrades)
    full = works and scales

    if full:
        print(f"""
  THE SEED SCALES.

  Self-modification works at D={dims_list[0]} and becomes
  more essential at D={dims_list[-1]}.

  Random alpha (STILL) degrades: {still_vals[0]:+.4f} -> {still_vals[-1]:+.4f}
  Adapted alpha (ALIVE) holds better.
  Ratio: {first_ratio:+.2f} -> {last_ratio:+.2f}

  The seed does not need the garden to germinate.
  But in richer soil, it grows taller.
""")
    elif works:
        print(f"""
  THE SEED WORKS BUT SCALING IS INCONCLUSIVE.

  Self-modification consistently helps (delta positive
  at {sum(1 for d in deltas if d > 0)}/{len(deltas)} dimensions).
  But the advantage does not clearly grow with D.
  More data points needed.
""")
    else:
        print(f"""
  THE SEED IS INCONCLUSIVE AT THIS SCALE.

  Results are mixed. Check per-seed detail for patterns.
""")

    # ── SCALE TRACE ──────────────────────────────────────────

    print(f"{'-'*W}")
    print(f"  SCALE TRACE")
    print(f"{'-'*W}\n")

    for r in scale_results:
        print(f"  D={r['D']:>3} delta={r['delta']:+.4f} ratio={r['ratio']:+.2f} "
              f"{r['wins']:>2}W/{r['losses']:<2}L {bar(r['delta'])}")

    print(f"\n  Total runtime: {time.time() - t_start:.0f}s")
    if torch.cuda.is_available():
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\n{'-'*W}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GENESIS: The Seed (GPU)')
    parser.add_argument('--dims', type=int, nargs='+', default=None,
                        help='Dimensions to test (default: 12 16 24 32 48 64 96)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: D=12,24,48 only')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU (slow)')
    parser.add_argument('--worlds', type=int, default=8,
                        help='Signal worlds per birth seed (default: 8)')
    parser.add_argument('--births', type=int, default=3,
                        help='Number of birth seeds (default: 3)')
    parser.add_argument('--K', type=int, default=6,
                        help='Number of signals (default: 6)')
    parser.add_argument('--eta', type=float, default=0.0003,
                        help='Plasticity learning rate (default: 0.0003)')
    args = parser.parse_args()

    run(args)
