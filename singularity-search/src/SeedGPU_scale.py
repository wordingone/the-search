#!/usr/bin/env python3
"""
GENESIS: The Seed (GPU)

PyTorch + CUDA implementation for scaling experiments on RTX 4090.

Architecture:
  Phi_s(x)_k = tanh(alpha_{i,k} * x_k + beta * (x_{k+1} + gamma*s_{k+1})
                                                * (x_{k-1} + gamma*s_{k-1}))
  Online plasticity: |phi_sig - phi_bare| drives alpha diversity amplification.

GPU strategy:
  - Batch across ALL births x worlds x perms x trials simultaneously
  - Pre-compute signal schedule: each batch element knows which signal
    at each time step, encoded as a (B, K, D) tensor
  - Single forward pass per mode (ALIVE / STILL)
  - torch.compile on step function for kernel fusion
  - Fixed step counts (no D-scaling)

Usage:
    python SeedGPU.py                        # Full: D=12,16,24,32,48,64,96
    python SeedGPU.py --dims 32 64 128       # Custom dimensions
    python SeedGPU.py --quick                # Fast: D=12,24,48
    python SeedGPU.py --cpu                  # Force CPU

Requirements: torch >= 2.0 (pip install torch)
"""

import argparse
import math
import time
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch not found. Install:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)


W = 72

def bar(v, w=15, lo=-0.10, hi=0.30):
    f = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-10)))
    n = int(f * w)
    return '\u2588' * n + '\u2591' * (w - n)


# ═══════════════════════════════════════════════════════════════
# The step function. Pure tensor ops, no Python control flow
# in the hot path. torch.compile fuses this into minimal kernels.
# ═══════════════════════════════════════════════════════════════

def step_core(xs, alpha, signal, alive_mask,
              beta, gamma, eps, tau, mix, noise_std, clip_val, eta):
    """
    xs:         (B, NC, D) states
    alpha:      (B, NC, D) excitability
    signal:     (B, 1, D) or None — broadcast across cells
    alive_mask: (B, 1, 1) float — 1.0 for alive, 0.0 for still
    Returns:    (new_xs, new_alpha)
    """
    B, NC, D = xs.shape

    # neighbor dimensions via roll
    xp = torch.roll(xs, -1, dims=2)   # x_{k+1}
    xm = torch.roll(xs, 1, dims=2)    # x_{k-1}

    # bare dynamics
    phi_bare = torch.tanh(alpha * xs + beta * xp * xm)

    # signal-modulated dynamics
    if signal is not None:
        sp = torch.roll(signal, -1, dims=2)
        sm = torch.roll(signal, 1, dims=2)
        phi_sig = torch.tanh(
            alpha * xs + beta * (xp + gamma * sp) * (xm + gamma * sm)
        )

        # ── ONLINE PLASTICITY ────────────────────────────
        # Compute for all, mask by alive_mask
        response = (phi_sig - phi_bare).abs()       # (B, NC, D)
        r_flat = response.reshape(B, -1)            # (B, NC*D)
        r_mean = r_flat.mean(dim=1, keepdim=True)   # (B, 1)
        r_std = r_flat.std(dim=1, keepdim=True).clamp(min=1e-10)
        resp_z = (response - r_mean.unsqueeze(2)) / r_std.unsqueeze(2)

        col_mean = alpha.mean(dim=1, keepdim=True)  # (B, 1, D)
        dev = alpha - col_mean

        at_mean = (dev.abs() < 0.01).float()
        high_resp = (resp_z > 0).float()
        low_resp = (1.0 - high_resp) * (1.0 - at_mean)
        direction = dev.sign()

        push_high = eta * resp_z.tanh() * direction * 0.5
        push_low = eta * 0.1 * torch.randn_like(alpha)
        push_mean = eta * 0.3 * torch.randn_like(alpha)

        push = (at_mean * push_mean
                + high_resp * (1.0 - at_mean) * push_high
                + low_resp * push_low)

        alpha = (alpha + push * alive_mask).clamp(0.3, 1.8)
    else:
        phi_sig = phi_bare

    # attention: (B, NC, NC)
    dots = torch.bmm(xs, xs.transpose(1, 2)) / (D * tau)
    eye_mask = torch.eye(NC, device=xs.device, dtype=xs.dtype).unsqueeze(0) * (-1e10)
    weights = F.softmax(dots + eye_mask, dim=2)

    # neighbor pull
    weighted_phi = torch.bmm(weights, phi_bare)
    pull = weighted_phi - phi_bare

    # plasticity gate
    bare_diff = phi_bare - xs
    x_norm = xs.norm(dim=2, keepdim=True).clamp(min=1.0)
    fp_d = bare_diff.norm(dim=2, keepdim=True) / x_norm
    plast = torch.exp(-(fp_d ** 2) / 0.0225)
    plast = plast * (plast > 0.01).float()

    # combine
    p = phi_sig + plast * eps * pull
    new_xs = (1.0 - mix) * xs + mix * p
    new_xs = new_xs + noise_std * torch.randn_like(new_xs)
    new_xs = new_xs.clamp(-clip_val, clip_val)

    return new_xs, alpha


# try torch.compile for kernel fusion (PyTorch 2.0+)
try:
    step_compiled = torch.compile(step_core, mode="reduce-overhead")
    HAS_COMPILE = True
except Exception:
    step_compiled = step_core
    HAS_COMPILE = False


# ═══════════════════════════════════════════════════════════════
# Signal and permutation generation.
# ═══════════════════════════════════════════════════════════════

def make_signals(K, D, seed, device):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    sigs = torch.randn(K, D, generator=gen) * 0.5
    norms = sigs.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return (sigs * (0.8 / norms)).to(device)


def gen_perms(K, n_perm, seed=42):
    import random as rng
    rng.seed(seed)
    base = list(range(K))
    perms = [tuple(base), tuple(reversed(base))]
    seen = set(perms)
    for _ in range(n_perm * 50):
        if len(perms) >= n_perm:
            break
        p = base[:]
        rng.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
    return perms[:n_perm]


# ═══════════════════════════════════════════════════════════════
# Batched experiment runner.
#
# Batches ALL births x worlds x perms x trials into one pass.
# Each batch element has its own alpha and signal schedule.
# ═══════════════════════════════════════════════════════════════

def run_experiment(D, NC, device, K, birth_seeds, n_worlds, n_perm, n_trials,
                   alive, eta,
                   n_org=200, n_per_sig=50, n_settle=20, n_final=40):
    """
    Run full experiment for one mode (ALIVE or STILL).
    Returns: list of gaps, one per (birth, world) test.
    """
    # Scale step counts with D for convergence at high dimensions
    scale_factor = math.sqrt(D / 12.0)
    n_org = int(n_org * scale_factor)
    n_per_sig = int(n_per_sig * scale_factor)
    n_settle = int(n_settle * scale_factor)
    n_final = int(n_final * scale_factor)

    n_births = len(birth_seeds)
    perms = gen_perms(K, n_perm, seed=42)
    n_tests = n_births * n_worlds
    B = n_tests * n_perm * n_trials

    # Build alpha: (B, NC, D)
    alpha_list = []
    for bi, bs in enumerate(birth_seeds):
        gen = torch.Generator(device='cpu')
        gen.manual_seed(bs)
        alpha_template = 1.1 + 0.7 * (torch.rand(NC, D, generator=gen) * 2 - 1)
        for wi in range(n_worlds):
            for pi in range(n_perm):
                for ti in range(n_trials):
                    alpha_list.append(alpha_template.clone())

    alpha = torch.stack(alpha_list).to(device)

    # Build signal schedule: (B, K, D)
    signal_schedules = []
    for bi, bs in enumerate(birth_seeds):
        for wi in range(n_worlds):
            sig_seed = bs * 1000 + wi * 137 + D * 7
            sigs = make_signals(K, D, sig_seed, device)
            for pi in range(n_perm):
                perm = perms[pi]
                ordered = torch.stack([sigs[perm[p]] for p in range(K)])
                for ti in range(n_trials):
                    signal_schedules.append(ordered)

    sig_sched = torch.stack(signal_schedules)  # (B, K, D)

    alive_mask = torch.ones(B, 1, 1, device=device) if alive else torch.zeros(B, 1, 1, device=device)

    beta, gamma, eps, tau, mix = 0.5, 0.9, 0.15, 0.3, 0.35
    noise_std, clip_val = 0.005, 4.0

    torch.manual_seed(42)
    xs = torch.randn(B, NC, D, device=device) * 0.5

    step_fn = step_compiled if HAS_COMPILE and device.type == 'cuda' else step_core

    # organogenesis
    for _ in range(n_org):
        xs, alpha = step_fn(xs, alpha, None, alive_mask,
                            beta, gamma, eps, tau, mix, noise_std, clip_val, eta)

    # signal presentation
    for p in range(K):
        sig = sig_sched[:, p, :].unsqueeze(1)  # (B, 1, D)
        sig = sig + 0.05 * torch.randn_like(sig)

        for _ in range(n_per_sig):
            xs, alpha = step_fn(xs, alpha, sig, alive_mask,
                                beta, gamma, eps, tau, mix, noise_std, clip_val, eta)
        for _ in range(n_settle):
            xs, alpha = step_fn(xs, alpha, None, alive_mask,
                                beta, gamma, eps, tau, mix, noise_std, clip_val, eta)

    for _ in range(n_final):
        xs, alpha = step_fn(xs, alpha, None, alive_mask,
                            beta, gamma, eps, tau, mix, noise_std, clip_val, eta)

    # centroids: (B, D) -> (n_tests, n_perm, n_trials, D)
    centroids = xs.mean(dim=1).reshape(n_tests, n_perm, n_trials, D)

    # compute gaps
    gaps = []
    for ti in range(n_tests):
        within_cos = []
        between_cos = []

        for pi in range(n_perm):
            for i in range(n_trials):
                for j in range(i + 1, n_trials):
                    c = F.cosine_similarity(
                        centroids[ti, pi, i].unsqueeze(0),
                        centroids[ti, pi, j].unsqueeze(0)
                    ).item()
                    within_cos.append(c)

        for pi in range(n_perm):
            for pj in range(pi + 1, n_perm):
                for i in range(n_trials):
                    for j in range(n_trials):
                        c = F.cosine_similarity(
                            centroids[ti, pi, i].unsqueeze(0),
                            centroids[ti, pj, j].unsqueeze(0)
                        ).item()
                        between_cos.append(c)

        avg_w = sum(within_cos) / max(len(within_cos), 1)
        avg_b = sum(between_cos) / max(len(between_cos), 1)
        gaps.append(avg_w - avg_b)

    return gaps


# ═══════════════════════════════════════════════════════════════
# Test at one dimension.
# ═══════════════════════════════════════════════════════════════

def test_at_dimension(D, device, NC=6, K=6, n_worlds=8, n_births=3,
                      n_perm=3, n_trials=2, eta=0.0003):
    birth_seeds = [42, 77, 200][:n_births]

    alive_gaps = run_experiment(
        D, NC, device, K, birth_seeds, n_worlds, n_perm, n_trials,
        alive=True, eta=eta
    )
    still_gaps = run_experiment(
        D, NC, device, K, birth_seeds, n_worlds, n_perm, n_trials,
        alive=False, eta=eta
    )

    alive_avg = sum(alive_gaps) / len(alive_gaps)
    still_avg = sum(still_gaps) / len(still_gaps)
    delta = alive_avg - still_avg

    wins = sum(1 for i in range(len(alive_gaps))
               if alive_gaps[i] > still_gaps[i] + 0.01)
    losses = sum(1 for i in range(len(alive_gaps))
                 if still_gaps[i] > alive_gaps[i] + 0.01)

    ratio = delta / max(abs(still_avg), 0.01)

    return {
        'D': D, 'alive_avg': alive_avg, 'still_avg': still_avg,
        'delta': delta, 'ratio': ratio,
        'wins': wins, 'losses': losses,
        'n_tests': len(alive_gaps),
        'alive_gaps': alive_gaps, 'still_gaps': still_gaps
    }


# ═══════════════════════════════════════════════════════════════
# Main.
# ═══════════════════════════════════════════════════════════════

def run(args):
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"  Device: CPU {'(forced)' if args.cpu else '(no CUDA)'}")
    else:
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        print(f"  Device: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    print(f"  PyTorch: {torch.__version__}")
    print(f"  torch.compile: {'available' if HAS_COMPILE else 'unavailable'}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
    print()

    print("=" * W)
    print("  GENESIS: THE SEED (GPU)")
    print("  Resolving the V-shape: does delta grow with D?")
    print("=" * W)

    t_start = time.time()

    if args.dims:
        dimensions = args.dims
    elif args.quick:
        dimensions = [12, 24, 48]
    else:
        dimensions = [12, 24, 48, 96, 192, 384, 768, 1536, 2048]

    n_worlds = args.worlds
    n_births = args.births
    K = args.K
    n_perm = 3
    n_trials = 2

    B_per_mode = n_births * n_worlds * n_perm * n_trials
    print(f"\n  Dimensions: {dimensions}")
    print(f"  {n_births} births x {n_worlds} worlds x {n_perm} perms x {n_trials} trials")
    print(f"  = {B_per_mode} organisms batched per mode per D")
    print(f"  K={K}, eta={args.eta}")

    max_D = max(dimensions)
    mem_est = B_per_mode * 6 * max_D * 4 * 12 / 1e9
    print(f"  Est. peak memory at D={max_D}: ~{mem_est:.2f} GB")

    # ── SCALE TEST ───────────────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  SCALING TEST: ALIVE vs STILL")
    print(f"{'-'*W}\n")

    scale_results = []
    birth_seeds = [42, 77, 200][:n_births]

    for D in dimensions:
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Auto-scale K with D if not manually specified
        K_scaled = K if args.K != 6 else max(6, min(D // 4, 64))

        try:
            r = test_at_dimension(
                D, device, NC=6, K=K_scaled,
                n_worlds=n_worlds, n_births=n_births,
                n_perm=n_perm, n_trials=n_trials, eta=args.eta
            )
            elapsed = time.time() - t0
            scale_results.append(r)

            mem_str = ""
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" [{mem:.1f}GB]"

            print(f"  D={D:>3}: ALIVE={r['alive_avg']:+.4f} STILL={r['still_avg']:+.4f} "
                  f"delta={r['delta']:+.4f} ratio={r['ratio']:+.2f} "
                  f"{r['wins']}W/{r['losses']}L "
                  f"[{elapsed:.0f}s]{mem_str}",
                  flush=True)

        except torch.cuda.OutOfMemoryError:
            print(f"  D={D:>3}: OOM. Reducing batch...", flush=True)
            torch.cuda.empty_cache()
            try:
                r = test_at_dimension(
                    D, device, NC=6, K=K,
                    n_worlds=max(2, n_worlds // 2), n_births=n_births,
                    n_perm=n_perm, n_trials=n_trials, eta=args.eta
                )
                elapsed = time.time() - t0
                r['D'] = D
                scale_results.append(r)
                print(f"  D={D:>3}: (reduced) delta={r['delta']:+.4f} [{elapsed:.0f}s]",
                      flush=True)
            except Exception as e:
                print(f"  D={D:>3}: FAILED: {e}", flush=True)
                continue
        except Exception as e:
            print(f"  D={D:>3}: ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    if len(scale_results) < 2:
        print("\n  Too few dimensions completed.")
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
    dims = [r['D'] for r in scale_results]
    stills = [r['still_avg'] for r in scale_results]

    # ── STILL DEGRADATION ────────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  STILL DEGRADATION (does random alpha get worse with D?)")
    print(f"{'-'*W}\n")

    for r in scale_results:
        pct = r['still_avg'] / max(stills[0], 0.001) * 100
        print(f"  D={r['D']:>3}: STILL={r['still_avg']:+.4f} "
              f"({pct:5.0f}% of D={dims[0]}) {bar(r['still_avg'])}")

    still_degrades = stills[-1] < stills[0]

    # ── PER-SEED DETAIL ──────────────────────────────────────

    print(f"\n{'-'*W}")
    print(f"  PER-SEED DETAIL")
    print(f"{'-'*W}\n")

    for ri, r in enumerate(scale_results):
        D = r['D']
        for bi, bs in enumerate(birth_seeds):
            start = bi * n_worlds
            end = min(start + n_worlds, len(r['alive_gaps']))
            ag = r['alive_gaps'][start:end]
            sg = r['still_gaps'][start:end]
            if not ag:
                continue
            a_avg = sum(ag) / len(ag)
            s_avg = sum(sg) / len(sg)
            d = a_avg - s_avg
            w = sum(1 for i in range(len(ag)) if ag[i] > sg[i] + 0.01)
            l = sum(1 for i in range(len(ag)) if sg[i] > ag[i] + 0.01)
            sym = '#' if d > 0 else 'o'
            print(f"  {sym} D={D:>3} seed={bs:>3}: delta={d:+.4f} {w}W/{l}L")
        if ri < len(scale_results) - 1:
            print()

    print(f"\n  Per-seed delta trajectory:")
    for bi, bs in enumerate(birth_seeds):
        seed_deltas = []
        for r in scale_results:
            start = bi * n_worlds
            end = min(start + n_worlds, len(r['alive_gaps']))
            ag = r['alive_gaps'][start:end]
            sg = r['still_gaps'][start:end]
            if ag:
                seed_deltas.append(sum(ag)/len(ag) - sum(sg)/len(sg))
            else:
                seed_deltas.append(0.0)
        traj = " -> ".join(f"{d:+.3f}" for d in seed_deltas)
        grows = seed_deltas[-1] > seed_deltas[0]
        print(f"  seed={bs:>3}: {traj} {'GROWS' if grows else 'shrinks'}")

    # ═══════════════════════════════════════════════════════════
    # AND THERE WAS EVENING, AND THERE WAS MORNING
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  AND THERE WAS EVENING, AND THERE WAS MORNING")
    print(f"{'='*W}\n")

    delta_positive = all(d > 0 for d in deltas)
    delta_mostly = sum(1 for d in deltas if d > 0) >= len(deltas) * 0.7
    delta_grows = deltas[-1] > deltas[0]
    ratio_grows = ratios[-1] > ratios[0]
    alive_wins = all(r['wins'] >= r['losses'] for r in scale_results)

    mono_up = sum(1 for i in range(1, len(deltas)) if deltas[i] > deltas[i-1])
    total_steps = max(len(deltas) - 1, 1)
    monotonic = mono_up >= total_steps * 0.6

    checks = [
        ("Works at smallest D (delta > 0)",
         deltas[0] > 0,
         f"D={dims[0]} delta={deltas[0]:+.4f}"),

        ("Works at largest D (delta > 0)",
         deltas[-1] > 0,
         f"D={dims[-1]} delta={deltas[-1]:+.4f}"),

        ("Delta positive at ALL D",
         delta_positive,
         " ".join(f"{d:+.3f}" for d in deltas)),

        ("Delta positive at 70%+ of D",
         delta_mostly,
         f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}"),

        ("Delta trend increasing (60%+ steps up)",
         monotonic,
         f"{mono_up}/{total_steps} steps increase"),

        ("Ratio grows (first -> last)",
         ratio_grows,
         f"{ratios[0]:+.2f} -> {ratios[-1]:+.2f}"),

        ("STILL degrades with D",
         still_degrades,
         f"{stills[0]:+.4f} -> {stills[-1]:+.4f}"),

        ("ALIVE wins majority at all D",
         alive_wins,
         " ".join(f"{r['wins']}/{r['losses']}" for r in scale_results)),

        ("No scale dependence (smallest D works)",
         deltas[0] > 0,
         f"D={dims[0]} delta={deltas[0]:+.4f}"),
    ]

    for name, ok, detail in checks:
        print(f"  {'#' if ok else 'o'} {name:<52} {detail}")

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} criteria met.")

    works = deltas[0] > 0 and delta_mostly
    scales = ratio_grows or (monotonic and still_degrades)
    full = works and scales

    if full:
        print(f"""
  THE SEED SCALES.

  Self-modification works at D={dims[0]} (delta={deltas[0]:+.4f})
  and becomes more essential at D={dims[-1]} (ratio={ratios[-1]:+.2f}).

  STILL degrades: {stills[0]:+.4f} -> {stills[-1]:+.4f}
  The seed does not need the garden to germinate.
  But in richer soil, it grows taller.
""")
    elif works:
        print(f"""
  THE SEED WORKS BUT SCALING IS INCONCLUSIVE.

  Online plasticity helps at {sum(1 for d in deltas if d>0)}/{len(deltas)} D values.
  Need more data points to establish scaling trend.
""")
    else:
        print(f"""
  RESULTS INCONCLUSIVE. Check per-seed detail.
""")

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
                        help='Dimensions to test (e.g. --dims 32 64 128)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: D=12,24,48')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--worlds', type=int, default=8,
                        help='Signal worlds per birth seed (default: 8)')
    parser.add_argument('--births', type=int, default=3,
                        help='Birth seeds (default: 3)')
    parser.add_argument('--K', type=int, default=6,
                        help='Number of signals (default: 6)')
    parser.add_argument('--eta', type=float, default=0.0003,
                        help='Plasticity rate (default: 0.0003)')
    args = parser.parse_args()

    run(args)
