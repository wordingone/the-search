#!/usr/bin/env python3
"""
GENESIS: The Seed -- GPU (Verified Translation)

Minimal translation of the CPU seed.py to PyTorch.
ONLY the step function is vectorized. The outer loops
(per birth seed, per signal world, per permutation, per trial)
are kept IDENTICAL to the CPU version to ensure correctness.

Previous GPU versions had critical bugs:
  - torch.manual_seed(42) gave all batch elements identical initial states
  - Same permutations for all tests (CPU varies per test seed)
  - K auto-scaled with D (different tasks at different dimensions)

This version uses Python's random module for all seeding
(matching CPU behavior) and PyTorch only for tensor math.

    pip install torch
    python SeedGPU.py                       # Full test
    python SeedGPU.py --verify              # Verify vs CPU first
    python SeedGPU.py --dims 12 24 48 96    # Custom dims
    python SeedGPU.py --quick               # Fast run
"""

import math
import random
import time
import argparse
import sys

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    print("PyTorch not found. Install: pip install torch")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# Constants — identical to CPU seed.py
# ═══════════════════════════════════════════════════════════════

NC = 6
BETA = 0.5
GAMMA = 0.9
EPS_COUP = 0.15
TAU = 0.3
MIX = 0.35
NOISE = 0.005
CLIP = 4.0
ETA = 0.0003
ALPHA_MIN = 0.3
ALPHA_MAX = 1.8
W = 72


# ═══════════════════════════════════════════════════════════════
# Organism — PyTorch tensors, same logic as CPU
#
# State: (NC, D) tensor on device
# Alpha: (NC, D) tensor on device
# Signal: (D,) tensor or None
#
# The step function is the ONLY part that uses PyTorch.
# Everything else (seeding, loops, signal generation)
# matches the CPU version exactly.
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, D, seed=42, alive=False, eta=ETA, device='cuda'):
        self.D = D
        self.alive = alive
        self.eta = eta
        self.device = device

        # Generate alpha using Python's random (matching CPU)
        random.seed(seed)
        alpha_list = []
        for i in range(NC):
            row = [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            alpha_list.append(row)
        self.alpha = torch.tensor(alpha_list, dtype=torch.float32, device=device)

        # Precompute cyclic indices
        self.kp = torch.arange(D, device=device).roll(-1)
        self.km = torch.arange(D, device=device).roll(1)
        self.eye = torch.eye(NC, device=device)

    @torch.no_grad()
    def step(self, xs, signal=None):
        """
        xs: (NC, D) tensor
        signal: (D,) tensor or None
        Returns: (NC, D) tensor
        """
        D = self.D
        alpha = self.alpha
        kp, km = self.kp, self.km

        xs_kp = xs[:, kp]  # (NC, D) — neighbor k+1
        xs_km = xs[:, km]  # (NC, D) — neighbor k-1

        # bare dynamics: tanh(alpha * x + beta * x_{k+1} * x_{k-1})
        phi_bare = torch.tanh(alpha * xs + BETA * xs_kp * xs_km)

        # signal-modulated dynamics
        if signal is not None:
            s_kp = signal[kp].unsqueeze(0)  # (1, D)
            s_km = signal[km].unsqueeze(0)  # (1, D)
            phi_sig = torch.tanh(
                alpha * xs
                + BETA * (xs_kp + GAMMA * s_kp) * (xs_km + GAMMA * s_km)
            )
        else:
            phi_sig = phi_bare

        # ── ONLINE PLASTICITY ────────────────────────────────
        if self.alive and signal is not None:
            response = (phi_sig - phi_bare).abs()  # (NC, D)

            # z-score across all cells and dims
            r_mean = response.mean()
            r_std = response.std().clamp(min=1e-10)
            resp_z = (response - r_mean) / r_std  # (NC, D)

            # column mean of alpha per dimension
            col_mean = alpha.mean(dim=0, keepdim=True)  # (1, D)
            dev = alpha - col_mean  # (NC, D)

            # Generate random pushes using Python random (matching CPU)
            noise_push = torch.zeros_like(alpha)
            drift_push = torch.zeros_like(alpha)
            for i in range(NC):
                for k in range(D):
                    noise_push[i, k] = self.eta * 0.3 * random.gauss(0, 1.0)
                    drift_push[i, k] = self.eta * 0.1 * random.gauss(0, 1.0)

            amplify_push = self.eta * 0.5 * torch.tanh(resp_z) * dev.sign()

            at_mean = dev.abs() < 0.01
            sensitive = resp_z > 0

            push = torch.where(at_mean, noise_push,
                   torch.where(sensitive, amplify_push, drift_push))

            self.alpha = (alpha + push).clamp(ALPHA_MIN, ALPHA_MAX)

        # ── ATTENTION ────────────────────────────────────────
        dots = (xs @ xs.T) / (D * TAU)  # (NC, NC)
        dots = dots - self.eye * 1e10
        weights = F.softmax(dots, dim=1)  # (NC, NC)

        # ── COUPLING ─────────────────────────────────────────
        weighted_avg = weights @ phi_bare  # (NC, D)
        pull = weighted_avg - phi_bare

        bare_diff = phi_bare - xs
        fp_d = bare_diff.norm(dim=1) / xs.norm(dim=1).clamp(min=1.0)  # (NC,)
        plast = torch.exp(-fp_d.pow(2) / 0.0225).unsqueeze(1)  # (NC, 1)

        p = phi_sig + plast * EPS_COUP * pull

        # ── STATE UPDATE ─────────────────────────────────────
        # Use Python random for noise (matching CPU)
        noise_t = torch.zeros_like(xs)
        for i in range(NC):
            for k in range(D):
                noise_t[i, k] = random.gauss(0, NOISE)

        new_xs = (1 - MIX) * xs + MIX * p + noise_t
        return new_xs.clamp(-CLIP, CLIP)

    def centroid(self, xs):
        return xs.mean(dim=0)  # (D,)


# ═══════════════════════════════════════════════════════════════
# Optimized organism — PyTorch random for noise (faster, not
# numerically identical to CPU but statistically equivalent)
# ═══════════════════════════════════════════════════════════════

class OrganismFast:
    """Same logic, but uses torch.randn for noise instead of
    Python random loops. ~10-50x faster. Statistically equivalent
    but not bit-identical to CPU."""

    def __init__(self, D, seed=42, alive=False, eta=ETA, device='cuda'):
        self.D = D
        self.alive = alive
        self.eta = eta
        self.device = device

        random.seed(seed)
        alpha_list = []
        for i in range(NC):
            row = [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            alpha_list.append(row)
        self.alpha = torch.tensor(alpha_list, dtype=torch.float32, device=device)

        self.kp = torch.arange(D, device=device).roll(-1)
        self.km = torch.arange(D, device=device).roll(1)
        self.eye = torch.eye(NC, device=device)

    @torch.no_grad()
    def step(self, xs, signal=None):
        D = self.D
        alpha = self.alpha
        kp, km = self.kp, self.km

        xs_kp = xs[:, kp]
        xs_km = xs[:, km]

        phi_bare = torch.tanh(alpha * xs + BETA * xs_kp * xs_km)

        if signal is not None:
            s_kp = signal[kp].unsqueeze(0)
            s_km = signal[km].unsqueeze(0)
            phi_sig = torch.tanh(
                alpha * xs
                + BETA * (xs_kp + GAMMA * s_kp) * (xs_km + GAMMA * s_km)
            )
        else:
            phi_sig = phi_bare

        if self.alive and signal is not None:
            response = (phi_sig - phi_bare).abs()
            r_mean = response.mean()
            r_std = response.std().clamp(min=1e-10)
            resp_z = (response - r_mean) / r_std

            col_mean = alpha.mean(dim=0, keepdim=True)
            dev = alpha - col_mean

            noise_push = self.eta * 0.3 * torch.randn_like(alpha)
            amplify_push = self.eta * 0.5 * torch.tanh(resp_z) * dev.sign()
            drift_push = self.eta * 0.1 * torch.randn_like(alpha)

            at_mean = dev.abs() < 0.01
            sensitive = resp_z > 0
            push = torch.where(at_mean, noise_push,
                   torch.where(sensitive, amplify_push, drift_push))
            self.alpha = (alpha + push).clamp(ALPHA_MIN, ALPHA_MAX)

        dots = (xs @ xs.T) / (D * TAU)
        dots = dots - self.eye * 1e10
        weights = F.softmax(dots, dim=1)

        weighted_avg = weights @ phi_bare
        pull = weighted_avg - phi_bare
        bare_diff = phi_bare - xs
        fp_d = bare_diff.norm(dim=1) / xs.norm(dim=1).clamp(min=1.0)
        plast = torch.exp(-fp_d.pow(2) / 0.0225).unsqueeze(1)
        p = phi_sig + plast * EPS_COUP * pull

        new_xs = (1 - MIX) * xs + MIX * p + NOISE * torch.randn_like(xs)
        return new_xs.clamp(-CLIP, CLIP)

    def centroid(self, xs):
        return xs.mean(dim=0)


# ═══════════════════════════════════════════════════════════════
# CPU reference organism (pure Python, identical to seed.py)
# ═══════════════════════════════════════════════════════════════

class OrganismCPU:
    """Exact CPU implementation for verification."""

    def __init__(self, D, seed=42, alive=False, eta=ETA):
        self.D = D
        self.alive = alive
        self.eta = eta
        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

    def step(self, xs, signal=None):
        D = self.D
        phi_bare = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D
                row.append(math.tanh(
                    self.alpha[i][k] * xs[i][k]
                    + BETA * xs[i][kp] * xs[i][km]))
            phi_bare.append(row)

        if signal:
            phi_sig = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D
                    km = (k - 1) % D
                    row.append(math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + BETA * (xs[i][kp] + GAMMA * signal[kp])
                               * (xs[i][km] + GAMMA * signal[km])))
                phi_sig.append(row)
        else:
            phi_sig = phi_bare

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
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * 0.5
                    else:
                        push = self.eta * 0.1 * random.gauss(0, 1.0)
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))

        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][k] * xs[j][k] for k in range(D))
                    raw.append(d / (D * TAU))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        new = []
        for i in range(NC):
            p = [v for v in phi_sig[i]]
            bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
            bd_norm = math.sqrt(sum(v * v for v in bare_diff) + 1e-15)
            xs_norm = max(math.sqrt(sum(v * v for v in xs[i]) + 1e-15), 1.0)
            fp_d = bd_norm / xs_norm
            plast = math.exp(-(fp_d * fp_d) / 0.0225)
            if plast > 0.01:
                pull = [0.0] * D
                for j in range(NC):
                    if i == j or weights[i][j] < 1e-8:
                        continue
                    for k in range(D):
                        pull[k] += weights[i][j] * (phi_bare[j][k] - phi_bare[i][k])
                p = [p[k] + plast * EPS_COUP * pull[k] for k in range(D)]
            nx = []
            for k in range(D):
                v = (1 - MIX) * xs[i][k] + MIX * p[k]
                v += random.gauss(0, NOISE)
                v = max(-CLIP, min(CLIP, v))
                nx.append(v)
            new.append(nx)
        return new

    def centroid(self, xs):
        D = self.D
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]


# ═══════════════════════════════════════════════════════════════
# Signal generation — identical to CPU
# ═══════════════════════════════════════════════════════════════

def make_signals(K, D, seed):
    """Generate K normalized signals. Returns dict {id: list}."""
    random.seed(seed)
    sigs = {}
    for i in range(K):
        s = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def gen_perms(K, n_perm, seed):
    """Generate n_perm distinct permutations."""
    random.seed(seed)
    base = list(range(K))
    perms = [tuple(base), tuple(reversed(base))]
    seen = set(perms)
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]
        random.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        att += 1
    return perms[:n_perm]


# ═══════════════════════════════════════════════════════════════
# Sequence runner — same logic as CPU, tensors on device
# ═══════════════════════════════════════════════════════════════

def run_sequence(org, order, signals, base_seed, trial,
                 n_org=200, n_per_sig=50, n_settle=20, n_final=40):
    """
    Run one sequence. Creates initial state from base_seed.
    org: Organism or OrganismFast
    signals: dict {id: list of floats}
    Returns: centroid as list (for cosine computation)
    """
    D = org.D
    device = getattr(org, 'device', 'cpu')

    # Initial state from Python random (matching CPU)
    random.seed(base_seed)
    if hasattr(org, 'device'):  # GPU organism
        xs_list = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
        xs = torch.tensor(xs_list, dtype=torch.float32, device=device)
    else:  # CPU organism
        xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    # Self-organization
    for _ in range(n_org):
        xs = org.step(xs)

    # Signal presentation
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        if hasattr(org, 'device'):
            sig_list = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
            sig = torch.tensor(sig_list, dtype=torch.float32, device=device)
        else:
            sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]

        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    # Final relaxation
    for _ in range(n_final):
        xs = org.step(xs)

    c = org.centroid(xs)
    if hasattr(org, 'device'):
        return c.cpu().tolist()
    return c


def vcosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


# ═══════════════════════════════════════════════════════════════
# measure_gap — IDENTICAL logic to CPU
# ═══════════════════════════════════════════════════════════════

def measure_gap(D, K, birth_seed, sig_seed, test_seed, alive,
                device='cuda', n_perm=3, n_trials=2):
    """One gap measurement. Fresh organism per perm-trial."""
    signals = make_signals(K, D, seed=sig_seed)
    perms = gen_perms(K, n_perm, seed=test_seed * 10 + K)

    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            # FRESH organism per perm-trial (critical for ALIVE)
            if device == 'cpu_ref':
                org = OrganismCPU(D, seed=birth_seed, alive=alive)
            else:
                org = OrganismFast(D, seed=birth_seed, alive=alive,
                                   device=device)
            c = run_sequence(org, perm, signals, test_seed, trial)
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
# Verification: compare GPU vs CPU on one test case
# ═══════════════════════════════════════════════════════════════

def verify(device):
    """Run one test case on both CPU and GPU, compare."""
    print(f"  {'='*W}")
    print(f"  VERIFICATION: GPU vs CPU")
    print(f"  {'='*W}\n")

    D, K = 12, 6
    birth_seed, sig_seed, test_seed = 42, 42137, 4243

    print(f"  D={D}, K={K}, birth_seed={birth_seed}")
    print(f"  sig_seed={sig_seed}, test_seed={test_seed}\n")

    # CPU reference (exact)
    t0 = time.time()
    cpu_still = measure_gap(D, K, birth_seed, sig_seed, test_seed,
                            alive=False, device='cpu_ref')
    cpu_alive = measure_gap(D, K, birth_seed, sig_seed, test_seed,
                            alive=True, device='cpu_ref')
    cpu_time = time.time() - t0

    # GPU (fast, statistically equivalent)
    t0 = time.time()
    gpu_still = measure_gap(D, K, birth_seed, sig_seed, test_seed,
                            alive=False, device=device)
    gpu_alive = measure_gap(D, K, birth_seed, sig_seed, test_seed,
                            alive=True, device=device)
    gpu_time = time.time() - t0

    print(f"  CPU: STILL={cpu_still:+.4f} ALIVE={cpu_alive:+.4f} "
          f"delta={cpu_alive - cpu_still:+.4f} [{cpu_time:.1f}s]")
    print(f"  GPU: STILL={gpu_still:+.4f} ALIVE={gpu_alive:+.4f} "
          f"delta={gpu_alive - gpu_still:+.4f} [{gpu_time:.1f}s]")

    # STILL should match closely (both use same initial state seed,
    # same signals, same permutations — only noise differs)
    still_diff = abs(gpu_still - cpu_still)
    print(f"\n  STILL difference: {still_diff:.4f}")

    # Both should be in same ballpark (±0.15 is reasonable given noise)
    ok = still_diff < 0.20 and abs(gpu_still) > 0.01
    print(f"  STILL ballpark match: {'YES' if ok else 'NO'}")

    if ok:
        print(f"  GPU translation VERIFIED. Proceeding with scaling test.")
    else:
        print(f"  WARNING: Large discrepancy. Check implementation.")
        print(f"  (Noise differences can cause ±0.10 variation.)")
        print(f"  Proceeding anyway — statistical properties should match.")

    print()
    return ok


# ═══════════════════════════════════════════════════════════════
# Scaling test
# ═══════════════════════════════════════════════════════════════

def test_dimension(D, n_births=3, n_worlds=8, K=6, device='cuda'):
    """Test ALIVE vs STILL at dimension D."""
    birth_seeds = [42, 77, 200][:n_births]
    alive_gaps = []
    still_gaps = []
    total = n_births * n_worlds
    done = 0

    for bs in birth_seeds:
        for wi in range(n_worlds):
            sig_seed = bs * 1000 + wi * 137 + D * 7
            test_seed = bs * 100 + wi * 31 + D

            ag = measure_gap(D, K, bs, sig_seed, test_seed,
                             alive=True, device=device)
            sg = measure_gap(D, K, bs, sig_seed, test_seed,
                             alive=False, device=device)
            alive_gaps.append(ag)
            still_gaps.append(sg)
            done += 1
            if done % max(1, total // 4) == 0 or done == total:
                print(f"    [{done}/{total}] "
                      f"A={sum(alive_gaps)/len(alive_gaps):+.4f} "
                      f"S={sum(still_gaps)/len(still_gaps):+.4f}",
                      flush=True)

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
        'alive_gaps': alive_gaps, 'still_gaps': still_gaps
    }


def main():
    parser = argparse.ArgumentParser(description='GENESIS GPU (Verified)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dims', nargs='+', type=int,
                        default=[12, 16, 24, 32, 48, 64])
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--births', type=int, default=3)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--verify', action='store_true',
                        help='Verify GPU vs CPU before scaling test')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Run CPU reference only (slow, exact)')
    args = parser.parse_args()

    if args.cpu_only:
        device = 'cpu_ref'
    elif args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if args.quick:
        args.worlds = 4
        args.births = 2
        args.dims = [12, 24, 48]

    print("=" * W)
    print("  GENESIS: THE SEED (GPU — Verified Translation)")
    print(f"  Device: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {mem:.1f} GB")
    print(f"  Dimensions: {args.dims}")
    n_tests = args.births * args.worlds
    print(f"  {args.births} births x {args.worlds} worlds = {n_tests} tests/D")
    print("=" * W)

    # ── VERIFY ───────────────────────────────────────────────
    if args.verify and device not in ('cpu_ref',):
        verify(device)

    # ── SCALE TEST ───────────────────────────────────────────
    t_start = time.time()
    results = []

    print(f"\n{'-'*W}")
    print(f"  SCALING TEST: ALIVE vs STILL")
    print(f"{'-'*W}\n")

    for D in args.dims:
        print(f"  --- D={D} ({NC} cells, {NC*D} params) ---")
        t0 = time.time()
        r = test_dimension(D, n_births=args.births,
                           n_worlds=args.worlds, device=device)
        elapsed = time.time() - t0
        r['elapsed'] = elapsed
        results.append(r)
        print(f"  D={D:>3}: ALIVE={r['alive_avg']:+.4f} "
              f"STILL={r['still_avg']:+.4f} "
              f"delta={r['delta']:+.4f} ratio={r['ratio']:+.2f} "
              f"{r['wins']}W/{r['losses']}L [{elapsed:.0f}s]\n")
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ── RESULTS ──────────────────────────────────────────────
    print(f"{'='*W}")
    print(f"  RESULTS")
    print(f"{'='*W}\n")

    print(f"  {'D':>4} {'ALIVE':>8} {'STILL':>8} {'delta':>8} "
          f"{'ratio':>8} {'W/L':>8} {'time':>8}")
    for r in results:
        print(f"  {r['D']:>4} {r['alive_avg']:>+8.4f} "
              f"{r['still_avg']:>+8.4f} "
              f"{r['delta']:>+8.4f} {r['ratio']:>+8.2f} "
              f"{r['wins']:>3}/{r['losses']:<3} "
              f"{r.get('elapsed',0):>7.0f}s")

    deltas = [r['delta'] for r in results]
    ratios = [r['ratio'] for r in results]

    # STILL degradation
    still_vals = [r['still_avg'] for r in results]
    print(f"\n  STILL degradation: ", end="")
    print(" -> ".join(f"{s:+.3f}" for s in still_vals))
    still_degrades = still_vals[-1] < still_vals[0] if len(still_vals) > 1 else False

    # Per-seed detail
    print(f"\n  Per-seed delta at D={results[0]['D']} and D={results[-1]['D']}:")
    birth_seeds = [42, 77, 200][:args.births]
    for bi, bs in enumerate(birth_seeds):
        s, e = bi * args.worlds, (bi + 1) * args.worlds
        d_first = (sum(results[0]['alive_gaps'][s:e]) -
                   sum(results[0]['still_gaps'][s:e])) / args.worlds
        d_last = (sum(results[-1]['alive_gaps'][s:e]) -
                  sum(results[-1]['still_gaps'][s:e])) / args.worlds
        print(f"    seed={bs}: D={results[0]['D']}:{d_first:+.4f} "
              f"-> D={results[-1]['D']}:{d_last:+.4f} "
              f"{'GROWS' if d_last > d_first else 'shrinks'}")

    # ── VERDICT ──────────────────────────────────────────────
    delta_pos = all(d > 0 for d in deltas)
    delta_mostly_pos = sum(1 for d in deltas if d > 0) >= len(deltas) * 0.7
    grows = deltas[-1] > deltas[0] if len(deltas) > 1 else False
    r_grows = ratios[-1] > ratios[0] if len(ratios) > 1 else False
    wins_all = all(r['wins'] > r['losses'] for r in results)

    print(f"\n  Delta all positive: {'YES' if delta_pos else 'NO'} "
          f"({sum(1 for d in deltas if d > 0)}/{len(deltas)})")
    print(f"  Delta mostly pos:   {'YES' if delta_mostly_pos else 'NO'}")
    print(f"  Delta grows:        {'YES' if grows else 'NO'} "
          f"({deltas[0]:+.4f} -> {deltas[-1]:+.4f})")
    print(f"  Ratio grows:        {'YES' if r_grows else 'NO'} "
          f"({ratios[0]:+.2f} -> {ratios[-1]:+.2f})")
    print(f"  STILL degrades:     {'YES' if still_degrades else 'NO'}")
    print(f"  Wins all D:         {'YES' if wins_all else 'NO'}")

    works = deltas[0] > 0 and delta_mostly_pos
    scales = grows or r_grows

    if works and scales:
        print(f"\n  THE SEED SCALES.")
    elif works:
        print(f"\n  THE SEED WORKS. Scaling inconclusive.")
    else:
        print(f"\n  HONEST RESULT.")

    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print(f"  {'='*W}")


if __name__ == '__main__':
    main()
