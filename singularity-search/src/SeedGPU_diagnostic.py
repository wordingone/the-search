#!/usr/bin/env python3
"""
SeedGPU Diagnostic: Measure scaling degradation causes

Instruments OrganismFast to track:
- Tanh saturation rates
- Response signal magnitude
- Attention entropy
- Alpha shift magnitude

Run: python SeedGPU_diagnostic.py --dims 12 24 48
"""

import math
import random
import time
import argparse
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)

# Constants
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


class OrganismDiagnostic:
    """Instrumented organism that tracks scaling metrics."""

    def __init__(self, D, seed=42, alive=False, eta=ETA, device='cuda'):
        self.D = D
        self.alive = alive
        self.eta = eta
        self.device = device

        # Diagnostic counters
        self.step_count = 0
        self.saturation_sum = 0.0
        self.response_sum = 0.0
        self.alpha_shift_sum = 0.0
        self.attention_entropy_sum = 0.0

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

        # DIAGNOSTIC: Measure saturation
        pre_tanh_bare = alpha * xs + BETA * xs_kp * xs_km
        saturation_rate = (pre_tanh_bare.abs() > 2.0).float().mean().item()
        self.saturation_sum += saturation_rate

        phi_bare = torch.tanh(pre_tanh_bare)

        if signal is not None:
            s_kp = signal[kp].unsqueeze(0)
            s_km = signal[km].unsqueeze(0)
            pre_tanh_sig = (alpha * xs
                           + BETA * (xs_kp + GAMMA * s_kp) * (xs_km + GAMMA * s_km))
            phi_sig = torch.tanh(pre_tanh_sig)
        else:
            phi_sig = phi_bare

        if self.alive and signal is not None:
            response = (phi_sig - phi_bare).abs()

            # DIAGNOSTIC: Response magnitude
            self.response_sum += response.mean().item()

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

            # DIAGNOSTIC: Alpha shift magnitude
            self.alpha_shift_sum += push.abs().mean().item()

            self.alpha = (alpha + push).clamp(ALPHA_MIN, ALPHA_MAX)

        # DIAGNOSTIC: Attention entropy
        dots = (xs @ xs.T) / (D * TAU)
        dots = dots - self.eye * 1e10
        weights = F.softmax(dots, dim=1)

        # Entropy per row, then mean
        entropy_per_row = -(weights * (weights + 1e-10).log()).sum(dim=1)
        self.attention_entropy_sum += entropy_per_row.mean().item()

        weighted_avg = weights @ phi_bare
        pull = weighted_avg - phi_bare
        bare_diff = phi_bare - xs
        fp_d = bare_diff.norm(dim=1) / xs.norm(dim=1).clamp(min=1.0)
        plast = torch.exp(-fp_d.pow(2) / 0.0225).unsqueeze(1)
        p = phi_sig + plast * EPS_COUP * pull

        new_xs = (1 - MIX) * xs + MIX * p + NOISE * torch.randn_like(xs)

        self.step_count += 1
        return new_xs.clamp(-CLIP, CLIP)

    def centroid(self, xs):
        return xs.mean(dim=0)

    def get_diagnostics(self):
        """Return average diagnostics since last reset."""
        if self.step_count == 0:
            return None
        return {
            'saturation_rate': self.saturation_sum / self.step_count,
            'response_magnitude': self.response_sum / max(1, self.step_count - 200),  # Exclude self-org
            'alpha_shift': self.alpha_shift_sum / max(1, self.step_count - 200),
            'attention_entropy': self.attention_entropy_sum / self.step_count,
            'steps': self.step_count
        }


def make_signals(K, D, seed):
    random.seed(seed)
    sigs = {}
    for i in range(K):
        s = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def gen_perms(K, n_perm, seed):
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


def run_sequence(org, order, signals, base_seed, trial,
                 n_org=200, n_per_sig=50, n_settle=20, n_final=40):
    D = org.D
    device = org.device

    random.seed(base_seed)
    xs_list = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    xs = torch.tensor(xs_list, dtype=torch.float32, device=device)

    for _ in range(n_org):
        xs = org.step(xs)

    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig_list = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        sig = torch.tensor(sig_list, dtype=torch.float32, device=device)

        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    for _ in range(n_final):
        xs = org.step(xs)

    c = org.centroid(xs)
    return c.cpu().tolist()


def vcosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def measure_gap_diagnostic(D, K, birth_seed, sig_seed, test_seed, alive,
                           device='cuda', n_perm=3, n_trials=2):
    """Measure gap and collect diagnostics."""
    signals = make_signals(K, D, seed=sig_seed)
    perms = gen_perms(K, n_perm, seed=test_seed * 10 + K)

    endpoints = {}
    all_diagnostics = []

    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            org = OrganismDiagnostic(D, seed=birth_seed, alive=alive, device=device)
            c = run_sequence(org, perm, signals, test_seed, trial)
            trials.append(c)

            diag = org.get_diagnostics()
            if diag:
                all_diagnostics.append(diag)
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
    gap = avg_w - avg_b

    # Average diagnostics
    avg_diag = {}
    if all_diagnostics:
        for key in all_diagnostics[0]:
            avg_diag[key] = sum(d[key] for d in all_diagnostics) / len(all_diagnostics)

    return gap, avg_diag


def test_dimension_diagnostic(D, device='cuda'):
    """Run one test per condition at dimension D, collect diagnostics."""
    print(f"\n  {'-'*W}")
    print(f"  D={D} DIAGNOSTIC")
    print(f"  {'-'*W}\n")

    K = 6
    birth_seed = 42
    sig_seed = birth_seed * 1000 + D * 7
    test_seed = birth_seed * 100 + D

    # STILL
    print(f"  Running STILL (alpha fixed)...")
    gap_still, diag_still = measure_gap_diagnostic(
        D, K, birth_seed, sig_seed, test_seed, alive=False, device=device
    )

    # ALIVE
    print(f"  Running ALIVE (alpha adaptive)...")
    gap_alive, diag_alive = measure_gap_diagnostic(
        D, K, birth_seed, sig_seed, test_seed, alive=True, device=device
    )

    print(f"\n  {'-'*40}")
    print(f"  RESULTS D={D}")
    print(f"  {'-'*40}")
    print(f"  Gap: STILL={gap_still:+.4f} ALIVE={gap_alive:+.4f} delta={gap_alive - gap_still:+.4f}")
    print(f"\n  STILL diagnostics:")
    for k, v in diag_still.items():
        if k != 'steps':
            print(f"    {k:25s}: {v:.6f}")

    print(f"\n  ALIVE diagnostics:")
    for k, v in diag_alive.items():
        if k != 'steps':
            print(f"    {k:25s}: {v:.6f}")

    # Compute ratios
    print(f"\n  ALIVE / STILL ratios:")
    for key in diag_alive:
        if key != 'steps' and diag_still[key] > 1e-10:
            ratio = diag_alive[key] / diag_still[key]
            print(f"    {key:25s}: {ratio:.3f}x")

    return {
        'D': D,
        'gap_still': gap_still,
        'gap_alive': gap_alive,
        'delta': gap_alive - gap_still,
        'diag_still': diag_still,
        'diag_alive': diag_alive
    }


def main():
    parser = argparse.ArgumentParser(description='SeedGPU Scaling Diagnostics')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dims', nargs='+', type=int, default=[12, 24, 48])
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * W)
    print("  SEEDGPU SCALING DIAGNOSTICS")
    print(f"  Device: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Testing dimensions: {args.dims}")
    print("=" * W)

    results = []
    for D in args.dims:
        r = test_dimension_diagnostic(D, device=device)
        results.append(r)
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Summary comparison
    print(f"\n{'='*W}")
    print(f"  CROSS-DIMENSION COMPARISON")
    print(f"{'='*W}\n")

    print(f"  {'D':>4} {'delta':>8} {'sat_A':>8} {'sat_S':>8} {'resp_A':>9} {'attn_ent_A':>11}")
    print(f"  {'-'*W}")
    for r in results:
        print(f"  {r['D']:>4} {r['delta']:>+8.4f} "
              f"{r['diag_alive']['saturation_rate']:>8.4f} "
              f"{r['diag_still']['saturation_rate']:>8.4f} "
              f"{r['diag_alive']['response_magnitude']:>9.6f} "
              f"{r['diag_alive']['attention_entropy']:>11.6f}")

    # Hypothesis testing
    print(f"\n  HYPOTHESIS CHECKS:")
    print(f"  {'-'*W}")

    # H1: Saturation increases with D?
    sats = [r['diag_alive']['saturation_rate'] for r in results]
    sat_increases = all(sats[i] <= sats[i+1] for i in range(len(sats)-1))
    print(f"  H4 (Saturation): {sats[0]:.4f} -> {sats[-1]:.4f} "
          f"{'INCREASES' if sat_increases else 'mixed'}")

    # H2: Response magnitude decreases with D?
    resps = [r['diag_alive']['response_magnitude'] for r in results]
    resp_decreases = all(resps[i] >= resps[i+1] for i in range(len(resps)-1))
    print(f"  H1 (Response):   {resps[0]:.6f} -> {resps[-1]:.6f} "
          f"{'DECREASES' if resp_decreases else 'mixed'}")

    # H3: Attention entropy increases (flatter)?
    ents = [r['diag_alive']['attention_entropy'] for r in results]
    ent_increases = all(ents[i] <= ents[i+1] for i in range(len(ents)-1))
    max_ent = math.log(NC - 1)  # Maximum possible entropy
    print(f"  H2 (Attn blur):  {ents[0]:.4f} -> {ents[-1]:.4f} (max={max_ent:.4f}) "
          f"{'INCREASES' if ent_increases else 'mixed'}")

    # Alpha shift
    shifts = [r['diag_alive']['alpha_shift'] for r in results]
    print(f"  Alpha shift:     {shifts[0]:.6f} -> {shifts[-1]:.6f}")

    print(f"\n  INTERPRETATION:")
    if sat_increases and resps[0] > 2 * resps[-1]:
        print(f"  Primary cause: Tanh saturation + weak response signal")
        print(f"  Recommendation: Scale beta by 1/sqrt(D) to reduce saturation")
    elif resp_decreases and resps[0] > 3 * resps[-1]:
        print(f"  Primary cause: Response signal vanishes at high D")
        print(f"  Recommendation: Scale eta by sqrt(D) to maintain effective learning rate")
    elif ent_increases and ents[-1] / max_ent > 0.8:
        print(f"  Primary cause: Attention becomes too diffuse")
        print(f"  Recommendation: Remove D normalization from attention or scale tau")
    else:
        print(f"  Cause unclear from diagnostics. Multiple factors may interact.")

    print(f"\n  {'='*W}")


if __name__ == '__main__':
    main()
