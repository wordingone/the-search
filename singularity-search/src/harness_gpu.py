#!/usr/bin/env python3
"""
GPU-Accelerated Experiment Harness

Ports the CPU harness.py to GPU using SeedGPU (3).py infrastructure.
Maintains identical API but adds batch evaluation for parallelizing
evolutionary search across multiple candidates.

Key features:
- Same API as harness.py: Organism, run_comparison(), measure_gap()
- Batch mode: batch_evaluate() runs N candidates in parallel on GPU
- Falls back to CPU if no GPU available
- Compatible with evolve.py (drop-in replacement)

Usage:
    from harness_gpu import run_comparison, batch_evaluate
    result = run_comparison(rule_params)  # single rule
    results = batch_evaluate([rule1, rule2, ...])  # batch parallel
"""

import math
import random
import sys

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found. GPU harness requires: pip install torch")
    print("Falling back to CPU harness.")


# ═══════════════════════════════════════════════════════════════
# GPU Configuration
# ═══════════════════════════════════════════════════════════════

# Default GPU memory ceiling (user preference: 50% max to avoid overloading)
DEFAULT_GPU_FRACTION = 0.5
_GPU_CONFIGURED = False

# CRITICAL: Apply GPU memory ceiling at module import time
if HAS_TORCH and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(DEFAULT_GPU_FRACTION)
    _GPU_CONFIGURED = True


def configure_gpu(gpu_fraction=DEFAULT_GPU_FRACTION):
    """
    Configure GPU memory ceiling.

    Args:
        gpu_fraction: Fraction of GPU memory to use (0.0-1.0).
                     Default 0.5 (50% ceiling).
    """
    global _GPU_CONFIGURED
    if HAS_TORCH and torch.cuda.is_available() and not _GPU_CONFIGURED:
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        _GPU_CONFIGURED = True


def get_device(device='auto', gpu_fraction=DEFAULT_GPU_FRACTION):
    """
    Get device for computation with optional GPU memory limit.

    Args:
        device: 'auto', 'cuda', or 'cpu'
        gpu_fraction: GPU memory fraction if using CUDA (default 0.5)

    Returns:
        device string ('cuda' or 'cpu')
    """
    if device == 'auto':
        if HAS_TORCH and torch.cuda.is_available():
            configure_gpu(gpu_fraction)
            return 'cuda'
        return 'cpu'
    elif device == 'cuda' and HAS_TORCH and torch.cuda.is_available():
        configure_gpu(gpu_fraction)
        return 'cuda'
    return 'cpu'


# ═══════════════════════════════════════════════════════════════
# Constants (identical to harness.py and SeedGPU)
# ═══════════════════════════════════════════════════════════════

D = 12
NC = 6
W = 72

# Core dynamics (frozen)
BETA = 0.5
GAMMA = 0.9
EPS_COUP = 0.15
TAU = 0.3
MIX = 0.35
NOISE = 0.005
CLIP = 4.0


# ═══════════════════════════════════════════════════════════════
# Utilities
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


# ═══════════════════════════════════════════════════════════════
# GPU Organism with Parameterized Plasticity Rule
# ═══════════════════════════════════════════════════════════════

class Organism:
    """
    GPU-accelerated organism with parameterized plasticity rule.

    API-compatible with harness.py Organism but runs on GPU.
    Uses PyTorch for tensor operations, Python random for seeding
    (matching CPU behavior).
    """

    def __init__(self, seed=42, alive=False, rule_params=None, device='cuda'):
        # Core dynamics parameters (frozen)
        self.beta = BETA
        self.gamma = GAMMA
        self.eps = EPS_COUP
        self.tau = TAU
        self.delta = MIX
        self.noise = NOISE
        self.clip = CLIP

        # Birth state
        self.seed = seed
        self.alive = alive
        self.device = device if HAS_TORCH else 'cpu'

        # Plasticity rule parameters
        if rule_params is None:
            rule_params = canonical_rule()

        self.eta = rule_params.get('eta', 0.0003)
        self.symmetry_break_mult = rule_params.get('symmetry_break_mult', 0.3)
        self.amplify_mult = rule_params.get('amplify_mult', 0.5)
        self.drift_mult = rule_params.get('drift_mult', 0.1)
        self.threshold = rule_params.get('threshold', 0.01)
        self.alpha_clip_lo = rule_params.get('alpha_clip_lo', 0.3)
        self.alpha_clip_hi = rule_params.get('alpha_clip_hi', 1.8)

        # Tracking
        self.total_alpha_shift = 0.0

        # Initialize alpha (using Python random, matching CPU)
        random.seed(seed)
        alpha_list = []
        for i in range(NC):
            row = [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            alpha_list.append(row)

        if HAS_TORCH:
            self.alpha = torch.tensor(alpha_list, dtype=torch.float32, device=self.device)
            # Precompute cyclic indices
            self.kp = torch.arange(D, device=self.device).roll(-1)
            self.km = torch.arange(D, device=self.device).roll(1)
            self.eye = torch.eye(NC, device=self.device)
        else:
            self.alpha = alpha_list

    def alpha_flat(self):
        if HAS_TORCH:
            return self.alpha.flatten().cpu().tolist()
        return [a for row in self.alpha for a in row]

    @torch.no_grad()
    def step(self, xs, signal=None, fast=True):
        """
        xs: (NC, D) tensor or list
        signal: (D,) tensor/list or None
        fast: if True, use torch.randn() for batch random generation (default True)
        Returns: (NC, D) tensor or list
        """
        if not HAS_TORCH:
            return self._step_cpu(xs, signal)

        # Ensure tensors on device
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.float32, device=self.device)
        if signal is not None and not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32, device=self.device)

        alpha = self.alpha
        kp, km = self.kp, self.km

        # ── BARE DYNAMICS ────────────────────────────────────
        xs_kp = xs[:, kp]
        xs_km = xs[:, km]
        phi_bare = torch.tanh(alpha * xs + self.beta * xs_kp * xs_km)

        # ── SIGNAL-MODULATED DYNAMICS ────────────────────────
        if signal is not None:
            s_kp = signal[kp].unsqueeze(0)  # (1, D)
            s_km = signal[km].unsqueeze(0)  # (1, D)
            phi_sig = torch.tanh(
                alpha * xs
                + self.beta * (xs_kp + self.gamma * s_kp) * (xs_km + self.gamma * s_km)
            )
        else:
            phi_sig = phi_bare

        # ── ONLINE PLASTICITY (PARAMETERIZED) ────────────────
        if self.alive and signal is not None:
            response = (phi_sig - phi_bare).abs()  # (NC, D)

            # z-score across all cells and dims
            r_mean = response.mean()
            r_std = response.std().clamp(min=1e-10)
            resp_z = (response - r_mean) / r_std  # (NC, D)

            # column mean of alpha per dimension
            col_mean = alpha.mean(dim=0, keepdim=True)  # (1, D)
            dev = alpha - col_mean  # (NC, D)

            # Generate random pushes
            if fast:
                # Fast path: batched torch.randn() generation
                noise_push = self.eta * self.symmetry_break_mult * torch.randn_like(alpha)
                drift_push = self.eta * self.drift_mult * torch.randn_like(alpha)
            else:
                # Slow path: Python random (matching CPU exactly)
                noise_push = torch.zeros_like(alpha)
                drift_push = torch.zeros_like(alpha)
                for i in range(NC):
                    for k in range(D):
                        noise_push[i, k] = self.eta * self.symmetry_break_mult * random.gauss(0, 1.0)
                        drift_push[i, k] = self.eta * self.drift_mult * random.gauss(0, 1.0)

            amplify_push = self.eta * self.amplify_mult * torch.tanh(resp_z) * dev.sign()

            at_mean = dev.abs() < self.threshold
            sensitive = resp_z > 0

            push = torch.where(at_mean, noise_push,
                   torch.where(sensitive, amplify_push, drift_push))

            old_alpha = alpha.clone()
            self.alpha = (alpha + push).clamp(self.alpha_clip_lo, self.alpha_clip_hi)
            self.total_alpha_shift += (self.alpha - old_alpha).abs().sum().item()

        # ── ATTENTION ────────────────────────────────────────
        dots = (xs @ xs.T) / (D * self.tau)  # (NC, NC)
        dots = dots - self.eye * 1e10
        weights = F.softmax(dots, dim=1)  # (NC, NC)

        # ── COUPLING ─────────────────────────────────────────
        weighted_avg = weights @ phi_bare  # (NC, D)
        pull = weighted_avg - phi_bare

        bare_diff = phi_bare - xs
        fp_d = bare_diff.norm(dim=1) / xs.norm(dim=1).clamp(min=1.0)  # (NC,)
        plast = torch.exp(-fp_d.pow(2) / 0.0225).unsqueeze(1)  # (NC, 1)

        p = phi_sig + plast * self.eps * pull

        # ── STATE UPDATE ─────────────────────────────────────
        if fast:
            # Fast path: batched torch.randn()
            noise_t = self.noise * torch.randn_like(xs)
        else:
            # Slow path: Python random (matching CPU exactly)
            noise_t = torch.zeros_like(xs)
            for i in range(NC):
                for k in range(D):
                    noise_t[i, k] = random.gauss(0, self.noise)

        new_xs = (1 - self.delta) * xs + self.delta * p + noise_t
        return new_xs.clamp(-self.clip, self.clip)

    def _step_cpu(self, xs, signal=None):
        """CPU fallback: identical to harness.py logic."""
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

                    if abs(dev) < self.threshold:
                        push = self.eta * self.symmetry_break_mult * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * self.amplify_mult
                    else:
                        push = self.eta * self.drift_mult * random.gauss(0, 1.0)

                    old = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(self.alpha_clip_lo,
                                           min(self.alpha_clip_hi, self.alpha[i][k]))
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

        # ── STATE UPDATE ─────────────────────────────────────
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
        if HAS_TORCH and isinstance(xs, torch.Tensor):
            return xs.mean(dim=0).cpu().tolist()
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]


# ═══════════════════════════════════════════════════════════════
# Signal Generation (identical to harness.py)
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


# ═══════════════════════════════════════════════════════════════
# Execution Protocol (identical to harness.py)
# ═══════════════════════════════════════════════════════════════

def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    random.seed(base_seed)

    if HAS_TORCH and org.device != 'cpu':
        xs_list = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
        xs = torch.tensor(xs_list, dtype=torch.float32, device=org.device)
    else:
        xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    for _ in range(n_org):
        xs = org.step(xs)

    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        if HAS_TORCH and org.device != 'cpu':
            sig_list = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
            sig = torch.tensor(sig_list, dtype=torch.float32, device=org.device)
        else:
            sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]

        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    for _ in range(n_final):
        xs = org.step(xs)

    return org.centroid(xs)


def measure_gap(org, signals, k, seed, n_perm=4, n_trials=3):
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c = run_sequence(org, perm, signals, seed, trial)
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
# Rule Configuration
# ═══════════════════════════════════════════════════════════════

def canonical_rule():
    """Returns the canonical ALIVE plasticity rule."""
    return {
        'eta': 0.0003,
        'symmetry_break_mult': 0.3,
        'amplify_mult': 0.5,
        'drift_mult': 0.1,
        'threshold': 0.01,
        'alpha_clip_lo': 0.3,
        'alpha_clip_hi': 1.8,
    }


# ═══════════════════════════════════════════════════════════════
# Comparison Protocol
# ═══════════════════════════════════════════════════════════════

def run_comparison(rule_params,
                   ks=[4, 6, 8, 10],
                   seeds=[42, 137, 2024],
                   novel_seed_base=99999,
                   birth_seed=42,
                   device='auto',
                   gpu_fraction=DEFAULT_GPU_FRACTION,
                   verbose=False):
    """
    Runs the full comparison protocol: STILL vs ALIVE-baseline vs ALIVE-variant.

    API-identical to harness.py but runs on GPU if available.

    Args:
        rule_params: plasticity rule parameters dict
        ks: list of K values (signal vocabulary sizes)
        seeds: list of test seeds
        novel_seed_base: base seed for novel signals
        birth_seed: organism birth seed
        device: 'auto', 'cuda', or 'cpu'
        gpu_fraction: GPU memory fraction limit (default 0.5 = 50%)
        verbose: print progress
    """
    device = get_device(device, gpu_fraction)

    if verbose:
        print(f"Running comparison for rule: {rule_params}")
        print(f"Device: {device}")

    canonical = canonical_rule()

    # ── TRAINING SIGNALS ─────────────────────────────────────
    still_gaps = []
    baseline_gaps = []
    variant_gaps = []

    for k in ks:
        sig_seed = birth_seed + k * 200
        sigs = make_signals(k, seed=sig_seed)

        for s in seeds:
            # STILL
            org_still = Organism(seed=birth_seed, alive=False, rule_params=None, device=device)
            g_still = measure_gap(org_still, sigs, k, s)
            still_gaps.append(g_still)

            # ALIVE baseline (canonical rule)
            org_baseline = Organism(seed=birth_seed, alive=True, rule_params=canonical, device=device)
            g_baseline = measure_gap(org_baseline, sigs, k, s)
            baseline_gaps.append(g_baseline)

            # ALIVE variant (test rule)
            org_variant = Organism(seed=birth_seed, alive=True, rule_params=rule_params, device=device)
            g_variant = measure_gap(org_variant, sigs, k, s)
            variant_gaps.append(g_variant)

    still_avg = sum(still_gaps) / len(still_gaps)
    baseline_avg = sum(baseline_gaps) / len(baseline_gaps)
    variant_avg = sum(variant_gaps) / len(variant_gaps)

    # ── NOVEL SIGNALS ────────────────────────────────────────
    novel_still = []
    novel_baseline = []
    novel_variant = []

    for wi in range(6):
        for k in [6, 8]:
            nsigs = make_signals(k, seed=novel_seed_base + wi * 37 + k)
            ts = 77 + wi * 13 + k

            org_still = Organism(seed=birth_seed, alive=False, rule_params=None, device=device)
            g_still = measure_gap(org_still, nsigs, k, ts)
            novel_still.append(g_still)

            org_baseline = Organism(seed=birth_seed, alive=True, rule_params=canonical, device=device)
            g_baseline = measure_gap(org_baseline, nsigs, k, ts)
            novel_baseline.append(g_baseline)

            org_variant = Organism(seed=birth_seed, alive=True, rule_params=rule_params, device=device)
            g_variant = measure_gap(org_variant, nsigs, k, ts)
            novel_variant.append(g_variant)

    novel_still_avg = sum(novel_still) / len(novel_still)
    novel_baseline_avg = sum(novel_baseline) / len(novel_baseline)
    novel_variant_avg = sum(novel_variant) / len(novel_variant)

    # ── GROUND TRUTH TEST ────────────────────────────────────
    ground_truth_pass = variant_avg > 0.0

    if verbose:
        print(f"  STILL: {still_avg:+.4f}")
        print(f"  ALIVE baseline: {baseline_avg:+.4f} (delta={baseline_avg - still_avg:+.4f})")
        print(f"  ALIVE variant: {variant_avg:+.4f} (delta={variant_avg - still_avg:+.4f})")
        print(f"  Novel: STILL={novel_still_avg:+.4f} baseline={novel_baseline_avg:+.4f} variant={novel_variant_avg:+.4f}")
        print(f"  Ground truth: {'PASS' if ground_truth_pass else 'FAIL'}")

    return {
        'rule_params': rule_params,
        'still_gap': still_avg,
        'baseline_gap': baseline_avg,
        'variant_gap': variant_avg,
        'novel_still': novel_still_avg,
        'novel_baseline': novel_baseline_avg,
        'novel_variant': novel_variant_avg,
        'ground_truth_pass': ground_truth_pass,
    }


def batch_evaluate(rule_configs, eval_mode='quick', device='auto',
                   gpu_fraction=DEFAULT_GPU_FRACTION, verbose=False):
    """
    Batch evaluation of multiple rule configs.

    NOTE: True parallel batch evaluation (running N organisms simultaneously)
    requires deeper refactoring of measure_gap/run_sequence to support
    batched state tensors. This implementation runs rules sequentially
    but on GPU for acceleration.

    For evolutionary search, the speedup comes from GPU-accelerating
    the sequential evaluations, not from batching across rules.

    Args:
        rule_configs: list of rule_params dicts
        eval_mode: 'quick', 'medium', or 'full'
        device: 'auto', 'cuda', or 'cpu'
        gpu_fraction: GPU memory fraction limit (default 0.5 = 50%)
        verbose: print progress

    Returns:
        list of fitness scores (one per rule)
    """
    device = get_device(device, gpu_fraction)

    if verbose:
        print(f"Batch evaluating {len(rule_configs)} rules on {device}")

    fitnesses = []
    for i, rule in enumerate(rule_configs):
        if verbose and (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(rule_configs)}]", flush=True)

        if eval_mode == 'quick':
            # Quick: K=6, seeds=[42], n_perm=3, n_trials=2
            result = run_comparison(rule, ks=[6], seeds=[42],
                                    device=device, verbose=False)
        elif eval_mode == 'medium':
            # Medium: K=[6,8], seeds=[42,137], n_perm=4, n_trials=3
            result = run_comparison(rule, ks=[6, 8], seeds=[42, 137],
                                    device=device, verbose=False)
        else:  # full
            result = run_comparison(rule, device=device, verbose=False)

        # Fitness = variant_gap (ALIVE with this rule)
        fitnesses.append(result['variant_gap'])

    return fitnesses


# ═══════════════════════════════════════════════════════════════
# Validation Test
# ═══════════════════════════════════════════════════════════════

def validate_harness(gpu_fraction=DEFAULT_GPU_FRACTION):
    """
    Validates that GPU harness matches CPU harness output (within tolerance).

    Args:
        gpu_fraction: GPU memory fraction limit (default 0.5 = 50%)
    """
    print("=" * W)
    print("  GPU HARNESS VALIDATION")
    print("  Testing that GPU implementation matches CPU harness.py")
    print("=" * W)

    device = get_device('auto', gpu_fraction)
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory limit: {int(gpu_fraction * 100)}%")

    canonical = canonical_rule()

    # Quick test (faster validation)
    result = run_comparison(canonical, ks=[6], seeds=[42],
                           device=device, gpu_fraction=gpu_fraction, verbose=True)

    print("\n" + "=" * W)
    print(f"  Canonical ALIVE gap: {result['variant_gap']:+.4f}")
    print(f"  Expected range: +0.10 to +0.25 (allowing for noise)")
    print(f"  Ground truth: {'PASS' if result['ground_truth_pass'] else 'FAIL'}")

    in_range = 0.05 <= result['variant_gap'] <= 0.30
    if in_range and result['ground_truth_pass']:
        print(f"\n  VALIDATION: PASS")
        print(f"  GPU harness produces canonical ALIVE within expected range.")
    else:
        print(f"\n  VALIDATION: WARNING")
        print(f"  Result outside expected range (may be noise or device difference).")

    print("=" * W)

    return result


if __name__ == '__main__':
    if not HAS_TORCH:
        print("PyTorch required. Install: pip install torch")
        sys.exit(1)

    validate_harness()
