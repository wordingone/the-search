#!/usr/bin/env python3
"""
Experiment Harness for Plasticity Rule Search

Parameterizes the plasticity rule from the_living_seed.py to enable
systematic exploration of the adaptation space.

Self-contained: copies all infrastructure from the_living_seed.py.
Zero imports from it.
"""

import math
import random


# ═══════════════════════════════════════════════════════════════
# Constants (frozen, from the_living_seed.py)
# ═══════════════════════════════════════════════════════════════

D = 12
NC = 6
W = 72


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Organism with Parameterized Plasticity Rule
# ═══════════════════════════════════════════════════════════════

class Organism:
    """
    Cellular automaton with parameterized plasticity rule.

    Plasticity rule parameters (from the_living_seed.py lines 157-190):
    - eta: learning rate (default 0.0003)
    - symmetry_break_mult: push scale at column mean (default 0.3)
    - amplify_mult: push scale when resp_z > 0 (default 0.5)
    - drift_mult: push scale when resp_z <= 0 (default 0.1)
    - threshold: "at column mean" cutoff (default 0.01)
    - alpha_clip_lo: lower alpha bound (default 0.3)
    - alpha_clip_hi: upper alpha bound (default 1.8)
    """

    def __init__(self, seed=42, alive=False, rule_params=None):
        # Core dynamics parameters (frozen)
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0

        # Birth state
        self.seed = seed
        self.alive = alive

        # Plasticity rule parameters
        if rule_params is None:
            rule_params = canonical_rule()

        # Stage 3 parameters
        self.stage3_enabled = rule_params.get('stage3_enabled', False)
        self.stage3_signal = rule_params.get('stage3_signal', 'delta_stability')  # 'delta_stability' or 'delta_correlation'
        self.alpha_meta = rule_params.get('alpha_meta', 0.05)  # Meta-learning rate
        self.eta_clip_lo = rule_params.get('eta_clip_lo', 0.001)
        self.eta_clip_hi = rule_params.get('eta_clip_hi', 0.1)

        # Eta: use mid-range for stage3, canonical otherwise
        if self.stage3_enabled:
            # Start at geometric mean of clip range for stage3
            default_eta = math.sqrt(self.eta_clip_lo * self.eta_clip_hi)
        else:
            default_eta = 0.0003  # Canonical value

        self.eta = rule_params.get('eta', default_eta)
        self.symmetry_break_mult = rule_params.get('symmetry_break_mult', 0.3)
        self.amplify_mult = rule_params.get('amplify_mult', 0.5)
        self.drift_mult = rule_params.get('drift_mult', 0.1)
        self.threshold = rule_params.get('threshold', 0.01)
        self.alpha_clip_lo = rule_params.get('alpha_clip_lo', 0.3)
        self.alpha_clip_hi = rule_params.get('alpha_clip_hi', 1.8)

        # Tracking
        self.total_alpha_shift = 0.0
        self.eta_history = []  # Track eta over time

        # Stage 3 history storage (delta_stability)
        self.phi_sig_prev = None  # Previous phi_sig (NC × D)
        self.phi_stability_prev = None  # Previous phi_stability value
        self.delta_stability_history = []  # Track delta_stability signal

        # Stage 3 history storage (delta_correlation)
        self.correlation_prev = None  # Previous error-push correlation
        self.delta_correlation_history = []  # Track delta_correlation signal

        # Resp_z tracking for contamination check
        self.resp_z_history = []  # Track mean(resp_z) at each step

        # Initialize alpha
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

        # ── ONLINE PLASTICITY (PARAMETERIZED) ────────────────
        push_mags = None  # Track for delta_correlation signal
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

            # Track push magnitudes for delta_correlation
            if self.stage3_enabled and self.stage3_signal == 'delta_correlation':
                push_mags = [0.0] * NC

            # Track resp_z for contamination check
            resp_z_values = []

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std
                    resp_z_values.append(resp_z)

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    if abs(dev) < self.threshold:
                        # at column mean: break symmetry
                        push = self.eta * self.symmetry_break_mult * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        # above-average response: amplify diversity
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * self.amplify_mult
                    else:
                        # below-average response: gentle drift
                        push = self.eta * self.drift_mult * random.gauss(0, 1.0)

                    old = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(self.alpha_clip_lo,
                                           min(self.alpha_clip_hi, self.alpha[i][k]))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old)

                    # Accumulate push magnitude for this cell
                    if push_mags is not None:
                        push_mags[i] += abs(push)

            # Store resp_z aggregate for contamination check
            # NOTE: mean(resp_z) ≡ 0 by z-scoring construction, so use
            # mean(|resp_z|) which captures the spread of responsiveness
            if self.stage3_enabled and len(resp_z_values) > 0:
                mean_abs_resp_z = sum(abs(v) for v in resp_z_values) / len(resp_z_values)
                self.resp_z_history.append(mean_abs_resp_z)

        # ── STAGE 3 ETA ADAPTATION ───────────────────────────
        if self.stage3_enabled and signal:
            if self.stage3_signal == 'delta_stability':
                # CANDIDATE #3: Delta_stability signal
                if self.phi_sig_prev is not None:
                    phi_change = sum(
                        (phi_sig[i][k] - self.phi_sig_prev[i][k]) ** 2
                        for i in range(NC) for k in range(D)
                    )
                    phi_stability = 1.0 / (1.0 + math.sqrt(phi_change))

                    if self.phi_stability_prev is not None:
                        delta_stability = phi_stability - self.phi_stability_prev
                        self.delta_stability_history.append(delta_stability)

                        # Two-way adaptation
                        eta_adjustment = self.alpha_meta * (-delta_stability)
                        new_eta = self.eta * (1.0 + eta_adjustment)
                        new_eta = max(self.eta_clip_lo, min(self.eta_clip_hi, new_eta))
                        self.eta = new_eta
                        self.eta_history.append(self.eta)

                    self.phi_stability_prev = phi_stability
                self.phi_sig_prev = [row[:] for row in phi_sig]

            elif self.stage3_signal == 'delta_correlation':
                # CANDIDATE #5: Delta_correlation signal
                # Compute error_mag per cell (from bare_diff in attention block below)
                # We need to compute it here before state update
                error_mags = []
                for i in range(NC):
                    bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
                    error_mag = vnorm(bare_diff)
                    error_mags.append(error_mag)

                # Compute Pearson correlation between error_mag and push_mag
                if push_mags is not None and len(error_mags) == NC and len(push_mags) == NC:
                    # Pearson correlation
                    mean_err = sum(error_mags) / NC
                    mean_push = sum(push_mags) / NC
                    std_err = math.sqrt(sum((e - mean_err)**2 for e in error_mags) / NC) + 1e-10
                    std_push = math.sqrt(sum((p - mean_push)**2 for p in push_mags) / NC) + 1e-10

                    cov = sum((error_mags[i] - mean_err) * (push_mags[i] - mean_push)
                              for i in range(NC)) / NC
                    correlation = cov / (std_err * std_push)

                    if self.correlation_prev is not None:
                        delta_correlation = correlation - self.correlation_prev
                        self.delta_correlation_history.append(delta_correlation)

                        # One-way adaptation: only decrease eta
                        eta_adjustment = self.alpha_meta * min(0, delta_correlation)
                        new_eta = self.eta * (1.0 + eta_adjustment)
                        new_eta = max(self.eta_clip_lo, min(self.eta_clip_hi, new_eta))
                        self.eta = new_eta
                        self.eta_history.append(self.eta)

                    self.correlation_prev = correlation

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
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]


# ═══════════════════════════════════════════════════════════════
# Signal Generation
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
# Execution Protocol
# ═══════════════════════════════════════════════════════════════

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


def measure_gap(org, signals, k, seed, n_perm=8, n_trials=6):
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
# Rule Configuration
# ═══════════════════════════════════════════════════════════════

def canonical_rule():
    """Returns the canonical ALIVE plasticity rule from the_living_seed.py."""
    return {
        'eta': 0.0003,
        'symmetry_break_mult': 0.3,
        'amplify_mult': 0.5,
        'drift_mult': 0.1,
        'threshold': 0.01,
        'alpha_clip_lo': 0.3,
        'alpha_clip_hi': 1.8,
    }


def stage3_delta_stability_rule(alpha_meta=0.05):
    """
    Returns a Stage 3 rule with delta_stability signal for eta adaptation.

    Args:
        alpha_meta: Meta-learning rate (scaling factor for delta_stability).
                    Default 0.05 from pipeline spec.

    Signal: delta_stability = phi_stability[t] - phi_stability[t-1]
            where phi_stability = mean(||phi[t] - phi[t-1]||²)

    Eta update: eta_new = clip(eta * (1 + alpha_meta * (-delta_stability)),
                               0.001, 0.1)

    Direction: delta_stability > 0 (settling) → decrease eta
               delta_stability < 0 (destabilizing) → increase eta
    """
    rule = canonical_rule()
    rule['stage3_enabled'] = True
    rule['stage3_signal'] = 'delta_stability'
    rule['alpha_meta'] = alpha_meta
    rule['eta_clip_lo'] = 0.0001
    rule['eta_clip_hi'] = 0.01
    # Start at canonical value so comparison is fair
    rule['eta'] = 0.0003
    return rule


def stage3_delta_correlation_rule(alpha_meta=0.05):
    """
    Returns a Stage 3 rule with delta_correlation signal for eta adaptation.

    Args:
        alpha_meta: Meta-learning rate (scaling factor for delta_correlation).
                    Default 0.05 from pipeline spec.

    Signal: delta_correlation = correlation[t] - correlation[t-1]
            where correlation = pearson_corr(error_mag, push_mag)
            error_mag[i] = ||phi_bare[i] - xs[i]||
            push_mag[i] = sum(|push[i][k]|)

    Eta update: eta_new = clip(eta * (1 + alpha_meta * min(0, delta_correlation)),
                               0.001, 0.1)

    Direction: delta_correlation < 0 (degrading) → decrease eta
               delta_correlation >= 0 (improving) → neutral (one-way adaptation)
    """
    rule = canonical_rule()
    rule['stage3_enabled'] = True
    rule['stage3_signal'] = 'delta_correlation'
    rule['alpha_meta'] = alpha_meta
    rule['eta_clip_lo'] = 0.0001
    rule['eta_clip_hi'] = 0.01
    # Start at canonical value so comparison is fair
    rule['eta'] = 0.0003
    return rule


# ═══════════════════════════════════════════════════════════════
# Comparison Protocol
# ═══════════════════════════════════════════════════════════════

def run_comparison(rule_params,
                   ks=[4, 6, 8, 10],
                   seeds=[42, 137, 2024],
                   novel_seed_base=99999,
                   birth_seed=42,
                   verbose=False,
                   n_perm=8,
                   n_trials=6):
    """
    Runs the full comparison protocol: STILL vs ALIVE-baseline vs ALIVE-variant.

    Args:
        rule_params: dict with plasticity rule parameters
        ks: list of K values (signal vocabulary sizes)
        seeds: list of test seeds for signal generation
        novel_seed_base: base seed for novel signal generation
        birth_seed: organism birth seed
        verbose: print progress

    Returns:
        dict with keys:
            - rule_params: input rule
            - still_gap: avg gap for STILL (alpha fixed)
            - baseline_gap: avg gap for ALIVE with canonical rule
            - variant_gap: avg gap for ALIVE with rule_params
            - novel_still: avg gap on novel signals (STILL)
            - novel_baseline: avg gap on novel signals (baseline ALIVE)
            - novel_variant: avg gap on novel signals (variant ALIVE)
            - eta_stats: {mean, std, min, max} of eta values (if heterogeneous)
            - ground_truth_pass: bool, whether variant passes ground truth test
    """

    if verbose:
        print(f"Running comparison for rule: {rule_params}")

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
            org_still = Organism(seed=birth_seed, alive=False, rule_params=None)
            g_still = measure_gap(org_still, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
            still_gaps.append(g_still)

            # ALIVE baseline (canonical rule)
            org_baseline = Organism(seed=birth_seed, alive=True, rule_params=canonical)
            g_baseline = measure_gap(org_baseline, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
            baseline_gaps.append(g_baseline)

            # ALIVE variant (test rule)
            org_variant = Organism(seed=birth_seed, alive=True, rule_params=rule_params)
            g_variant = measure_gap(org_variant, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
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

            org_still = Organism(seed=birth_seed, alive=False, rule_params=None)
            g_still = measure_gap(org_still, nsigs, k, ts, n_perm=n_perm, n_trials=n_trials)
            novel_still.append(g_still)

            org_baseline = Organism(seed=birth_seed, alive=True, rule_params=canonical)
            g_baseline = measure_gap(org_baseline, nsigs, k, ts, n_perm=n_perm, n_trials=n_trials)
            novel_baseline.append(g_baseline)

            org_variant = Organism(seed=birth_seed, alive=True, rule_params=rule_params)
            g_variant = measure_gap(org_variant, nsigs, k, ts, n_perm=n_perm, n_trials=n_trials)
            novel_variant.append(g_variant)

    novel_still_avg = sum(novel_still) / len(novel_still)
    novel_baseline_avg = sum(novel_baseline) / len(novel_baseline)
    novel_variant_avg = sum(novel_variant) / len(novel_variant)

    # ── ETA STATISTICS (if heterogeneous eta) ────────────────
    eta_stats = None
    if isinstance(rule_params.get('eta'), list):
        etas_flat = rule_params['eta']
        eta_stats = {
            'mean': sum(etas_flat) / len(etas_flat),
            'std': math.sqrt(sum((e - sum(etas_flat)/len(etas_flat))**2
                                for e in etas_flat) / len(etas_flat)),
            'min': min(etas_flat),
            'max': max(etas_flat),
        }

    # ── GROUND TRUTH TEST ────────────────────────────────────
    # Variant must produce distinguishable final states for distinguishable
    # input sequences, measured by gap > 0
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
        'eta_stats': eta_stats,
        'ground_truth_pass': ground_truth_pass,
    }


def batch_comparison(rule_configs, **kwargs):
    """
    Runs comparison protocol for a list of rule configurations.

    Args:
        rule_configs: list of rule_params dicts
        **kwargs: passed to run_comparison

    Returns:
        list of result dicts (one per rule_config)
    """
    results = []
    for i, rule in enumerate(rule_configs):
        print(f"\n[{i+1}/{len(rule_configs)}] Testing rule...")
        res = run_comparison(rule, verbose=True, **kwargs)
        results.append(res)
    return results


# ═══════════════════════════════════════════════════════════════
# Validation Test
# ═══════════════════════════════════════════════════════════════

def validate_harness():
    """
    Validates that harness with canonical rule matches the_living_seed.py.
    """
    print("=" * W)
    print("  HARNESS VALIDATION")
    print("  Testing that canonical rule reproduces ALIVE from the_living_seed.py")
    print("=" * W)

    canonical = canonical_rule()
    result = run_comparison(canonical, verbose=True)

    # Expected values from the_living_seed.py Session 1:
    # ALIVE eta=0.0003: overall +0.168 (from CLAUDE.md / state.md)
    # The exact value depends on randomness, but should be in the ballpark

    print("\n" + "=" * W)
    print(f"  Canonical ALIVE gap: {result['variant_gap']:+.4f}")
    print(f"  Expected range: +0.15 to +0.20 (from session history)")
    print(f"  Ground truth: {'PASS' if result['ground_truth_pass'] else 'FAIL'}")

    in_range = 0.10 <= result['variant_gap'] <= 0.25
    if in_range and result['ground_truth_pass']:
        print(f"\n  VALIDATION: PASS")
        print(f"  Harness reproduces canonical ALIVE within expected noise.")
    else:
        print(f"\n  VALIDATION: FAIL")
        print(f"  Harness may have implementation errors.")

    print("=" * W)

    return result


def compute_pearson_correlation(x, y):
    """
    Compute Pearson correlation coefficient between two lists.

    Args:
        x, y: Lists of equal length

    Returns:
        float: Pearson correlation coefficient, or 0.0 if lists are constant
    """
    n = len(x)
    if n == 0 or len(y) != n:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((x[i] - mean_x)**2 for i in range(n)))
    std_y = math.sqrt(sum((y[i] - mean_y)**2 for i in range(n)))

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0  # Constant signal

    return cov / (std_x * std_y)


def quick_eval(rule_params, **kwargs):
    """
    Quick evaluation mode with lower exposure for fast iteration.
    Uses n_perm=4, n_trials=3 (old defaults).
    """
    return run_comparison(rule_params, n_perm=4, n_trials=3, **kwargs)


# ═══════════════════════════════════════════════════════════════
# Stage 3 Phase 2 Validation
# ═══════════════════════════════════════════════════════════════

def phase2_validate_delta_stability(alpha_meta=0.05, seeds=[42, 137, 2024], verbose=True):
    """
    Phase 2 validation for delta_stability signal (Candidate #3).

    Requirements from stage3_experiment_pipeline.md:
    1. Implement signal computation (DONE in Organism class)
    2. Implement global eta update (DONE in Organism class)
    3. Run 3-seed ground truth check (must pass all 3)
    4. Compute resp_z contamination check: corr(delta_stability, resp_z) < 0.7
    5. Measure eta trajectory: must not be trivial
    6. Compare training gap: adaptive eta vs fixed eta=0.0003

    Args:
        alpha_meta: Meta-learning rate for delta_stability signal
        seeds: List of seeds to test (default: 3 seeds)
        verbose: Print progress

    Returns:
        dict with validation results
    """
    if verbose:
        print("=" * W)
        print(f"  PHASE 2 VALIDATION: delta_stability (alpha_meta={alpha_meta})")
        print("=" * W)

    # Create rules
    canonical = canonical_rule()
    stage3_rule = stage3_delta_stability_rule(alpha_meta=alpha_meta)

    # Test 1: 3-seed ground truth check
    ground_truth_passes = []
    for seed in seeds:
        if verbose:
            print(f"\n[Seed {seed}] Running ground truth check...")
        # Phase 2: cheap eval per spec (n_perm=4, n_trials=3, K=[6,8])
        result = run_comparison(stage3_rule, ks=[6], seeds=[seed],
                                n_perm=4, n_trials=3, verbose=False)
        ground_truth_passes.append(result['ground_truth_pass'])
        if verbose:
            print(f"  Ground truth: {'PASS' if result['ground_truth_pass'] else 'FAIL'}")
            print(f"  Variant gap: {result['variant_gap']:+.4f}")

    # Test 2: Eta trajectory analysis
    if verbose:
        print(f"\n[Eta Trajectory] Analyzing eta evolution...")

    # Run one full sequence to extract eta history
    org = Organism(seed=seeds[0], alive=True, rule_params=stage3_rule)
    sigs = make_signals(6, seed=seeds[0] + 200)
    perms = gen_perms(6, n_perm=4, seed=seeds[0] * 10)
    run_sequence(org, perms[0], sigs, seeds[0], trial=0)

    eta_history = org.eta_history
    delta_stability_history = org.delta_stability_history

    if len(eta_history) > 0:
        eta_mean = sum(eta_history) / len(eta_history)
        eta_std = math.sqrt(sum((e - eta_mean)**2 for e in eta_history) / len(eta_history))
        eta_min = min(eta_history)
        eta_max = max(eta_history)

        # Check for trivial convergence (against actual clip bounds)
        at_lower_bound = sum(1 for e in eta_history if abs(e - stage3_rule['eta_clip_lo']) < 1e-8)
        at_upper_bound = sum(1 for e in eta_history if abs(e - stage3_rule['eta_clip_hi']) < 1e-8)
        all_equal = eta_std < 1e-6

        eta_nontrivial = (not all_equal and
                          at_lower_bound < len(eta_history) * 0.9 and
                          at_upper_bound < len(eta_history) * 0.9)

        if verbose:
            print(f"  Eta mean: {eta_mean:.6f}, std: {eta_std:.6f}")
            print(f"  Eta range: [{eta_min:.6f}, {eta_max:.6f}]")
            print(f"  Eta trajectory: {'NON-TRIVIAL' if eta_nontrivial else 'TRIVIAL (FAIL)'}")
    else:
        eta_nontrivial = False
        if verbose:
            print(f"  Eta history empty (no signal-bearing steps with history)")

    # Test 3: resp_z contamination check
    # Compute correlation between delta_stability and mean(resp_z) across steps
    if verbose:
        print(f"\n[Contamination Check]")

    resp_z_history = org.resp_z_history
    contamination_pass = True
    contamination_corr = 0.0

    if len(delta_stability_history) > 1 and len(resp_z_history) > 1:
        # Align histories (delta_stability starts one step later than resp_z)
        min_len = min(len(delta_stability_history), len(resp_z_history))
        ds_aligned = delta_stability_history[:min_len]
        rz_aligned = resp_z_history[-min_len:]  # Take last min_len elements

        contamination_corr = compute_pearson_correlation(ds_aligned, rz_aligned)
        contamination_pass = abs(contamination_corr) < 0.7

        if verbose:
            print(f"  corr(delta_stability, mean(resp_z)) = {contamination_corr:+.4f}")
            print(f"  Contamination check: {'PASS' if contamination_pass else 'FAIL'} (|r| < 0.7)")

        # Also report signal dynamic range
        ds_mean = sum(delta_stability_history) / len(delta_stability_history)
        ds_std = math.sqrt(sum((d - ds_mean)**2 for d in delta_stability_history) / len(delta_stability_history))
        ds_min = min(delta_stability_history)
        ds_max = max(delta_stability_history)

        if verbose:
            print(f"\n[Delta_stability Signal]")
            print(f"  Mean: {ds_mean:+.6f}, Std: {ds_std:.6f}")
            print(f"  Range: [{ds_min:+.6f}, {ds_max:+.6f}]")
            print(f"  Dynamic range: {'SUFFICIENT' if ds_std > 1e-6 else 'INSUFFICIENT (FAIL)'}")

        signal_sufficient = ds_std > 1e-6
    else:
        signal_sufficient = False
        if verbose:
            print(f"  Insufficient history for contamination check")

    # Test 4: Compare adaptive vs fixed eta (full comparison)
    if verbose:
        print(f"\n[Training Gap Comparison]")
        print(f"  Running adaptive eta (stage3) vs fixed eta (canonical)...")

    # Phase 2: cheap eval per spec (n_perm=4, n_trials=3, K=[6,8])
    result_adaptive = run_comparison(stage3_rule, ks=[6, 8], seeds=seeds,
                                     n_perm=4, n_trials=3, verbose=False)
    result_fixed = run_comparison(canonical, ks=[6, 8], seeds=seeds,
                                  n_perm=4, n_trials=3, verbose=False)

    adaptive_gap = result_adaptive['variant_gap']
    fixed_gap = result_fixed['variant_gap']
    improvement = adaptive_gap - fixed_gap

    if verbose:
        print(f"  Fixed eta=0.0003: {fixed_gap:+.4f}")
        print(f"  Adaptive eta (delta_stability): {adaptive_gap:+.4f}")
        print(f"  Improvement: {improvement:+.4f} ({'POSITIVE' if improvement > 0 else 'NEGATIVE'})")

    # Phase 2 pass criteria
    phase2_pass = (all(ground_truth_passes) and
                   contamination_pass and
                   eta_nontrivial and
                   signal_sufficient and
                   improvement >= 0)  # Phase 2 only requires non-negative

    if verbose:
        print("\n" + "=" * W)
        print(f"  PHASE 2 VALIDATION: {'PASS' if phase2_pass else 'FAIL'}")
        print(f"  - Ground truth (3/3): {'PASS' if all(ground_truth_passes) else 'FAIL'}")
        print(f"  - Contamination |r| < 0.7: {'PASS' if contamination_pass else 'FAIL'}")
        print(f"  - Eta non-trivial: {'PASS' if eta_nontrivial else 'FAIL'}")
        print(f"  - Signal sufficient: {'PASS' if signal_sufficient else 'FAIL'}")
        print(f"  - Improvement >= 0: {'PASS' if improvement >= 0 else 'FAIL'}")
        print("=" * W)

    return {
        'phase2_pass': phase2_pass,
        'ground_truth_passes': ground_truth_passes,
        'contamination_pass': contamination_pass,
        'contamination_corr': contamination_corr,
        'eta_nontrivial': eta_nontrivial,
        'signal_sufficient': signal_sufficient,
        'eta_history': eta_history,
        'delta_stability_history': delta_stability_history,
        'resp_z_history': resp_z_history,
        'adaptive_gap': adaptive_gap,
        'fixed_gap': fixed_gap,
        'improvement': improvement,
        'alpha_meta': alpha_meta,
    }


def phase2_validate_delta_correlation(alpha_meta=0.05, seeds=[42, 137, 2024], verbose=True):
    """
    Phase 2 validation for delta_correlation signal (Candidate #5).

    Requirements from stage3_experiment_pipeline.md:
    1. Implement signal computation (DONE in Organism class)
    2. Implement global eta update (DONE in Organism class)
    3. Run 3-seed ground truth check (must pass all 3)
    4. Compute resp_z contamination check: corr(delta_correlation, resp_z) < 0.7
    5. Measure eta trajectory: must not be trivial
    6. Compare training gap: adaptive eta vs fixed eta=0.0003

    Args:
        alpha_meta: Meta-learning rate for delta_correlation signal
        seeds: List of seeds to test (default: 3 seeds)
        verbose: Print progress

    Returns:
        dict with validation results
    """
    if verbose:
        print("=" * W)
        print(f"  PHASE 2 VALIDATION: delta_correlation (alpha_meta={alpha_meta})")
        print("=" * W)

    # Create rules
    canonical = canonical_rule()
    stage3_rule = stage3_delta_correlation_rule(alpha_meta=alpha_meta)

    # Test 1: 3-seed ground truth check
    ground_truth_passes = []
    for seed in seeds:
        if verbose:
            print(f"\n[Seed {seed}] Running ground truth check...")
        # Phase 2: cheap eval per spec (n_perm=4, n_trials=3, K=[6,8])
        result = run_comparison(stage3_rule, ks=[6], seeds=[seed],
                                n_perm=4, n_trials=3, verbose=False)
        ground_truth_passes.append(result['ground_truth_pass'])
        if verbose:
            print(f"  Ground truth: {'PASS' if result['ground_truth_pass'] else 'FAIL'}")
            print(f"  Variant gap: {result['variant_gap']:+.4f}")

    # Test 2: Eta trajectory analysis
    if verbose:
        print(f"\n[Eta Trajectory] Analyzing eta evolution...")

    org = Organism(seed=seeds[0], alive=True, rule_params=stage3_rule)
    sigs = make_signals(6, seed=seeds[0] + 200)
    perms = gen_perms(6, n_perm=4, seed=seeds[0] * 10)
    run_sequence(org, perms[0], sigs, seeds[0], trial=0)

    eta_history = org.eta_history
    delta_correlation_history = org.delta_correlation_history

    if len(eta_history) > 0:
        eta_mean = sum(eta_history) / len(eta_history)
        eta_std = math.sqrt(sum((e - eta_mean)**2 for e in eta_history) / len(eta_history))
        eta_min = min(eta_history)
        eta_max = max(eta_history)

        at_lower_bound = sum(1 for e in eta_history if abs(e - stage3_rule['eta_clip_lo']) < 1e-8)
        at_upper_bound = sum(1 for e in eta_history if abs(e - stage3_rule['eta_clip_hi']) < 1e-8)
        all_equal = eta_std < 1e-6

        eta_nontrivial = (not all_equal and
                          at_lower_bound < len(eta_history) * 0.9 and
                          at_upper_bound < len(eta_history) * 0.9)

        if verbose:
            print(f"  Eta mean: {eta_mean:.6f}, std: {eta_std:.6f}")
            print(f"  Eta range: [{eta_min:.6f}, {eta_max:.6f}]")
            print(f"  Eta trajectory: {'NON-TRIVIAL' if eta_nontrivial else 'TRIVIAL (FAIL)'}")
    else:
        eta_nontrivial = False
        if verbose:
            print(f"  Eta history empty (no signal-bearing steps with history)")

    # Test 3: resp_z contamination check
    if verbose:
        print(f"\n[Contamination Check]")

    resp_z_history = org.resp_z_history
    contamination_pass = True
    contamination_corr = 0.0

    if len(delta_correlation_history) > 1 and len(resp_z_history) > 1:
        # Align histories (delta_correlation starts one step later than resp_z)
        min_len = min(len(delta_correlation_history), len(resp_z_history))
        dc_aligned = delta_correlation_history[:min_len]
        rz_aligned = resp_z_history[-min_len:]

        contamination_corr = compute_pearson_correlation(dc_aligned, rz_aligned)
        contamination_pass = abs(contamination_corr) < 0.7

        if verbose:
            print(f"  corr(delta_correlation, mean(resp_z)) = {contamination_corr:+.4f}")
            print(f"  Contamination check: {'PASS' if contamination_pass else 'FAIL'} (|r| < 0.7)")

        # Also report signal dynamic range
        dc_mean = sum(delta_correlation_history) / len(delta_correlation_history)
        dc_std = math.sqrt(sum((d - dc_mean)**2 for d in delta_correlation_history) / len(delta_correlation_history))
        dc_min = min(delta_correlation_history)
        dc_max = max(delta_correlation_history)

        if verbose:
            print(f"\n[Delta_correlation Signal]")
            print(f"  Mean: {dc_mean:+.6f}, Std: {dc_std:.6f}")
            print(f"  Range: [{dc_min:+.6f}, {dc_max:+.6f}]")
            print(f"  Dynamic range: {'SUFFICIENT' if dc_std > 1e-6 else 'INSUFFICIENT (FAIL)'}")

        signal_sufficient = dc_std > 1e-6
    else:
        signal_sufficient = False
        if verbose:
            print(f"  Insufficient history for contamination check")

    # Test 4: Compare adaptive vs fixed eta
    if verbose:
        print(f"\n[Training Gap Comparison]")
        print(f"  Running adaptive eta (stage3) vs fixed eta (canonical)...")

    # Phase 2: cheap eval per spec (n_perm=4, n_trials=3, K=[6,8])
    result_adaptive = run_comparison(stage3_rule, ks=[6, 8], seeds=seeds,
                                     n_perm=4, n_trials=3, verbose=False)
    result_fixed = run_comparison(canonical, ks=[6, 8], seeds=seeds,
                                  n_perm=4, n_trials=3, verbose=False)

    adaptive_gap = result_adaptive['variant_gap']
    fixed_gap = result_fixed['variant_gap']
    improvement = adaptive_gap - fixed_gap

    if verbose:
        print(f"  Fixed eta=0.0003: {fixed_gap:+.4f}")
        print(f"  Adaptive eta (delta_correlation): {adaptive_gap:+.4f}")
        print(f"  Improvement: {improvement:+.4f} ({'POSITIVE' if improvement > 0 else 'NEGATIVE'})")

    # Phase 2 pass criteria
    phase2_pass = (all(ground_truth_passes) and
                   contamination_pass and
                   eta_nontrivial and
                   signal_sufficient and
                   improvement >= 0)

    if verbose:
        print("\n" + "=" * W)
        print(f"  PHASE 2 VALIDATION: {'PASS' if phase2_pass else 'FAIL'}")
        print(f"  - Ground truth (3/3): {'PASS' if all(ground_truth_passes) else 'FAIL'}")
        print(f"  - Contamination |r| < 0.7: {'PASS' if contamination_pass else 'FAIL'}")
        print(f"  - Eta non-trivial: {'PASS' if eta_nontrivial else 'FAIL'}")
        print(f"  - Signal sufficient: {'PASS' if signal_sufficient else 'FAIL'}")
        print(f"  - Improvement >= 0: {'PASS' if improvement >= 0 else 'FAIL'}")
        print("=" * W)

    return {
        'phase2_pass': phase2_pass,
        'ground_truth_passes': ground_truth_passes,
        'contamination_pass': contamination_pass,
        'contamination_corr': contamination_corr,
        'eta_nontrivial': eta_nontrivial,
        'signal_sufficient': signal_sufficient,
        'eta_history': eta_history,
        'delta_correlation_history': delta_correlation_history,
        'resp_z_history': resp_z_history,
        'adaptive_gap': adaptive_gap,
        'fixed_gap': fixed_gap,
        'improvement': improvement,
        'alpha_meta': alpha_meta,
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("=" * W)
        print("  QUICK MODE (n_perm=4, n_trials=3)")
        print("=" * W)
        canonical = canonical_rule()
        result = quick_eval(canonical, verbose=True)
        print("\n" + "=" * W)
        print(f"  Quick eval gap: {result['variant_gap']:+.4f}")
        print("=" * W)
    elif len(sys.argv) > 1 and sys.argv[1] == '--phase2':
        # Phase 2 validation for Stage 3 signals
        signal_type = 'delta_stability'  # Default
        alpha_meta_values = [0.01, 0.05, 0.1]  # Test all three per strategist spec
        test_all = True

        if len(sys.argv) > 2:
            signal_type = sys.argv[2]  # 'delta_stability' or 'delta_correlation'
        if len(sys.argv) > 3:
            # Single alpha_meta value specified
            alpha_meta_values = [float(sys.argv[3])]
            test_all = False

        # Test each alpha_meta value
        results = []
        for alpha_meta in alpha_meta_values:
            if test_all:
                print(f"\n{'='*W}")
                print(f"  Testing alpha_meta = {alpha_meta}")
                print(f"{'='*W}\n")

            if signal_type == 'delta_correlation':
                result = phase2_validate_delta_correlation(alpha_meta=alpha_meta, verbose=True)
            else:
                result = phase2_validate_delta_stability(alpha_meta=alpha_meta, verbose=True)

            results.append(result)

        # Summary report if testing multiple values
        if test_all:
            print(f"\n{'='*W}")
            print(f"  ALPHA_META SWEEP SUMMARY ({signal_type})")
            print(f"{'='*W}")
            for i, alpha_meta in enumerate(alpha_meta_values):
                r = results[i]
                status = 'PASS' if r['phase2_pass'] else 'FAIL'
                print(f"  alpha_meta={alpha_meta:5.2f}: {status:4s} | "
                      f"contam={r['contamination_corr']:+.3f} | "
                      f"improvement={r['improvement']:+.4f}")
            print(f"{'='*W}")
    else:
        validate_harness()
