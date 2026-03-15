#!/usr/bin/env python3
"""
LOCAL PROXY CORRELATION TEST

The central question: Can a LOCAL statistic (per-cell, O(1), intrinsic)
serve as a proxy for GLOBAL MI to guide beta/gamma adaptation?

Alpha adapts locally via per-cell resp_z signal (works).
Beta/gamma are shared parameters that currently require expensive global MI measurement.

If a local proxy exists that correlates strongly with MI (|r| > 0.7),
then beta/gamma can adapt locally too → strong thesis preserved.

If no correlate exists → beta/gamma require extrinsic measurement → strong thesis fails.

This script sweeps (beta, gamma) across a grid, measures MI and 4 local proxy
candidates, and reports Pearson correlation between each proxy and MI.
"""

import math
import random
import sys
import os

# Import Organism from the_living_seed
sys.path.insert(0, os.path.dirname(__file__))
from the_living_seed import Organism, make_signals, run_sequence, vcosine

D = 12
NC = 6


# ═══════════════════════════════════════════════════════════════
# MUTUAL INFORMATION ESTIMATION
# ═══════════════════════════════════════════════════════════════

def measure_mi(org, signals, k, seed, n_perm=4, n_trials=3):
    """
    Measure MI via within-sequence vs between-sequence endpoint similarity.
    Higher gap = higher MI (sequence discriminability).
    """
    from the_living_seed import gen_perms

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
    return avg_w - avg_b  # MI proxy: within - between similarity


# ═══════════════════════════════════════════════════════════════
# LOCAL PROXY CANDIDATES
# ═══════════════════════════════════════════════════════════════

def measure_local_proxies(org, signals, k, seed, n_trials=3):
    """
    Measure 4 local proxy candidates across multiple signal presentations.
    Returns dict of proxy_name -> value.
    """
    from the_living_seed import gen_perms

    # Run a few sequences to accumulate local statistics
    perms = gen_perms(k, n_perm=2, seed=seed * 10 + k)

    all_activation_fracs = []
    all_response_entropies = []
    all_neighbor_corrs = []
    all_response_vars = []

    for pi, perm in enumerate(perms[:2]):  # 2 permutations
        for trial in range(n_trials):
            # Create fresh organism for each trial
            test_org = Organism(seed=org.seed, alive=False)
            test_org.beta = org.beta
            test_org.gamma = org.gamma

            # Run sequence and collect response statistics
            random.seed(seed * 1000 + pi * 100 + trial)
            xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

            # Settle
            for _ in range(100):
                xs = test_org.step(xs)

            # Signal presentations with response tracking
            responses_per_sig = []
            for idx, sid in enumerate(perm):
                random.seed(seed * 1000 + sid * 100 + idx * 10 + trial)
                sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]

                # Compute bare and signal-modulated responses manually
                beta, gamma = test_org.beta, test_org.gamma

                # Bare dynamics
                phi_bare = []
                for i in range(NC):
                    row = []
                    for k in range(D):
                        kp = (k + 1) % D
                        km = (k - 1) % D
                        row.append(math.tanh(
                            test_org.alpha[i][k] * xs[i][k]
                            + beta * xs[i][kp] * xs[i][km]))
                    phi_bare.append(row)

                # Signal-modulated dynamics
                phi_sig = []
                for i in range(NC):
                    row = []
                    for k in range(D):
                        kp = (k + 1) % D
                        km = (k - 1) % D
                        row.append(math.tanh(
                            test_org.alpha[i][k] * xs[i][k]
                            + beta * (xs[i][kp] + gamma * sig[kp])
                                   * (xs[i][km] + gamma * sig[km])))
                    phi_sig.append(row)

                # Response differences
                response = []
                for i in range(NC):
                    response.append([abs(phi_sig[i][k] - phi_bare[i][k])
                                     for k in range(D)])

                responses_per_sig.append(response)

                # Step with signal
                for _ in range(20):
                    xs = test_org.step(xs, sig)

            # Compute local proxies from collected responses
            # Flatten all responses
            all_resp = []
            for resp in responses_per_sig:
                for i in range(NC):
                    for k in range(D):
                        all_resp.append(resp[i][k])

            if len(all_resp) == 0:
                continue

            # 1. Activation fraction: fraction with |response| > threshold
            threshold = 0.5
            activation_frac = sum(1 for r in all_resp if abs(r) > threshold) / len(all_resp)
            all_activation_fracs.append(activation_frac)

            # 2. Response entropy: entropy of binned response distribution
            n_bins = 10
            resp_min = min(all_resp)
            resp_max = max(all_resp)
            if resp_max - resp_min < 1e-10:
                entropy = 0.0
            else:
                bins = [0] * n_bins
                for r in all_resp:
                    bin_idx = int((r - resp_min) / (resp_max - resp_min + 1e-10) * n_bins)
                    bin_idx = max(0, min(n_bins - 1, bin_idx))
                    bins[bin_idx] += 1
                probs = [b / len(all_resp) for b in bins]
                entropy = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
            all_response_entropies.append(entropy)

            # 3. Neighbor correlation: mean correlation between adjacent cells' responses
            # Average across all signal presentations
            neighbor_corrs = []
            for resp in responses_per_sig:
                for i in range(NC - 1):
                    r1 = resp[i]
                    r2 = resp[i + 1]
                    corr = vcosine(r1, r2)
                    neighbor_corrs.append(corr)
            avg_neighbor_corr = sum(neighbor_corrs) / max(len(neighbor_corrs), 1)
            all_neighbor_corrs.append(avg_neighbor_corr)

            # 4. Response variance: var(phi_sig - phi_bare) across cells
            # Flatten response differences
            resp_diffs = []
            for resp in responses_per_sig:
                for i in range(NC):
                    for k in range(D):
                        resp_diffs.append(resp[i][k])
            if len(resp_diffs) > 1:
                mean_rd = sum(resp_diffs) / len(resp_diffs)
                var_rd = sum((rd - mean_rd) ** 2 for rd in resp_diffs) / len(resp_diffs)
            else:
                var_rd = 0.0
            all_response_vars.append(var_rd)

    # Average across trials
    return {
        'activation_fraction': sum(all_activation_fracs) / max(len(all_activation_fracs), 1),
        'response_entropy': sum(all_response_entropies) / max(len(all_response_entropies), 1),
        'neighbor_correlation': sum(all_neighbor_corrs) / max(len(all_neighbor_corrs), 1),
        'response_variance': sum(all_response_vars) / max(len(all_response_vars), 1),
    }


# ═══════════════════════════════════════════════════════════════
# ADVANCED LOCAL PROXY CANDIDATES (from researcher findings)
# ═══════════════════════════════════════════════════════════════

def measure_advanced_proxies(org, signals, k, seed, n_trials=3):
    """
    Measure advanced proxy candidates:
    1. Local mismatch statistics (max, 95th percentile, std)
    2. Resistance distance histogram (RDH) variance

    Returns dict of proxy_name -> value.
    """
    from the_living_seed import gen_perms

    perms = gen_perms(k, n_perm=2, seed=seed * 10 + k)

    all_mismatch_max = []
    all_mismatch_p95 = []
    all_mismatch_std = []
    all_rdh_variance = []

    for pi, perm in enumerate(perms[:2]):
        for trial in range(n_trials):
            test_org = Organism(seed=org.seed, alive=False)
            test_org.beta = org.beta
            test_org.gamma = org.gamma

            random.seed(seed * 1000 + pi * 100 + trial)
            xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

            for _ in range(100):
                xs = test_org.step(xs)

            responses_per_sig = []
            attention_weights_per_sig = []

            for idx, sid in enumerate(perm):
                random.seed(seed * 1000 + sid * 100 + idx * 10 + trial)
                sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]

                beta, gamma = test_org.beta, test_org.gamma

                # Bare dynamics
                phi_bare = []
                for i in range(NC):
                    row = []
                    for k in range(D):
                        kp = (k + 1) % D
                        km = (k - 1) % D
                        row.append(math.tanh(
                            test_org.alpha[i][k] * xs[i][k]
                            + beta * xs[i][kp] * xs[i][km]))
                    phi_bare.append(row)

                # Signal-modulated dynamics
                phi_sig = []
                for i in range(NC):
                    row = []
                    for k in range(D):
                        kp = (k + 1) % D
                        km = (k - 1) % D
                        row.append(math.tanh(
                            test_org.alpha[i][k] * xs[i][k]
                            + beta * (xs[i][kp] + gamma * sig[kp])
                                   * (xs[i][km] + gamma * sig[km])))
                    phi_sig.append(row)

                # Response differences
                response = []
                for i in range(NC):
                    response.append([abs(phi_sig[i][k] - phi_bare[i][k])
                                     for k in range(D)])

                responses_per_sig.append(response)

                # Compute attention weights (from Organism.step logic)
                weights = []
                for i in range(NC):
                    raw = []
                    for j in range(NC):
                        if i == j:
                            raw.append(-1e10)
                        else:
                            d = sum(xs[i][k] * xs[j][k] for k in range(D))
                            raw.append(d / (D * test_org.tau))
                    mx = max(raw)
                    exps = [math.exp(min(v - mx, 50)) for v in raw]
                    s = sum(exps) + 1e-15
                    weights.append([e / s for e in exps])

                attention_weights_per_sig.append(weights)

                for _ in range(20):
                    xs = test_org.step(xs, sig)

            # === LOCAL MISMATCH STATISTICS ===
            # Per-element mismatch from mean
            all_resp = []
            for resp in responses_per_sig:
                for i in range(NC):
                    for k in range(D):
                        all_resp.append(resp[i][k])

            if len(all_resp) > 0:
                mean_resp = sum(all_resp) / len(all_resp)
                mismatches = [abs(r - mean_resp) for r in all_resp]

                # Max mismatch
                mismatch_max = max(mismatches)
                all_mismatch_max.append(mismatch_max)

                # 95th percentile mismatch
                sorted_mismatches = sorted(mismatches)
                p95_idx = int(0.95 * len(sorted_mismatches))
                mismatch_p95 = sorted_mismatches[p95_idx]
                all_mismatch_p95.append(mismatch_p95)

                # Std of mismatches
                mean_mismatch = sum(mismatches) / len(mismatches)
                var_mismatch = sum((m - mean_mismatch) ** 2 for m in mismatches) / len(mismatches)
                mismatch_std = math.sqrt(var_mismatch)
                all_mismatch_std.append(mismatch_std)

            # === RESISTANCE DISTANCE HISTOGRAM (RDH) ===
            # Use attention weights as inverse distances
            # Compute effective resistance between all pairs
            if len(attention_weights_per_sig) > 0:
                # Average attention weights across signals
                avg_weights = [[0.0] * NC for _ in range(NC)]
                for weights in attention_weights_per_sig:
                    for i in range(NC):
                        for j in range(NC):
                            avg_weights[i][j] += weights[i][j]
                for i in range(NC):
                    for j in range(NC):
                        avg_weights[i][j] /= len(attention_weights_per_sig)

                # Compute resistance distances (simplified: use 1/weight as resistance)
                # For small NC, just compute all pairwise distances
                resistances = []
                for i in range(NC):
                    for j in range(i + 1, NC):
                        # Direct resistance (inverse of attention weight)
                        if avg_weights[i][j] > 1e-10:
                            resistance = 1.0 / avg_weights[i][j]
                        else:
                            resistance = 1e10  # infinite resistance
                        resistances.append(resistance)

                # RDH variance (variance of resistance distribution)
                if len(resistances) > 1:
                    mean_r = sum(resistances) / len(resistances)
                    var_r = sum((r - mean_r) ** 2 for r in resistances) / len(resistances)
                    all_rdh_variance.append(var_r)

    # Average across trials
    result = {}
    if len(all_mismatch_max) > 0:
        result['mismatch_max'] = sum(all_mismatch_max) / len(all_mismatch_max)
    else:
        result['mismatch_max'] = 0.0

    if len(all_mismatch_p95) > 0:
        result['mismatch_p95'] = sum(all_mismatch_p95) / len(all_mismatch_p95)
    else:
        result['mismatch_p95'] = 0.0

    if len(all_mismatch_std) > 0:
        result['mismatch_std'] = sum(all_mismatch_std) / len(all_mismatch_std)
    else:
        result['mismatch_std'] = 0.0

    if len(all_rdh_variance) > 0:
        result['rdh_variance'] = sum(all_rdh_variance) / len(all_rdh_variance)
    else:
        result['rdh_variance'] = 0.0

    return result


# ═══════════════════════════════════════════════════════════════
# GRID SWEEP AND CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════

def pearson_correlation(xs, ys):
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0

    return cov / (std_x * std_y)


def run_sweep():
    """
    Sweep beta x gamma grid, measure MI and local proxies, compute correlations.
    """
    print("=" * 72)
    print("  LOCAL PROXY CORRELATION TEST")
    print("  Can a local statistic proxy global MI for beta/gamma adaptation?")
    print("=" * 72)
    print()

    # Grid parameters
    betas = [0.1, 0.5, 1.0, 1.5, 2.0]
    gammas = [0.1, 0.5, 0.9, 1.5, 2.0]
    k = 6  # Signal count
    n_seeds = 3
    base_seed = 42

    print(f"Grid: {len(betas)} betas x {len(gammas)} gammas = {len(betas) * len(gammas)} points")
    print(f"Seeds per point: {n_seeds}")
    print(f"K (signals): {k}")
    print()

    # Storage
    mi_values = []
    proxy_values = {
        'activation_fraction': [],
        'response_entropy': [],
        'neighbor_correlation': [],
        'response_variance': [],
        'mismatch_max': [],
        'mismatch_p95': [],
        'mismatch_std': [],
        'rdh_variance': [],
    }

    # Grid sweep
    print("Sweeping grid (basic + advanced proxies)...")
    print()
    point_idx = 0
    for beta in betas:
        for gamma in gammas:
            point_idx += 1
            print(f"[{point_idx:2d}/{len(betas)*len(gammas)}] beta={beta:.1f} gamma={gamma:.1f}...", end=" ", flush=True)

            # Accumulate MI and proxies across seeds
            mi_per_seed = []
            proxies_per_seed = {pname: [] for pname in proxy_values.keys()}

            for seed_idx in range(n_seeds):
                seed = base_seed + seed_idx * 100 + point_idx

                # Create organism with these parameters
                org = Organism(seed=seed, alive=False)
                org.beta = beta
                org.gamma = gamma

                # Generate signals
                sigs = make_signals(k, seed=seed + k * 200)

                # Measure MI
                mi = measure_mi(org, sigs, k, seed, n_perm=3, n_trials=2)
                mi_per_seed.append(mi)

                # Measure basic local proxies
                proxies = measure_local_proxies(org, sigs, k, seed, n_trials=2)
                for pname, pval in proxies.items():
                    proxies_per_seed[pname].append(pval)

                # Measure advanced proxies
                advanced_proxies = measure_advanced_proxies(org, sigs, k, seed, n_trials=2)
                for pname, pval in advanced_proxies.items():
                    proxies_per_seed[pname].append(pval)

            # Average across seeds
            avg_mi = sum(mi_per_seed) / len(mi_per_seed)
            mi_values.append(avg_mi)

            for pname in proxy_values.keys():
                if len(proxies_per_seed[pname]) > 0:
                    avg_proxy = sum(proxies_per_seed[pname]) / len(proxies_per_seed[pname])
                else:
                    avg_proxy = 0.0
                proxy_values[pname].append(avg_proxy)

            print(f"MI={avg_mi:+.4f}", flush=True)

    print()
    print("-" * 72)
    print("  CORRELATION ANALYSIS")
    print("-" * 72)
    print()

    # Compute correlations
    correlations = {}
    for pname, pvals in proxy_values.items():
        r = pearson_correlation(mi_values, pvals)
        correlations[pname] = r

    # Report
    basic_proxies = ['activation_fraction', 'response_entropy', 'neighbor_correlation', 'response_variance']
    advanced_proxies = ['mismatch_max', 'mismatch_p95', 'mismatch_std', 'rdh_variance']

    print("BASIC PROXIES (linear/simple):")
    print(f"{'Proxy':<25} {'Pearson r':>12} {'|r|':>8} {'Strong?':>10}")
    print("-" * 72)
    for pname in basic_proxies:
        if pname in correlations:
            r = correlations[pname]
            abs_r = abs(r)
            strong = '  YES' if abs_r > 0.7 else '   no'
            print(f"{pname:<25} {r:+12.4f} {abs_r:8.4f} {strong:>10}")

    print()
    print("ADVANCED PROXIES (non-linear/graph-based):")
    print(f"{'Proxy':<25} {'Pearson r':>12} {'|r|':>8} {'Strong?':>10}")
    print("-" * 72)
    for pname in advanced_proxies:
        if pname in correlations:
            r = correlations[pname]
            abs_r = abs(r)
            strong = '  YES' if abs_r > 0.7 else '   no'
            print(f"{pname:<25} {r:+12.4f} {abs_r:8.4f} {strong:>10}")

    print()
    print("-" * 72)
    print("  CONCLUSION")
    print("-" * 72)
    print()

    # Find best proxy
    best_proxy = max(correlations.keys(), key=lambda p: abs(correlations[p]))
    best_r = correlations[best_proxy]

    print(f"Best proxy: {best_proxy}")
    print(f"Pearson r:  {best_r:+.4f}")
    print(f"|r|:        {abs(best_r):.4f}")
    print()

    if abs(best_r) > 0.7:
        print("RESULT: STRONG LOCAL CORRELATE EXISTS")
        print()
        print(f"  {best_proxy} correlates with MI (|r| = {abs(best_r):.3f} > 0.7).")
        print()
        print("  Implication: Beta/gamma CAN adapt locally via this proxy.")
        print("               Shared parameters do not require extrinsic measurement.")
        print("               STRONG THESIS PRESERVED: computation IS adaptation,")
        print("               even for shared parameters.")
        print()
    else:
        print("RESULT: NO STRONG LOCAL CORRELATE FOUND")
        print()
        print(f"  Best proxy {best_proxy} has |r| = {abs(best_r):.3f} < 0.7.")
        print()
        print("  Implication: Beta/gamma adaptation requires GLOBAL MI measurement,")
        print("               which is expensive and episodic (not per-step).")
        print("               STRONG THESIS FAILS for shared parameters:")
        print("               Alpha adapts locally (computation IS adaptation),")
        print("               but beta/gamma require extrinsic measurement.")
        print()

    print("=" * 72)

    return {
        'mi_values': mi_values,
        'proxy_values': proxy_values,
        'correlations': correlations,
        'best_proxy': best_proxy,
        'best_r': best_r,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    results = run_sweep()
