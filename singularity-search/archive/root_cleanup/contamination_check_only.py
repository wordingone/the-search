"""
Quick contamination check for both Stage 3 signals.
Tests corr(signal, mean(resp_z)) across multiple seeds without full Phase 2 validation.
"""

import sys
sys.path.insert(0, 'src')

from harness import (
    stage3_delta_stability_rule, stage3_delta_correlation_rule,
    Organism, make_signals, gen_perms, run_sequence, compute_pearson_correlation
)

def check_contamination(signal_name, rule_func, alpha_meta, seeds=[42, 137, 2024]):
    """Run contamination check for a signal across multiple seeds."""
    print(f"\n{'='*80}")
    print(f"  CONTAMINATION CHECK: {signal_name} (alpha_meta={alpha_meta})")
    print(f"{'='*80}")

    correlations = []

    for seed in seeds:
        rule = rule_func(alpha_meta=alpha_meta)
        org = Organism(seed=seed, alive=True, rule_params=rule)

        sigs = make_signals(6, seed=seed + 200)
        perms = gen_perms(6, n_perm=4, seed=seed * 10)
        run_sequence(org, perms[0], sigs, seed, trial=0)

        # Get histories
        if signal_name == 'delta_stability':
            signal_history = org.delta_stability_history
        else:
            signal_history = org.delta_correlation_history

        resp_z_history = org.resp_z_history

        # Compute correlation
        if len(signal_history) > 1 and len(resp_z_history) > 1:
            min_len = min(len(signal_history), len(resp_z_history))
            sig_aligned = signal_history[:min_len]
            rz_aligned = resp_z_history[-min_len:]

            corr = compute_pearson_correlation(sig_aligned, rz_aligned)
            correlations.append(corr)

            print(f"  Seed {seed:5d}: corr = {corr:+.4f} | "
                  f"signal_len={len(signal_history):3d} | "
                  f"resp_z_len={len(resp_z_history):3d}")
        else:
            print(f"  Seed {seed:5d}: INSUFFICIENT HISTORY")

    # Summary
    if correlations:
        mean_corr = sum(correlations) / len(correlations)
        max_abs_corr = max(abs(c) for c in correlations)
        contamination_pass = max_abs_corr < 0.7

        print(f"\n  Mean correlation: {mean_corr:+.4f}")
        print(f"  Max |correlation|: {max_abs_corr:+.4f}")
        print(f"  Threshold: |r| < 0.7")
        print(f"  Result: {'PASS' if contamination_pass else 'FAIL'}")

        return {
            'correlations': correlations,
            'mean_corr': mean_corr,
            'max_abs_corr': max_abs_corr,
            'contamination_pass': contamination_pass
        }
    else:
        print("  NO VALID CORRELATIONS COMPUTED")
        return None


# Test delta_stability
print("="*80)
print("  STAGE 3 SIGNAL CONTAMINATION CHECK")
print("="*80)

ds_result = check_contamination('delta_stability', stage3_delta_stability_rule, alpha_meta=0.05)

# Test delta_correlation
dc_result = check_contamination('delta_correlation', stage3_delta_correlation_rule, alpha_meta=0.05)

# Final summary
print("\n" + "="*80)
print("  CONTAMINATION SUMMARY (alpha_meta=0.05)")
print("="*80)

if ds_result:
    status = 'PASS' if ds_result['contamination_pass'] else 'FAIL'
    print(f"  delta_stability:    {status} | max|r| = {ds_result['max_abs_corr']:.4f}")

if dc_result:
    status = 'PASS' if dc_result['contamination_pass'] else 'FAIL'
    print(f"  delta_correlation:  {status} | max|r| = {dc_result['max_abs_corr']:.4f}")

print("="*80)

# Save results
with open('contamination_results.txt', 'w') as f:
    if ds_result:
        f.write(f"delta_stability: {'PASS' if ds_result['contamination_pass'] else 'FAIL'}\n")
        f.write(f"  max|r| = {ds_result['max_abs_corr']:.4f}\n")
        f.write(f"  correlations = {ds_result['correlations']}\n")
    if dc_result:
        f.write(f"delta_correlation: {'PASS' if dc_result['contamination_pass'] else 'FAIL'}\n")
        f.write(f"  max|r| = {dc_result['max_abs_corr']:.4f}\n")
        f.write(f"  correlations = {dc_result['correlations']}\n")

print("\nResults saved to contamination_results.txt")
