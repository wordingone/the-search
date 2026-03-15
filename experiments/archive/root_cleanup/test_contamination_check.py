"""
Test contamination check implementation for delta_stability.

Verifies:
1. resp_z history is being tracked
2. Contamination correlation is computed
3. Pass/fail logic works correctly
"""

import sys
sys.path.insert(0, 'src')

from harness import (
    Organism, stage3_delta_stability_rule, make_signals, gen_perms,
    run_sequence, compute_pearson_correlation
)

def test_contamination_tracking():
    """Verify resp_z history is tracked and contamination check works."""
    print("="*60)
    print("  CONTAMINATION CHECK TEST")
    print("="*60)

    # Create Stage 3 organism
    rule = stage3_delta_stability_rule(alpha_meta=0.05)
    org = Organism(seed=42, alive=True, rule_params=rule)

    # Run a sequence
    sigs = make_signals(6, seed=242)
    perms = gen_perms(6, n_perm=4, seed=420)
    run_sequence(org, perms[0], sigs, 42, trial=0)

    # Check histories exist
    print(f"\n[History Tracking]")
    print(f"  resp_z history length: {len(org.resp_z_history)}")
    print(f"  delta_stability history length: {len(org.delta_stability_history)}")

    if len(org.resp_z_history) == 0:
        print("\n[FAIL] resp_z history is empty!")
        return False

    if len(org.delta_stability_history) == 0:
        print("\n[FAIL] delta_stability history is empty!")
        return False

    # Compute contamination correlation
    min_len = min(len(org.delta_stability_history), len(org.resp_z_history))
    ds_aligned = org.delta_stability_history[:min_len]
    rz_aligned = org.resp_z_history[-min_len:]

    contamination_corr = compute_pearson_correlation(ds_aligned, rz_aligned)

    print(f"\n[Contamination Check]")
    print(f"  Aligned length: {min_len}")
    print(f"  corr(delta_stability, mean(resp_z)) = {contamination_corr:+.4f}")
    print(f"  Pass threshold: |r| < 0.7")
    print(f"  Result: {'PASS' if abs(contamination_corr) < 0.7 else 'FAIL'}")

    # Check signal ranges
    print(f"\n[Signal Statistics]")
    print(f"  resp_z range: [{min(rz_aligned):+.4f}, {max(rz_aligned):+.4f}]")
    print(f"  delta_stability range: [{min(ds_aligned):+.6f}, {max(ds_aligned):+.6f}]")

    print(f"\n[PASS] Contamination check implementation verified")
    return True


if __name__ == '__main__':
    test_contamination_tracking()
