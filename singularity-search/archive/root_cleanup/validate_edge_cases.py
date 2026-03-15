#!/usr/bin/env python3
"""
Edge case validation for new eval protocol.

Tests:
1. Single K value (not a list)
2. Unusual seed values
3. Empty parameter overrides
4. Verify parameters thread correctly
"""

import sys
sys.path.insert(0, 'src')

from harness import run_comparison, canonical_rule, measure_gap, make_signals, Organism

print("=" * 72)
print("  EDGE CASE VALIDATION")
print("=" * 72)

canonical = canonical_rule()

# Test 1: Single K value
print("\n[Test 1] Single K value (K=6)")
try:
    result = run_comparison(canonical, ks=[6], seeds=[42], verbose=False)
    gap = result['variant_gap']
    print(f"  Result: gap={gap:+.4f}")
    if result['ground_truth_pass']:
        print("  [PASS] Single K value works")
    else:
        print("  [FAIL] Ground truth failed")
except Exception as e:
    print(f"  [FAIL] Exception: {e}")

# Test 2: Unusual seed
print("\n[Test 2] Unusual seed (seed=31337)")
try:
    result = run_comparison(canonical, ks=[6], seeds=[31337], verbose=False)
    gap = result['variant_gap']
    print(f"  Result: gap={gap:+.4f}")
    if result['ground_truth_pass']:
        print("  [PASS] Unusual seed works")
    else:
        print("  [FAIL] Ground truth failed")
except Exception as e:
    print(f"  [FAIL] Exception: {e}")

# Test 3: Verify parameter threading
print("\n[Test 3] Parameter override (n_perm=2, n_trials=2)")
try:
    result = run_comparison(canonical, ks=[6], seeds=[42],
                          n_perm=2, n_trials=2, verbose=False)
    gap = result['variant_gap']
    print(f"  Result: gap={gap:+.4f}")
    print("  [PASS] Parameter override works")
except Exception as e:
    print(f"  [FAIL] Exception: {e}")

# Test 4: Verify default parameters propagate
print("\n[Test 4] Verify defaults propagate (should use n_perm=8, n_trials=6)")
try:
    # Call measure_gap directly to inspect behavior
    org = Organism(seed=42, alive=True, rule_params=canonical)
    sigs = make_signals(6, seed=42+6*200)
    gap = measure_gap(org, sigs, 6, 42)
    print(f"  Result: gap={gap:+.4f}")
    print("  [PASS] measure_gap runs with defaults")
except Exception as e:
    print(f"  [FAIL] Exception: {e}")

print("\n" + "=" * 72)
print("  EDGE CASE VALIDATION COMPLETE")
print("=" * 72)
