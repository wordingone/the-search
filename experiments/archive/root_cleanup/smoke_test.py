#!/usr/bin/env python3
"""Smoke test: verify harness.py defaults are correct."""

import sys
sys.path.insert(0, 'src')

import inspect
from harness import measure_gap, run_comparison

# Check measure_gap signature
sig = inspect.signature(measure_gap)
n_perm_default = sig.parameters['n_perm'].default
n_trials_default = sig.parameters['n_trials'].default

print("measure_gap defaults:")
print(f"  n_perm = {n_perm_default} (expected: 8)")
print(f"  n_trials = {n_trials_default} (expected: 6)")
print()

# Check run_comparison signature
sig2 = inspect.signature(run_comparison)
n_perm_default2 = sig2.parameters['n_perm'].default
n_trials_default2 = sig2.parameters['n_trials'].default

print("run_comparison defaults:")
print(f"  n_perm = {n_perm_default2} (expected: 8)")
print(f"  n_trials = {n_trials_default2} (expected: 6)")
print()

# Verify
if n_perm_default == 8 and n_trials_default == 6:
    print("[PASS] measure_gap defaults CORRECT")
else:
    print("[FAIL] measure_gap defaults WRONG")
    sys.exit(1)

if n_perm_default2 == 8 and n_trials_default2 == 6:
    print("[PASS] run_comparison defaults CORRECT")
else:
    print("[FAIL] run_comparison defaults WRONG")
    sys.exit(1)

print()
print("SUCCESS: All defaults verified as n_perm=8, n_trials=6")
