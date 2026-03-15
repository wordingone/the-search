#!/usr/bin/env python3
"""
Quick validation of Candidate #2 (reduced seed count for fast testing)
"""

import math
import sys
from scipy import stats
from harness import run_comparison, canonical_rule, measure_gap, make_signals, Organism

# Candidate #2 parameters from Session 7
CANDIDATE2 = {
    'eta': 0.000526,
    'symmetry_break_mult': 0.53345,
    'amplify_mult': 0.387234,
    'drift_mult': 0.157323,
    'threshold': 0.001777,
    'alpha_clip_lo': 0.358291,
    'alpha_clip_hi': 1.909935,
}

# Quick test: 3 seeds
SEEDS = [42, 137, 2024]
KS = [4, 6, 8, 10]
BIRTH_SEED = 42
NOVEL_SEED_BASE = 99999


def per_seed_gap(rule_params, ks, seed, birth_seed, alive=True):
    """Computes training gap for a single seed across all K values."""
    gaps = []
    for k in ks:
        sig_seed = birth_seed + k * 200
        sigs = make_signals(k, seed=sig_seed)
        org = Organism(seed=birth_seed, alive=alive, rule_params=rule_params)
        g = measure_gap(org, sigs, k, seed)
        gaps.append(g)
    return sum(gaps) / len(gaps)


def per_seed_novel_gap(rule_params, birth_seed, novel_seed_base, alive=True):
    """Computes novel signal gap."""
    gaps = []
    for wi in range(6):
        for k in [6, 8]:
            nsigs = make_signals(k, seed=novel_seed_base + wi * 37 + k)
            ts = 77 + wi * 13 + k
            org = Organism(seed=birth_seed, alive=alive, rule_params=rule_params)
            g = measure_gap(org, nsigs, k, ts)
            gaps.append(g)
    return sum(gaps) / len(gaps)


def main():
    canonical = canonical_rule()

    print("=" * 80)
    print("  QUICK VALIDATION: CANDIDATE #2 (3 seeds)")
    print("=" * 80)
    print("\nCandidate #2 parameters:")
    for k, v in CANDIDATE2.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nSeeds: {SEEDS}")
    print(f"K values: {KS}")
    print()

    # Training gaps
    print("Running training gap tests...")
    canonical_training = []
    candidate2_training = []

    for i, seed in enumerate(SEEDS):
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed}...", flush=True)

        g_canonical = per_seed_gap(canonical, KS, seed, BIRTH_SEED, alive=True)
        canonical_training.append(g_canonical)

        g_candidate2 = per_seed_gap(CANDIDATE2, KS, seed, BIRTH_SEED, alive=True)
        candidate2_training.append(g_candidate2)

        print(f"      canonical: {g_canonical:+.4f}  |  candidate2: {g_candidate2:+.4f}  |  delta: {g_candidate2 - g_canonical:+.4f}")

    # Training summary
    print("\n" + "-" * 80)
    print("TRAINING GAP SUMMARY:")
    print("-" * 80)
    mean_canonical = sum(canonical_training) / len(canonical_training)
    mean_candidate2 = sum(candidate2_training) / len(candidate2_training)
    print(f"Canonical:    mean={mean_canonical:+.4f}")
    print(f"Candidate #2: mean={mean_candidate2:+.4f}")
    print(f"Improvement:  {mean_candidate2 - mean_canonical:+.4f} ({100*(mean_candidate2 - mean_canonical)/abs(mean_canonical):+.1f}%)")

    # Novel gaps (just one run for quick test)
    print("\n" + "=" * 80)
    print("Running novel signal gap test...")
    print("  Computing novel gap for canonical...", flush=True)
    g_canonical_novel = per_seed_novel_gap(canonical, BIRTH_SEED, NOVEL_SEED_BASE, alive=True)

    print("  Computing novel gap for Candidate #2...", flush=True)
    g_candidate2_novel = per_seed_novel_gap(CANDIDATE2, BIRTH_SEED, NOVEL_SEED_BASE, alive=True)

    print(f"\n  canonical: {g_canonical_novel:+.4f}")
    print(f"  candidate2: {g_candidate2_novel:+.4f}")
    print(f"  improvement: {g_candidate2_novel - g_canonical_novel:+.4f} ({100*(g_candidate2_novel - g_canonical_novel)/abs(g_canonical_novel):+.1f}%)")

    print("\n" + "=" * 80)
    print("QUICK VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nTraining: Candidate #2 {'>' if mean_candidate2 > mean_canonical else '<'} canonical by {abs(mean_candidate2 - mean_canonical):.4f}")
    print(f"Novel:    Candidate #2 {'>' if g_candidate2_novel > g_canonical_novel else '<'} canonical by {abs(g_candidate2_novel - g_canonical_novel):.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
