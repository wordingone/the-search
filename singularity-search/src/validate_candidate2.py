#!/usr/bin/env python3
"""
Statistical Validation of Candidate #2 from Session 7

Runs full-protocol validation with extended seed set (10 seeds) to establish
statistical significance of the breakthrough parameters found in Session 7.

Candidate #2 parameters:
- eta=0.000526
- symmetry_break_mult=0.53345
- amplify_mult=0.387234
- drift_mult=0.157323
- threshold=0.001777
- alpha_clip_lo=0.358291
- alpha_clip_hi=1.909935

Session 7 single-seed result:
- Training: +0.1813 (+7.6% vs canonical)
- Novel: +0.1024 (+37.6% vs canonical)

This script:
1. Runs BOTH canonical and Candidate #2 on the same 10 seeds
2. Uses K=[4,6,8,10], full novel signal protocol
3. Computes per-seed training gap and novel gap for both rules
4. Computes mean, std, 95% CI for both
5. Runs paired t-test for training and novel separately
6. Reports effect size (Cohen's d)

Acceptance criterion: Candidate #2 beats canonical with p < 0.05 on BOTH training AND novel.
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

# Extended seed set (10 seeds)
SEEDS = [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
KS = [4, 6, 8, 10]
BIRTH_SEED = 42
NOVEL_SEED_BASE = 99999


def per_seed_gap(rule_params, ks, seed, birth_seed, alive=True):
    """
    Computes training gap for a single seed across all K values.

    Returns average gap across all K values for this seed.
    """
    gaps = []
    for k in ks:
        sig_seed = birth_seed + k * 200
        sigs = make_signals(k, seed=sig_seed)
        org = Organism(seed=birth_seed, alive=alive, rule_params=rule_params)
        g = measure_gap(org, sigs, k, seed)
        gaps.append(g)
    return sum(gaps) / len(gaps)


def per_seed_novel_gap(rule_params, birth_seed, novel_seed_base, alive=True):
    """
    Computes novel signal gap for a single configuration.

    Returns average gap across all novel signal windows.
    """
    gaps = []
    for wi in range(6):
        for k in [6, 8]:
            nsigs = make_signals(k, seed=novel_seed_base + wi * 37 + k)
            ts = 77 + wi * 13 + k
            org = Organism(seed=birth_seed, alive=alive, rule_params=rule_params)
            g = measure_gap(org, nsigs, k, ts)
            gaps.append(g)
    return sum(gaps) / len(gaps)


def paired_ttest_summary(x, y, name_x, name_y):
    """
    Runs paired t-test and computes effect size (Cohen's d).

    Args:
        x: list of values for condition X
        y: list of values for condition Y
        name_x: label for X
        name_y: label for Y

    Returns:
        dict with keys: t_stat, p_value, cohens_d, mean_diff, ci_95
    """
    diff = [yi - xi for xi, yi in zip(x, y)]
    mean_diff = sum(diff) / len(diff)

    # Standard deviation of differences
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diff) / (len(diff) - 1))

    # 95% confidence interval
    se = std_diff / math.sqrt(len(diff))
    t_crit = stats.t.ppf(0.975, len(diff) - 1)
    ci_95 = (mean_diff - t_crit * se, mean_diff + t_crit * se)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(y, x)

    # Cohen's d (effect size)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': mean_diff,
        'ci_95': ci_95,
    }


def main():
    canonical = canonical_rule()

    print("=" * 80)
    print("  STATISTICAL VALIDATION: CANDIDATE #2")
    print("=" * 80)
    print("\nCandidate #2 parameters:")
    for k, v in CANDIDATE2.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nSeeds: {SEEDS}")
    print(f"K values: {KS}")
    print()

    # ── TRAINING GAPS ────────────────────────────────────────────
    print("Running training gap tests...")
    canonical_training = []
    candidate2_training = []

    for i, seed in enumerate(SEEDS):
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed}")

        # Canonical
        g_canonical = per_seed_gap(canonical, KS, seed, BIRTH_SEED, alive=True)
        canonical_training.append(g_canonical)

        # Candidate #2
        g_candidate2 = per_seed_gap(CANDIDATE2, KS, seed, BIRTH_SEED, alive=True)
        candidate2_training.append(g_candidate2)

        print(f"      canonical: {g_canonical:+.4f}  |  candidate2: {g_candidate2:+.4f}  |  delta: {g_candidate2 - g_canonical:+.4f}")

    # Training summary
    print("\n" + "-" * 80)
    print("TRAINING GAP SUMMARY:")
    print("-" * 80)
    print(f"Canonical:    mean={sum(canonical_training)/len(canonical_training):+.4f}, "
          f"std={math.sqrt(sum((x - sum(canonical_training)/len(canonical_training))**2 for x in canonical_training) / (len(canonical_training)-1)):.4f}")
    print(f"Candidate #2: mean={sum(candidate2_training)/len(candidate2_training):+.4f}, "
          f"std={math.sqrt(sum((x - sum(candidate2_training)/len(candidate2_training))**2 for x in candidate2_training) / (len(candidate2_training)-1)):.4f}")

    training_stats = paired_ttest_summary(canonical_training, candidate2_training,
                                          "canonical", "candidate2")
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {training_stats['t_stat']:.4f}")
    print(f"  p-value: {training_stats['p_value']:.6f}")
    print(f"  Mean difference: {training_stats['mean_diff']:+.4f}")
    print(f"  95% CI: ({training_stats['ci_95'][0]:+.4f}, {training_stats['ci_95'][1]:+.4f})")
    print(f"  Cohen's d: {training_stats['cohens_d']:.4f}")

    training_significant = training_stats['p_value'] < 0.05 and training_stats['mean_diff'] > 0
    print(f"\nTraining result: {'SIGNIFICANT IMPROVEMENT' if training_significant else 'NOT SIGNIFICANT'}")

    # ── NOVEL GAPS ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Running novel signal gap tests...")
    canonical_novel = []
    candidate2_novel = []

    # For novel signals, we just run once per rule (not per-seed, as novel signals
    # have their own seed structure)
    print("  Computing novel gap for canonical...")
    g_canonical_novel = per_seed_novel_gap(canonical, BIRTH_SEED, NOVEL_SEED_BASE, alive=True)

    print("  Computing novel gap for Candidate #2...")
    g_candidate2_novel = per_seed_novel_gap(CANDIDATE2, BIRTH_SEED, NOVEL_SEED_BASE, alive=True)

    print(f"\n  canonical: {g_canonical_novel:+.4f}")
    print(f"  candidate2: {g_candidate2_novel:+.4f}")
    print(f"  delta: {g_candidate2_novel - g_canonical_novel:+.4f}")

    # For novel, we need multiple runs to get variance estimate
    # Let's run 5 times with different birth seeds
    print("\nRunning replications with different birth seeds for variance estimate...")
    birth_seeds = [42, 137, 314, 2024, 7]

    for bs in birth_seeds:
        g_can = per_seed_novel_gap(canonical, bs, NOVEL_SEED_BASE, alive=True)
        g_cand = per_seed_novel_gap(CANDIDATE2, bs, NOVEL_SEED_BASE, alive=True)
        canonical_novel.append(g_can)
        candidate2_novel.append(g_cand)
        print(f"  birth_seed={bs}: canonical={g_can:+.4f}, candidate2={g_cand:+.4f}, delta={g_cand - g_can:+.4f}")

    # Novel summary
    print("\n" + "-" * 80)
    print("NOVEL GAP SUMMARY:")
    print("-" * 80)
    print(f"Canonical:    mean={sum(canonical_novel)/len(canonical_novel):+.4f}, "
          f"std={math.sqrt(sum((x - sum(canonical_novel)/len(canonical_novel))**2 for x in canonical_novel) / (len(canonical_novel)-1)):.4f}")
    print(f"Candidate #2: mean={sum(candidate2_novel)/len(candidate2_novel):+.4f}, "
          f"std={math.sqrt(sum((x - sum(candidate2_novel)/len(candidate2_novel))**2 for x in candidate2_novel) / (len(candidate2_novel)-1)):.4f}")

    novel_stats = paired_ttest_summary(canonical_novel, candidate2_novel,
                                       "canonical", "candidate2")
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {novel_stats['t_stat']:.4f}")
    print(f"  p-value: {novel_stats['p_value']:.6f}")
    print(f"  Mean difference: {novel_stats['mean_diff']:+.4f}")
    print(f"  95% CI: ({novel_stats['ci_95'][0]:+.4f}, {novel_stats['ci_95'][1]:+.4f})")
    print(f"  Cohen's d: {novel_stats['cohens_d']:.4f}")

    novel_significant = novel_stats['p_value'] < 0.05 and novel_stats['mean_diff'] > 0
    print(f"\nNovel result: {'SIGNIFICANT IMPROVEMENT' if novel_significant else 'NOT SIGNIFICANT'}")

    # ── FINAL VERDICT ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL VERDICT:")
    print("=" * 80)
    print(f"Training: {'PASS' if training_significant else 'FAIL'} (p={training_stats['p_value']:.6f}, d={training_stats['cohens_d']:.4f})")
    print(f"Novel:    {'PASS' if novel_significant else 'FAIL'} (p={novel_stats['p_value']:.6f}, d={novel_stats['cohens_d']:.4f})")

    overall_pass = training_significant and novel_significant
    print(f"\nOverall: {'VALIDATED' if overall_pass else 'NOT VALIDATED'}")

    if overall_pass:
        print("\nCandidate #2 statistically beats canonical on BOTH training AND novel signals.")
        print("Ready to proceed to Stage 3 with Candidate #2 as baseline.")
    else:
        print("\nCandidate #2 does NOT meet validation criteria.")
        print("Further investigation required before proceeding to Stage 3.")

    print("=" * 80)

    return {
        'training_significant': training_significant,
        'novel_significant': novel_significant,
        'training_stats': training_stats,
        'novel_stats': novel_stats,
        'overall_pass': overall_pass,
    }


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result['overall_pass'] else 1)
