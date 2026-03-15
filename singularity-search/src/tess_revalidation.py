#!/usr/bin/env python3
"""
Tess QA: Stage 1 revalidation with dual-timescale I.

Task 4: Two tests
  Test 1: Backward compatibility — tau_slow=0.0 → alive_gap ≈ +0.1256 (±0.005)
  Test 2: Dual-timescale I active — tau_slow=0.005 → alive_gap > 0

Protocol: 10 seeds [42, 137, 2024, 999, 7, 314, 1618, 2718, 1414, 577]
          K=[4,6,8,10], n_perm=8, n_trials=6.

Acceptance criteria:
  Test 1: alive_gap within ±0.005 of 0.1256
  Test 2: alive_gap > 0
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from harness import make_signals, gen_perms, vcosine, D, NC
from anima_organism import AnimaOrganism

# Protocol constants (per session-lead instructions)
SEEDS = [42, 137, 2024, 999, 7, 314, 1618, 2718, 1414, 577]
KS = [4, 6, 8, 10]
N_PERM = 8
N_TRIALS = 6
BIRTH_SEED = 42

# Canonical params from Session 19
BASE_PARAMS = {
    'w_lr':   0.0003,
    'tau':    0.3,
    'gamma':  3.0,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}


def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    """Run one sequence through org. Returns (centroid, xs)."""
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    # Organism warm-up (no signal)
    for _ in range(n_org):
        xs = org.step(xs)

    # Signal sequence
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    # Final settle
    for _ in range(n_final):
        xs = org.step(xs)

    return org.centroid(xs), xs


def measure_gap(org, signals, k, seed, n_perm=8, n_trials=6):
    """MI gap = avg_within_cosine - avg_between_cosine."""
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


def run_comparison(rule_params, label="", verbose=True):
    """Run STILL vs ALIVE comparison with given rule_params."""
    still_gaps = []
    alive_gaps = []

    total = len(KS) * len(SEEDS)
    done = 0

    for k in KS:
        sig_seed = BIRTH_SEED + k * 200
        sigs = make_signals(k, seed=sig_seed)

        for s in SEEDS:
            done += 1
            if verbose:
                print(f"  [{done:2d}/{total}] K={k} seed={s} ...", flush=True)

            org_still = AnimaOrganism(seed=BIRTH_SEED, alive=False, rule_params=rule_params)
            g_still = measure_gap(org_still, sigs, k, s, n_perm=N_PERM, n_trials=N_TRIALS)
            still_gaps.append(g_still)

            org_alive = AnimaOrganism(seed=BIRTH_SEED, alive=True, rule_params=rule_params)
            g_alive = measure_gap(org_alive, sigs, k, s, n_perm=N_PERM, n_trials=N_TRIALS)
            alive_gaps.append(g_alive)

            if verbose:
                print(f"         still={g_still:+.4f}  alive={g_alive:+.4f}  delta={g_alive - g_still:+.4f}")

    still_avg = sum(still_gaps) / len(still_gaps)
    alive_avg = sum(alive_gaps) / len(alive_gaps)
    gap_delta = alive_avg - still_avg

    still_std = math.sqrt(sum((g - still_avg)**2 for g in still_gaps) / len(still_gaps))
    alive_std = math.sqrt(sum((g - alive_avg)**2 for g in alive_gaps) / len(alive_gaps))

    pooled_std = math.sqrt((still_std**2 + alive_std**2) / 2.0 + 1e-15)
    d_stat = gap_delta / pooled_std

    ground_truth_pass = alive_avg > 0.0

    return {
        'label': label,
        'still_gap': still_avg,
        'still_std': still_std,
        'alive_gap': alive_avg,
        'alive_std': alive_std,
        'gap_delta': gap_delta,
        'd_stat': d_stat,
        'ground_truth_pass': ground_truth_pass,
    }


def print_result(r, acceptance_check=None):
    print(f"\n  --- {r['label']} ---")
    print(f"  STILL  gap: {r['still_gap']:+.4f}  (std={r['still_std']:.4f})")
    print(f"  ALIVE  gap: {r['alive_gap']:+.4f}  (std={r['alive_std']:.4f})")
    print(f"  Delta:      {r['gap_delta']:+.4f}")
    print(f"  d-stat:     {r['d_stat']:+.4f}")
    print(f"  Ground truth (alive_gap > 0): {'PASS' if r['ground_truth_pass'] else 'FAIL'}")
    if acceptance_check:
        ok, msg = acceptance_check(r)
        print(f"  Acceptance: {'PASS' if ok else 'FAIL'} — {msg}")
    return r


if __name__ == '__main__':
    print("=" * 72)
    print("  TESS QA — Stage 1 Revalidation: Dual-Timescale I")
    print("  Protocol: 10 seeds, K=[4,6,8,10], n_perm=8, n_trials=6")
    print("=" * 72)

    # ── Test 1: Backward compatibility (tau_slow=0.0) ──────────────────
    print("\n\n[TEST 1] Backward compatibility: tau_slow=0.0")
    print("  Expected: alive_gap ~= +0.1256 (+/-0.005)")
    print("-" * 72)

    params_t1 = dict(BASE_PARAMS, tau_slow=0.0)
    r1 = run_comparison(params_t1, label="Test1 tau_slow=0.0 (backward compat)")

    EXPECTED_GAP = 0.1256
    TOLERANCE = 0.005

    def check_t1(r):
        deviation = abs(r['alive_gap'] - EXPECTED_GAP)
        ok = r['ground_truth_pass'] and deviation <= TOLERANCE
        return ok, f"alive_gap={r['alive_gap']:+.4f}, deviation={deviation:.4f}, tolerance={TOLERANCE}"

    print_result(r1, check_t1)
    t1_ok, _ = check_t1(r1)

    # ── Test 2: Dual-timescale I active (tau_slow=0.005) ───────────────
    print("\n\n[TEST 2] Dual-timescale I active: tau_slow=0.005")
    print("  Expected: alive_gap > 0")
    print("-" * 72)

    params_t2 = dict(BASE_PARAMS, tau_slow=0.005)
    r2 = run_comparison(params_t2, label="Test2 tau_slow=0.005 (dual-timescale)")

    def check_t2(r):
        ok = r['alive_gap'] > 0.0
        return ok, f"alive_gap={r['alive_gap']:+.4f}"

    print_result(r2, check_t2)
    t2_ok, _ = check_t2(r2)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n")
    print("=" * 72)
    print("  TASK 4 SUMMARY")
    print("=" * 72)
    print(f"  Test 1 (backward compat): {'PASS' if t1_ok else 'FAIL'}")
    print(f"  Test 2 (dual-timescale):  {'PASS' if t2_ok else 'FAIL'}")
    overall = t1_ok and t2_ok
    print(f"  Overall: {'PASS — ready for Task 5' if overall else 'FAIL — halt, report to team-lead'}")
    print("=" * 72)

    if not overall:
        sys.exit(1)
