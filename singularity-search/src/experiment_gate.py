#!/usr/bin/env python3
"""
Experiment Gate: Phased Compute Budget Protocol

Enforces a 3-phase validation pipeline before expensive experiments.
Born from Sessions 7-8 where ~45% compute was wasted on 3-seed false
positives that required demolition sessions to retract.

Phase 1: THEORY (seconds)
    Does the signal/approach have the right mathematical properties?
    - Check against known constraints
    - Verify autocorrelation structure
    - Verify Principle II compliance (self-generated signal)
    - Verify Principle IV (net frozen frame reduction)
    Cost: 1 simulation run or pure analysis

Phase 2: SINGLE-CANDIDATE VALIDATION (minutes)
    One candidate, full statistical protocol.
    - 10 seeds, 2x exposure (n_perm=8, n_trials=6)
    - Paired comparison vs canonical baseline
    - Compute effect size (Cohen's d) and p-value
    Cost: ~10x a 3-seed run

Phase 3: SEARCH (hours)
    Only if Phase 2 shows genuine signal.
    - Evolutionary/grid search with 10+ seed eval per candidate
    - Full validation of best candidates
    Cost: hundreds of simulation runs

Usage:
    from experiment_gate import ExperimentGate
    gate = ExperimentGate("my_experiment")
    gate.phase1_theory(signal_props, frozen_frame_delta, principle_ii=True)
    gate.phase2_validate(candidate_rule, baseline_rule)
    gate.phase3_search(search_fn)  # only unlocks if phase2 passed
"""

import json
import math
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

VALIDATION_SEEDS = [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
VALIDATION_N_PERM = 8
VALIDATION_N_TRIALS = 6
PHASE2_MIN_EFFECT_SIZE = 0.3   # Cohen's d threshold for "worth searching"
PHASE2_MIN_P_TREND = 0.20      # p-value threshold for "not obviously noise"


# ═══════════════════════════════════════════════════════════════
# Phase 1: Theory Check
# ═══════════════════════════════════════════════════════════════

class Phase1Result:
    def __init__(self):
        self.passed = False
        self.checks = {}
        self.blockers = []
        self.warnings = []

    def summary(self):
        status = "PASS" if self.passed else "BLOCKED"
        lines = [f"Phase 1 (Theory): {status}"]
        for name, result in self.checks.items():
            mark = "+" if result['passed'] else "X"
            lines.append(f"  [{mark}] {name}: {result['detail']}")
        if self.blockers:
            lines.append(f"  BLOCKERS: {'; '.join(self.blockers)}")
        if self.warnings:
            lines.append(f"  WARNINGS: {'; '.join(self.warnings)}")
        return "\n".join(lines)


def check_constraints(approach_tags: List[str]) -> tuple:
    """Check approach against known constraints."""
    project_root = Path(__file__).parent.parent
    constraints_path = project_root / ".knowledge" / "constraints.json"

    if not constraints_path.exists():
        return True, "constraints.json not found (skipping)"

    with open(constraints_path) as f:
        constraints = json.load(f)

    violations = []
    for c in constraints:
        if not c.get("active", True):
            continue
        # Check if any approach tag matches constraint tags
        overlap = set(approach_tags) & set(c.get("tags", []))
        if overlap:
            violations.append(f"[{c['id']}] {c['rule']} (tags: {overlap})")

    if violations:
        return False, violations
    return True, "No constraint violations"


def check_autocorrelation(signal_autocorr: float, min_autocorr: float = 0.3) -> tuple:
    """Check if signal has sufficient temporal structure."""
    if signal_autocorr >= min_autocorr:
        return True, f"r={signal_autocorr:.3f} >= {min_autocorr} (sufficient structure)"
    return False, f"r={signal_autocorr:.3f} < {min_autocorr} (signal is noise at this timescale)"


def check_frozen_frame_delta(added: int, removed: int) -> tuple:
    """Check Principle IV: net frozen frame reduction."""
    if added == 0 and removed > 0:
        return True, f"Net: -{removed} frozen elements (pure reduction)"
    if removed >= 2 * added:
        return True, f"Net: +{added} -{removed} = {added - removed} (2:1 ratio satisfied)"
    if removed > added:
        return True, f"Net: +{added} -{removed} = {added - removed} (net reduction)"
    return False, f"Net: +{added} -{removed} = {added - removed} (frozen frame grows or stalls)"


def check_principle_ii(is_self_generated: bool, signal_description: str = "") -> tuple:
    """Check Principle II: signal from computation, not beside it."""
    if is_self_generated:
        return True, f"Signal is self-generated{': ' + signal_description if signal_description else ''}"
    return False, f"Signal requires external evaluator{': ' + signal_description if signal_description else ''}"


# ═══════════════════════════════════════════════════════════════
# Phase 2: Single-Candidate Validation
# ═══════════════════════════════════════════════════════════════

class Phase2Result:
    def __init__(self):
        self.passed = False
        self.effect_size = 0.0
        self.p_value = 1.0
        self.variant_gaps = []
        self.baseline_gaps = []
        self.mean_delta = 0.0
        self.cv_variant = 0.0
        self.cv_baseline = 0.0
        self.n_seeds = 0
        self.wall_time = 0.0

    def summary(self):
        status = "PASS" if self.passed else "BLOCKED"
        lines = [
            f"Phase 2 (Validation): {status}",
            f"  Seeds: {self.n_seeds}",
            f"  Baseline: mean={_mean(self.baseline_gaps):+.4f}, CV={self.cv_baseline:.1%}",
            f"  Variant:  mean={_mean(self.variant_gaps):+.4f}, CV={self.cv_variant:.1%}",
            f"  Delta: {self.mean_delta:+.4f}",
            f"  Effect size (d): {self.effect_size:.3f} (threshold: {PHASE2_MIN_EFFECT_SIZE})",
            f"  p-value: {self.p_value:.4f} (threshold: {PHASE2_MIN_P_TREND})",
            f"  Wall time: {self.wall_time:.1f}s",
        ]
        if not self.passed:
            if self.effect_size < PHASE2_MIN_EFFECT_SIZE:
                lines.append(f"  BLOCKED: Effect size too small (d={self.effect_size:.3f} < {PHASE2_MIN_EFFECT_SIZE})")
            if self.p_value > PHASE2_MIN_P_TREND:
                lines.append(f"  BLOCKED: p-value too high (p={self.p_value:.4f} > {PHASE2_MIN_P_TREND})")
        return "\n".join(lines)


def paired_t_test(x: List[float], y: List[float]) -> tuple:
    """Paired t-test. Returns (t_stat, p_value, cohen_d)."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0, 1.0, 0.0

    diffs = [xi - yi for xi, yi in zip(x, y)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d) if var_d > 0 else 1e-10
    se_d = sd_d / math.sqrt(n)

    t_stat = mean_d / se_d if se_d > 1e-15 else 0.0

    # Two-tailed p-value approximation (t-distribution with n-1 df)
    # Using the approximation from Abramowitz and Stegun
    df = n - 1
    p_value = _t_to_p(abs(t_stat), df) * 2  # two-tailed

    # Cohen's d (paired)
    cohen_d = mean_d / sd_d if sd_d > 1e-15 else 0.0

    return t_stat, min(p_value, 1.0), cohen_d


def _t_to_p(t, df):
    """Approximate one-tailed p-value from t-statistic and degrees of freedom."""
    # Welch-Satterthwaite approximation using normal for large df
    if df >= 30:
        # Normal approximation
        return 0.5 * math.erfc(t / math.sqrt(2))

    # For small df, use a rough beta-function approximation
    x = df / (df + t * t)
    # Regularized incomplete beta function approximation
    # This is rough but sufficient for gating decisions
    if t < 0.5:
        return 0.5
    elif t < 1.0:
        return 0.5 * x ** (df / 2)
    elif t < 2.0:
        return 0.25 * x ** (df / 2)
    elif t < 3.0:
        return 0.05 * x ** (df / 2)
    else:
        return 0.01 * x ** (df / 2)


# ═══════════════════════════════════════════════════════════════
# Phase 3: Search Gate
# ═══════════════════════════════════════════════════════════════

class Phase3Result:
    def __init__(self):
        self.passed = False
        self.reason = ""

    def summary(self):
        status = "UNLOCKED" if self.passed else "LOCKED"
        return f"Phase 3 (Search): {status} — {self.reason}"


# ═══════════════════════════════════════════════════════════════
# Main Gate
# ═══════════════════════════════════════════════════════════════

class ExperimentGate:
    """
    Enforces phased compute budgeting.

    Usage:
        gate = ExperimentGate("stage3_windowed_delta_rz")

        # Phase 1: Theory
        p1 = gate.phase1_theory(
            approach_tags=["stage3", "eta", "delta_rz"],
            signal_autocorr=0.45,
            frozen_added=1,
            frozen_removed=2,
            principle_ii=True,
            signal_description="windowed average of delta_rz over 5 steps"
        )
        print(p1.summary())

        # Phase 2: Validate (only if Phase 1 passed)
        if gate.phase1_passed:
            p2 = gate.phase2_validate(variant_rule, eval_fn)
            print(p2.summary())

        # Phase 3: Search (only if Phase 2 passed)
        if gate.phase2_passed:
            p3 = gate.phase3_approve()
            print(p3.summary())
    """

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.phase1_passed = False
        self.phase2_passed = False
        self.phase3_passed = False
        self.results = {}
        self.log = []

    def _log(self, msg: str):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def phase1_theory(
        self,
        approach_tags: List[str],
        signal_autocorr: Optional[float] = None,
        frozen_added: int = 0,
        frozen_removed: int = 0,
        principle_ii: bool = True,
        signal_description: str = "",
        custom_checks: Optional[Dict[str, tuple]] = None,
    ) -> Phase1Result:
        """
        Phase 1: Theory check. Cheap, seconds.

        Args:
            approach_tags: Tags describing the approach (matched against constraints)
            signal_autocorr: Lag-1 autocorrelation of the driving signal (None to skip)
            frozen_added: Number of new frozen frame elements introduced
            frozen_removed: Number of frozen frame elements eliminated
            principle_ii: Whether the signal is self-generated
            signal_description: Human-readable description of the signal source
            custom_checks: Dict of {name: (passed, detail)} for additional checks

        Returns:
            Phase1Result
        """
        self._log(f"Phase 1 starting for '{self.name}'")
        result = Phase1Result()

        # Check 1: Constraint violations
        passed, detail = check_constraints(approach_tags)
        if not passed:
            # Constraint matches are warnings, not blockers —
            # the approach might use the same tags but a different method
            result.checks["constraints"] = {
                "passed": True,  # tag overlap is informational
                "detail": f"Potential overlaps (review manually): {len(detail)} matches"
            }
            result.warnings.extend(detail)
        else:
            result.checks["constraints"] = {"passed": True, "detail": detail}

        # Check 2: Signal autocorrelation
        if signal_autocorr is not None:
            passed, detail = check_autocorrelation(signal_autocorr)
            result.checks["autocorrelation"] = {"passed": passed, "detail": detail}
            if not passed:
                result.blockers.append(f"Signal autocorrelation too low: r={signal_autocorr:.3f}")

        # Check 3: Frozen frame delta (Principle IV)
        passed, detail = check_frozen_frame_delta(frozen_added, frozen_removed)
        result.checks["frozen_frame"] = {"passed": passed, "detail": detail}
        if not passed:
            result.blockers.append(f"Frozen frame violation: +{frozen_added} -{frozen_removed}")

        # Check 4: Principle II
        passed, detail = check_principle_ii(principle_ii, signal_description)
        result.checks["principle_ii"] = {"passed": passed, "detail": detail}
        if not passed:
            result.blockers.append("Principle II violation: external signal")

        # Custom checks
        if custom_checks:
            for name, (passed, detail) in custom_checks.items():
                result.checks[name] = {"passed": passed, "detail": detail}
                if not passed:
                    result.blockers.append(f"Custom check '{name}' failed: {detail}")

        result.passed = len(result.blockers) == 0
        self.phase1_passed = result.passed
        self.results["phase1"] = result

        self._log(f"Phase 1: {'PASS' if result.passed else 'BLOCKED'} ({len(result.blockers)} blockers)")
        return result

    def phase2_validate(
        self,
        variant_rule: Dict[str, Any],
        eval_fn: Callable,
        seeds: Optional[List[int]] = None,
        birth_seed: int = 42,
    ) -> Phase2Result:
        """
        Phase 2: Single-candidate validation. Full protocol, 10 seeds.

        Args:
            variant_rule: Rule configuration dict for the variant
            eval_fn: Function(rule_params, seed, birth_seed) -> float (training gap)
                Must accept a rule_params dict, an evaluation seed, and a birth_seed.
                Returns the training gap (higher = better).
            seeds: Evaluation seeds (default: VALIDATION_SEEDS)
            birth_seed: Organism birth seed

        Returns:
            Phase2Result
        """
        if not self.phase1_passed:
            result = Phase2Result()
            result.passed = False
            self._log("Phase 2 SKIPPED: Phase 1 not passed")
            self.results["phase2"] = result
            return result

        self._log(f"Phase 2 starting for '{self.name}'")
        t0 = time.time()

        if seeds is None:
            seeds = VALIDATION_SEEDS

        result = Phase2Result()
        result.n_seeds = len(seeds)

        # Import harness for canonical rule
        from harness import canonical_rule
        baseline_rule = canonical_rule()

        # Evaluate both rules on all seeds
        for seed in seeds:
            baseline_gap = eval_fn(baseline_rule, seed, birth_seed)
            variant_gap = eval_fn(variant_rule, seed, birth_seed)
            result.baseline_gaps.append(baseline_gap)
            result.variant_gaps.append(variant_gap)

        # Statistics
        t_stat, p_value, cohen_d = paired_t_test(result.variant_gaps, result.baseline_gaps)
        result.effect_size = cohen_d
        result.p_value = p_value
        result.mean_delta = _mean(result.variant_gaps) - _mean(result.baseline_gaps)
        result.cv_baseline = _cv(result.baseline_gaps)
        result.cv_variant = _cv(result.variant_gaps)
        result.wall_time = time.time() - t0

        # Gate decision
        result.passed = (
            result.effect_size >= PHASE2_MIN_EFFECT_SIZE
            and result.p_value <= PHASE2_MIN_P_TREND
        )
        self.phase2_passed = result.passed
        self.results["phase2"] = result

        self._log(f"Phase 2: {'PASS' if result.passed else 'BLOCKED'} "
                  f"(d={cohen_d:.3f}, p={p_value:.4f}, delta={result.mean_delta:+.4f})")
        return result

    def phase3_approve(self) -> Phase3Result:
        """
        Phase 3: Check if search is approved.

        Returns Phase3Result indicating whether expensive search can proceed.
        """
        result = Phase3Result()

        if not self.phase1_passed:
            result.passed = False
            result.reason = "Phase 1 not passed — fix theory issues first"
        elif not self.phase2_passed:
            result.passed = False
            result.reason = "Phase 2 not passed — single candidate did not show genuine signal"
        else:
            result.passed = True
            result.reason = (
                f"Phases 1+2 passed. "
                f"Effect size d={self.results['phase2'].effect_size:.3f}, "
                f"p={self.results['phase2'].p_value:.4f}. "
                f"Search approved with 10+ seed evaluation per candidate."
            )

        self.phase3_passed = result.passed
        self.results["phase3"] = result
        self._log(f"Phase 3: {'UNLOCKED' if result.passed else 'LOCKED'}")
        return result

    def full_report(self) -> str:
        """Generate complete gate report."""
        lines = [
            "=" * 60,
            f"  EXPERIMENT GATE REPORT: {self.name}",
            "=" * 60,
        ]

        for phase_name in ["phase1", "phase2", "phase3"]:
            if phase_name in self.results:
                lines.append("")
                lines.append(self.results[phase_name].summary())

        lines.append("")
        lines.append("-" * 60)
        lines.append("Log:")
        for entry in self.log:
            lines.append(f"  {entry}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate state for knowledge base ingestion."""
        d = {
            "experiment": self.name,
            "phase1_passed": self.phase1_passed,
            "phase2_passed": self.phase2_passed,
            "phase3_passed": self.phase3_passed,
        }
        if "phase2" in self.results:
            p2 = self.results["phase2"]
            d["phase2_stats"] = {
                "effect_size": p2.effect_size,
                "p_value": p2.p_value,
                "mean_delta": p2.mean_delta,
                "cv_baseline": p2.cv_baseline,
                "cv_variant": p2.cv_variant,
                "n_seeds": p2.n_seeds,
                "wall_time": p2.wall_time,
            }
        return d


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _cv(xs):
    m = _mean(xs)
    if abs(m) < 1e-15:
        return float('inf')
    return _std(xs) / abs(m)


# ═══════════════════════════════════════════════════════════════
# Standalone eval helper (wraps harness for Phase 2)
# ═══════════════════════════════════════════════════════════════

def make_eval_fn(ks=(4, 6, 8, 10), n_perm=VALIDATION_N_PERM, n_trials=VALIDATION_N_TRIALS):
    """
    Creates an evaluation function compatible with Phase 2.

    Returns:
        fn(rule_params, seed, birth_seed) -> float (training gap)
    """
    from harness import Organism, make_signals, measure_gap

    def eval_fn(rule_params, seed, birth_seed):
        gaps = []
        for k in ks:
            sig_seed = birth_seed + k * 200
            sigs = make_signals(k, seed=sig_seed)
            org = Organism(seed=birth_seed, alive=True, rule_params=rule_params)
            g = measure_gap(org, sigs, k, seed, n_perm=n_perm, n_trials=n_trials)
            gaps.append(g)
        return _mean(gaps)

    return eval_fn


# ═══════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EXPERIMENT GATE SELF-TEST")
    print("=" * 60)

    # Test 1: Phase 1 that should pass
    print("\n[Test 1] Phase 1: Valid approach")
    gate = ExperimentGate("test_valid")
    p1 = gate.phase1_theory(
        approach_tags=["stage3", "eta"],
        signal_autocorr=0.45,
        frozen_added=1,
        frozen_removed=2,
        principle_ii=True,
        signal_description="windowed delta_rz average"
    )
    print(p1.summary())
    assert p1.passed, "Expected Phase 1 to pass"

    # Test 2: Phase 1 that should fail (low autocorrelation)
    print("\n[Test 2] Phase 1: Low autocorrelation")
    gate2 = ExperimentGate("test_low_autocorr")
    p1 = gate2.phase1_theory(
        approach_tags=["stage3"],
        signal_autocorr=0.05,
        frozen_added=0,
        frozen_removed=1,
        principle_ii=True,
    )
    print(p1.summary())
    assert not p1.passed, "Expected Phase 1 to fail"

    # Test 3: Phase 1 that should fail (Principle II violation)
    print("\n[Test 3] Phase 1: Principle II violation")
    gate3 = ExperimentGate("test_external")
    p1 = gate3.phase1_theory(
        approach_tags=["stage3"],
        signal_autocorr=0.8,
        frozen_added=0,
        frozen_removed=1,
        principle_ii=False,
        signal_description="finite-diff MI gradient"
    )
    print(p1.summary())
    assert not p1.passed, "Expected Phase 1 to fail"

    # Test 4: Phase 1 that should fail (frozen frame grows)
    print("\n[Test 4] Phase 1: Frozen frame grows")
    gate4 = ExperimentGate("test_frozen_grows")
    p1 = gate4.phase1_theory(
        approach_tags=["stage3"],
        signal_autocorr=0.5,
        frozen_added=3,
        frozen_removed=1,
        principle_ii=True,
    )
    print(p1.summary())
    assert not p1.passed, "Expected Phase 1 to fail"

    # Test 5: Phase 2 blocked because Phase 1 failed
    print("\n[Test 5] Phase 2: Blocked by Phase 1")
    p2 = gate4.phase2_validate({}, lambda r, s, b: 0.0)
    assert not p2.passed, "Expected Phase 2 to be blocked"
    print("  Phase 2 correctly blocked.")

    # Test 6: Phase 3 locked
    print("\n[Test 6] Phase 3: Locked")
    p3 = gate4.phase3_approve()
    print(p3.summary())
    assert not p3.passed, "Expected Phase 3 to be locked"

    # Test 7: Paired t-test
    print("\n[Test 7] Paired t-test")
    x = [0.2, 0.3, 0.25, 0.28, 0.32]
    y = [0.1, 0.15, 0.12, 0.14, 0.18]
    t, p, d = paired_t_test(x, y)
    print(f"  t={t:.3f}, p={p:.4f}, d={d:.3f}")
    assert d > 1.0, f"Expected large effect size, got d={d:.3f}"
    assert p < 0.05, f"Expected significant p-value, got p={p:.4f}"

    print("\n" + "=" * 60)
    print("  ALL SELF-TESTS PASSED")
    print("=" * 60)
