"""
Constraint Checker

Validates rule configurations against experimentally-derived constraints.
Each constraint represents a path that failed in previous experiments.

Input: A rule_config dict with plasticity parameters
Output: List of violations (or empty if valid)
"""

import json
from typing import Dict, List, Any, Tuple
from pathlib import Path


class Violation:
    """Represents a single constraint violation."""

    def __init__(self, constraint_id: str, rule: str, reason: str):
        self.constraint_id = constraint_id
        self.rule = rule
        self.reason = reason

    def __repr__(self):
        return f"Violation({self.constraint_id}: {self.reason})"

    def __str__(self):
        return f"[{self.constraint_id}] {self.rule}\n  Reason: {self.reason}"


class ConstraintChecker:
    """
    Loads constraints from .knowledge/constraints.json and validates rule configs.
    """

    def __init__(self, constraints_path: str = None):
        if constraints_path is None:
            # Default to project root .knowledge/constraints.json
            project_root = Path(__file__).parent.parent
            constraints_path = project_root / ".knowledge" / "constraints.json"

        self.constraints_path = Path(constraints_path)
        self.constraints = self._load_constraints()

    def _load_constraints(self) -> List[Dict[str, Any]]:
        """Load constraints from JSON file."""
        if not self.constraints_path.exists():
            raise FileNotFoundError(f"Constraints file not found: {self.constraints_path}")

        with open(self.constraints_path, 'r') as f:
            return json.load(f)

    def check_rule(self, rule_config: Dict[str, Any]) -> List[Violation]:
        """
        Check a rule config against all active constraints.

        Args:
            rule_config: Dict with keys:
                - eta: float (base learning rate)
                - symmetry_break_mult: float
                - amplify_mult: float
                - drift_mult: float
                - threshold: float
                - alpha_clip_lo: float
                - alpha_clip_hi: float
                - eta_adaptive: bool (whether eta adapts)
                - eta_method: str ("none", "self_referential", "delta_rz", "external")
                - beta_adaptive: bool (whether beta is per-cell adaptive)
                - gamma_adaptive: bool (whether gamma is per-cell adaptive)
                - beta_method: str (how beta adapts)
                - gamma_method: str (how gamma adapts)

        Returns:
            List of Violation objects (empty if valid)
        """
        violations = []

        # Check each active constraint
        for constraint in self.constraints:
            if not constraint.get("active", True):
                continue

            constraint_id = constraint["id"]
            rule_text = constraint["rule"]

            # Map constraint ID to check function
            check_result = self._check_constraint(constraint_id, rule_config)

            if check_result:  # If check returns a reason string, it's a violation
                violations.append(Violation(constraint_id, rule_text, check_result))

        return violations

    def is_valid(self, rule_config: Dict[str, Any]) -> bool:
        """Quick pass/fail check. Returns True if no violations."""
        return len(self.check_rule(rule_config)) == 0

    def _check_constraint(self, constraint_id: str, config: Dict[str, Any]) -> str:
        """
        Check a single constraint. Returns violation reason (str) or None if passes.

        This is where constraint IDs map to actual validation logic.
        """

        # ── c001: Response-weighted analytical gradients ──
        if constraint_id == "c001":
            # Observational: analytical gradients don't work for beta/gamma
            # This constraint is about METHOD choice, not config parameters
            # Not directly checkable from rule_config alone
            return None  # Cannot validate from static config

        # ── c002: Analytical gradient magnitude too weak ──
        if constraint_id == "c002":
            # Observational: gradient magnitude issue
            # Not directly checkable from static config
            return None

        # ── c003: No local proxy for beta/gamma ──
        if constraint_id == "c003":
            # Check if beta or gamma uses local signal method
            beta_method = config.get("beta_method", "none")
            gamma_method = config.get("gamma_method", "none")

            local_methods = ["local_proxy", "activation_frac", "response_entropy",
                           "neighbor_corr", "response_variance", "state_entropy",
                           "cross_entropy", "response_divergence"]

            if beta_method in local_methods:
                return f"beta_method='{beta_method}' uses local proxy (none exceed r>0.7 correlation)"
            if gamma_method in local_methods:
                return f"gamma_method='{gamma_method}' uses local proxy (none exceed r>0.7 correlation)"

            return None

        # ── c004: Finite-diff MI violates Principle II ──
        if constraint_id == "c004":
            # Reject finite-difference MI gradient (external measurement)
            beta_method = config.get("beta_method", "none")
            gamma_method = config.get("gamma_method", "none")

            if beta_method == "finite_diff_mi":
                return "beta_method='finite_diff_mi' violates Principle II (external signal, not self-generated)"
            if gamma_method == "finite_diff_mi":
                return "gamma_method='finite_diff_mi' violates Principle II (external signal, not self-generated)"

            return None

        # ── c005: Multiple local maxima ──
        if constraint_id == "c005":
            # Observational: MI landscape structure
            # Warning rather than hard constraint
            # Could check if beta/gamma search is enabled and warn
            return None  # Cannot prevent, only document

        # ── c006: Per-cell beta/gamma destroys performance ──
        if constraint_id == "c006":
            # Reject per-cell beta or gamma
            beta_adaptive = config.get("beta_adaptive", False)
            gamma_adaptive = config.get("gamma_adaptive", False)

            if beta_adaptive:
                return "beta_adaptive=True causes 53% MI loss (per-cell decomposition destroys global coupling)"
            if gamma_adaptive:
                return "gamma_adaptive=True causes 53% MI loss (per-cell decomposition destroys global coupling)"

            return None

        # ── c007: Beta/gamma fundamentally global ──
        if constraint_id == "c007":
            # Conceptual constraint, enforced by c006
            # Per-cell arrays violate this
            beta_adaptive = config.get("beta_adaptive", False)
            gamma_adaptive = config.get("gamma_adaptive", False)

            if beta_adaptive or gamma_adaptive:
                return "Beta/gamma are global coupling parameters, not locally decomposable"

            return None

        # ── c008: Self-referential eta causes bang-bang oscillation ──
        if constraint_id == "c008":
            # Reject self-referential eta method
            eta_method = config.get("eta_method", "none")

            if eta_method == "self_referential":
                return "eta_method='self_referential' causes bang-bang oscillation (multiplicative feedback loop)"

            return None

        # ── c009: Stage 3 meta-rate must be external to eta ──
        if constraint_id == "c009":
            # Reject eta as its own meta-rate (redundant with c008 but more explicit)
            eta_method = config.get("eta_method", "none")

            if eta_method == "self_referential":
                return "Meta-rate cannot be eta itself (Stage 3 requires external signal like delta_rz)"

            return None

        # Unknown constraint: warn but don't block
        return None


def check_rule(rule_config: Dict[str, Any]) -> List[Violation]:
    """
    Convenience function: check a rule config against constraints.
    Creates a checker instance and runs validation.
    """
    checker = ConstraintChecker()
    return checker.check_rule(rule_config)


def is_valid(rule_config: Dict[str, Any]) -> bool:
    """Convenience function: quick pass/fail check."""
    checker = ConstraintChecker()
    return checker.is_valid(rule_config)


if __name__ == "__main__":
    # Self-test: validate canonical config and known-bad configs

    print("=" * 60)
    print("CONSTRAINT CHECKER SELF-TEST")
    print("=" * 60)

    # ── Test 1: Canonical config (should pass) ──
    print("\n[Test 1] Canonical config (all defaults)")
    canonical = {
        "eta": 0.0003,
        "symmetry_break_mult": 0.3,
        "amplify_mult": 0.5,
        "drift_mult": 0.1,
        "threshold": 0.01,
        "alpha_clip_lo": 0.3,
        "alpha_clip_hi": 1.8,
        "eta_adaptive": False,
        "eta_method": "none",
        "beta_adaptive": False,
        "gamma_adaptive": False,
        "beta_method": "none",
        "gamma_method": "none",
    }

    violations = check_rule(canonical)
    if violations:
        print(f"FAIL: Expected no violations, got {len(violations)}:")
        for v in violations:
            print(f"  {v}")
    else:
        print("PASS: No violations")

    # ── Test 2: Self-referential eta (should fail c008, c009) ──
    print("\n[Test 2] Self-referential eta")
    bad_eta = canonical.copy()
    bad_eta["eta_adaptive"] = True
    bad_eta["eta_method"] = "self_referential"

    violations = check_rule(bad_eta)
    if not violations:
        print("FAIL: Expected violations for self_referential eta")
    else:
        print(f"PASS: {len(violations)} violations detected:")
        for v in violations:
            print(f"  {v}")

    # ── Test 3: Per-cell beta (should fail c006, c007) ──
    print("\n[Test 3] Per-cell beta adaptation")
    bad_beta = canonical.copy()
    bad_beta["beta_adaptive"] = True
    bad_beta["beta_method"] = "local_proxy"

    violations = check_rule(bad_beta)
    if not violations:
        print("FAIL: Expected violations for per-cell beta")
    else:
        print(f"PASS: {len(violations)} violations detected:")
        for v in violations:
            print(f"  {v}")

    # ── Test 4: Finite-diff MI (should fail c004) ──
    print("\n[Test 4] Finite-difference MI gradient")
    bad_mi = canonical.copy()
    bad_mi["gamma_method"] = "finite_diff_mi"

    violations = check_rule(bad_mi)
    if not violations:
        print("FAIL: Expected violation for finite_diff_mi")
    else:
        print(f"PASS: {len(violations)} violations detected:")
        for v in violations:
            print(f"  {v}")

    # ── Test 5: Valid Stage 3 config (delta_rz method) ──
    print("\n[Test 5] Valid Stage 3 config (delta_rz eta)")
    valid_s3 = canonical.copy()
    valid_s3["eta_adaptive"] = True
    valid_s3["eta_method"] = "delta_rz"

    violations = check_rule(valid_s3)
    if violations:
        print(f"FAIL: Expected no violations, got {len(violations)}:")
        for v in violations:
            print(f"  {v}")
    else:
        print("PASS: No violations")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Self-test complete. Review results above.")
    print("=" * 60)
