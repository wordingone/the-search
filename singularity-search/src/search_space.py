#!/usr/bin/env python3
"""
Search Space for Plasticity Rules

Defines the space of possible plasticity rules and operations for exploring it.

Encodes constraints from .knowledge/constraints.json:
- c008: Avoid multiplicative self-referential meta-rates
- c009: Meta-rate must be external to eta
"""

import random
import math


# ═══════════════════════════════════════════════════════════════
# Parameter Bounds
# ═══════════════════════════════════════════════════════════════

PARAM_BOUNDS = {
    # Learning rate: canonical=0.0003, tested range [0.0001, 0.001]
    'eta': {
        'min': 0.00005,
        'max': 0.005,
        'canonical': 0.0003,
        'log_scale': True,  # search in log space
    },

    # Symmetry breaking multiplier: push scale at column mean
    # canonical=0.3, needs to be strong enough to break symmetry
    'symmetry_break_mult': {
        'min': 0.05,
        'max': 1.0,
        'canonical': 0.3,
        'log_scale': False,
    },

    # Amplification multiplier: push scale when resp_z > 0
    # canonical=0.5, amplifies diversity on sensitive dimensions
    'amplify_mult': {
        'min': 0.1,
        'max': 2.0,
        'canonical': 0.5,
        'log_scale': False,
    },

    # Drift multiplier: push scale when resp_z <= 0
    # canonical=0.1, gentle random drift
    'drift_mult': {
        'min': 0.01,
        'max': 0.5,
        'canonical': 0.1,
        'log_scale': False,
    },

    # Threshold for "at column mean" detection
    # canonical=0.01, too large -> never triggers symmetry break
    'threshold': {
        'min': 0.001,
        'max': 0.1,
        'canonical': 0.01,
        'log_scale': True,
    },

    # Alpha clip bounds: prevent runaway growth
    # canonical: [0.3, 1.8], needs headroom for specialization
    'alpha_clip_lo': {
        'min': 0.1,
        'max': 0.8,
        'canonical': 0.3,
        'log_scale': False,
    },

    'alpha_clip_hi': {
        'min': 1.2,
        'max': 3.0,
        'canonical': 1.8,
        'log_scale': False,
    },
}


# ═══════════════════════════════════════════════════════════════
# Constraints
# ═══════════════════════════════════════════════════════════════

def validate_rule(rule_params):
    """
    Checks if a rule satisfies hard constraints.

    Returns: (is_valid, violations)
    """
    violations = []

    # Constraint: alpha_clip_lo < alpha_clip_hi
    if rule_params['alpha_clip_lo'] >= rule_params['alpha_clip_hi']:
        violations.append('alpha_clip_lo must be < alpha_clip_hi')

    # Constraint: clip bounds must contain the canonical birth alpha range [0.4, 1.8]
    # (from the_living_seed.py: 1.1 ± 0.7 = [0.4, 1.8])
    if rule_params['alpha_clip_lo'] > 0.4:
        violations.append('alpha_clip_lo > 0.4 (birth alphas start at 1.1±0.7)')

    if rule_params['alpha_clip_hi'] < 1.8:
        violations.append('alpha_clip_hi < 1.8 (birth alphas start at 1.1±0.7)')

    # Constraint: eta must be positive
    if rule_params['eta'] <= 0:
        violations.append('eta must be > 0')

    # Constraint: multipliers must be non-negative
    for key in ['symmetry_break_mult', 'amplify_mult', 'drift_mult']:
        if rule_params[key] < 0:
            violations.append(f'{key} must be >= 0')

    # Constraint: threshold must be positive
    if rule_params['threshold'] <= 0:
        violations.append('threshold must be > 0')

    # Constraint: amplify_mult should be larger than drift_mult
    # (otherwise strong responses get weaker pushes than weak responses)
    if rule_params['amplify_mult'] < rule_params['drift_mult']:
        violations.append('amplify_mult should be >= drift_mult (otherwise inverted incentives)')

    return len(violations) == 0, violations


# ═══════════════════════════════════════════════════════════════
# Rule Configuration API
# ═══════════════════════════════════════════════════════════════

def canonical_rule():
    """Returns the canonical ALIVE plasticity rule from the_living_seed.py."""
    return {
        'eta': 0.0003,
        'symmetry_break_mult': 0.3,
        'amplify_mult': 0.5,
        'drift_mult': 0.1,
        'threshold': 0.01,
        'alpha_clip_lo': 0.3,
        'alpha_clip_hi': 1.8,
    }


def sample_rule(seed=None):
    """
    Samples a random rule from the search space.

    Returns: rule_params dict
    """
    if seed is not None:
        random.seed(seed)

    rule = {}
    for param, bounds in PARAM_BOUNDS.items():
        if bounds['log_scale']:
            # Sample uniformly in log space
            log_min = math.log(bounds['min'])
            log_max = math.log(bounds['max'])
            log_val = random.uniform(log_min, log_max)
            rule[param] = math.exp(log_val)
        else:
            # Sample uniformly in linear space
            rule[param] = random.uniform(bounds['min'], bounds['max'])

    # Resample until constraints are satisfied (max 100 attempts)
    for _ in range(100):
        is_valid, _ = validate_rule(rule)
        if is_valid:
            return rule
        # Resample failed constraints
        rule['alpha_clip_lo'] = random.uniform(
            PARAM_BOUNDS['alpha_clip_lo']['min'],
            min(PARAM_BOUNDS['alpha_clip_lo']['max'], 0.4)
        )
        rule['alpha_clip_hi'] = random.uniform(
            max(PARAM_BOUNDS['alpha_clip_hi']['min'], 1.8),
            PARAM_BOUNDS['alpha_clip_hi']['max']
        )
        if rule['amplify_mult'] < rule['drift_mult']:
            rule['amplify_mult'], rule['drift_mult'] = rule['drift_mult'], rule['amplify_mult']

    # Fallback: return canonical if sampling fails
    return canonical_rule()


def mutate_rule(rule, sigma=0.2, seed=None):
    """
    Applies gaussian perturbation to a rule.

    Args:
        rule: rule_params dict
        sigma: perturbation scale (fraction of parameter range)
        seed: random seed

    Returns: mutated rule_params dict
    """
    if seed is not None:
        random.seed(seed)

    mutated = {}
    for param, bounds in PARAM_BOUNDS.items():
        val = rule[param]

        if bounds['log_scale']:
            # Perturb in log space
            log_val = math.log(val)
            log_range = math.log(bounds['max']) - math.log(bounds['min'])
            log_val += random.gauss(0, sigma * log_range)
            log_val = max(math.log(bounds['min']), min(math.log(bounds['max']), log_val))
            mutated[param] = math.exp(log_val)
        else:
            # Perturb in linear space
            param_range = bounds['max'] - bounds['min']
            val += random.gauss(0, sigma * param_range)
            val = max(bounds['min'], min(bounds['max'], val))
            mutated[param] = val

    # Enforce constraints
    mutated['alpha_clip_lo'] = min(mutated['alpha_clip_lo'], 0.4)
    mutated['alpha_clip_hi'] = max(mutated['alpha_clip_hi'], 1.8)
    if mutated['amplify_mult'] < mutated['drift_mult']:
        mutated['amplify_mult'], mutated['drift_mult'] = mutated['drift_mult'], mutated['amplify_mult']

    return mutated


def crossover_rule(rule1, rule2, seed=None):
    """
    Blends two rules via uniform crossover.

    Args:
        rule1, rule2: rule_params dicts
        seed: random seed

    Returns: offspring rule_params dict
    """
    if seed is not None:
        random.seed(seed)

    offspring = {}
    for param in PARAM_BOUNDS.keys():
        # Blend with random weight
        w = random.random()
        val1 = rule1[param]
        val2 = rule2[param]

        if PARAM_BOUNDS[param]['log_scale']:
            # Blend in log space
            log_val = w * math.log(val1) + (1 - w) * math.log(val2)
            offspring[param] = math.exp(log_val)
        else:
            # Blend in linear space
            offspring[param] = w * val1 + (1 - w) * val2

    # Enforce constraints
    offspring['alpha_clip_lo'] = min(offspring['alpha_clip_lo'], 0.4)
    offspring['alpha_clip_hi'] = max(offspring['alpha_clip_hi'], 1.8)
    if offspring['amplify_mult'] < offspring['drift_mult']:
        offspring['amplify_mult'], offspring['drift_mult'] = offspring['drift_mult'], offspring['amplify_mult']

    return offspring


def grid_sample(param_name, n_samples):
    """
    Samples a parameter along a grid, holding others at canonical values.

    Args:
        param_name: parameter to vary
        n_samples: number of grid points

    Returns: list of rule_params dicts
    """
    if param_name not in PARAM_BOUNDS:
        raise ValueError(f"Unknown parameter: {param_name}")

    bounds = PARAM_BOUNDS[param_name]
    canonical = canonical_rule()
    rules = []

    if bounds['log_scale']:
        # Grid in log space
        log_min = math.log(bounds['min'])
        log_max = math.log(bounds['max'])
        log_vals = [log_min + i * (log_max - log_min) / (n_samples - 1)
                    for i in range(n_samples)]
        vals = [math.exp(lv) for lv in log_vals]
    else:
        # Grid in linear space
        vals = [bounds['min'] + i * (bounds['max'] - bounds['min']) / (n_samples - 1)
                for i in range(n_samples)]

    for val in vals:
        rule = canonical.copy()
        rule[param_name] = val
        rules.append(rule)

    return rules


# ═══════════════════════════════════════════════════════════════
# Constrained Search Spaces
# ═══════════════════════════════════════════════════════════════

def eta_sweep_rules(etas=None):
    """
    Generates rules for eta sweep (Session 1 style).

    Args:
        etas: list of eta values (default: [0.0001, 0.0003, 0.001])

    Returns: list of rule_params dicts
    """
    if etas is None:
        etas = [0.0001, 0.0003, 0.001]

    canonical = canonical_rule()
    rules = []
    for eta in etas:
        rule = canonical.copy()
        rule['eta'] = eta
        rules.append(rule)
    return rules


def balance_sweep_rules(n_samples=7):
    """
    Sweeps the amplify_mult / drift_mult balance.

    Explores the tradeoff between amplifying strong responses vs
    allowing drift on weak responses.

    Returns: list of rule_params dicts
    """
    canonical = canonical_rule()
    rules = []

    # Hold sum constant, vary ratio
    total = canonical['amplify_mult'] + canonical['drift_mult']  # 0.6
    ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99][:n_samples]

    for r in ratios:
        rule = canonical.copy()
        rule['amplify_mult'] = r * total
        rule['drift_mult'] = (1 - r) * total
        rules.append(rule)

    return rules


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def rule_distance(rule1, rule2):
    """
    Computes normalized L2 distance between two rules.

    Returns: float in [0, 1] (approximately)
    """
    diffs = []
    for param, bounds in PARAM_BOUNDS.items():
        val1 = rule1[param]
        val2 = rule2[param]

        if bounds['log_scale']:
            # Distance in log space
            d = (math.log(val1) - math.log(val2)) / (
                math.log(bounds['max']) - math.log(bounds['min'])
            )
        else:
            # Distance in linear space
            d = (val1 - val2) / (bounds['max'] - bounds['min'])

        diffs.append(d * d)

    return math.sqrt(sum(diffs) / len(diffs))


def format_rule(rule):
    """Pretty-prints a rule configuration."""
    lines = []
    for param in sorted(PARAM_BOUNDS.keys()):
        val = rule[param]
        canonical = PARAM_BOUNDS[param]['canonical']
        delta = ((val - canonical) / canonical * 100) if canonical > 0 else 0
        lines.append(f"  {param:20} = {val:9.6f}  (canonical={canonical:.6f}, delta={delta:+6.1f}%)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Validation Test
# ═══════════════════════════════════════════════════════════════

def validate_search_space():
    """Tests the search space API."""
    print("=" * 72)
    print("  SEARCH SPACE VALIDATION")
    print("=" * 72)

    # Test 1: Canonical rule
    canonical = canonical_rule()
    is_valid, violations = validate_rule(canonical)
    print(f"\nCanonical rule valid: {is_valid}")
    if not is_valid:
        print(f"  Violations: {violations}")
    print(format_rule(canonical))

    # Test 2: Random sampling
    print("\n" + "-" * 72)
    print("Random sampling (5 rules):")
    for i in range(5):
        rule = sample_rule(seed=100 + i)
        is_valid, violations = validate_rule(rule)
        dist = rule_distance(rule, canonical)
        print(f"\n  Rule {i+1}: valid={is_valid}, distance={dist:.3f}")
        if not is_valid:
            print(f"    Violations: {violations}")

    # Test 3: Mutation
    print("\n" + "-" * 72)
    print("Mutation (sigma=0.2, 3 steps from canonical):")
    rule = canonical.copy()
    for i in range(3):
        rule = mutate_rule(rule, sigma=0.2, seed=200 + i)
        is_valid, _ = validate_rule(rule)
        dist = rule_distance(rule, canonical)
        print(f"  Step {i+1}: valid={is_valid}, distance={dist:.3f}")

    # Test 4: Crossover
    print("\n" + "-" * 72)
    print("Crossover (2 random parents):")
    r1 = sample_rule(seed=300)
    r2 = sample_rule(seed=301)
    offspring = crossover_rule(r1, r2, seed=302)
    is_valid, _ = validate_rule(offspring)
    d1 = rule_distance(offspring, r1)
    d2 = rule_distance(offspring, r2)
    print(f"  Offspring: valid={is_valid}, dist_to_p1={d1:.3f}, dist_to_p2={d2:.3f}")

    # Test 5: Grid sampling
    print("\n" + "-" * 72)
    print("Grid sampling (eta, 5 points):")
    rules = grid_sample('eta', 5)
    for i, rule in enumerate(rules):
        print(f"  {i+1}: eta={rule['eta']:.6f}")

    # Test 6: Specialized sweeps
    print("\n" + "-" * 72)
    print("Eta sweep (3 values):")
    rules = eta_sweep_rules()
    for i, rule in enumerate(rules):
        print(f"  {i+1}: eta={rule['eta']:.6f}")

    print("\n" + "-" * 72)
    print("Balance sweep (amplify vs drift, 5 ratios):")
    rules = balance_sweep_rules(n_samples=5)
    for i, rule in enumerate(rules):
        ratio = rule['amplify_mult'] / (rule['amplify_mult'] + rule['drift_mult'])
        print(f"  {i+1}: amplify={rule['amplify_mult']:.3f}, drift={rule['drift_mult']:.3f}, ratio={ratio:.2f}")

    print("\n" + "=" * 72)
    print("  VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    validate_search_space()
