#!/usr/bin/env python3
"""
Evolutionary Search Loop for Plasticity Rules

Uses harness.py, search_space.py, and constraint_checker.py to search
the space of plasticity rules via evolutionary optimization.

Core loop:
1. Initialize population (canonical + random samples)
2. Evaluate via harness.run_comparison()
3. Filter via constraint_checker.is_valid()
4. Select top K by variant_gap
5. Reproduce via mutation and crossover
6. Repeat for G generations

Checkpoints state every generation for resume capability.
"""

import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    run_comparison, Organism, make_signals, measure_gap, canonical_rule as _harness_canonical
)
from search_space import (
    canonical_rule, sample_rule, mutate_rule, crossover_rule, rule_distance, format_rule
)
from constraint_checker import is_valid


# ═══════════════════════════════════════════════════════════════
# Evolution Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    'population_size': 20,
    'generations': 10,
    'top_k': 5,
    'mutation_sigma': 0.3,
    'crossover_prob': 0.5,
    'seeds': [42, 137, 2024],
    'k_values': [4, 6, 8, 10],
    'novel_seed_base': 99999,
    'birth_seed': 42,
    'early_stop_patience': 3,
    'checkpoint_every': 1,
}


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def augment_rule_for_checker(rule_params):
    """
    Adds fields expected by constraint_checker to a rule from search_space.

    For basic parameter search, we're not varying structural aspects like
    eta_adaptive or beta_adaptive, so set them to safe defaults.
    """
    return {
        **rule_params,
        'eta_adaptive': False,
        'eta_method': 'none',
        'beta_adaptive': False,
        'gamma_adaptive': False,
        'beta_method': 'none',
        'gamma_method': 'none',
    }


def quick_evaluate(rule_params, birth_seed=42):
    """
    Fast evaluation: K=6, single seed, reduced perms/trials.
    ~6 run_sequence calls vs ~864 for full comparison.

    Returns dict matching evaluate_population result format.
    """
    k = 6
    sig_seed = birth_seed + k * 200
    sigs = make_signals(k, seed=sig_seed)

    org_still = Organism(seed=birth_seed, alive=False)
    g_still = measure_gap(org_still, sigs, k, 42, n_perm=3, n_trials=2)

    org_var = Organism(seed=birth_seed, alive=True, rule_params=rule_params)
    g_var = measure_gap(org_var, sigs, k, 42, n_perm=3, n_trials=2)

    return {
        'variant_gap': g_var,
        'still_gap': g_still,
        'novel_variant': g_var,  # approximate, no separate novel test
        'ground_truth_pass': g_var > 0.0,
    }


def medium_evaluate(rule_params, birth_seed=42):
    """
    Medium evaluation: K=[4,6,8], two seeds, no novel signals.
    ~36 run_sequence calls — more robust than quick, cheaper than full.
    """
    ks = [4, 6, 8]
    seeds = [42, 137]
    canonical = canonical_rule()

    still_gaps = []
    base_gaps = []
    var_gaps = []

    for k in ks:
        sig_seed = birth_seed + k * 200
        sigs = make_signals(k, seed=sig_seed)
        for s in seeds:
            still_gaps.append(measure_gap(
                Organism(seed=birth_seed, alive=False), sigs, k, s, n_perm=3, n_trials=2))
            base_gaps.append(measure_gap(
                Organism(seed=birth_seed, alive=True, rule_params=canonical), sigs, k, s, n_perm=3, n_trials=2))
            var_gaps.append(measure_gap(
                Organism(seed=birth_seed, alive=True, rule_params=rule_params), sigs, k, s, n_perm=3, n_trials=2))

    sv = sum(still_gaps) / len(still_gaps)
    bv = sum(base_gaps) / len(base_gaps)
    vv = sum(var_gaps) / len(var_gaps)

    return {
        'variant_gap': vv,
        'baseline_gap': bv,
        'still_gap': sv,
        'novel_variant': vv,  # approximate
        'ground_truth_pass': vv > 0.0,
    }


def compute_diversity(population_results):
    """
    Computes population diversity as mean pairwise distance.
    """
    rules = [r['rule'] for r in population_results]
    if len(rules) < 2:
        return 0.0

    distances = []
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            distances.append(rule_distance(rules[i], rules[j]))

    return sum(distances) / len(distances)


# ═══════════════════════════════════════════════════════════════
# Core Evolution Loop
# ═══════════════════════════════════════════════════════════════

def initialize_population(config):
    """
    Creates initial population: canonical + random samples, or seeded around a center rule.

    If config['seed_rule'] is provided, initializes population with that rule
    plus perturbations around it using config['seed_sigma'] (default 0.1).
    """
    population = []

    # Check if we're seeding around a specific rule
    seed_rule = config.get('seed_rule', None)

    if seed_rule is not None:
        # LOCAL SEARCH MODE: seed around center rule
        seed_sigma = config.get('seed_sigma', 0.1)

        # Add the center rule itself
        population.append(seed_rule)

        # Add perturbed versions
        for i in range(config['population_size'] - 1):
            attempt = 0
            while attempt < 100:
                candidate = mutate_rule(seed_rule, sigma=seed_sigma, seed=i * 137 + 42 + attempt)
                augmented = augment_rule_for_checker(candidate)
                if is_valid(augmented):
                    population.append(candidate)
                    break
                attempt += 1

            # Fallback if mutation fails
            if len(population) <= i:
                population.append(seed_rule)
    else:
        # GLOBAL SEARCH MODE: canonical + random samples
        # Add canonical rule
        population.append(canonical_rule())

        # Add random samples
        for i in range(config['population_size'] - 1):
            while True:
                candidate = sample_rule(seed=i * 137 + 42)
                augmented = augment_rule_for_checker(candidate)
                if is_valid(augmented):
                    population.append(candidate)
                    break
                # If invalid, resample (rare with current constraints)

    return population


def evaluate_population(population, config, gen_num, verbose=True):
    """
    Evaluates all candidates in population.

    Uses eval_mode from config: "quick", "medium", or "full" (default).

    Returns: list of dicts with keys:
        - rule: rule_params
        - result: harness result dict
        - variant_gap: ALIVE-variant gap
        - novel_gap: ALIVE-variant gap on novel signals
        - fitness: primary fitness metric (variant_gap)
    """
    results = []
    eval_mode = config.get('eval_mode', 'full')

    for i, rule in enumerate(population):
        t0 = time.time()
        if verbose:
            print(f"  Gen {gen_num}: [{i+1:2d}/{len(population)}] ...", end="", flush=True)

        if eval_mode == "quick":
            result = quick_evaluate(rule, config['birth_seed'])
        elif eval_mode == "medium":
            result = medium_evaluate(rule, config['birth_seed'])
        else:
            result = run_comparison(
                rule,
                ks=config['k_values'],
                seeds=config['seeds'],
                novel_seed_base=config['novel_seed_base'],
                birth_seed=config['birth_seed'],
                verbose=False,
            )

        elapsed = time.time() - t0
        vgap = result['variant_gap']
        gt = result.get('ground_truth_pass', vgap > 0.0)

        results.append({
            'rule': rule,
            'result': result,
            'variant_gap': vgap,
            'novel_gap': result.get('novel_variant', vgap),
            'fitness': vgap if gt else vgap - 1.0,  # Penalize ground truth failures
        })

        if verbose:
            marker = "OK" if gt else "FAIL"
            print(f" gap={vgap:+.4f} gt={marker} ({elapsed:.1f}s)", flush=True)

    # Sort by fitness (descending)
    results.sort(key=lambda x: x['fitness'], reverse=True)

    return results


def select_parents(population_results, top_k):
    """
    Selects top K individuals by fitness.
    """
    return population_results[:top_k]


def reproduce(parents, population_size, config):
    """
    Generates offspring via mutation and crossover.

    Returns: new population (list of rule_params)
    """
    offspring = []

    # Keep elite (top parent)
    offspring.append(parents[0]['rule'])

    # Generate offspring
    while len(offspring) < population_size:
        if random.random() < config['crossover_prob'] and len(parents) >= 2:
            # Crossover
            p1 = random.choice(parents)['rule']
            p2 = random.choice(parents)['rule']
            child = crossover_rule(p1, p2)
        else:
            # Mutation
            parent = random.choice(parents)['rule']
            child = mutate_rule(parent, sigma=config['mutation_sigma'])

        # Validate
        augmented = augment_rule_for_checker(child)
        if is_valid(augmented):
            offspring.append(child)
        # If invalid, retry (should be rare)

    return offspring[:population_size]


def run_evolution(config=None, checkpoint_path=None, verbose=True):
    """
    Main evolutionary search loop.

    Args:
        config: dict with evolution parameters (uses DEFAULT_CONFIG if None)
        checkpoint_path: path to resume from (if exists)
        verbose: print progress

    Returns:
        dict with keys:
            - history: list of generation summaries
            - final_population: final population results
            - best_rule: best rule found
            - best_fitness: best fitness found
    """

    # ── SETUP ────────────────────────────────────────────────
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if verbose:
        print("=" * 72)
        print("  EVOLUTIONARY SEARCH FOR PLASTICITY RULES")
        print("=" * 72)
        print(f"\nConfiguration:")
        for k, v in config.items():
            print(f"  {k:20} = {v}")
        print()

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Checkpoint path
    if checkpoint_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = results_dir / f"evolution_run_{timestamp}.json"
    else:
        checkpoint_path = Path(checkpoint_path)

    # ── RESUME OR INITIALIZE ─────────────────────────────────
    if checkpoint_path.exists():
        if verbose:
            print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)

        population = state['population']
        history = state['history']
        start_gen = state['generation'] + 1
        best_ever_fitness = state['best_ever_fitness']
        best_ever_rule = state['best_ever_rule']
        no_improvement_count = state['no_improvement_count']
    else:
        if verbose:
            print(f"Initializing population (size={config['population_size']})...")

        population = initialize_population(config)
        history = []
        start_gen = 0
        best_ever_fitness = -float('inf')
        best_ever_rule = None
        no_improvement_count = 0

    # ── EVOLUTION LOOP ───────────────────────────────────────
    for gen in range(start_gen, config['generations']):
        gen_start = time.time()

        if verbose:
            print(f"\n{'-' * 72}")
            print(f"Generation {gen + 1}/{config['generations']}")
            print(f"{'-' * 72}")

        # Evaluate population
        population_results = evaluate_population(population, config, gen + 1, verbose=verbose)

        # Statistics
        fitnesses = [r['fitness'] for r in population_results]
        best_fitness = fitnesses[0]
        mean_fitness = sum(fitnesses) / len(fitnesses)
        diversity = compute_diversity(population_results)

        # Track best ever
        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_rule = population_results[0]['rule']
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Log generation summary
        gen_summary = {
            'generation': gen,
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'diversity': diversity,
            'best_rule': population_results[0]['rule'],
            'best_novel_gap': population_results[0]['novel_gap'],
            'time_seconds': time.time() - gen_start,
        }
        history.append(gen_summary)

        if verbose:
            print(f"\n  Best fitness:  {best_fitness:+.4f}")
            print(f"  Mean fitness:  {mean_fitness:+.4f}")
            print(f"  Diversity:     {diversity:.4f}")
            print(f"  Novel gap:     {population_results[0]['novel_gap']:+.4f}")
            print(f"  Time:          {gen_summary['time_seconds']:.1f}s")
            print(f"  Best ever:     {best_ever_fitness:+.4f} (no improvement for {no_improvement_count} gen)")

        # Save checkpoint
        if (gen + 1) % config['checkpoint_every'] == 0:
            state = {
                'generation': gen,
                'population': [r['rule'] for r in population_results],
                'history': history,
                'best_ever_fitness': best_ever_fitness,
                'best_ever_rule': best_ever_rule,
                'no_improvement_count': no_improvement_count,
                'config': config,
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2)

            if verbose:
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if no_improvement_count >= config['early_stop_patience']:
            if verbose:
                print(f"\n  Early stopping: no improvement for {config['early_stop_patience']} generations")
            break

        # Reproduce for next generation
        if gen + 1 < config['generations']:
            parents = select_parents(population_results, config['top_k'])
            population = reproduce(parents, config['population_size'], config)

    # ── FINAL RESULTS ────────────────────────────────────────
    if verbose:
        print("\n" + "=" * 72)
        print("  EVOLUTION COMPLETE")
        print("=" * 72)

        print(f"\nBest rule found (fitness={best_ever_fitness:+.4f}):")
        for param, val in best_ever_rule.items():
            print(f"  {param:20} = {val:.6f}")

        print(f"\nTop 5 final population:")
        for i, res in enumerate(population_results[:5]):
            print(f"  {i+1}. fitness={res['fitness']:+.4f}, novel_gap={res['novel_gap']:+.4f}")

    # Save final results
    final_results = {
        'config': config,
        'history': history,
        'final_population': [
            {
                'rule': r['rule'],
                'fitness': r['fitness'],
                'variant_gap': r['variant_gap'],
                'novel_gap': r['novel_gap'],
            }
            for r in population_results
        ],
        'best_rule': best_ever_rule,
        'best_fitness': best_ever_fitness,
    }

    final_path = checkpoint_path.parent / f"final_{checkpoint_path.name}"
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    if verbose:
        print(f"\nFinal results saved: {final_path}")

    return final_results


# ═══════════════════════════════════════════════════════════════
# Command-line interface
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evolutionary search over plasticity rules")
    parser.add_argument("--fast", action="store_true", help="Quick test (3 gen, pop 8, quick eval)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint file")
    parser.add_argument("--generations", type=int, help="Number of generations")
    parser.add_argument("--population", type=int, help="Population size")
    parser.add_argument("--eval", choices=["quick", "medium", "full"], help="Evaluation mode")
    parser.add_argument("--stagnation", type=int, help="Early stop patience")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.fast:
        config['population_size'] = 8
        config['generations'] = 3
        config['top_k'] = 3
        config['early_stop_patience'] = 2
        config['eval_mode'] = 'quick'

    if args.generations:
        config['generations'] = args.generations
    if args.population:
        config['population_size'] = args.population
    if args.eval:
        config['eval_mode'] = args.eval
    if args.stagnation:
        config['early_stop_patience'] = args.stagnation

    run_evolution(config=config, checkpoint_path=args.resume)
