"""
Anima Variant Comparison Test
=============================

Runs all variants under identical conditions to compare:
- Survival rates
- Learning dynamics
- Memory accumulation
- State stability

Variants tested:
1. Base Anima (immortal, no energy constraint)
2. Mortal (energy depletion → death)
3. Metamorphic (energy depletion → transform)
4. Collective (shared pool, cooperation)
"""

import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import time
import json
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anima.core.base import Anima, AnimaConfig
from anima.core.environment import AnimaEnvironment
from anima.variants.mortal import AnimaMortal, AnimaMortalConfig
from anima.variants.metamorphic import AnimaMetamorphic, AnimaMetamorphicConfig
from anima.variants.collective import AnimaCollective, AnimaCollectiveConfig, AnimaSwarm


@dataclass
class VariantResult:
    """Result from running one variant."""
    name: str
    survived: bool
    total_steps: int
    total_cycles: int
    final_energy: Optional[float]
    death_cause: Optional[str]

    # State metrics
    final_world_norm: float
    final_internal_norm: float
    world_memory_norm: float
    internal_memory_norm: float

    # Learning metrics
    avg_prediction_error: float

    # Variant-specific
    transformations: int = 0
    final_population: int = 1
    cooperation_rate: float = 0.0

    runtime_seconds: float = 0.0


def run_base_anima(env: AnimaEnvironment, steps: int, perturb_step: int, perturb_mag: float) -> VariantResult:
    """Run base Anima (immortal)."""
    start = time.time()

    config = AnimaConfig(
        world_dim=64,
        internal_dim=64,
        sensory_dim=8,
        action_dim=4,
        use_energy=False,
    )

    entity = Anima(config)

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        if step == perturb_step:
            env.perturb(perturb_mag)

        status = entity.step(s_tensor)
        if not status['alive']:
            break

    env.reset()
    report = entity.get_final_report()

    return VariantResult(
        name='Base (Immortal)',
        survived=report['alive'],
        total_steps=report['total_steps'],
        total_cycles=report['total_cycles'],
        final_energy=None,
        death_cause=report.get('death_cause'),
        final_world_norm=report['final_world_norm'],
        final_internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        runtime_seconds=time.time() - start,
    )


def run_mortal_anima(env: AnimaEnvironment, steps: int, perturb_step: int, perturb_mag: float) -> VariantResult:
    """Run mortal Anima."""
    start = time.time()

    config = AnimaMortalConfig(
        world_dim=64,
        internal_dim=64,
        sensory_dim=8,
        action_dim=4,
    )

    entity = AnimaMortal(config)

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        if step == perturb_step:
            env.perturb(perturb_mag)

        status = entity.step(s_tensor)
        if not status['alive']:
            break

    env.reset()
    report = entity.get_final_report()

    return VariantResult(
        name='Mortal',
        survived=report['alive'],
        total_steps=report['total_steps'],
        total_cycles=report['total_cycles'],
        final_energy=report.get('energy'),
        death_cause=report.get('death_cause'),
        final_world_norm=report['final_world_norm'],
        final_internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        runtime_seconds=time.time() - start,
    )


def run_metamorphic_anima(env: AnimaEnvironment, steps: int, perturb_step: int, perturb_mag: float) -> VariantResult:
    """Run metamorphic Anima."""
    start = time.time()

    config = AnimaMetamorphicConfig(
        world_dim=64,
        internal_dim=64,
        sensory_dim=8,
        action_dim=4,
    )

    entity = AnimaMetamorphic(config)

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        if step == perturb_step:
            env.perturb(perturb_mag)

        status = entity.step(s_tensor)
        if not status['alive']:
            break

    env.reset()
    report = entity.get_final_report()

    return VariantResult(
        name='Metamorphic',
        survived=report['alive'],
        total_steps=report['total_steps'],
        total_cycles=report['total_cycles'],
        final_energy=report.get('energy'),
        death_cause=report.get('death_cause'),
        final_world_norm=report['final_world_norm'],
        final_internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        transformations=report.get('transformation_count', 0),
        runtime_seconds=time.time() - start,
    )


def run_collective_anima(env: AnimaEnvironment, steps: int, perturb_step: int, perturb_mag: float) -> VariantResult:
    """Run collective Anima swarm."""
    start = time.time()

    config = AnimaCollectiveConfig(
        world_dim=64,
        internal_dim=64,
        sensory_dim=8,
        action_dim=4,
        swarm_size=5,
    )

    swarm = AnimaSwarm(config)

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        if step == perturb_step:
            env.perturb(perturb_mag)

        status = swarm.step(s_tensor)
        if not status['alive']:
            break

    env.reset()
    report = swarm.get_report()

    return VariantResult(
        name='Collective',
        survived=report['final_population'] > 0,
        total_steps=report['total_steps'],
        total_cycles=0,
        final_energy=report['final_E_pool'],
        death_cause='Extinction' if report['final_population'] == 0 else None,
        final_world_norm=0,
        final_internal_norm=0,
        world_memory_norm=0,
        internal_memory_norm=0,
        avg_prediction_error=0,
        final_population=report['final_population'],
        cooperation_rate=report['cooperation_rate'],
        runtime_seconds=time.time() - start,
    )


def run_variant_comparison(
    steps: int = 2000,
    perturb_step: int = 1000,
    perturb_mag: float = 0.5,
) -> Dict[str, Any]:
    """
    Run all variants under identical conditions.
    """
    print('\n' + '=' * 70)
    print('A N I M A  V A R I A N T  C O M P A R I S O N')
    print('=' * 70)
    print(f'Steps: {steps}')
    print(f'Perturbation: step={perturb_step}, magnitude={perturb_mag}')
    print('=' * 70)

    # Shared environment (reset between runs)
    env = AnimaEnvironment(sensory_dim=8)

    results = []

    # Run each variant
    variants = [
        ('Base (Immortal)', run_base_anima),
        ('Mortal', run_mortal_anima),
        ('Metamorphic', run_metamorphic_anima),
        ('Collective', run_collective_anima),
    ]

    for name, runner in variants:
        print(f'\n[RUNNING] {name}...')
        result = runner(env, steps, perturb_step, perturb_mag)
        results.append(result)
        print(f'  Survived: {result.survived}')
        print(f'  Steps: {result.total_steps}')
        if result.transformations > 0:
            print(f'  Transformations: {result.transformations}')
        if result.final_population != 1:
            print(f'  Final population: {result.final_population}')
        if result.cooperation_rate > 0:
            print(f'  Cooperation rate: {result.cooperation_rate:.1%}')

    # Summary table
    print('\n' + '=' * 70)
    print('C O M P A R I S O N  S U M M A R Y')
    print('=' * 70)

    print(f'\n{"Variant":<18} | {"Survived":<8} | {"Steps":<6} | {"Special":<25}')
    print('-' * 70)

    for r in results:
        survived = 'YES' if r.survived else 'NO'

        if r.name == 'Metamorphic':
            special = f'Transforms: {r.transformations}'
        elif r.name == 'Collective':
            special = f'Pop: {r.final_population}/5, Coop: {r.cooperation_rate:.0%}'
        elif r.name == 'Mortal':
            special = f'Energy: {r.final_energy:.3f}' if r.final_energy else 'Depleted'
        else:
            special = f'Cycles: {r.total_cycles}'

        print(f'{r.name:<18} | {survived:<8} | {r.total_steps:<6} | {special:<25}')

    print('-' * 70)

    # Analysis
    print('\n' + '=' * 70)
    print('B E H A V I O R A L  A N A L Y S I S')
    print('=' * 70)

    survivors = [r for r in results if r.survived]
    print(f'\nSurvival rate: {len(survivors)}/{len(results)}')

    if any(r.name == 'Mortal' for r in results):
        mortal = next(r for r in results if r.name == 'Mortal')
        if not mortal.survived:
            print(f'\nMortality effect: Died at step {mortal.total_steps}')
            print('  Energy pressure creates urgency but limits lifespan')

    if any(r.name == 'Metamorphic' for r in results):
        meta = next(r for r in results if r.name == 'Metamorphic')
        if meta.transformations > 0:
            print(f'\nTransformation effect: {meta.transformations} metamorphoses')
            print('  Transformation allows survival through resource crises')

    if any(r.name == 'Collective' for r in results):
        coll = next(r for r in results if r.name == 'Collective')
        print(f'\nCooperation effect: {coll.cooperation_rate:.1%} cooperation rate')
        print(f'  {coll.final_population}/{5} members survived')

    print('=' * 70 + '\n')

    return {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {'steps': steps, 'perturb_step': perturb_step, 'perturb_mag': perturb_mag},
        'results': [asdict(r) for r in results],
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Anima Variant Comparison')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--perturb-step', type=int, default=1000)
    parser.add_argument('--perturb-mag', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='variant_comparison.json')

    args = parser.parse_args()

    results = run_variant_comparison(
        steps=args.steps,
        perturb_step=args.perturb_step,
        perturb_mag=args.perturb_mag,
    )

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Results saved to: {args.output}')
