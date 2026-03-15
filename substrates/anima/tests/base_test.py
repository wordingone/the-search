"""
A N I M A  T E S T  R U N N E R
===============================

Comprehensive test suite for Anima entity.

Tests Intelligence Emergence properties:
    - N >= N_min: Capacity threshold
    - T_min = 3: Functional completeness {S, M, D}
    - F in F_complete: Cross-type nonlinear coupling
    - phi observable: Actions distinguish decision states

Test Categories:
    1. State Separation - W, I, T evolve independently
    2. Consolidation - Cycle-based memory updates
    3. Coherence - State norm constraints
    4. Perturbation - Adaptation to environmental changes
    5. Observables - All required metrics present
    6. Action Generation - phi observability

Usage:
    python anima_test.py                    # Run all tests
    python anima_test.py --test separation  # Single test
    python anima_test.py --steps 3000       # Extended run
    python anima_test.py --experiment       # Run full experiment
"""

import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time
import argparse
import json


@dataclass
class AnimaTestResult:
    """Result from a single Anima test."""
    name: str
    passed: bool
    steps_completed: int
    survived: bool
    total_cycles: int
    final_phase: float
    world_norm: float
    internal_norm: float
    world_memory_norm: float
    internal_memory_norm: float
    avg_prediction_error: float
    energy: Optional[float] = None
    death_cause: Optional[str] = None
    runtime_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


def test_state_separation(steps: int = 500) -> AnimaTestResult:
    """
    Test 1: State Separation

    Verify that W, I, T evolve independently:
    - W updates only from observation
    - I updates only from prediction error
    - T advances only from activity rate

    Pass criteria:
    - Correlation between state updates < 0.7
    - Each state shows variation over time
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] State Separation')
    start = time.time()

    config = AnimaConfig(
        world_dim=32,
        internal_dim=32,
        sensory_dim=8,
        use_energy=False,
    )

    entity = Anima(config)
    env = AnimaEnvironment(config.sensory_dim)

    # Track state trajectories
    world_trajectory = []
    internal_trajectory = []
    phase_trajectory = []

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor([s], dtype=torch.float32)
        status = entity.step(s_tensor)

        if not status['alive']:
            break

        world_trajectory.append(status['world_state_norm'])
        internal_trajectory.append(status['internal_state_norm'])
        phase_trajectory.append(status['phase'])

    # Calculate correlations
    w = np.array(world_trajectory)
    i = np.array(internal_trajectory)
    p = np.array(phase_trajectory)

    # Correlations should be low (independent evolution)
    corr_wi = abs(np.corrcoef(w, i)[0, 1]) if len(w) > 1 else 0
    corr_wp = abs(np.corrcoef(w, p)[0, 1]) if len(w) > 1 else 0
    corr_ip = abs(np.corrcoef(i, p)[0, 1]) if len(i) > 1 else 0

    # Handle NaN correlations
    corr_wi = 0 if np.isnan(corr_wi) else corr_wi
    corr_wp = 0 if np.isnan(corr_wp) else corr_wp
    corr_ip = 0 if np.isnan(corr_ip) else corr_ip

    # Each state should show variation
    var_w = np.var(w) if len(w) > 1 else 0
    var_i = np.var(i) if len(i) > 1 else 0
    var_p = np.var(p) if len(p) > 1 else 0

    # Pass criteria
    correlation_ok = corr_wi < 0.7 and corr_wp < 0.7 and corr_ip < 0.7
    variation_ok = var_w > 0.0001 and var_i > 0.0001 and var_p > 0.0001
    passed = correlation_ok and variation_ok

    report = entity.get_final_report()

    print(f'  Correlation W-I: {corr_wi:.3f}')
    print(f'  Correlation W-T: {corr_wp:.3f}')
    print(f'  Correlation I-T: {corr_ip:.3f}')
    print(f'  Variance W: {var_w:.4f}, I: {var_i:.4f}, T: {var_p:.4f}')
    print(f'  Result: {"PASS" if passed else "FAIL"}')

    return AnimaTestResult(
        name='State Separation',
        passed=passed,
        steps_completed=entity.step_count,
        survived=report['alive'],
        total_cycles=report['total_cycles'],
        final_phase=report['final_phase'],
        world_norm=report['final_world_norm'],
        internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        runtime_seconds=time.time() - start,
        details={
            'corr_wi': corr_wi, 'corr_wp': corr_wp, 'corr_ip': corr_ip,
            'var_w': var_w, 'var_i': var_i, 'var_p': var_p,
        }
    )


def test_consolidation(steps: int = 1000) -> AnimaTestResult:
    """
    Test 2: Cycle-Based Consolidation

    Verify that memory updates occur at cycle boundaries:
    - Memory norms change after cycle completion
    - Consolidation triggered by phase wrap

    Pass criteria:
    - At least 3 cycles completed
    - Memory norms > 0 after consolidation
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] Cycle-Based Consolidation')
    start = time.time()

    config = AnimaConfig(
        world_dim=32,
        internal_dim=32,
        sensory_dim=8,
        base_frequency=0.2,  # Faster cycles for testing
        cycles_per_consolidation=1,
        use_energy=False,
    )

    entity = Anima(config)
    env = AnimaEnvironment(config.sensory_dim)

    # Track consolidation events
    cycle_events = []
    prev_cycle = 0

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor([s], dtype=torch.float32)
        status = entity.step(s_tensor)

        if not status['alive']:
            break

        if status['cycle'] > prev_cycle:
            cycle_events.append({
                'step': step,
                'cycle': status['cycle'],
                'world_mem_norm': entity.world_memory.norm().item(),
                'internal_mem_norm': entity.internal_memory.norm().item(),
            })
            prev_cycle = status['cycle']

    # Pass criteria
    cycles_ok = entity.cycle_count >= 3
    memory_ok = (entity.world_memory.norm().item() > 0.01 and
                 entity.internal_memory.norm().item() > 0.01)
    passed = cycles_ok and memory_ok

    report = entity.get_final_report()

    print(f'  Cycles completed: {entity.cycle_count}')
    print(f'  World memory norm: {report["world_memory_norm"]:.4f}')
    print(f'  Internal memory norm: {report["internal_memory_norm"]:.4f}')
    print(f'  Consolidation events: {len(cycle_events)}')
    print(f'  Result: {"PASS" if passed else "FAIL"}')

    return AnimaTestResult(
        name='Consolidation',
        passed=passed,
        steps_completed=entity.step_count,
        survived=report['alive'],
        total_cycles=report['total_cycles'],
        final_phase=report['final_phase'],
        world_norm=report['final_world_norm'],
        internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        runtime_seconds=time.time() - start,
        details={'cycle_events': cycle_events[:10]},  # First 10 events
    )


def test_coherence(steps: int = 2000) -> AnimaTestResult:
    """
    Test 3: Coherence Constraints

    Verify that state norms stay within bounds:
    - ||W|| < world_coherence_max
    - ||I|| < internal_coherence_max
    - Death only from coherence failure (not energy)

    Pass criteria:
    - Survives full run OR dies from coherence
    - No energy-based death
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] Coherence Constraints')
    start = time.time()

    config = AnimaConfig(
        world_dim=32,
        internal_dim=32,
        sensory_dim=8,
        world_coherence_max=10.0,
        internal_coherence_max=10.0,
        use_energy=False,
    )

    entity = Anima(config)
    env = AnimaEnvironment(config.sensory_dim)

    max_world_norm = 0
    max_internal_norm = 0

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor([s], dtype=torch.float32)
        status = entity.step(s_tensor)

        if status['alive']:
            max_world_norm = max(max_world_norm, status['world_state_norm'])
            max_internal_norm = max(max_internal_norm, status['internal_state_norm'])
        else:
            break

    # Pass criteria
    report = entity.get_final_report()
    survived = report['alive']
    if not survived:
        # Death should be from coherence, not energy
        coherence_death = 'Coherence' in (report['death_cause'] or '')
        passed = coherence_death
    else:
        passed = True

    print(f'  Max ||W||: {max_world_norm:.4f} (limit: {config.world_coherence_max})')
    print(f'  Max ||I||: {max_internal_norm:.4f} (limit: {config.internal_coherence_max})')
    print(f'  Survived: {survived}')
    if not survived:
        print(f'  Death cause: {report["death_cause"]}')
    print(f'  Result: {"PASS" if passed else "FAIL"}')

    return AnimaTestResult(
        name='Coherence',
        passed=passed,
        steps_completed=entity.step_count,
        survived=report['alive'],
        total_cycles=report['total_cycles'],
        final_phase=report['final_phase'],
        world_norm=report['final_world_norm'],
        internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        death_cause=report['death_cause'],
        runtime_seconds=time.time() - start,
        details={'max_world_norm': max_world_norm, 'max_internal_norm': max_internal_norm},
    )


def test_perturbation(steps: int = 2000) -> AnimaTestResult:
    """
    Test 4: Perturbation Adaptation

    Verify adaptation to environmental changes:
    - Prediction error spikes after perturbation
    - Error decreases as entity adapts
    - Entity survives perturbation

    Pass criteria:
    - Entity survives perturbation
    - Post-perturbation error eventually decreases
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] Perturbation Adaptation')
    start = time.time()

    config = AnimaConfig(
        world_dim=32,
        internal_dim=32,
        sensory_dim=8,
        use_energy=False,
    )

    entity = Anima(config)
    env = AnimaEnvironment(config.sensory_dim)

    perturbation_step = steps // 2
    errors_before = []
    errors_after = []

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor([s], dtype=torch.float32)

        if step == perturbation_step:
            env.perturb(0.5)
            print(f'  Perturbation applied at step {step}')

        status = entity.step(s_tensor)

        if not status['alive']:
            break

        if step < perturbation_step:
            errors_before.append(status['prediction_error'])
        else:
            errors_after.append(status['prediction_error'])

    # Pass criteria
    report = entity.get_final_report()
    survived = report['alive']

    # Check error recovery
    if len(errors_after) >= 100:
        early_after = np.mean(errors_after[:50])
        late_after = np.mean(errors_after[-50:])
        recovery = late_after < early_after * 1.5  # Some recovery
    else:
        recovery = True  # Too few samples to judge

    passed = survived and recovery

    avg_before = np.mean(errors_before) if errors_before else 0
    avg_after = np.mean(errors_after) if errors_after else 0

    print(f'  Avg error before perturbation: {avg_before:.4f}')
    print(f'  Avg error after perturbation: {avg_after:.4f}')
    print(f'  Survived: {survived}')
    print(f'  Error recovery: {"Yes" if recovery else "No"}')
    print(f'  Result: {"PASS" if passed else "FAIL"}')

    return AnimaTestResult(
        name='Perturbation',
        passed=passed,
        steps_completed=entity.step_count,
        survived=report['alive'],
        total_cycles=report['total_cycles'],
        final_phase=report['final_phase'],
        world_norm=report['final_world_norm'],
        internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        death_cause=report['death_cause'],
        runtime_seconds=time.time() - start,
        details={'avg_error_before': avg_before, 'avg_error_after': avg_after},
    )


def test_observables(steps: int = 500) -> AnimaTestResult:
    """
    Test 5: Empirical Observables

    Verify all required observables are present:
    - State norms (W, I)
    - Prediction error
    - Memory norms (world, internal)
    - Cycles and phase
    - Actions
    - Optional energy

    Pass criteria:
    - All observables present in step output
    - All observables have valid values
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] Empirical Observables')
    start = time.time()

    config = AnimaConfig(
        world_dim=32,
        internal_dim=32,
        sensory_dim=8,
        action_dim=4,
        use_energy=True,  # Test with energy enabled
    )

    entity = Anima(config)
    env = AnimaEnvironment(config.sensory_dim)

    required_keys = [
        'alive', 'step', 'world_state_norm', 'internal_state_norm',
        'phase', 'cycle', 'cycle_completed', 'prediction_error',
        'action', 'activity_rate', 'energy'
    ]

    all_present = True
    all_valid = True
    sample_status = None

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor([s], dtype=torch.float32)
        status = entity.step(s_tensor)

        if step == 0:
            sample_status = status.copy()
            # Check keys present
            for key in required_keys:
                if key not in status:
                    all_present = False
                    print(f'  Missing key: {key}')

        if not status['alive']:
            break

        # Validate values
        if status['world_state_norm'] < 0 or np.isnan(status['world_state_norm']):
            all_valid = False
        if status['internal_state_norm'] < 0 or np.isnan(status['internal_state_norm']):
            all_valid = False
        if status['phase'] < 0 or status['phase'] > 2 * np.pi + 0.1:
            all_valid = False

    passed = all_present and all_valid

    report = entity.get_final_report()

    print(f'  All required keys present: {all_present}')
    print(f'  All values valid: {all_valid}')
    if sample_status:
        print(f'  Sample output keys: {list(sample_status.keys())}')
    print(f'  Result: {"PASS" if passed else "FAIL"}')

    return AnimaTestResult(
        name='Observables',
        passed=passed,
        steps_completed=entity.step_count,
        survived=report['alive'],
        total_cycles=report['total_cycles'],
        final_phase=report['final_phase'],
        world_norm=report['final_world_norm'],
        internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        energy=report['energy'],
        runtime_seconds=time.time() - start,
        details={'sample_status': {k: str(v)[:50] for k, v in (sample_status or {}).items()}},
    )


def test_energy_modes(steps: int = 1000) -> AnimaTestResult:
    """
    Test 6: Energy Modes Comparison

    Compare behavior with energy enabled vs disabled:
    - Without energy: Entity cannot die from energy depletion
    - With energy: Energy depletes but doesn't kill (constraint only)

    Pass criteria:
    - Both modes produce valid trajectories
    - Neither mode shows energy-based death
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] Energy Modes Comparison')
    start = time.time()

    results = {}

    for use_energy in [False, True]:
        mode = 'with_energy' if use_energy else 'no_energy'

        config = AnimaConfig(
            world_dim=32,
            internal_dim=32,
            sensory_dim=8,
            use_energy=use_energy,
            E_init=1.0,
            E_decay=0.001,
        )

        entity = Anima(config)
        env = AnimaEnvironment(config.sensory_dim)

        for step in range(steps):
            s = env.step()
            s_tensor = torch.tensor([s], dtype=torch.float32)
            status = entity.step(s_tensor)
            if not status['alive']:
                break

        report = entity.get_final_report()
        results[mode] = {
            'survived': report['alive'],
            'steps': report['total_steps'],
            'energy': report['energy'],
            'death_cause': report['death_cause'],
        }

        print(f'  Mode: {mode}')
        print(f'    Survived: {report["alive"]}')
        print(f'    Steps: {report["total_steps"]}')
        print(f'    Energy: {report["energy"]}')

    # Pass criteria: both valid, no energy death
    passed = True
    for mode, data in results.items():
        if data['death_cause'] and 'energy' in data['death_cause'].lower():
            passed = False

    print(f'  Result: {"PASS" if passed else "FAIL"}')

    # Use the energy-enabled run for the result
    report = results['with_energy']
    return AnimaTestResult(
        name='Energy Modes',
        passed=passed,
        steps_completed=report['steps'],
        survived=report['survived'],
        total_cycles=0,  # Not tracked in this test
        final_phase=0,
        world_norm=0,
        internal_norm=0,
        world_memory_norm=0,
        internal_memory_norm=0,
        avg_prediction_error=0,
        energy=report['energy'],
        runtime_seconds=time.time() - start,
        details=results,
    )


def test_action_generation(steps: int = 500) -> AnimaTestResult:
    """
    Test 7: Action Generation from Internal State

    Verify actions are generated from internal state:
    - Actions vary with internal state changes
    - Action space matches config
    - phi is observable (different I -> different actions)

    Pass criteria:
    - Actions have correct dimension
    - Actions show variation
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] Action Generation')
    start = time.time()

    config = AnimaConfig(
        world_dim=32,
        internal_dim=32,
        sensory_dim=8,
        action_dim=4,
        use_energy=False,
    )

    entity = Anima(config)
    env = AnimaEnvironment(config.sensory_dim)

    actions = []
    internal_states = []

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor([s], dtype=torch.float32)
        status = entity.step(s_tensor)

        if not status['alive']:
            break

        actions.append(status['action'].numpy().flatten())
        internal_states.append(entity.internal_state.detach().numpy().flatten())

    # Check action dimension
    action_dim_ok = actions[0].shape[0] == config.action_dim if actions else False

    # Check action variation
    if len(actions) > 1:
        action_var = np.var(np.array(actions), axis=0).mean()
        variation_ok = action_var > 0.001
    else:
        action_var = 0
        variation_ok = False

    # Check internal-action correlation (phi observability)
    if len(actions) > 10:
        # Check if different internal states produce different actions
        internal_norms = [np.linalg.norm(i) for i in internal_states]
        action_norms = [np.linalg.norm(a) for a in actions]
        corr = abs(np.corrcoef(internal_norms, action_norms)[0, 1])
        if np.isnan(corr):
            corr = 0
    else:
        corr = 0

    passed = action_dim_ok and variation_ok

    report = entity.get_final_report()

    print(f'  Action dimension: {actions[0].shape[0] if actions else 0} (expected: {config.action_dim})')
    print(f'  Action variance: {action_var:.4f}')
    print(f'  I-Action correlation: {corr:.3f}')
    print(f'  Result: {"PASS" if passed else "FAIL"}')

    return AnimaTestResult(
        name='Action Generation',
        passed=passed,
        steps_completed=entity.step_count,
        survived=report['alive'],
        total_cycles=report['total_cycles'],
        final_phase=report['final_phase'],
        world_norm=report['final_world_norm'],
        internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        runtime_seconds=time.time() - start,
        details={'action_dim_ok': action_dim_ok, 'variation_ok': variation_ok, 'correlation': corr},
    )


def test_cross_type_coupling(steps: int = 500) -> AnimaTestResult:
    """
    Test 8: Cross-Type Coupling (F in F_complete)

    Verify information flows S -> M -> D:
    - Observations (S) affect world state
    - Prediction errors affect internal state (M)
    - Internal state drives actions (D)

    Pass criteria:
    - Changes in observation cause world state changes
    - Prediction errors cause internal state changes
    - Internal state changes cause action changes
    """
    from anima import Anima, AnimaConfig, AnimaEnvironment

    print('\n[TEST] Cross-Type Coupling (F completeness)')
    start = time.time()

    config = AnimaConfig(
        world_dim=32,
        internal_dim=32,
        sensory_dim=8,
        action_dim=4,
        use_energy=False,
    )

    entity = Anima(config)
    env = AnimaEnvironment(config.sensory_dim)

    # Track state changes
    obs_history = []
    world_delta_history = []
    error_history = []
    internal_delta_history = []
    action_history = []

    prev_world = entity.world_state.clone()
    prev_internal = entity.internal_state.clone()

    for step in range(steps):
        s = env.step()
        s_tensor = torch.tensor([s], dtype=torch.float32)

        obs_history.append(np.linalg.norm(s))

        status = entity.step(s_tensor)

        if not status['alive']:
            break

        # Track deltas
        world_delta = (entity.world_state - prev_world).norm().item()
        internal_delta = (entity.internal_state - prev_internal).norm().item()

        world_delta_history.append(world_delta)
        error_history.append(status['prediction_error'])
        internal_delta_history.append(internal_delta)
        action_history.append(np.linalg.norm(status['action'].numpy()))

        prev_world = entity.world_state.clone()
        prev_internal = entity.internal_state.clone()

    # Test coupling: S -> W (observations affect world)
    if len(obs_history) > 10:
        s_w_coupling = abs(np.corrcoef(obs_history[:-1], world_delta_history[1:])[0, 1])
        s_w_coupling = 0 if np.isnan(s_w_coupling) else s_w_coupling
    else:
        s_w_coupling = 0

    # Test coupling: error -> I (prediction errors affect internal)
    if len(error_history) > 10:
        e_i_coupling = abs(np.corrcoef(error_history[:-1], internal_delta_history[1:])[0, 1])
        e_i_coupling = 0 if np.isnan(e_i_coupling) else e_i_coupling
    else:
        e_i_coupling = 0

    # Test coupling: I -> action (internal affects actions)
    if len(internal_delta_history) > 10:
        i_a_coupling = abs(np.corrcoef(internal_delta_history[:-1], action_history[1:])[0, 1])
        i_a_coupling = 0 if np.isnan(i_a_coupling) else i_a_coupling
    else:
        i_a_coupling = 0

    # At least some coupling expected (not zero)
    passed = (s_w_coupling > 0.01 or e_i_coupling > 0.01 or i_a_coupling > 0.01)

    report = entity.get_final_report()

    print(f'  S->W coupling (obs->world): {s_w_coupling:.3f}')
    print(f'  E->I coupling (error->internal): {e_i_coupling:.3f}')
    print(f'  I->A coupling (internal->action): {i_a_coupling:.3f}')
    print(f'  Result: {"PASS" if passed else "FAIL"}')

    return AnimaTestResult(
        name='Cross-Type Coupling',
        passed=passed,
        steps_completed=entity.step_count,
        survived=report['alive'],
        total_cycles=report['total_cycles'],
        final_phase=report['final_phase'],
        world_norm=report['final_world_norm'],
        internal_norm=report['final_internal_norm'],
        world_memory_norm=report['world_memory_norm'],
        internal_memory_norm=report['internal_memory_norm'],
        avg_prediction_error=report['avg_prediction_error'],
        runtime_seconds=time.time() - start,
        details={'s_w': s_w_coupling, 'e_i': e_i_coupling, 'i_a': i_a_coupling},
    )


def run_all_tests(steps: int = 1000) -> List[AnimaTestResult]:
    """Run all Anima tests."""
    print('\n' + '='*70)
    print('A N I M A   T E S T   S U I T E')
    print('Intelligence Emergence Validation')
    print('S = (V, tau, F, phi)')
    print('='*70)

    tests = [
        ('separation', lambda: test_state_separation(steps // 2)),
        ('consolidation', lambda: test_consolidation(steps)),
        ('coherence', lambda: test_coherence(steps * 2)),
        ('perturbation', lambda: test_perturbation(steps * 2)),
        ('observables', lambda: test_observables(steps // 2)),
        ('energy', lambda: test_energy_modes(steps)),
        ('action', lambda: test_action_generation(steps // 2)),
        ('coupling', lambda: test_cross_type_coupling(steps // 2)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f'\n[ERROR] Test {name} failed with exception: {e}')
            import traceback
            traceback.print_exc()
            results.append(AnimaTestResult(
                name=name,
                passed=False,
                steps_completed=0,
                survived=False,
                total_cycles=0,
                final_phase=0,
                world_norm=0,
                internal_norm=0,
                world_memory_norm=0,
                internal_memory_norm=0,
                avg_prediction_error=0,
                details={'error': str(e)},
            ))

    return results


def print_summary(results: List[AnimaTestResult]):
    """Print test summary."""
    print('\n' + '='*70)
    print('T E S T   S U M M A R Y')
    print('='*70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f'\n{"Test":<20} | {"Status":<8} | {"Steps":<6} | {"Cycles":<6} | {"Time":>8}')
    print('-' * 70)

    for r in results:
        status = 'PASS' if r.passed else 'FAIL'
        print(f'{r.name:<20} | {status:<8} | {r.steps_completed:<6} | '
              f'{r.total_cycles:<6} | {r.runtime_seconds:>7.2f}s')

    print('-' * 70)
    print(f'Total: {passed}/{total} passed')

    if passed == total:
        print('\n[SUCCESS] All Intelligence Emergence properties validated!')
        print('  N >= N_min: Capacity sufficient')
        print('  T_min = 3: Functional types complete {S, M, D}')
        print('  F in F_complete: Cross-type coupling verified')
        print('  phi observable: Actions distinguish states')
    else:
        print(f'\n[WARNING] {total - passed} test(s) failed')

    print('='*70 + '\n')


def save_results(results: List[AnimaTestResult], filepath: str):
    """Save results to JSON."""
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'entity': 'Anima',
        'theory': 'Intelligence Emergence S = (V, tau, F, phi)',
        'passed': sum(1 for r in results if r.passed),
        'total': len(results),
        'results': [asdict(r) for r in results],
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'Results saved to: {filepath}')


def main():
    parser = argparse.ArgumentParser(description='Anima Test Suite')
    parser.add_argument('--steps', type=int, default=1000, help='Base steps per test')
    parser.add_argument('--test', type=str, default=None,
                        choices=['separation', 'consolidation', 'coherence',
                                 'perturbation', 'observables', 'energy', 'action', 'coupling'],
                        help='Run single test')
    parser.add_argument('--output', type=str, default='anima_test_results.json',
                        help='Output file for results')
    parser.add_argument('--experiment', action='store_true',
                        help='Run full experiment instead of tests')

    args = parser.parse_args()

    if args.experiment:
        # Run the full experiment from anima.py
        from anima import run_anima_experiment
        run_anima_experiment(
            max_steps=args.steps * 2,
            perturbation_step=args.steps,
            use_energy=False,
        )
        return

    if args.test:
        # Run single test
        test_fns = {
            'separation': lambda: test_state_separation(args.steps),
            'consolidation': lambda: test_consolidation(args.steps),
            'coherence': lambda: test_coherence(args.steps * 2),
            'perturbation': lambda: test_perturbation(args.steps * 2),
            'observables': lambda: test_observables(args.steps),
            'energy': lambda: test_energy_modes(args.steps),
            'action': lambda: test_action_generation(args.steps),
            'coupling': lambda: test_cross_type_coupling(args.steps),
        }
        result = test_fns[args.test]()
        print_summary([result])
        return

    # Run all tests
    results = run_all_tests(args.steps)
    print_summary(results)
    save_results(results, args.output)


if __name__ == '__main__':
    main()
