"""
A L L  V A R I A N T S  B E N C H M A R K
=========================================

Tests ALL Anima variants against:
1. Text-based reasoning (4 tasks)
2. Physical simulation (4 tasks)

Variants:
1. Base (Original) - W/I/T state separation, immortal
2. Mortal - Energy depletion → death
3. Metamorphic - Energy crisis → transform
4. Collective - Shared pool + cooperation
5. Neuroplastic - Dynamic N (grow/prune modules)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time
import json
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anima.core.base import Anima, AnimaConfig
from anima.variants.mortal import AnimaMortal, AnimaMortalConfig
from anima.variants.metamorphic import AnimaMetamorphic, AnimaMetamorphicConfig
from anima.variants.collective import AnimaCollective, AnimaCollectiveConfig
from anima.variants.neuroplastic import AnimaNeuroplastic, AnimaNeuroplasticConfig


# ============================================================================
# REASONING TEST COMPONENTS
# ============================================================================

class SymbolicEnvironment:
    """Environment that presents symbolic sequences."""

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
        self.current_sequence = []
        self.position = 0
        self.target = None

    def encode(self, symbol: int) -> np.ndarray:
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        vec[symbol] = 1.0
        return vec

    def decode(self, vec: np.ndarray) -> int:
        return int(np.argmax(vec))

    def set_sequence(self, sequence: List[int], target: int):
        self.current_sequence = sequence
        self.target = target
        self.position = 0

    def step(self) -> Tuple[np.ndarray, bool, int]:
        if self.position < len(self.current_sequence):
            symbol = self.current_sequence[self.position]
            self.position += 1
            return self.encode(symbol), False, self.target
        else:
            return np.zeros(self.vocab_size, dtype=np.float32), True, self.target

    def reset(self):
        self.position = 0


class ReasoningTask:
    def generate_example(self) -> Tuple[List[int], int]:
        raise NotImplementedError
    def name(self) -> str:
        raise NotImplementedError


class SequenceCompletionTask(ReasoningTask):
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str:
        return "Sequence"
    def generate_example(self) -> Tuple[List[int], int]:
        start = np.random.randint(0, 4)
        step = np.random.randint(1, 4)
        seq = [(start + i * step) % self.vocab_size for i in range(3)]
        target = (start + 3 * step) % self.vocab_size
        return seq, target


class PatternRepetitionTask(ReasoningTask):
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str:
        return "Pattern"
    def generate_example(self) -> Tuple[List[int], int]:
        a = np.random.randint(0, self.vocab_size)
        b = np.random.randint(0, self.vocab_size)
        while b == a:
            b = np.random.randint(0, self.vocab_size)
        return [a, b, a, b], a


class ConditionalTask(ReasoningTask):
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str:
        return "Conditional"
    def generate_example(self) -> Tuple[List[int], int]:
        trigger = np.random.randint(0, self.vocab_size - 1)
        target = trigger + 1
        seq = [np.random.randint(0, self.vocab_size) for _ in range(4)]
        seq[np.random.randint(0, 4)] = trigger
        return seq, target


class AnalogyTask(ReasoningTask):
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str:
        return "Analogy"
    def generate_example(self) -> Tuple[List[int], int]:
        a = np.random.randint(1, self.vocab_size - 4)
        diff = np.random.randint(1, 4)
        b = a + diff
        c = np.random.randint(1, self.vocab_size - 4)
        target = (c + diff) % self.vocab_size
        return [a, b, c], target


# ============================================================================
# PHYSICS TEST COMPONENTS
# ============================================================================

@dataclass
class PhysicsObject:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

    def update(self, dt: float = 0.1):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def distance_to(self, other: 'PhysicsObject') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class PhysicsEnvironment:
    def __init__(self, state_dim: int = 16):
        self.state_dim = state_dim
        self.agent = PhysicsObject(0.5, 0.5)
        self.objects: List[PhysicsObject] = []
        self.goal: Optional[PhysicsObject] = None
        self.dt = 0.1

    def reset(self):
        self.agent = PhysicsObject(0.5, 0.5)
        self.objects = []
        self.goal = None

    def add_object(self, obj: PhysicsObject):
        self.objects.append(obj)

    def set_goal(self, x: float, y: float):
        self.goal = PhysicsObject(x, y)

    def get_state(self) -> np.ndarray:
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0:4] = [self.agent.x, self.agent.y, self.agent.vx, self.agent.vy]
        if self.objects:
            state[4:8] = [self.objects[0].x, self.objects[0].y,
                         self.objects[0].vx, self.objects[0].vy]
        if self.goal:
            state[8:10] = [self.goal.x, self.goal.y]
            state[10] = self.agent.distance_to(self.goal)
        if self.objects:
            state[11] = min(self.agent.distance_to(o) for o in self.objects)
        return state

    def step(self, action: Optional[np.ndarray] = None) -> np.ndarray:
        if action is not None and len(action) >= 2:
            self.agent.vx = np.clip(action[0], -0.5, 0.5)
            self.agent.vy = np.clip(action[1], -0.5, 0.5)
        self.agent.update(self.dt)
        self.agent.x = np.clip(self.agent.x, 0, 1)
        self.agent.y = np.clip(self.agent.y, 0, 1)
        for obj in self.objects:
            obj.update(self.dt)
        return self.get_state()

    def check_collision(self, threshold: float = 0.12) -> bool:
        return any(self.agent.distance_to(o) < threshold for o in self.objects)


class PhysicsTask:
    def setup(self, env: PhysicsEnvironment): raise NotImplementedError
    def evaluate(self, env: PhysicsEnvironment) -> float: raise NotImplementedError
    def name(self) -> str: raise NotImplementedError


class ProjectileTask(PhysicsTask):
    def name(self) -> str: return "Projectile"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        obj = PhysicsObject(np.random.uniform(0.1, 0.3), np.random.uniform(0.5, 0.8),
                           np.random.uniform(0.2, 0.5), np.random.uniform(-0.1, 0.1))
        env.add_object(obj)
        env.set_goal(np.clip(obj.x + obj.vx * 0.5, 0, 1),
                    np.clip(obj.y + obj.vy * 0.5, 0, 1))
        env.agent.x, env.agent.y = np.random.uniform(0.5, 0.9), np.random.uniform(0.2, 0.5)
    def evaluate(self, env: PhysicsEnvironment) -> float:
        return max(0, 1 - env.agent.distance_to(env.goal) * 2) if env.goal else 0


class CollisionTask(PhysicsTask):
    def name(self) -> str: return "Collision"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        env.agent.x, env.agent.y = 0.5, 0.5
        angle = np.random.uniform(0, 2 * np.pi)
        env.add_object(PhysicsObject(0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle),
                                     -0.3 * np.cos(angle), -0.3 * np.sin(angle)))
    def evaluate(self, env: PhysicsEnvironment) -> float:
        return 0.0 if env.check_collision() else 1.0


class GoalTask(PhysicsTask):
    def name(self) -> str: return "Goal"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        env.agent.x, env.agent.y = np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4)
        env.set_goal(np.random.uniform(0.6, 0.9), np.random.uniform(0.6, 0.9))
    def evaluate(self, env: PhysicsEnvironment) -> float:
        return max(0, 1 - env.agent.distance_to(env.goal)) if env.goal else 0


class MomentumTask(PhysicsTask):
    def name(self) -> str: return "Momentum"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        env.add_object(PhysicsObject(0.5, 0.5, np.random.uniform(-0.3, 0.3),
                                     np.random.uniform(-0.3, 0.3)))
        env.agent.x, env.agent.y, env.agent.vx, env.agent.vy = 0.3, 0.3, 0, 0
    def evaluate(self, env: PhysicsEnvironment) -> float:
        if env.objects:
            vel_diff = np.sqrt((env.agent.vx - env.objects[0].vx)**2 +
                              (env.agent.vy - env.objects[0].vy)**2)
            return max(0, 1 - vel_diff * 2)
        return 0


# ============================================================================
# VARIANT CREATION
# ============================================================================

def create_variant(name: str, sensory_dim: int, action_dim: int):
    """Create a variant by name with specified dimensions."""

    if name == "Base":
        config = AnimaConfig(
            world_dim=64, internal_dim=64, time_dim=16,
            sensory_dim=sensory_dim, action_dim=action_dim,
            n_world_slots=8, n_internal_slots=6,
            use_energy=False
        )
        return Anima(config)

    elif name == "Mortal":
        config = AnimaMortalConfig(
            world_dim=64, internal_dim=64, time_dim=16,
            sensory_dim=sensory_dim, action_dim=action_dim,
            n_world_slots=8, n_internal_slots=6,
            E_init=1.0, E_decay=0.0005  # Slow decay for testing
        )
        return AnimaMortal(config)

    elif name == "Metamorphic":
        config = AnimaMetamorphicConfig(
            world_dim=64, internal_dim=64, time_dim=16,
            sensory_dim=sensory_dim, action_dim=action_dim,
            n_world_slots=8, n_internal_slots=6,
            E_init=1.0, E_decay=0.0005
        )
        return AnimaMetamorphic(config)

    elif name == "Collective":
        # For collective, we use a single member (not full swarm) for fair comparison
        config = AnimaCollectiveConfig(
            world_dim=64, internal_dim=64, time_dim=16,
            sensory_dim=sensory_dim, action_dim=action_dim,
            n_world_slots=8, n_internal_slots=6,
            E_init=1.0, E_decay=0.0005
        )
        return AnimaCollective(config, member_id=0)

    elif name == "Neuroplastic":
        config = AnimaNeuroplasticConfig(
            world_dim=64, internal_dim=64, time_dim=16,
            sensory_dim=sensory_dim, action_dim=action_dim,
            n_world_slots=8, n_internal_slots=6,
            min_world_modules=1, max_world_modules=4,
            min_internal_modules=1, max_internal_modules=4,
            growth_error_threshold=0.7,
            prune_contribution_threshold=0.02,
            use_energy=False
        )
        return AnimaNeuroplastic(config)

    else:
        raise ValueError(f"Unknown variant: {name}")


# ============================================================================
# TEST RUNNERS
# ============================================================================

def run_reasoning_test(entity, env: SymbolicEnvironment, task: ReasoningTask,
                       n_train: int = 50, n_eval: int = 100) -> Dict[str, float]:
    """Run reasoning test for one entity on one task."""

    # Training
    for _ in range(n_train):
        seq, target = task.generate_example()
        env.set_sequence(seq, target)
        for _ in range(len(seq) + 1):
            obs, _, _ = env.step()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            try:
                entity.step(obs_tensor)
            except:
                pass  # Handle any variant-specific issues
        env.reset()

    # Evaluation
    correct = 0
    total = 0

    for _ in range(n_eval):
        seq, target = task.generate_example()
        env.set_sequence(seq, target)

        for _ in range(len(seq) + 1):
            obs, is_query, tgt = env.step()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            try:
                status = entity.step(obs_tensor)

                if is_query and status.get('alive', True):
                    action = status.get('action')
                    if action is not None:
                        action = action.detach().numpy().flatten()
                        if len(action) >= env.vocab_size:
                            pred = env.decode(action[:env.vocab_size])
                        else:
                            extended = np.zeros(env.vocab_size)
                            extended[:len(action)] = action
                            pred = env.decode(extended)

                        if pred == tgt:
                            correct += 1
                        total += 1
            except:
                pass

        env.reset()

    accuracy = correct / total if total > 0 else 0
    return {'accuracy': accuracy, 'correct': correct, 'total': total}


def run_physics_test(entity, env: PhysicsEnvironment, task: PhysicsTask,
                     n_train: int = 30, n_eval: int = 50, steps: int = 15) -> Dict[str, float]:
    """Run physics test for one entity on one task."""

    # Training
    for _ in range(n_train):
        task.setup(env)
        for _ in range(steps):
            state = env.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            try:
                status = entity.step(state_tensor)
                if status.get('alive', True):
                    action = status.get('action')
                    if action is not None:
                        env.step(action.detach().numpy().flatten())
            except:
                pass

    # Evaluation
    scores = []

    for _ in range(n_eval):
        task.setup(env)

        for _ in range(steps):
            state = env.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            try:
                status = entity.step(state_tensor)
                if status.get('alive', True):
                    action = status.get('action')
                    if action is not None:
                        env.step(action.detach().numpy().flatten())
            except:
                pass

        scores.append(task.evaluate(env))

    return {'score': np.mean(scores), 'std': np.std(scores)}


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_all_variants_benchmark(n_reasoning_eval: int = 100, n_physics_eval: int = 50):
    """Run full benchmark across all variants."""

    print('\n' + '=' * 80)
    print('A L L  V A R I A N T S  B E N C H M A R K')
    print('=' * 80)

    variants = ["Base", "Mortal", "Metamorphic", "Collective", "Neuroplastic"]
    vocab_size = 16
    state_dim = 16
    action_dim = 16  # Use vocab_size for reasoning compatibility

    reasoning_tasks = [
        SequenceCompletionTask(vocab_size),
        PatternRepetitionTask(vocab_size),
        ConditionalTask(vocab_size),
        AnalogyTask(vocab_size),
    ]

    physics_tasks = [
        ProjectileTask(),
        CollisionTask(),
        GoalTask(),
        MomentumTask(),
    ]

    results = {}

    for variant_name in variants:
        print(f'\n{"=" * 80}')
        print(f'VARIANT: {variant_name}')
        print('=' * 80)

        results[variant_name] = {
            'reasoning': {},
            'physics': {},
            'summary': {}
        }

        # Create entity
        entity = create_variant(variant_name, sensory_dim=vocab_size, action_dim=action_dim)
        param_count = sum(p.numel() for p in entity.parameters())
        print(f'Parameters: {param_count:,}')

        # Reasoning tests
        print(f'\n--- Reasoning Tests ---')
        sym_env = SymbolicEnvironment(vocab_size)
        reasoning_scores = []

        for task in reasoning_tasks:
            result = run_reasoning_test(entity, sym_env, task,
                                        n_train=50, n_eval=n_reasoning_eval)
            results[variant_name]['reasoning'][task.name()] = result['accuracy']
            reasoning_scores.append(result['accuracy'])
            status = "PASS" if result['accuracy'] > 1/vocab_size else "FAIL"
            print(f"  {task.name():<12}: {result['accuracy']:>6.1%} [{status}]")

        # Physics tests
        print(f'\n--- Physics Tests ---')
        phys_env = PhysicsEnvironment(state_dim)
        physics_scores = []

        # Recreate entity for physics (fresh state)
        entity = create_variant(variant_name, sensory_dim=state_dim, action_dim=4)

        for task in physics_tasks:
            result = run_physics_test(entity, phys_env, task,
                                      n_train=30, n_eval=n_physics_eval)
            results[variant_name]['physics'][task.name()] = result['score']
            physics_scores.append(result['score'])
            status = "PASS" if result['score'] > 0.25 else "FAIL"
            print(f"  {task.name():<12}: {result['score']:>6.1%} [{status}]")

        # Summary
        results[variant_name]['summary'] = {
            'reasoning_avg': np.mean(reasoning_scores),
            'physics_avg': np.mean(physics_scores),
            'overall_avg': np.mean(reasoning_scores + physics_scores),
            'param_count': param_count,
        }

        print(f'\n  Reasoning avg: {np.mean(reasoning_scores):.1%}')
        print(f'  Physics avg:   {np.mean(physics_scores):.1%}')

    # Final comparison table
    print('\n' + '=' * 80)
    print('C O M P A R I S O N  T A B L E')
    print('=' * 80)

    # Reasoning header
    print(f'\n{"REASONING":<14}', end='')
    for task in reasoning_tasks:
        print(f' | {task.name():>10}', end='')
    print(f' | {"AVG":>8}')
    print('-' * 80)

    for variant_name in variants:
        print(f'{variant_name:<14}', end='')
        for task in reasoning_tasks:
            score = results[variant_name]['reasoning'][task.name()]
            print(f' | {score:>9.1%}', end='')
        avg = results[variant_name]['summary']['reasoning_avg']
        print(f' | {avg:>7.1%}')

    # Physics header
    print(f'\n{"PHYSICS":<14}', end='')
    for task in physics_tasks:
        print(f' | {task.name():>10}', end='')
    print(f' | {"AVG":>8}')
    print('-' * 80)

    for variant_name in variants:
        print(f'{variant_name:<14}', end='')
        for task in physics_tasks:
            score = results[variant_name]['physics'][task.name()]
            print(f' | {score:>9.1%}', end='')
        avg = results[variant_name]['summary']['physics_avg']
        print(f' | {avg:>7.1%}')

    # Overall ranking
    print(f'\n{"=" * 80}')
    print('O V E R A L L  R A N K I N G')
    print('=' * 80)

    ranked = sorted(variants, key=lambda v: results[v]['summary']['overall_avg'], reverse=True)

    print(f'\n{"Rank":<6} | {"Variant":<14} | {"Reasoning":>10} | {"Physics":>10} | {"Overall":>10}')
    print('-' * 60)

    for i, variant_name in enumerate(ranked, 1):
        s = results[variant_name]['summary']
        print(f'{i:<6} | {variant_name:<14} | {s["reasoning_avg"]:>9.1%} | {s["physics_avg"]:>9.1%} | {s["overall_avg"]:>9.1%}')

    print('=' * 80)
    print(f'\nBest overall: {ranked[0]}')
    print(f'Best reasoning: {max(variants, key=lambda v: results[v]["summary"]["reasoning_avg"])}')
    print(f'Best physics: {max(variants, key=lambda v: results[v]["summary"]["physics_avg"])}')
    print('=' * 80 + '\n')

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='All Variants Benchmark')
    parser.add_argument('--reasoning-trials', type=int, default=100)
    parser.add_argument('--physics-trials', type=int, default=50)
    parser.add_argument('--output', type=str, default='all_variants_results.json')

    args = parser.parse_args()

    results = run_all_variants_benchmark(
        n_reasoning_eval=args.reasoning_trials,
        n_physics_eval=args.physics_trials
    )

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Results saved to: {output_path}')
