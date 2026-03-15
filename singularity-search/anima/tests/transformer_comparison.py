"""
T R A N S F O R M E R  vs  A N I M A  C O M P A R I S O N
==========================================================

Benchmarks a standard tiny transformer (~103k params) against
all Anima variants on the same reasoning and physics tasks.

Key Question: Does Anima's W/I/T architecture provide any
advantage over a standard transformer at the same parameter budget?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anima.core.base import Anima, AnimaConfig
from anima.variants.mortal import AnimaMortal, AnimaMortalConfig
from anima.variants.metamorphic import AnimaMetamorphic, AnimaMetamorphicConfig
from anima.variants.collective import AnimaCollective, AnimaCollectiveConfig
from anima.variants.neuroplastic import AnimaNeuroplastic, AnimaNeuroplasticConfig
from anima.tests.tiny_transformer_baseline import TinyTransformer, TinyTransformerConfig


# ============================================================================
# REASONING TEST COMPONENTS (copied from all_variants_benchmark)
# ============================================================================

class SymbolicEnvironment:
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


class SequenceTask:
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str: return "Sequence"
    def generate_example(self) -> Tuple[List[int], int]:
        start = np.random.randint(0, 4)
        step = np.random.randint(1, 4)
        seq = [(start + i * step) % self.vocab_size for i in range(3)]
        target = (start + 3 * step) % self.vocab_size
        return seq, target


class PatternTask:
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str: return "Pattern"
    def generate_example(self) -> Tuple[List[int], int]:
        a = np.random.randint(0, self.vocab_size)
        b = np.random.randint(0, self.vocab_size)
        while b == a: b = np.random.randint(0, self.vocab_size)
        return [a, b, a, b], a


class ConditionalTask:
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str: return "Conditional"
    def generate_example(self) -> Tuple[List[int], int]:
        trigger = np.random.randint(0, self.vocab_size - 1)
        target = trigger + 1
        seq = [np.random.randint(0, self.vocab_size) for _ in range(4)]
        seq[np.random.randint(0, 4)] = trigger
        return seq, target


class AnalogyTask:
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
    def name(self) -> str: return "Analogy"
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


class ProjectilePhysicsTask:
    def name(self) -> str: return "Projectile"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        obj = PhysicsObject(np.random.uniform(0.1, 0.3), np.random.uniform(0.5, 0.8),
                           np.random.uniform(0.2, 0.5), np.random.uniform(-0.1, 0.1))
        env.add_object(obj)
        env.set_goal(np.clip(obj.x + obj.vx * 0.5, 0, 1), np.clip(obj.y + obj.vy * 0.5, 0, 1))
        env.agent.x, env.agent.y = np.random.uniform(0.5, 0.9), np.random.uniform(0.2, 0.5)
    def evaluate(self, env: PhysicsEnvironment) -> float:
        return max(0, 1 - env.agent.distance_to(env.goal) * 2) if env.goal else 0


class CollisionPhysicsTask:
    def name(self) -> str: return "Collision"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        env.agent.x, env.agent.y = 0.5, 0.5
        angle = np.random.uniform(0, 2 * np.pi)
        env.add_object(PhysicsObject(0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle),
                                     -0.3 * np.cos(angle), -0.3 * np.sin(angle)))
    def evaluate(self, env: PhysicsEnvironment) -> float:
        return 0.0 if env.check_collision() else 1.0


class GoalPhysicsTask:
    def name(self) -> str: return "Goal"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        env.agent.x, env.agent.y = np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4)
        env.set_goal(np.random.uniform(0.6, 0.9), np.random.uniform(0.6, 0.9))
    def evaluate(self, env: PhysicsEnvironment) -> float:
        return max(0, 1 - env.agent.distance_to(env.goal)) if env.goal else 0


class MomentumPhysicsTask:
    def name(self) -> str: return "Momentum"
    def setup(self, env: PhysicsEnvironment):
        env.reset()
        env.add_object(PhysicsObject(0.5, 0.5, np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3)))
        env.agent.x, env.agent.y, env.agent.vx, env.agent.vy = 0.3, 0.3, 0, 0
    def evaluate(self, env: PhysicsEnvironment) -> float:
        if env.objects:
            vel_diff = np.sqrt((env.agent.vx - env.objects[0].vx)**2 + (env.agent.vy - env.objects[0].vy)**2)
            return max(0, 1 - vel_diff * 2)
        return 0


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_anima_variant(name: str, sensory_dim: int, action_dim: int):
    """Create Anima variant by name."""
    if name == "Base":
        config = AnimaConfig(world_dim=64, internal_dim=64, time_dim=16,
                            sensory_dim=sensory_dim, action_dim=action_dim,
                            n_world_slots=8, n_internal_slots=6, use_energy=False)
        return Anima(config)
    elif name == "Mortal":
        config = AnimaMortalConfig(world_dim=64, internal_dim=64, time_dim=16,
                                   sensory_dim=sensory_dim, action_dim=action_dim,
                                   n_world_slots=8, n_internal_slots=6, E_init=1.0, E_decay=0.0005)
        return AnimaMortal(config)
    elif name == "Metamorphic":
        config = AnimaMetamorphicConfig(world_dim=64, internal_dim=64, time_dim=16,
                                        sensory_dim=sensory_dim, action_dim=action_dim,
                                        n_world_slots=8, n_internal_slots=6, E_init=1.0, E_decay=0.0005)
        return AnimaMetamorphic(config)
    elif name == "Collective":
        config = AnimaCollectiveConfig(world_dim=64, internal_dim=64, time_dim=16,
                                       sensory_dim=sensory_dim, action_dim=action_dim,
                                       n_world_slots=8, n_internal_slots=6, E_init=1.0, E_decay=0.0005)
        return AnimaCollective(config, member_id=0)
    elif name == "Neuroplastic":
        config = AnimaNeuroplasticConfig(world_dim=64, internal_dim=64, time_dim=16,
                                         sensory_dim=sensory_dim, action_dim=action_dim,
                                         n_world_slots=8, n_internal_slots=6,
                                         min_world_modules=1, max_world_modules=4,
                                         min_internal_modules=1, max_internal_modules=4,
                                         use_energy=False)
        return AnimaNeuroplastic(config)
    else:
        raise ValueError(f"Unknown variant: {name}")


def create_transformer(sensory_dim: int, action_dim: int):
    """Create matched transformer."""
    # Config tuned to ~103k params
    config = TinyTransformerConfig(
        input_dim=sensory_dim,
        output_dim=action_dim,
        d_model=72,
        n_heads=4,
        n_layers=2,
        d_ff=192,
        max_seq_len=32,
    )
    return TinyTransformer(config)


# ============================================================================
# TEST RUNNERS
# ============================================================================

def run_reasoning_test(model, env: SymbolicEnvironment, task,
                       n_train: int = 50, n_eval: int = 100, is_transformer: bool = False) -> float:
    """Run reasoning test."""
    # Reset transformer history if applicable
    if is_transformer and hasattr(model, 'reset'):
        model.reset()

    # Training
    for _ in range(n_train):
        seq, target = task.generate_example()
        env.set_sequence(seq, target)
        if is_transformer:
            model.reset()
        for _ in range(len(seq) + 1):
            obs, _, _ = env.step()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            try:
                model.step(obs_tensor)
            except:
                pass
        env.reset()

    # Evaluation
    correct = 0
    total = 0

    for _ in range(n_eval):
        seq, target = task.generate_example()
        env.set_sequence(seq, target)
        if is_transformer:
            model.reset()

        for _ in range(len(seq) + 1):
            obs, is_query, tgt = env.step()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            try:
                status = model.step(obs_tensor)
                if is_query:
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

    return correct / total if total > 0 else 0


def run_physics_test(model, env: PhysicsEnvironment, task,
                     n_train: int = 30, n_eval: int = 50, steps: int = 15,
                     is_transformer: bool = False) -> float:
    """Run physics test."""
    # Training
    for _ in range(n_train):
        task.setup(env)
        if is_transformer:
            model.reset()
        for _ in range(steps):
            state = env.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            try:
                status = model.step(state_tensor)
                action = status.get('action')
                if action is not None:
                    env.step(action.detach().numpy().flatten())
            except:
                pass

    # Evaluation
    scores = []
    for _ in range(n_eval):
        task.setup(env)
        if is_transformer:
            model.reset()
        for _ in range(steps):
            state = env.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            try:
                status = model.step(state_tensor)
                action = status.get('action')
                if action is not None:
                    env.step(action.detach().numpy().flatten())
            except:
                pass
        scores.append(task.evaluate(env))

    return np.mean(scores)


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_transformer_comparison(n_reasoning: int = 100, n_physics: int = 50):
    """Compare transformer baseline against all Anima variants."""

    print('\n' + '=' * 80)
    print('T R A N S F O R M E R  vs  A N I M A  C O M P A R I S O N')
    print('Same parameter budget (~103k), different architectures')
    print('=' * 80)

    vocab_size = 16
    state_dim = 16

    models = {
        "Transformer": ("transformer", None),
        "Base": ("anima", "Base"),
        "Mortal": ("anima", "Mortal"),
        "Metamorphic": ("anima", "Metamorphic"),
        "Collective": ("anima", "Collective"),
        "Neuroplastic": ("anima", "Neuroplastic"),
    }

    reasoning_tasks = [SequenceTask(vocab_size), PatternTask(vocab_size),
                       ConditionalTask(vocab_size), AnalogyTask(vocab_size)]
    physics_tasks = [ProjectilePhysicsTask(), CollisionPhysicsTask(),
                     GoalPhysicsTask(), MomentumPhysicsTask()]

    results = {}

    for model_name, (model_type, variant) in models.items():
        print(f'\n{"=" * 80}')
        print(f'MODEL: {model_name}')
        print('=' * 80)

        results[model_name] = {'reasoning': {}, 'physics': {}}

        # Create model for reasoning
        if model_type == "transformer":
            model = create_transformer(vocab_size, vocab_size)
            is_transformer = True
        else:
            model = create_anima_variant(variant, vocab_size, vocab_size)
            is_transformer = False

        param_count = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {param_count:,}')

        # Reasoning tests
        print(f'\n--- Reasoning Tests ---')
        env = SymbolicEnvironment(vocab_size)

        for task in reasoning_tasks:
            score = run_reasoning_test(model, env, task, n_train=50, n_eval=n_reasoning,
                                       is_transformer=is_transformer)
            results[model_name]['reasoning'][task.name()] = score
            status = "PASS" if score > 1/vocab_size else "FAIL"
            print(f"  {task.name():<12}: {score:>6.1%} [{status}]")

        # Physics tests (recreate model for fresh state)
        print(f'\n--- Physics Tests ---')
        if model_type == "transformer":
            model = create_transformer(state_dim, 4)
        else:
            model = create_anima_variant(variant, state_dim, 4)

        phys_env = PhysicsEnvironment(state_dim)

        for task in physics_tasks:
            score = run_physics_test(model, phys_env, task, n_train=30, n_eval=n_physics,
                                     is_transformer=is_transformer)
            results[model_name]['physics'][task.name()] = score
            status = "PASS" if score > 0.25 else "FAIL"
            print(f"  {task.name():<12}: {score:>6.1%} [{status}]")

        # Calculate averages
        results[model_name]['reasoning_avg'] = np.mean(list(results[model_name]['reasoning'].values()))
        results[model_name]['physics_avg'] = np.mean(list(results[model_name]['physics'].values()))
        results[model_name]['overall'] = (results[model_name]['reasoning_avg'] +
                                          results[model_name]['physics_avg']) / 2

    # Comparison tables
    print('\n' + '=' * 80)
    print('C O M P A R I S O N  T A B L E S')
    print('=' * 80)

    # Reasoning comparison
    print(f'\n{"REASONING":<14}', end='')
    for task in reasoning_tasks:
        print(f' | {task.name():>10}', end='')
    print(f' | {"AVG":>8}')
    print('-' * 80)

    for model_name in models.keys():
        print(f'{model_name:<14}', end='')
        for task in reasoning_tasks:
            score = results[model_name]['reasoning'][task.name()]
            print(f' | {score:>9.1%}', end='')
        print(f' | {results[model_name]["reasoning_avg"]:>7.1%}')

    # Physics comparison
    print(f'\n{"PHYSICS":<14}', end='')
    for task in physics_tasks:
        print(f' | {task.name():>10}', end='')
    print(f' | {"AVG":>8}')
    print('-' * 80)

    for model_name in models.keys():
        print(f'{model_name:<14}', end='')
        for task in physics_tasks:
            score = results[model_name]['physics'][task.name()]
            print(f' | {score:>9.1%}', end='')
        print(f' | {results[model_name]["physics_avg"]:>7.1%}')

    # Overall ranking
    print(f'\n{"=" * 80}')
    print('O V E R A L L  R A N K I N G')
    print('=' * 80)

    ranked = sorted(models.keys(), key=lambda m: results[m]['overall'], reverse=True)

    print(f'\n{"Rank":<6} | {"Model":<14} | {"Reasoning":>10} | {"Physics":>10} | {"Overall":>10}')
    print('-' * 65)

    transformer_rank = None
    for i, model_name in enumerate(ranked, 1):
        r = results[model_name]
        marker = " <-- TRANSFORMER" if model_name == "Transformer" else ""
        print(f'{i:<6} | {model_name:<14} | {r["reasoning_avg"]:>9.1%} | {r["physics_avg"]:>9.1%} | {r["overall"]:>9.1%}{marker}')
        if model_name == "Transformer":
            transformer_rank = i

    # Analysis
    print(f'\n{"=" * 80}')
    print('A N A L Y S I S')
    print('=' * 80)

    anima_variants = [m for m in models.keys() if m != "Transformer"]
    anima_better_count = sum(1 for v in anima_variants if results[v]['overall'] > results['Transformer']['overall'])

    print(f'\nTransformer rank: {transformer_rank}/{len(models)}')
    print(f'Anima variants beating Transformer: {anima_better_count}/{len(anima_variants)}')

    # Task-by-task comparison
    print(f'\nTask-by-task (Transformer vs Best Anima):')

    for task in reasoning_tasks:
        t_score = results['Transformer']['reasoning'][task.name()]
        best_anima = max(anima_variants, key=lambda v: results[v]['reasoning'][task.name()])
        a_score = results[best_anima]['reasoning'][task.name()]
        winner = "Transformer" if t_score > a_score else best_anima
        print(f"  {task.name()}: Transformer {t_score:.1%} vs {best_anima} {a_score:.1%} -> {winner}")

    for task in physics_tasks:
        t_score = results['Transformer']['physics'][task.name()]
        best_anima = max(anima_variants, key=lambda v: results[v]['physics'][task.name()])
        a_score = results[best_anima]['physics'][task.name()]
        winner = "Transformer" if t_score > a_score else best_anima
        print(f"  {task.name()}: Transformer {t_score:.1%} vs {best_anima} {a_score:.1%} -> {winner}")

    print('=' * 80)

    # Verdict
    if transformer_rank == 1:
        print('\nVERDICT: Transformer architecture wins at this parameter scale.')
    elif transformer_rank == len(models):
        print('\nVERDICT: Anima W/I/T architecture outperforms standard transformer!')
    else:
        print(f'\nVERDICT: Mixed results - Transformer ranked {transformer_rank}/{len(models)}')

    print('=' * 80 + '\n')

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transformer vs Anima Comparison')
    parser.add_argument('--reasoning-trials', type=int, default=100)
    parser.add_argument('--physics-trials', type=int, default=50)
    parser.add_argument('--output', type=str, default='transformer_comparison.json')

    args = parser.parse_args()

    results = run_transformer_comparison(
        n_reasoning=args.reasoning_trials,
        n_physics=args.physics_trials
    )

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Results saved to: {output_path}')
