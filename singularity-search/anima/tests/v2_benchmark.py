"""
V 2  C O M P R E H E N S I V E  B E N C H M A R K
=================================================

Tests ALL Anima versions against reasoning and physics tasks:

V1 Baseline:
- Base (original)

V1 Variants:
- Mortal, Metamorphic, Collective, Neuroplastic

V2 Baseline (synthesized):
- AnimaV2 (trunk with urgency + compression + coherence + stability)

V2 Hybrids:
- Adaptive (Neuroplastic + Metamorphic)
- Resonant (Collective + Internal harmony)
- Phoenix (Metamorphic + Staged growth)
- Pressured (Mortal + Collective stability)

External Baseline:
- Tiny Transformer (~103k params)

Total: 11 models compared
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# V1
from anima.core.base import Anima, AnimaConfig
from anima.core.anima_v2 import AnimaV2, AnimaV2Config
from anima.variants.mortal import AnimaMortal, AnimaMortalConfig
from anima.variants.metamorphic import AnimaMetamorphic, AnimaMetamorphicConfig
from anima.variants.collective import AnimaCollective, AnimaCollectiveConfig
from anima.variants.neuroplastic import AnimaNeuroplastic, AnimaNeuroplasticConfig

# V2 Hybrids
from anima.variants.v2_hybrids import (
    AnimaAdaptive, AnimaAdaptiveConfig,
    AnimaResonant, AnimaResonantConfig,
    AnimaPhoenix, AnimaPhoenixConfig,
    AnimaPressured, AnimaPressuredConfig,
)

# Transformer baseline
from anima.tests.tiny_transformer_baseline import TinyTransformer, TinyTransformerConfig


# ============================================================================
# TEST ENVIRONMENTS (reused from previous benchmarks)
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


class PhysicsObject:
    def __init__(self, x, y, vx=0, vy=0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy

    def update(self, dt=0.1):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class PhysicsEnvironment:
    def __init__(self, state_dim=16):
        self.state_dim = state_dim
        self.agent = PhysicsObject(0.5, 0.5)
        self.objects = []
        self.goal = None
        self.dt = 0.1

    def reset(self):
        self.agent = PhysicsObject(0.5, 0.5)
        self.objects = []
        self.goal = None

    def add_object(self, obj): self.objects.append(obj)
    def set_goal(self, x, y): self.goal = PhysicsObject(x, y)

    def get_state(self):
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

    def step(self, action=None):
        if action is not None and len(action) >= 2:
            self.agent.vx = np.clip(action[0], -0.5, 0.5)
            self.agent.vy = np.clip(action[1], -0.5, 0.5)
        self.agent.update(self.dt)
        self.agent.x = np.clip(self.agent.x, 0, 1)
        self.agent.y = np.clip(self.agent.y, 0, 1)
        for obj in self.objects:
            obj.update(self.dt)
        return self.get_state()

    def check_collision(self, threshold=0.12):
        return any(self.agent.distance_to(o) < threshold for o in self.objects)


# ============================================================================
# TASKS
# ============================================================================

# Reasoning tasks
class SequenceTask:
    def __init__(self, vs=16): self.vs = vs
    def name(self): return "Sequence"
    def generate(self):
        s, d = np.random.randint(0, 4), np.random.randint(1, 4)
        return [(s + i*d) % self.vs for i in range(3)], (s + 3*d) % self.vs

class PatternTask:
    def __init__(self, vs=16): self.vs = vs
    def name(self): return "Pattern"
    def generate(self):
        a, b = np.random.randint(0, self.vs), np.random.randint(0, self.vs)
        while b == a: b = np.random.randint(0, self.vs)
        return [a, b, a, b], a

class ConditionalTask:
    def __init__(self, vs=16): self.vs = vs
    def name(self): return "Conditional"
    def generate(self):
        t = np.random.randint(0, self.vs - 1)
        seq = [np.random.randint(0, self.vs) for _ in range(4)]
        seq[np.random.randint(0, 4)] = t
        return seq, t + 1

class AnalogyTask:
    def __init__(self, vs=16): self.vs = vs
    def name(self): return "Analogy"
    def generate(self):
        a, d = np.random.randint(1, self.vs-4), np.random.randint(1, 4)
        c = np.random.randint(1, self.vs-4)
        return [a, a+d, c], (c + d) % self.vs

# Physics tasks
class ProjectileTask:
    def name(self): return "Projectile"
    def setup(self, env):
        env.reset()
        obj = PhysicsObject(np.random.uniform(0.1, 0.3), np.random.uniform(0.5, 0.8),
                           np.random.uniform(0.2, 0.5), np.random.uniform(-0.1, 0.1))
        env.add_object(obj)
        env.set_goal(np.clip(obj.x + obj.vx*0.5, 0, 1), np.clip(obj.y + obj.vy*0.5, 0, 1))
        env.agent.x, env.agent.y = np.random.uniform(0.5, 0.9), np.random.uniform(0.2, 0.5)
    def evaluate(self, env):
        return max(0, 1 - env.agent.distance_to(env.goal)*2) if env.goal else 0

class CollisionTask:
    def name(self): return "Collision"
    def setup(self, env):
        env.reset()
        env.agent.x, env.agent.y = 0.5, 0.5
        a = np.random.uniform(0, 2*np.pi)
        env.add_object(PhysicsObject(0.5+0.4*np.cos(a), 0.5+0.4*np.sin(a), -0.3*np.cos(a), -0.3*np.sin(a)))
    def evaluate(self, env):
        return 0.0 if env.check_collision() else 1.0

class GoalTask:
    def name(self): return "Goal"
    def setup(self, env):
        env.reset()
        env.agent.x, env.agent.y = np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4)
        env.set_goal(np.random.uniform(0.6, 0.9), np.random.uniform(0.6, 0.9))
    def evaluate(self, env):
        return max(0, 1 - env.agent.distance_to(env.goal)) if env.goal else 0

class MomentumTask:
    def name(self): return "Momentum"
    def setup(self, env):
        env.reset()
        env.add_object(PhysicsObject(0.5, 0.5, np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3)))
        env.agent.x, env.agent.y, env.agent.vx, env.agent.vy = 0.3, 0.3, 0, 0
    def evaluate(self, env):
        if env.objects:
            vd = np.sqrt((env.agent.vx-env.objects[0].vx)**2 + (env.agent.vy-env.objects[0].vy)**2)
            return max(0, 1 - vd*2)
        return 0


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(name: str, sensory_dim: int, action_dim: int):
    """Create model by name."""

    # V1 Models
    if name == "V1-Base":
        return Anima(AnimaConfig(world_dim=64, internal_dim=64, time_dim=16,
                                 sensory_dim=sensory_dim, action_dim=action_dim,
                                 n_world_slots=8, n_internal_slots=6, use_energy=False))

    elif name == "V1-Mortal":
        return AnimaMortal(AnimaMortalConfig(world_dim=64, internal_dim=64, time_dim=16,
                                             sensory_dim=sensory_dim, action_dim=action_dim,
                                             n_world_slots=8, n_internal_slots=6,
                                             E_init=1.0, E_decay=0.0005))

    elif name == "V1-Metamorphic":
        return AnimaMetamorphic(AnimaMetamorphicConfig(world_dim=64, internal_dim=64, time_dim=16,
                                                        sensory_dim=sensory_dim, action_dim=action_dim,
                                                        n_world_slots=8, n_internal_slots=6,
                                                        E_init=1.0, E_decay=0.0005))

    elif name == "V1-Collective":
        return AnimaCollective(AnimaCollectiveConfig(world_dim=64, internal_dim=64, time_dim=16,
                                                     sensory_dim=sensory_dim, action_dim=action_dim,
                                                     n_world_slots=8, n_internal_slots=6,
                                                     E_init=1.0, E_decay=0.0005), member_id=0)

    elif name == "V1-Neuroplastic":
        return AnimaNeuroplastic(AnimaNeuroplasticConfig(world_dim=64, internal_dim=64, time_dim=16,
                                                          sensory_dim=sensory_dim, action_dim=action_dim,
                                                          n_world_slots=8, n_internal_slots=6,
                                                          use_energy=False))

    # V2 Models
    elif name == "V2-Core":
        return AnimaV2(AnimaV2Config(world_dim=64, internal_dim=64, time_dim=16,
                                     sensory_dim=sensory_dim, action_dim=action_dim,
                                     n_world_slots=8, n_internal_slots=6))

    elif name == "V2-Adaptive":
        return AnimaAdaptive(AnimaAdaptiveConfig(world_dim=64, internal_dim=64, time_dim=16,
                                                  sensory_dim=sensory_dim, action_dim=action_dim,
                                                  n_world_slots=8, n_internal_slots=6))

    elif name == "V2-Resonant":
        return AnimaResonant(AnimaResonantConfig(world_dim=64, internal_dim=64, time_dim=16,
                                                  sensory_dim=sensory_dim, action_dim=action_dim,
                                                  n_world_slots=8, n_internal_slots=6))

    elif name == "V2-Phoenix":
        return AnimaPhoenix(AnimaPhoenixConfig(world_dim=64, internal_dim=64, time_dim=16,
                                                sensory_dim=sensory_dim, action_dim=action_dim,
                                                n_world_slots=8, n_internal_slots=6))

    elif name == "V2-Pressured":
        return AnimaPressured(AnimaPressuredConfig(world_dim=64, internal_dim=64, time_dim=16,
                                                    sensory_dim=sensory_dim, action_dim=action_dim,
                                                    n_world_slots=8, n_internal_slots=6))

    elif name == "Transformer":
        config = TinyTransformerConfig(input_dim=sensory_dim, output_dim=action_dim,
                                       d_model=72, n_heads=4, n_layers=2, d_ff=192, max_seq_len=32)
        return TinyTransformer(config)

    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================================
# TEST RUNNERS
# ============================================================================

def run_reasoning(model, env, task, n_train=50, n_eval=100, is_transformer=False):
    """Run reasoning test."""
    if is_transformer and hasattr(model, 'reset'):
        model.reset()

    # Train
    for _ in range(n_train):
        seq, tgt = task.generate()
        env.set_sequence(seq, tgt)
        if is_transformer: model.reset()
        for _ in range(len(seq) + 1):
            obs, _, _ = env.step()
            try:
                model.step(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            except: pass
        env.reset()

    # Eval
    correct, total = 0, 0
    for _ in range(n_eval):
        seq, tgt = task.generate()
        env.set_sequence(seq, tgt)
        if is_transformer: model.reset()
        for _ in range(len(seq) + 1):
            obs, is_q, t = env.step()
            try:
                status = model.step(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                if is_q:
                    action = status.get('action')
                    if action is not None:
                        a = action.detach().numpy().flatten()
                        if len(a) >= env.vocab_size:
                            pred = env.decode(a[:env.vocab_size])
                        else:
                            ext = np.zeros(env.vocab_size)
                            ext[:len(a)] = a
                            pred = env.decode(ext)
                        if pred == t: correct += 1
                        total += 1
            except: pass
        env.reset()
    return correct / total if total > 0 else 0


def run_physics(model, env, task, n_train=30, n_eval=50, steps=15, is_transformer=False):
    """Run physics test."""
    # Train
    for _ in range(n_train):
        task.setup(env)
        if is_transformer: model.reset()
        for _ in range(steps):
            s = env.get_state()
            try:
                status = model.step(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                action = status.get('action')
                if action is not None:
                    env.step(action.detach().numpy().flatten())
            except: pass

    # Eval
    scores = []
    for _ in range(n_eval):
        task.setup(env)
        if is_transformer: model.reset()
        for _ in range(steps):
            s = env.get_state()
            try:
                status = model.step(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                action = status.get('action')
                if action is not None:
                    env.step(action.detach().numpy().flatten())
            except: pass
        scores.append(task.evaluate(env))
    return np.mean(scores)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_v2_benchmark(n_reason=100, n_phys=50):
    """Run comprehensive V2 benchmark."""

    print('\n' + '=' * 90)
    print('V 2  C O M P R E H E N S I V E  B E N C H M A R K')
    print('V1 vs V2 vs Transformer - All Models Compared')
    print('=' * 90)

    vocab_size = 16
    state_dim = 16

    models = [
        # V1
        "V1-Base", "V1-Mortal", "V1-Metamorphic", "V1-Collective", "V1-Neuroplastic",
        # V2
        "V2-Core", "V2-Adaptive", "V2-Resonant", "V2-Phoenix", "V2-Pressured",
        # External
        "Transformer",
    ]

    reasoning_tasks = [SequenceTask(vocab_size), PatternTask(vocab_size),
                       ConditionalTask(vocab_size), AnalogyTask(vocab_size)]
    physics_tasks = [ProjectileTask(), CollisionTask(), GoalTask(), MomentumTask()]

    results = {}

    for model_name in models:
        print(f'\n{"=" * 90}')
        print(f'MODEL: {model_name}')
        print('=' * 90)

        is_transformer = model_name == "Transformer"
        results[model_name] = {'reasoning': {}, 'physics': {}}

        # Reasoning
        print(f'\n--- Reasoning ---')
        model = create_model(model_name, vocab_size, vocab_size)
        params = sum(p.numel() for p in model.parameters())
        print(f'Parameters: {params:,}')

        env = SymbolicEnvironment(vocab_size)
        for task in reasoning_tasks:
            score = run_reasoning(model, env, task, n_train=50, n_eval=n_reason, is_transformer=is_transformer)
            results[model_name]['reasoning'][task.name()] = score
            status = "PASS" if score > 1/vocab_size else "FAIL"
            print(f"  {task.name():<12}: {score:>6.1%} [{status}]")

        # Physics (fresh model)
        print(f'\n--- Physics ---')
        model = create_model(model_name, state_dim, 4)
        phys_env = PhysicsEnvironment(state_dim)

        for task in physics_tasks:
            score = run_physics(model, phys_env, task, n_train=30, n_eval=n_phys, is_transformer=is_transformer)
            results[model_name]['physics'][task.name()] = score
            status = "PASS" if score > 0.25 else "FAIL"
            print(f"  {task.name():<12}: {score:>6.1%} [{status}]")

        # Summary
        results[model_name]['reasoning_avg'] = np.mean(list(results[model_name]['reasoning'].values()))
        results[model_name]['physics_avg'] = np.mean(list(results[model_name]['physics'].values()))
        results[model_name]['overall'] = (results[model_name]['reasoning_avg'] + results[model_name]['physics_avg']) / 2
        results[model_name]['params'] = params

    # ========== COMPARISON TABLES ==========
    print('\n' + '=' * 90)
    print('R E A S O N I N G  C O M P A R I S O N')
    print('=' * 90)

    print(f'\n{"Model":<16}', end='')
    for t in reasoning_tasks:
        print(f' | {t.name():>10}', end='')
    print(f' | {"AVG":>8}')
    print('-' * 90)

    for m in models:
        print(f'{m:<16}', end='')
        for t in reasoning_tasks:
            print(f' | {results[m]["reasoning"][t.name()]:>9.1%}', end='')
        print(f' | {results[m]["reasoning_avg"]:>7.1%}')

    print('\n' + '=' * 90)
    print('P H Y S I C S  C O M P A R I S O N')
    print('=' * 90)

    print(f'\n{"Model":<16}', end='')
    for t in physics_tasks:
        print(f' | {t.name():>10}', end='')
    print(f' | {"AVG":>8}')
    print('-' * 90)

    for m in models:
        print(f'{m:<16}', end='')
        for t in physics_tasks:
            print(f' | {results[m]["physics"][t.name()]:>9.1%}', end='')
        print(f' | {results[m]["physics_avg"]:>7.1%}')

    # ========== OVERALL RANKING ==========
    print('\n' + '=' * 90)
    print('O V E R A L L  R A N K I N G')
    print('=' * 90)

    ranked = sorted(models, key=lambda m: results[m]['overall'], reverse=True)

    print(f'\n{"Rank":<6} | {"Model":<16} | {"Reasoning":>10} | {"Physics":>10} | {"Overall":>10} | {"Params":>10}')
    print('-' * 80)

    for i, m in enumerate(ranked, 1):
        r = results[m]
        marker = ""
        if m.startswith("V2-"): marker = " [V2]"
        elif m == "Transformer": marker = " [EXT]"
        print(f'{i:<6} | {m:<16} | {r["reasoning_avg"]:>9.1%} | {r["physics_avg"]:>9.1%} | {r["overall"]:>9.1%} | {r["params"]:>10,}{marker}')

    # ========== V1 vs V2 ANALYSIS ==========
    print('\n' + '=' * 90)
    print('V 1  vs  V 2  A N A L Y S I S')
    print('=' * 90)

    v1_models = [m for m in models if m.startswith("V1-")]
    v2_models = [m for m in models if m.startswith("V2-")]

    best_v1 = max(v1_models, key=lambda m: results[m]['overall'])
    best_v2 = max(v2_models, key=lambda m: results[m]['overall'])
    transformer = "Transformer"

    print(f'\nBest V1: {best_v1} ({results[best_v1]["overall"]:.1%})')
    print(f'Best V2: {best_v2} ({results[best_v2]["overall"]:.1%})')
    print(f'Transformer: {results[transformer]["overall"]:.1%}')

    v1_avg = np.mean([results[m]['overall'] for m in v1_models])
    v2_avg = np.mean([results[m]['overall'] for m in v2_models])

    print(f'\nV1 average: {v1_avg:.1%}')
    print(f'V2 average: {v2_avg:.1%}')
    print(f'V2 improvement: {((v2_avg/v1_avg)-1)*100:+.1f}%')

    v2_beat_transformer = sum(1 for m in v2_models if results[m]['overall'] > results[transformer]['overall'])
    print(f'\nV2 models beating Transformer: {v2_beat_transformer}/{len(v2_models)}')

    print('=' * 90)

    # Verdict
    if results[best_v2]['overall'] > results[best_v1]['overall'] and \
       results[best_v2]['overall'] > results[transformer]['overall']:
        print('\nVERDICT: V2 synthesis SUCCESSFUL - Best V2 outperforms V1 and Transformer!')
    elif results[best_v2]['overall'] > results[best_v1]['overall']:
        print('\nVERDICT: V2 synthesis PARTIAL - Best V2 outperforms V1 but not Transformer')
    else:
        print('\nVERDICT: V2 synthesis NEEDS WORK - V1 still competitive')

    print('=' * 90 + '\n')

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='V2 Comprehensive Benchmark')
    parser.add_argument('--reasoning-trials', type=int, default=100)
    parser.add_argument('--physics-trials', type=int, default=50)
    parser.add_argument('--output', type=str, default='v2_benchmark_results.json')

    args = parser.parse_args()

    results = run_v2_benchmark(n_reason=args.reasoning_trials, n_phys=args.physics_trials)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Results saved to: {output_path}')
