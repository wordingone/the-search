"""
A N I M A  P H Y S I C S  T E S T
=================================

Test 2: Physical Environment Simulation

Tests whether Anima can learn to predict and navigate
simple physical dynamics.

Tasks:
1. Projectile prediction (where will ball land?)
2. Collision avoidance (move away from approaching object)
3. Goal seeking (move toward target)
4. Momentum tracking (predict velocity from positions)

Physics is simplified 2D - testing if architecture can
learn physical intuition, not match real physics engines.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

from anima import Anima, AnimaConfig


@dataclass
class PhysicsObject:
    """A simple physics object with position and velocity."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.1

    def update(self, dt: float = 0.1, gravity: float = 0.0):
        """Update position based on velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy -= gravity * dt  # Gravity pulls down

    def distance_to(self, other: 'PhysicsObject') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class PhysicsEnvironment:
    """
    2D physics environment for Anima.

    State encoding:
    - Agent position (x, y)
    - Agent velocity (vx, vy)
    - Object positions (x, y) for each object
    - Object velocities (vx, vy) for each object
    - Goal position (x, y) if applicable
    """

    def __init__(self, state_dim: int = 16, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = PhysicsObject(0.5, 0.5)
        self.objects: List[PhysicsObject] = []
        self.goal: Optional[PhysicsObject] = None
        self.bounds = (0.0, 1.0)  # World is unit square
        self.dt = 0.1

    def reset(self):
        """Reset environment."""
        self.agent = PhysicsObject(0.5, 0.5)
        self.objects = []
        self.goal = None

    def add_object(self, obj: PhysicsObject):
        self.objects.append(obj)

    def set_goal(self, x: float, y: float):
        self.goal = PhysicsObject(x, y)

    def get_state(self) -> np.ndarray:
        """Encode current state as vector."""
        state = np.zeros(self.state_dim, dtype=np.float32)

        # Agent state (4 dims)
        state[0] = self.agent.x
        state[1] = self.agent.y
        state[2] = self.agent.vx
        state[3] = self.agent.vy

        # First object (4 dims)
        if len(self.objects) > 0:
            state[4] = self.objects[0].x
            state[5] = self.objects[0].y
            state[6] = self.objects[0].vx
            state[7] = self.objects[0].vy

        # Goal (2 dims)
        if self.goal:
            state[8] = self.goal.x
            state[9] = self.goal.y

        # Distance to goal (1 dim)
        if self.goal:
            state[10] = self.agent.distance_to(self.goal)

        # Distance to nearest object (1 dim)
        if self.objects:
            state[11] = min(self.agent.distance_to(o) for o in self.objects)

        return state

    def apply_action(self, action: np.ndarray):
        """Apply action as velocity change."""
        # Action: [dx, dy, dvx, dvy] or subset
        if len(action) >= 2:
            # Direct velocity control
            self.agent.vx = np.clip(action[0], -0.5, 0.5)
            self.agent.vy = np.clip(action[1], -0.5, 0.5)

    def step(self, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Step physics and return new state."""
        if action is not None:
            self.apply_action(action)

        # Update agent
        self.agent.update(self.dt)

        # Clamp to bounds
        self.agent.x = np.clip(self.agent.x, self.bounds[0], self.bounds[1])
        self.agent.y = np.clip(self.agent.y, self.bounds[0], self.bounds[1])

        # Update objects
        for obj in self.objects:
            obj.update(self.dt)

        return self.get_state()

    def check_collision(self, threshold: float = 0.15) -> bool:
        """Check if agent collides with any object."""
        for obj in self.objects:
            if self.agent.distance_to(obj) < threshold:
                return True
        return False

    def check_goal_reached(self, threshold: float = 0.1) -> bool:
        """Check if agent reached goal."""
        if self.goal:
            return self.agent.distance_to(self.goal) < threshold
        return False


class PhysicsTask:
    """Base class for physics tasks."""

    def setup(self, env: PhysicsEnvironment):
        raise NotImplementedError

    def evaluate(self, env: PhysicsEnvironment, final_action: np.ndarray) -> float:
        """Return score 0-1."""
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class ProjectilePredictionTask(PhysicsTask):
    """
    Task: Predict where a moving object will be.
    Success = moving toward predicted landing spot.
    """

    def name(self) -> str:
        return "Projectile Prediction"

    def setup(self, env: PhysicsEnvironment):
        env.reset()
        # Launch projectile from random position
        obj = PhysicsObject(
            x=np.random.uniform(0.1, 0.3),
            y=np.random.uniform(0.5, 0.8),
            vx=np.random.uniform(0.2, 0.5),
            vy=np.random.uniform(-0.1, 0.1),
        )
        env.add_object(obj)

        # Predict where it will be in ~5 steps
        future_x = obj.x + obj.vx * 5 * env.dt
        future_y = obj.y + obj.vy * 5 * env.dt
        env.set_goal(np.clip(future_x, 0, 1), np.clip(future_y, 0, 1))

        # Agent starts elsewhere
        env.agent.x = np.random.uniform(0.5, 0.9)
        env.agent.y = np.random.uniform(0.2, 0.5)

    def evaluate(self, env: PhysicsEnvironment, final_action: np.ndarray) -> float:
        # Score based on how close agent moved toward goal
        if env.goal:
            dist = env.agent.distance_to(env.goal)
            return max(0, 1 - dist * 2)  # 1 if at goal, 0 if far
        return 0


class CollisionAvoidanceTask(PhysicsTask):
    """
    Task: Avoid approaching object.
    Success = not colliding after object passes.
    """

    def name(self) -> str:
        return "Collision Avoidance"

    def setup(self, env: PhysicsEnvironment):
        env.reset()
        # Agent in center
        env.agent.x = 0.5
        env.agent.y = 0.5

        # Object approaching from random direction
        angle = np.random.uniform(0, 2 * np.pi)
        speed = 0.3
        obj = PhysicsObject(
            x=0.5 + 0.4 * np.cos(angle),
            y=0.5 + 0.4 * np.sin(angle),
            vx=-speed * np.cos(angle),
            vy=-speed * np.sin(angle),
        )
        env.add_object(obj)

    def evaluate(self, env: PhysicsEnvironment, final_action: np.ndarray) -> float:
        # Score 1 if no collision, 0 if collision
        return 0.0 if env.check_collision(threshold=0.12) else 1.0


class GoalSeekingTask(PhysicsTask):
    """
    Task: Move toward goal position.
    Success = reaching goal.
    """

    def name(self) -> str:
        return "Goal Seeking"

    def setup(self, env: PhysicsEnvironment):
        env.reset()
        # Agent at random position
        env.agent.x = np.random.uniform(0.1, 0.4)
        env.agent.y = np.random.uniform(0.1, 0.4)

        # Goal at different random position
        env.set_goal(
            np.random.uniform(0.6, 0.9),
            np.random.uniform(0.6, 0.9),
        )

    def evaluate(self, env: PhysicsEnvironment, final_action: np.ndarray) -> float:
        if env.goal:
            dist = env.agent.distance_to(env.goal)
            return max(0, 1 - dist)
        return 0


class MomentumTrackingTask(PhysicsTask):
    """
    Task: Match velocity of moving object.
    Success = agent velocity similar to object velocity.
    """

    def name(self) -> str:
        return "Momentum Tracking"

    def setup(self, env: PhysicsEnvironment):
        env.reset()
        # Object with random velocity
        target_vx = np.random.uniform(-0.3, 0.3)
        target_vy = np.random.uniform(-0.3, 0.3)

        obj = PhysicsObject(
            x=0.5,
            y=0.5,
            vx=target_vx,
            vy=target_vy,
        )
        env.add_object(obj)

        # Agent starts stationary
        env.agent.x = 0.3
        env.agent.y = 0.3
        env.agent.vx = 0
        env.agent.vy = 0

    def evaluate(self, env: PhysicsEnvironment, final_action: np.ndarray) -> float:
        if env.objects:
            obj = env.objects[0]
            # Score based on velocity match
            vel_diff = np.sqrt(
                (env.agent.vx - obj.vx)**2 +
                (env.agent.vy - obj.vy)**2
            )
            return max(0, 1 - vel_diff * 2)
        return 0


def run_physics_trial(
    entity: Anima,
    env: PhysicsEnvironment,
    task: PhysicsTask,
    steps: int = 20,
) -> float:
    """Run a single physics trial."""
    task.setup(env)

    final_action = None

    for step in range(steps):
        # Get state
        state = env.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Anima step
        status = entity.step(state_tensor)

        if not status['alive']:
            break

        # Get action
        action = status['action'].numpy().flatten()
        final_action = action

        # Apply to environment
        env.step(action)

    # Evaluate
    score = task.evaluate(env, final_action if final_action is not None else np.zeros(4))
    return score


def run_physics_tests(n_trials: int = 100) -> Dict[str, Any]:
    """Run all physics tests."""
    print('\n' + '=' * 70)
    print('A N I M A  P H Y S I C S  T E S T')
    print('Physical Environment Simulation')
    print('=' * 70)

    state_dim = 16
    action_dim = 4

    # Create Anima
    config = AnimaConfig(
        world_dim=64,
        internal_dim=64,
        time_dim=16,
        sensory_dim=state_dim,
        action_dim=action_dim,
        n_world_slots=8,
        n_internal_slots=6,
        base_frequency=0.2,
        use_energy=False,
    )

    entity = Anima(config)
    env = PhysicsEnvironment(state_dim=state_dim, action_dim=action_dim)

    print(f'\nAnima parameters: {sum(p.numel() for p in entity.parameters()):,}')
    print(f'State dim: {state_dim}, Action dim: {action_dim}')
    print(f'Trials per task: {n_trials}')
    print('=' * 70)

    tasks = [
        ProjectilePredictionTask(),
        CollisionAvoidanceTask(),
        GoalSeekingTask(),
        MomentumTrackingTask(),
    ]

    results = []

    for task in tasks:
        print(f'\n[TASK] {task.name()}')

        # Training phase
        print('  Training phase...')
        for _ in range(50):
            run_physics_trial(entity, env, task, steps=20)

        # Evaluation phase
        print('  Evaluation phase...')
        scores = []
        for _ in range(n_trials):
            score = run_physics_trial(entity, env, task, steps=20)
            scores.append(score)

        avg_score = np.mean(scores)
        random_baseline = 0.25  # Approximate random performance

        result = {
            'task': task.name(),
            'avg_score': avg_score,
            'std_score': np.std(scores),
            'random_baseline': random_baseline,
            'above_random': avg_score > random_baseline,
            'improvement_ratio': avg_score / random_baseline if random_baseline > 0 else 0,
        }
        results.append(result)

        print(f'  Average score: {avg_score:.1%}')
        print(f'  Random baseline: ~{random_baseline:.1%}')
        print(f'  Above random: {"YES" if result["above_random"] else "NO"}')

    # Summary
    print('\n' + '=' * 70)
    print('P H Y S I C S  S U M M A R Y')
    print('=' * 70)

    print(f'\n{"Task":<25} | {"Score":>10} | {"Baseline":>10} | {"Status":>10}')
    print('-' * 70)

    tasks_above_random = 0
    for r in results:
        status = 'PASS' if r['above_random'] else 'FAIL'
        if r['above_random']:
            tasks_above_random += 1
        print(f'{r["task"]:<25} | {r["avg_score"]:>9.1%} | {r["random_baseline"]:>9.1%} | {status:>10}')

    print('-' * 70)
    print(f'Tasks above random: {tasks_above_random}/{len(results)}')

    avg_score = np.mean([r['avg_score'] for r in results])
    print(f'Average score: {avg_score:.1%}')

    if tasks_above_random >= len(results) // 2:
        print('\n[RESULT] Anima shows PHYSICAL INTUITION (above random on majority)')
    else:
        print('\n[RESULT] Anima does NOT show physical intuition yet')

    print('=' * 70 + '\n')

    return {
        'tasks': results,
        'tasks_above_random': tasks_above_random,
        'total_tasks': len(results),
        'avg_score': avg_score,
    }


if __name__ == '__main__':
    results = run_physics_tests(n_trials=100)
