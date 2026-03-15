"""
A N I M A  R E A S O N I N G  T E S T
=====================================

Test 1: Text-Based Reasoning Capability

Tests whether Anima can learn sequential patterns and make
predictions that demonstrate reasoning-like behavior.

Tasks:
1. Sequence completion (A B C -> D)
2. Pattern recognition (1 2 1 2 -> 1)
3. Simple logic (IF X THEN Y)
4. Analogy-like patterns (A:B :: C:?)

Note: Anima is a tiny model (~10k params) - we're testing
whether the ARCHITECTURE supports reasoning, not whether
it matches GPT-4. Success = better than random.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import time

from anima import Anima, AnimaConfig


class SymbolicEnvironment:
    """
    Environment that presents symbolic sequences.
    Encodes symbols as one-hot vectors.
    """

    def __init__(self, vocab_size: int = 16, sequence_length: int = 4):
        self.vocab_size = vocab_size
        self.seq_len = sequence_length
        self.current_sequence = []
        self.position = 0
        self.target = None

    def encode(self, symbol: int) -> np.ndarray:
        """One-hot encode a symbol."""
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        vec[symbol] = 1.0
        return vec

    def decode(self, vec: np.ndarray) -> int:
        """Decode vector to symbol (argmax)."""
        return int(np.argmax(vec))

    def set_sequence(self, sequence: List[int], target: int):
        """Set the current sequence and target."""
        self.current_sequence = sequence
        self.target = target
        self.position = 0

    def step(self) -> Tuple[np.ndarray, bool, int]:
        """
        Return next symbol in sequence.
        Returns: (encoded_symbol, is_query_position, target)
        """
        if self.position < len(self.current_sequence):
            symbol = self.current_sequence[self.position]
            self.position += 1
            return self.encode(symbol), False, self.target
        else:
            # Query position - return zeros (asking for prediction)
            return np.zeros(self.vocab_size, dtype=np.float32), True, self.target

    def reset(self):
        self.position = 0


class ReasoningTask:
    """Base class for reasoning tasks."""

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size

    def generate_example(self) -> Tuple[List[int], int]:
        """Generate (sequence, target) pair."""
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


class SequenceCompletionTask(ReasoningTask):
    """
    Task: Complete arithmetic sequences.
    Example: [2, 4, 6] -> 8
    """

    def name(self) -> str:
        return "Sequence Completion"

    def generate_example(self) -> Tuple[List[int], int]:
        # Generate arithmetic sequence
        start = np.random.randint(0, 4)
        step = np.random.randint(1, 4)
        seq = [start + i * step for i in range(3)]
        target = start + 3 * step

        # Clamp to vocab size
        seq = [s % self.vocab_size for s in seq]
        target = target % self.vocab_size

        return seq, target


class PatternRepetitionTask(ReasoningTask):
    """
    Task: Predict next in repeating pattern.
    Example: [1, 2, 1, 2] -> 1
    """

    def name(self) -> str:
        return "Pattern Repetition"

    def generate_example(self) -> Tuple[List[int], int]:
        # Generate repeating pattern of length 2
        a = np.random.randint(0, self.vocab_size)
        b = np.random.randint(0, self.vocab_size)
        while b == a:
            b = np.random.randint(0, self.vocab_size)

        seq = [a, b, a, b]
        target = a  # Pattern repeats

        return seq, target


class ConditionalTask(ReasoningTask):
    """
    Task: Learn IF-THEN rules.
    IF symbol 0 appears -> target is 1
    IF symbol 2 appears -> target is 3
    etc.
    """

    def name(self) -> str:
        return "Conditional Logic"

    def generate_example(self) -> Tuple[List[int], int]:
        # Rule: target = trigger + 1
        trigger = np.random.randint(0, self.vocab_size - 1)
        target = trigger + 1

        # Generate sequence with trigger at random position
        seq_len = 4
        seq = [np.random.randint(0, self.vocab_size) for _ in range(seq_len)]
        trigger_pos = np.random.randint(0, seq_len)
        seq[trigger_pos] = trigger

        return seq, target


class AnalogyTask(ReasoningTask):
    """
    Task: A:B :: C:?
    Pattern: difference between A and B applied to C gives target.
    Example: 2:4 :: 3:? -> 5 (difference is +2)
    """

    def name(self) -> str:
        return "Analogy Pattern"

    def generate_example(self) -> Tuple[List[int], int]:
        # A:B with some relationship
        a = np.random.randint(1, self.vocab_size - 4)
        diff = np.random.randint(1, 4)
        b = a + diff

        # C:? with same relationship
        c = np.random.randint(1, self.vocab_size - 4)
        target = (c + diff) % self.vocab_size

        seq = [a, b, c]
        return seq, target


def evaluate_reasoning(
    entity: Anima,
    task: ReasoningTask,
    env: SymbolicEnvironment,
    n_trials: int = 100,
    training_steps: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate Anima on a reasoning task.

    Process:
    1. Present sequence symbols one by one
    2. At query position, check if Anima's action matches target
    3. Repeat for many trials

    Returns accuracy and comparison to random baseline.
    """
    correct = 0
    total = 0

    for trial in range(n_trials):
        # Generate example
        seq, target = task.generate_example()
        env.set_sequence(seq, target)

        # Present sequence to Anima
        for _ in range(len(seq) + 1):
            obs, is_query, tgt = env.step()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Step Anima
            status = entity.step(obs_tensor)

            if is_query and status['alive']:
                # Get Anima's prediction from action
                action = status['action'].numpy().flatten()

                # Map action to prediction (use first vocab_size dims)
                if len(action) >= env.vocab_size:
                    pred = env.decode(action[:env.vocab_size])
                else:
                    # Extend action to vocab size
                    extended = np.zeros(env.vocab_size)
                    extended[:len(action)] = action
                    pred = env.decode(extended)

                if pred == tgt:
                    correct += 1
                total += 1

        env.reset()

    accuracy = correct / total if total > 0 else 0
    random_baseline = 1.0 / env.vocab_size

    return {
        'task': task.name(),
        'accuracy': accuracy,
        'random_baseline': random_baseline,
        'above_random': accuracy > random_baseline,
        'improvement_ratio': accuracy / random_baseline if random_baseline > 0 else 0,
        'correct': correct,
        'total': total,
    }


def run_reasoning_tests(n_trials: int = 200) -> Dict[str, Any]:
    """Run all reasoning tests."""
    print('\n' + '=' * 70)
    print('A N I M A  R E A S O N I N G  T E S T')
    print('Text-Based Sequential Pattern Learning')
    print('=' * 70)

    vocab_size = 16

    # Create Anima with vocab-sized output
    config = AnimaConfig(
        world_dim=64,
        internal_dim=64,
        time_dim=16,
        sensory_dim=vocab_size,
        action_dim=vocab_size,  # Output matches vocab for prediction
        n_world_slots=8,
        n_internal_slots=6,
        base_frequency=0.15,
        use_energy=False,
    )

    entity = Anima(config)
    env = SymbolicEnvironment(vocab_size=vocab_size)

    print(f'\nAnima parameters: {sum(p.numel() for p in entity.parameters()):,}')
    print(f'Vocab size: {vocab_size}')
    print(f'Trials per task: {n_trials}')
    print('=' * 70)

    tasks = [
        SequenceCompletionTask(vocab_size),
        PatternRepetitionTask(vocab_size),
        ConditionalTask(vocab_size),
        AnalogyTask(vocab_size),
    ]

    results = []

    for task in tasks:
        print(f'\n[TASK] {task.name()}')

        # Warm up Anima on this task type
        print('  Training phase...')
        for _ in range(100):
            seq, target = task.generate_example()
            env.set_sequence(seq, target)
            for _ in range(len(seq) + 1):
                obs, _, _ = env.step()
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                entity.step(obs_tensor)
            env.reset()

        # Evaluate
        print('  Evaluation phase...')
        result = evaluate_reasoning(entity, task, env, n_trials=n_trials)
        results.append(result)

        print(f'  Accuracy: {result["accuracy"]:.1%}')
        print(f'  Random baseline: {result["random_baseline"]:.1%}')
        print(f'  Above random: {"YES" if result["above_random"] else "NO"}')
        print(f'  Improvement: {result["improvement_ratio"]:.2f}x')

    # Summary
    print('\n' + '=' * 70)
    print('R E A S O N I N G  S U M M A R Y')
    print('=' * 70)

    print(f'\n{"Task":<25} | {"Accuracy":>10} | {"Random":>10} | {"Status":>10}')
    print('-' * 70)

    tasks_above_random = 0
    for r in results:
        status = 'PASS' if r['above_random'] else 'FAIL'
        if r['above_random']:
            tasks_above_random += 1
        print(f'{r["task"]:<25} | {r["accuracy"]:>9.1%} | {r["random_baseline"]:>9.1%} | {status:>10}')

    print('-' * 70)
    print(f'Tasks above random: {tasks_above_random}/{len(results)}')

    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_baseline = np.mean([r['random_baseline'] for r in results])
    print(f'Average accuracy: {avg_accuracy:.1%} (baseline: {avg_baseline:.1%})')

    if tasks_above_random >= len(results) // 2:
        print('\n[RESULT] Anima shows REASONING capability (above random on majority)')
    else:
        print('\n[RESULT] Anima does NOT show reasoning capability yet')

    print('=' * 70 + '\n')

    return {
        'tasks': results,
        'tasks_above_random': tasks_above_random,
        'total_tasks': len(results),
        'avg_accuracy': avg_accuracy,
        'avg_baseline': avg_baseline,
    }


if __name__ == '__main__':
    results = run_reasoning_tests(n_trials=200)
