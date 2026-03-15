"""
ANIMA Evaluation Harness
========================

Multi-benchmark evaluation for ANIMA architectures.
"""

import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from tqdm import tqdm
from datetime import datetime

from .metrics import (
    accuracy, mse, mae, perplexity,
    task_specific_accuracy, MetricTracker
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""
    name: str
    task_type: str  # sequence, pattern, collision, goal, etc.
    num_samples: int = 100
    seq_len: int = 10
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'mse'])


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    benchmarks: List[BenchmarkConfig] = field(default_factory=list)
    batch_size: int = 32
    device: str = 'auto'
    output_dir: str = 'eval_results'
    save_predictions: bool = False


class BenchmarkTask:
    """Base class for benchmark tasks."""

    def __init__(self, config: BenchmarkConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)

    def generate_data(self) -> tuple:
        """Generate (inputs, targets) for the benchmark."""
        raise NotImplementedError

    def evaluate(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate model on this benchmark."""
        inputs, targets = self.generate_data()
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.eval()
        with torch.no_grad():
            # Handle ANIMA vs standard models
            if hasattr(model, 'reset'):
                model.reset(inputs.shape[0], device)
                outputs = []
                for t in range(inputs.shape[1]):
                    result = model.step(inputs[:, t])
                    outputs.append(result['action'])
                predictions = torch.stack(outputs, dim=1)
            else:
                predictions = model(inputs)

        # Compute metrics
        results = {}

        # Extract last timestep predictions and targets
        pred_final = predictions[:, -1] if predictions.dim() > 2 else predictions
        tgt_final = targets[:, -1] if targets.dim() > 2 else targets

        # Ensure matching shapes for metrics
        if pred_final.shape != tgt_final.shape:
            min_dim = min(pred_final.shape[-1], tgt_final.shape[-1])
            pred_final = pred_final[..., :min_dim]
            tgt_final = tgt_final[..., :min_dim]

        if 'accuracy' in self.config.metrics:
            results['accuracy'] = task_specific_accuracy(
                pred_final, tgt_final,
                self.config.task_type
            )

        if 'mse' in self.config.metrics:
            results['mse'] = mse(pred_final, tgt_final)

        if 'mae' in self.config.metrics:
            results['mae'] = mae(pred_final, tgt_final)

        return results


class SequenceBenchmark(BenchmarkTask):
    """Arithmetic sequence prediction."""

    def generate_data(self) -> tuple:
        n = self.config.num_samples
        seq_len = self.config.seq_len

        inputs = []
        targets = []

        for _ in range(n):
            start = torch.rand(1, generator=self.rng).item() * 10
            step = torch.rand(1, generator=self.rng).item() * 5 + 0.5

            seq = torch.tensor([start + i * step for i in range(seq_len + 1)]) / 100.0
            inp = torch.zeros(seq_len, 8)
            inp[:, 0] = seq[:-1]
            tgt = torch.zeros(seq_len, 4)
            tgt[:, 0] = seq[1:]

            inputs.append(inp)
            targets.append(tgt)

        return torch.stack(inputs), torch.stack(targets)


class PatternBenchmark(BenchmarkTask):
    """Repeating pattern recognition."""

    def generate_data(self) -> tuple:
        n = self.config.num_samples
        seq_len = self.config.seq_len

        inputs = []
        targets = []

        for _ in range(n):
            plen = int(torch.randint(2, 5, (1,), generator=self.rng).item())
            pattern = torch.rand(plen, generator=self.rng)

            repeats = (seq_len // plen) + 2
            full = pattern.repeat(repeats)[:seq_len + 1]

            inp = torch.zeros(seq_len, 8)
            inp[:, 0] = full[:-1]
            tgt = torch.zeros(seq_len, 4)
            tgt[:, 0] = full[1:]

            inputs.append(inp)
            targets.append(tgt)

        return torch.stack(inputs), torch.stack(targets)


class CollisionBenchmark(BenchmarkTask):
    """Collision prediction (binary classification)."""

    def generate_data(self) -> tuple:
        n = self.config.num_samples
        seq_len = self.config.seq_len

        inputs = []
        targets = []

        for _ in range(n):
            x1 = torch.rand(1, generator=self.rng).item() * 0.5
            v1 = torch.rand(1, generator=self.rng).item() * 0.3 + 0.1
            x2 = torch.rand(1, generator=self.rng).item() * 0.5 + 0.5
            v2 = -torch.rand(1, generator=self.rng).item() * 0.3 - 0.1

            # Will collide if closing velocity > 0
            will_collide = 1.0 if v1 > -v2 else 0.0

            inp = torch.zeros(seq_len, 8)
            inp[0, :4] = torch.tensor([x1, v1, x2, v2])
            # Target is just the collision label (scalar per sample)
            tgt = torch.tensor([will_collide])

            inputs.append(inp)
            targets.append(tgt)

        return torch.stack(inputs), torch.stack(targets)


class GoalBenchmark(BenchmarkTask):
    """Goal-directed navigation."""

    def generate_data(self) -> tuple:
        n = self.config.num_samples
        seq_len = self.config.seq_len

        inputs = []
        targets = []

        for _ in range(n):
            pos = torch.rand(2, generator=self.rng) - 0.5
            goal = torch.rand(2, generator=self.rng) - 0.5

            # Direction to goal
            direction = goal - pos
            direction = direction / (direction.norm() + 1e-6)

            inp = torch.zeros(seq_len, 8)
            inp[0, :2] = pos
            inp[0, 2:4] = goal
            tgt = torch.zeros(seq_len, 4)
            tgt[-1, :2] = direction

            inputs.append(inp)
            targets.append(tgt)

        return torch.stack(inputs), torch.stack(targets)


class ProjectileBenchmark(BenchmarkTask):
    """Projectile trajectory prediction."""

    def generate_data(self) -> tuple:
        n = self.config.num_samples
        seq_len = self.config.seq_len

        inputs = []
        targets = []

        for _ in range(n):
            v0 = torch.rand(1, generator=self.rng).item() * 10 + 5
            t = torch.linspace(0, 1, seq_len + 1)
            positions = v0 * t / 20.0

            inp = torch.zeros(seq_len, 8)
            inp[:, 0] = positions[:-1]
            tgt = torch.zeros(seq_len, 4)
            tgt[:, 0] = positions[1:]

            inputs.append(inp)
            targets.append(tgt)

        return torch.stack(inputs), torch.stack(targets)


class MomentumBenchmark(BenchmarkTask):
    """Momentum conservation prediction."""

    def generate_data(self) -> tuple:
        n = self.config.num_samples
        seq_len = self.config.seq_len

        inputs = []
        targets = []

        for _ in range(n):
            m1 = torch.rand(1, generator=self.rng).item() * 0.8 + 0.2
            v1 = torch.rand(1, generator=self.rng).item() * 0.8 + 0.2
            m2 = torch.rand(1, generator=self.rng).item() * 0.8 + 0.2
            v2 = -torch.rand(1, generator=self.rng).item() * 0.8 - 0.2

            # Final velocity (inelastic)
            v_final = (m1 * v1 + m2 * v2) / (m1 + m2)
            target_norm = (v_final + 1) / 2  # Normalize to [0, 1]

            inp = torch.zeros(seq_len, 8)
            inp[0, :4] = torch.tensor([m1, v1, m2, v2])
            tgt = torch.zeros(seq_len, 4)
            tgt[-1, 0] = target_norm

            inputs.append(inp)
            targets.append(tgt)

        return torch.stack(inputs), torch.stack(targets)


BENCHMARK_REGISTRY = {
    'sequence': SequenceBenchmark,
    'pattern': PatternBenchmark,
    'collision': CollisionBenchmark,
    'goal': GoalBenchmark,
    'projectile': ProjectileBenchmark,
    'momentum': MomentumBenchmark,
}


class EvaluationHarness:
    """
    Multi-benchmark evaluation harness.

    Runs a model through multiple benchmark tasks and reports results.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()

        # Setup device
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        # Setup output
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default benchmarks if none specified
        if not self.config.benchmarks:
            self.config.benchmarks = [
                BenchmarkConfig('sequence', 'sequence'),
                BenchmarkConfig('pattern', 'pattern'),
                BenchmarkConfig('collision', 'collision'),
                BenchmarkConfig('goal', 'goal'),
                BenchmarkConfig('projectile', 'projectile'),
                BenchmarkConfig('momentum', 'momentum'),
            ]

    def evaluate(
        self,
        model: nn.Module,
        model_name: str = 'model',
    ) -> Dict[str, Any]:
        """
        Run all benchmarks on a model.

        Args:
            model: Model to evaluate
            model_name: Name for logging/saving

        Returns:
            Results dictionary
        """
        model = model.to(self.device)
        model.eval()

        logger.info(f"Evaluating {model_name} on {len(self.config.benchmarks)} benchmarks")
        logger.info(f"Device: {self.device}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'benchmarks': {},
        }

        # Categorize results
        reasoning_tasks = ['sequence', 'pattern', 'conditional', 'analogy']
        physics_tasks = ['projectile', 'collision', 'goal', 'momentum']

        reasoning_scores = []
        physics_scores = []

        for benchmark_config in tqdm(self.config.benchmarks, desc="Benchmarks"):
            # Create benchmark task
            if benchmark_config.task_type not in BENCHMARK_REGISTRY:
                logger.warning(f"Unknown benchmark type: {benchmark_config.task_type}")
                continue

            benchmark = BENCHMARK_REGISTRY[benchmark_config.task_type](benchmark_config)

            # Run evaluation
            try:
                task_results = benchmark.evaluate(model, self.device)
                results['benchmarks'][benchmark_config.name] = task_results

                # Track category scores
                if benchmark_config.task_type in reasoning_tasks:
                    reasoning_scores.append(task_results.get('accuracy', 0))
                elif benchmark_config.task_type in physics_tasks:
                    physics_scores.append(task_results.get('accuracy', 0))

                logger.info(f"  {benchmark_config.name}: {task_results}")

            except Exception as e:
                logger.error(f"Benchmark {benchmark_config.name} failed: {e}")
                results['benchmarks'][benchmark_config.name] = {'error': str(e)}

        # Aggregate scores
        results['summary'] = {
            'reasoning_avg': sum(reasoning_scores) / max(len(reasoning_scores), 1),
            'physics_avg': sum(physics_scores) / max(len(physics_scores), 1),
        }
        results['summary']['overall'] = (
            results['summary']['reasoning_avg'] + results['summary']['physics_avg']
        ) / 2

        logger.info(f"\nSummary:")
        logger.info(f"  Reasoning: {results['summary']['reasoning_avg']:.1%}")
        logger.info(f"  Physics: {results['summary']['physics_avg']:.1%}")
        logger.info(f"  Overall: {results['summary']['overall']:.1%}")

        # Save results
        output_path = self.output_dir / f"{model_name}_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        return results

    def compare(
        self,
        models: Dict[str, nn.Module],
    ) -> Dict[str, Any]:
        """
        Compare multiple models on all benchmarks.

        Args:
            models: Dict mapping model names to models

        Returns:
            Comparison results
        """
        all_results = {}

        for name, model in models.items():
            all_results[name] = self.evaluate(model, name)

        # Create comparison table
        comparison = {
            'models': list(models.keys()),
            'benchmarks': {},
            'summary': {},
        }

        for benchmark_config in self.config.benchmarks:
            benchmark_name = benchmark_config.name
            comparison['benchmarks'][benchmark_name] = {
                name: all_results[name]['benchmarks'].get(benchmark_name, {}).get('accuracy', 0)
                for name in models.keys()
            }

        for key in ['reasoning_avg', 'physics_avg', 'overall']:
            comparison['summary'][key] = {
                name: all_results[name]['summary'].get(key, 0)
                for name in models.keys()
            }

        # Save comparison
        output_path = self.output_dir / 'comparison.json'
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)

        return comparison


def evaluate_model(
    model: nn.Module,
    benchmarks: Optional[List[str]] = None,
    device: str = 'auto',
) -> Dict[str, float]:
    """
    Quick evaluation function.

    Args:
        model: Model to evaluate
        benchmarks: List of benchmark names (default: all)
        device: Device to use

    Returns:
        Results dictionary
    """
    if benchmarks is None:
        benchmarks = ['sequence', 'pattern', 'collision', 'goal', 'projectile', 'momentum']

    config = EvalConfig(
        benchmarks=[BenchmarkConfig(name=b, task_type=b) for b in benchmarks],
        device=device,
    )

    harness = EvaluationHarness(config)
    return harness.evaluate(model)
