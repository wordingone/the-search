"""
ANIMA Evaluation Suite
======================

Multi-benchmark evaluation harness for ANIMA architectures.
"""

from .harness import EvaluationHarness, evaluate_model
from .metrics import accuracy, mse, perplexity, retrieval_recall
from .benchmark_standard import (
    BenchmarkStandard,
    TransformerBaselineSpec,
    StandardTransformer,
    TaskSpec,
    TASK_SPECIFICATIONS,
    OFFICIAL_TRANSFORMER_BASELINE,
    OFFICIAL_ANIMA_ZERO_BASELINE,
    OFFICIAL_ANIMA_ONE_BASELINE,
    get_standard,
    get_transformer_spec,
    verify_params,
    print_standard_info,
)

__all__ = [
    'EvaluationHarness',
    'evaluate_model',
    'accuracy',
    'mse',
    'perplexity',
    'retrieval_recall',
    # Benchmark Standard v1.0
    'BenchmarkStandard',
    'TransformerBaselineSpec',
    'StandardTransformer',
    'TaskSpec',
    'TASK_SPECIFICATIONS',
    'OFFICIAL_TRANSFORMER_BASELINE',
    'OFFICIAL_ANIMA_ZERO_BASELINE',
    'OFFICIAL_ANIMA_ONE_BASELINE',
    'get_standard',
    'get_transformer_spec',
    'verify_params',
    'print_standard_info',
]
