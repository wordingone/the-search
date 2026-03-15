#!/usr/bin/env python
"""
Generate comprehensive report from all quantization experiment results.

Usage:
    python genesis/experiments/quantization/generate_report.py > QUANTIZATION_RESULTS.md
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def load_results(exp_dir: Path) -> Optional[Dict]:
    """Load results from an experiment's results directory."""
    results_files = list(exp_dir.glob("results/*.json"))
    if not results_files:
        return None

    # Load the most recent result file
    latest = max(results_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


def format_pass_fail(passed: bool) -> str:
    """Format pass/fail status."""
    return "PASS" if passed else "FAIL"


def generate_report() -> str:
    """Generate the full report."""
    base_dir = Path(__file__).parent

    lines = [
        "# Quantization Hypothesis Test Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Hypothesis",
        "",
        "> Fully quantized training for Genesis is plausible **only in structured,",
        "> sparse, or discrete latent spaces with bounded activations and local propagation**",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Sub-Problem | Condition | Result | Key Metric |",
        "|-------------|-----------|--------|------------|",
    ]

    # Load and summarize each experiment
    experiments = [
        ("exp1_structured", "Structured (3D grid)", "structured_q8_ratio"),
        ("exp2_sparse", "Sparse (<5% occupancy)", "q8_error_at_5pct_sparsity"),
        ("exp3_discrete", "Discrete (FSQ)", "codebook_utilization"),
        ("exp4_bounded", "Bounded activations", "bounded_pass"),
        ("exp5_local", "Local propagation", "windowed_growth_rate"),
    ]

    all_passed = True
    exp_details = []

    for exp_name, condition, key_metric in experiments:
        exp_dir = base_dir / exp_name
        results = load_results(exp_dir)

        if results is None:
            lines.append(f"| {exp_name} | {condition} | NOT RUN | - |")
            all_passed = False
            continue

        evaluation = results.get("evaluation", {})
        passed = evaluation.get("passed", False)
        all_passed = all_passed and passed

        # Get the key metric value
        metric_value = evaluation.get(key_metric, "N/A")
        if isinstance(metric_value, float):
            metric_value = f"{metric_value:.4f}"
        elif isinstance(metric_value, bool):
            metric_value = "Yes" if metric_value else "No"

        lines.append(f"| {exp_name} | {condition} | {format_pass_fail(passed)} | {metric_value} |")
        exp_details.append((exp_name, condition, results, evaluation))

    # Integration test
    integration_dir = base_dir / "final_integration"
    integration_results = load_results(integration_dir)

    if integration_results:
        eval_int = integration_results.get("evaluation", {})
        int_passed = eval_int.get("hypothesis_confirmed", False)
        q8_ratio = eval_int.get("q8_full_ratio", "N/A")
        if isinstance(q8_ratio, float):
            q8_ratio = f"{q8_ratio:.2f}x"
        lines.append(f"| **Integration** | **All conditions** | **{format_pass_fail(int_passed)}** | **Q8 ratio: {q8_ratio}** |")
        all_passed = all_passed and int_passed
    else:
        lines.append("| **Integration** | **All conditions** | **NOT RUN** | - |")
        all_passed = False

    lines.extend([
        "",
        f"### Overall Verdict: {'HYPOTHESIS CONFIRMED' if all_passed else 'HYPOTHESIS NEEDS REFINEMENT'}",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ])

    # Detailed results for each experiment
    for exp_name, condition, results, evaluation in exp_details:
        lines.extend([
            f"### {exp_name}: {condition}",
            "",
            f"**Status:** {format_pass_fail(evaluation.get('passed', False))}",
            "",
            f"**Interpretation:** {evaluation.get('interpretation', 'N/A')}",
            "",
        ])

        # Add experiment-specific details
        if exp_name == "exp1_structured":
            lines.extend([
                "| Metric | Structured (Windowed) | Unstructured (Full) |",
                "|--------|----------------------|---------------------|",
                f"| Q8/FP32 Loss Ratio | {evaluation.get('structured_q8_ratio', 'N/A'):.4f} | {evaluation.get('unstructured_q8_ratio', 'N/A'):.4f} |",
                "",
            ])

        elif exp_name == "exp2_sparse":
            lines.extend([
                "| Sparsity | Q8 Error |",
                "|----------|----------|",
            ])
            summary = results.get("results", {}).get("summary", {}).get("q8", {})
            for sparsity in ["1", "5", "10", "25", "50", "100"]:
                error = summary.get(sparsity, "N/A")
                if isinstance(error, float):
                    error = f"{error:.6f}"
                marker = " (target)" if sparsity == "5" else ""
                lines.append(f"| {sparsity}%{marker} | {error} |")
            lines.append("")

        elif exp_name == "exp3_discrete":
            lines.extend([
                "| Mode | Loss Ratio | Codebook Util |",
                "|------|------------|---------------|",
                f"| FP32 (baseline) | 1.00x | {results.get('results', {}).get('summary', {}).get('fp32_codebook_util', 'N/A'):.2%} |",
                f"| Q8 gradients | {evaluation.get('q8_grad_loss_ratio', 'N/A'):.2f}x | {results.get('results', {}).get('summary', {}).get('q8_grad_codebook_util', 'N/A'):.2%} |",
                f"| Q8 full | {evaluation.get('q8_full_loss_ratio', 'N/A'):.2f}x | {evaluation.get('codebook_utilization', 'N/A'):.2%} |",
                "",
            ])

        elif exp_name == "exp4_bounded":
            lines.extend([
                "| Activation | Bounded | Q8/FP32 Ratio | Pass |",
                "|------------|---------|---------------|------|",
            ])
            for r in evaluation.get("bounded_results", []):
                ratio = "diverged" if r.get("diverged") else f"{r.get('ratio', 'N/A'):.2f}x"
                lines.append(f"| {r.get('activation', 'N/A')} | Yes | {ratio} | {format_pass_fail(r.get('pass', False))} |")
            for r in evaluation.get("unbounded_results", []):
                ratio = "diverged" if r.get("diverged") else f"{r.get('ratio', 'N/A'):.2f}x"
                expected = "Yes" if r.get("fails_as_expected") else "No"
                lines.append(f"| {r.get('activation', 'N/A')} | No | {ratio} | Fails as expected: {expected} |")
            lines.append("")

        elif exp_name == "exp5_local":
            lines.extend([
                "| Attention Type | Growth Rate | Pattern |",
                "|----------------|-------------|---------|",
                f"| Windowed | {evaluation.get('windowed_growth_rate', 'N/A'):.3f}x/layer | {evaluation.get('windowed_pattern', 'N/A')} |",
                f"| Full | {evaluation.get('full_growth_rate', 'N/A'):.3f}x/layer | {evaluation.get('full_pattern', 'N/A')} |",
                "",
            ])

        lines.append("")

    # Integration details
    if integration_results:
        eval_int = integration_results.get("evaluation", {})
        summary = integration_results.get("summary", {})

        lines.extend([
            "### Final Integration Test",
            "",
            f"**Hypothesis Confirmed:** {format_pass_fail(eval_int.get('hypothesis_confirmed', False))}",
            "",
            f"**Interpretation:** {eval_int.get('interpretation', 'N/A')}",
            "",
            "| Mode | Final Loss | Ratio vs FP32 | Codebook Util |",
            "|------|------------|---------------|---------------|",
        ])

        fp32_loss = summary.get("fp32", {}).get("final_loss", 1.0)
        for mode in ["fp32", "q8_full", "mixed", "q8_sensitive"]:
            data = summary.get(mode, {})
            loss = data.get("final_loss", "N/A")
            ratio = loss / fp32_loss if isinstance(loss, float) else "N/A"
            util = data.get("codebook_utilization", "N/A")

            if isinstance(loss, float):
                loss = f"{loss:.6f}"
            if isinstance(ratio, float):
                ratio = f"{ratio:.2f}x"
            if isinstance(util, float):
                util = f"{util:.2%}"

            lines.append(f"| {mode} | {loss} | {ratio} | {util} |")

        lines.append("")

    lines.extend([
        "---",
        "",
        "## Conclusions",
        "",
    ])

    if all_passed:
        lines.extend([
            "The hypothesis is **CONFIRMED**. Fully quantized (Q8) training is feasible when:",
            "",
            "1. **Structured**: Using 3D voxel grids with spatial locality",
            "2. **Sparse**: Operating on <5% occupied voxels",
            "3. **Discrete**: Using FSQ-style quantized latents with STE",
            "4. **Bounded**: Using bounded activations (HardTanh, Sigmoid, Tanh)",
            "5. **Local**: Using windowed attention to limit error propagation",
            "",
            "### Implications for Genesis",
            "",
            "- The architecture is inherently quantization-friendly",
            "- INT8 inference should be achievable with minimal quality loss",
            "- Mixed precision training (Q8 weights, FP32 accumulators) recommended",
            "- LayerNorm and Softmax should remain in FP32 for stability",
        ])
    else:
        lines.extend([
            "The hypothesis **NEEDS REFINEMENT**. Some conditions failed:",
            "",
            "Review the detailed results above to identify:",
            "- Which conditions are necessary vs sufficient",
            "- Which operations have irreducible precision requirements",
            "- What modifications might enable full quantization",
        ])

    lines.extend([
        "",
        "---",
        "",
        "*Generated by `genesis/experiments/quantization/generate_report.py`*",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_report())
