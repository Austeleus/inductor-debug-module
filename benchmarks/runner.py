#!/usr/bin/env python3
"""
Benchmark Runner for TorchInductor Debug Module.

Runs all benchmarks and generates a summary report.

Usage:
    python -m benchmarks.runner --all
    python -m benchmarks.runner --model bert
    python -m benchmarks.runner --model resnet --model mamba
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.base import BenchmarkResult


# Color output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class WandbLogger:
    """Optional Weights & Biases logger for benchmark runs."""

    def __init__(self, run, wandb_module):
        self._run = run
        self._wandb = wandb_module

    @classmethod
    def create(
        cls,
        enabled: bool,
        project: Optional[str],
        run_name: Optional[str],
        tags: Optional[List[str]],
        mode: Optional[str],
        config: Dict[str, Any],
    ):
        if not enabled:
            return None
        try:
            import wandb  # type: ignore
        except ImportError:
            print(f"{Colors.YELLOW}Warning: wandb not installed; disabling --wandb flag{Colors.END}")
            return None

        run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "inductor-debug-module"),
            name=run_name,
            tags=tags,
            mode=mode,
            config=config,
        )
        return cls(run, wandb)

    def log_benchmark(self, result):
        metrics = {
            "model_name": result.model_name,
            "model_type": result.model_type,
            "parameter_count": result.parameter_count,
            "device": result.device,
            "kerneldiff_passed": result.kerneldiff_passed,
            "max_absolute_error": result.max_absolute_error,
            "mean_absolute_error": result.mean_absolute_error,
            "graph_count": result.graph_count,
            "graph_break_count": result.graph_break_count,
        }

        def add_backend(prefix: str, backend_result):
            if backend_result is None:
                return
            metrics[f"{prefix}/compile_time_s"] = backend_result.compile_time
            metrics[f"{prefix}/avg_inference_ms"] = backend_result.avg_inference_time * 1000
            metrics[f"{prefix}/std_inference_ms"] = backend_result.std_inference_time * 1000
            metrics[f"{prefix}/success"] = float(bool(backend_result.success))
            if backend_result.constraint_warnings:
                metrics[f"{prefix}/constraint_warnings"] = len(backend_result.constraint_warnings)

        add_backend("eager", result.eager_result)
        add_backend("inductor", result.inductor_result)
        add_backend("mock", result.mock_result)

        self._run.log(metrics)

        summary_html = f"<pre>{result.summary()}</pre>"
        self._run.log({f"summary/{result.model_name}": self._wandb.Html(summary_html)})

    def log_summary_report(self, text: str):
        self._run.log({"benchmark/summary_report": self._wandb.Html(f"<pre>{text}</pre>")})

    def finish(self):
        self._run.finish()


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")


def run_bert_benchmark(**kwargs) -> BenchmarkResult:
    """Run BERT benchmark."""
    from benchmarks.bert import run_bert_benchmark as _run
    return _run(**kwargs)


def run_resnet_benchmark(**kwargs) -> BenchmarkResult:
    """Run ResNet benchmark."""
    from benchmarks.resnet import run_resnet_benchmark as _run
    return _run(**kwargs)


def run_mamba_benchmark(**kwargs) -> BenchmarkResult:
    """Run Mamba benchmark."""
    from benchmarks.mamba import run_mamba_benchmark as _run
    # Use custom small model by default since HuggingFace Mamba has torch.compile issues
    kwargs.setdefault("use_pretrained", False)
    return _run(**kwargs)


BENCHMARK_REGISTRY = {
    "bert": {
        "name": "BERT-base-multilingual",
        "runner": run_bert_benchmark,
        "description": "Transformer encoder model (~178M params)",
    },
    "resnet": {
        "name": "ResNet-18",
        "runner": run_resnet_benchmark,
        "description": "Convolutional neural network (~11M params)",
    },
    "mamba": {
        "name": "Mamba-Small (Custom)",
        "runner": run_mamba_benchmark,
        "description": "State space model (custom config, torch.compile compatible)",
    },
}


def generate_summary_report(results: List[BenchmarkResult], output_dir: str) -> str:
    """Generate a comprehensive summary report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 80,
        "TORCHINDUCTOR DEBUG MODULE - BENCHMARK SUMMARY REPORT",
        "=" * 80,
        f"Generated: {timestamp}",
        f"Models Tested: {len(results)}",
        "",
    ]

    # Overview table
    lines.append("OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"{'Model':<25} {'Type':<12} {'Params':<12} {'KernelDiff':<12} {'Warnings':<10}")
    lines.append("-" * 80)

    for r in results:
        params_str = f"{r.parameter_count / 1e6:.1f}M"
        kd_status = "PASS" if r.kerneldiff_passed else "FAIL"
        warnings = len(r.mock_result.constraint_warnings) if r.mock_result else 0
        lines.append(f"{r.model_name:<25} {r.model_type:<12} {params_str:<12} {kd_status:<12} {warnings:<10}")

    lines.append("-" * 80)
    lines.append("")

    # Compilation times
    lines.append("COMPILATION TIMES (seconds)")
    lines.append("-" * 80)
    lines.append(f"{'Model':<25} {'Eager':<12} {'Inductor':<12} {'Mock':<12}")
    lines.append("-" * 80)

    for r in results:
        eager_t = f"{r.eager_result.compile_time:.3f}" if r.eager_result else "N/A"
        ind_t = f"{r.inductor_result.compile_time:.3f}" if r.inductor_result and r.inductor_result.success else "FAIL"
        mock_t = f"{r.mock_result.compile_time:.3f}" if r.mock_result and r.mock_result.success else "FAIL"
        lines.append(f"{r.model_name:<25} {eager_t:<12} {ind_t:<12} {mock_t:<12}")

    lines.append("-" * 80)
    lines.append("")

    # Inference times
    lines.append("INFERENCE TIMES (milliseconds, avg ± std)")
    lines.append("-" * 80)
    lines.append(f"{'Model':<25} {'Eager':<18} {'Inductor':<18} {'Mock':<18}")
    lines.append("-" * 80)

    for r in results:
        def fmt_time(backend_result):
            if backend_result and backend_result.success:
                avg = backend_result.avg_inference_time * 1000
                std = backend_result.std_inference_time * 1000
                return f"{avg:.1f} ± {std:.1f}"
            return "FAIL"

        lines.append(f"{r.model_name:<25} {fmt_time(r.eager_result):<18} {fmt_time(r.inductor_result):<18} {fmt_time(r.mock_result):<18}")

    lines.append("-" * 80)
    lines.append("")

    # Graph analysis
    lines.append("GRAPH ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"{'Model':<25} {'Graphs':<10} {'Breaks':<10} {'Warnings':<10}")
    lines.append("-" * 80)

    for r in results:
        warnings = len(r.mock_result.constraint_warnings) if r.mock_result else 0
        lines.append(f"{r.model_name:<25} {r.graph_count:<10} {r.graph_break_count:<10} {warnings:<10}")

    lines.append("-" * 80)
    lines.append("")

    # KernelDiff results
    lines.append("KERNELDIFF RESULTS (Mock vs Eager)")
    lines.append("-" * 80)
    lines.append(f"{'Model':<25} {'Status':<10} {'Max Error':<15} {'Mean Error':<15}")
    lines.append("-" * 80)

    for r in results:
        status = "PASS" if r.kerneldiff_passed else "FAIL"
        max_err = f"{r.max_absolute_error:.2e}"
        mean_err = f"{r.mean_absolute_error:.2e}"
        lines.append(f"{r.model_name:<25} {status:<10} {max_err:<15} {mean_err:<15}")

    lines.append("-" * 80)
    lines.append("")

    # Constraint warnings detail
    lines.append("CONSTRAINT WARNINGS (Top warnings per model)")
    lines.append("-" * 80)

    for r in results:
        if r.mock_result and r.mock_result.constraint_warnings:
            lines.append(f"\n{r.model_name}:")
            # Group similar warnings
            warning_counts: Dict[str, int] = {}
            for w in r.mock_result.constraint_warnings:
                # Extract warning type
                key = w.split(" ")[0] if w else "Unknown"
                warning_counts[key] = warning_counts.get(key, 0) + 1

            for warning_type, count in sorted(warning_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {warning_type}: {count} occurrences")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"benchmark_summary_{int(time.time())}.txt")

    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n{Colors.GREEN}Summary report saved to: {report_path}{Colors.END}")

    # Also save as JSON
    json_report = {
        "timestamp": timestamp,
        "models_tested": len(results),
        "results": [r.to_dict() for r in results],
    }

    json_path = report_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"{Colors.GREEN}JSON report saved to: {json_path}{Colors.END}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for TorchInductor Debug Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmarks.runner --all
    python -m benchmarks.runner --model bert --model resnet
    python -m benchmarks.runner --model mamba --device cuda
        """,
    )

    parser.add_argument(
        "--model", "-m",
        action="append",
        choices=list(BENCHMARK_REGISTRY.keys()),
        help="Model(s) to benchmark. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu/cuda)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of inference runs (default: 5)",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "aot"],
        default="mock",
        help="Test backend for benchmarks (default: mock)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and exit",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging for benchmark runs",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override the W&B project name (defaults to WANDB_PROJECT env or inductor-debug-module)",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional run name to show in the W&B UI",
    )
    parser.add_argument(
        "--wandb-tag",
        action="append",
        dest="wandb_tags",
        help="Additional W&B tags (can be provided multiple times)",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Pass-through for wandb.init(mode=...)",
    )

    args = parser.parse_args()

    # List benchmarks
    if args.list:
        print("\nAvailable benchmarks:")
        print("-" * 50)
        for key, info in BENCHMARK_REGISTRY.items():
            print(f"  {key:<10} - {info['name']}")
            print(f"             {info['description']}")
        print()
        return 0

    # Determine which benchmarks to run
    models_to_run = []
    if args.all:
        models_to_run = list(BENCHMARK_REGISTRY.keys())
    elif args.model:
        models_to_run = args.model
    else:
        parser.print_help()
        print(f"\n{Colors.YELLOW}Please specify --all or --model <name>{Colors.END}")
        return 1

    # Run benchmarks
    print_header("TorchInductor Debug Module Benchmarks")

    print(f"Models to benchmark: {', '.join(models_to_run)}")
    print(f"Device: {args.device}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Inference runs: {args.runs}")
    print(f"Output directory: {args.output_dir}")

    wandb_logger = WandbLogger.create(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config={
            "device": args.device,
            "warmup_runs": args.warmup,
            "inference_runs": args.runs,
            "models": models_to_run,
        },
    )

    results: List[BenchmarkResult] = []

    for model_key in models_to_run:
        info = BENCHMARK_REGISTRY[model_key]
        print_header(f"Benchmarking: {info['name']}")

        try:
            result = info["runner"](
                num_warmup=args.warmup,
                num_inference=args.runs,
                device=args.device,
                backend=args.backend,
                save_results=True,
            )
            results.append(result)
            if wandb_logger:
                wandb_logger.log_benchmark(result)
            print(f"\n{Colors.GREEN}✓ {info['name']} benchmark complete{Colors.END}")

        except Exception as e:
            print(f"\n{Colors.RED}✗ {info['name']} benchmark FAILED: {e}{Colors.END}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    if results:
        print_header("Generating Summary Report")
        report = generate_summary_report(results, args.output_dir)
        print("\n" + report)
        if wandb_logger:
            wandb_logger.log_summary_report(report)

    # Final summary
    print_header("Benchmark Complete")
    print(f"Total models tested: {len(results)}/{len(models_to_run)}")

    passed = sum(1 for r in results if r.kerneldiff_passed)
    print(f"KernelDiff passed: {passed}/{len(results)}")

    if wandb_logger:
        wandb_logger.finish()

    return 0 if len(results) == len(models_to_run) else 1


if __name__ == "__main__":
    sys.exit(main())
