"""
Base classes and utilities for benchmarking.

Provides:
- BenchmarkResult: Dataclass for storing benchmark results
- BaseBenchmark: Abstract base class for model benchmarks
"""

import torch
import torch.nn as nn
import time
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager
import traceback

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BackendResult:
    """Results from running a single backend."""

    name: str
    compile_time: float = 0.0
    inference_times: List[float] = field(default_factory=list)
    avg_inference_time: float = 0.0
    std_inference_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    # Mock backend specific
    constraint_warnings: List[str] = field(default_factory=list)

    def compute_stats(self):
        """Compute average and std deviation of inference times."""
        if self.inference_times:
            self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            if len(self.inference_times) > 1:
                variance = sum((t - self.avg_inference_time) ** 2 for t in self.inference_times) / len(self.inference_times)
                self.std_inference_time = variance ** 0.5


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a model."""

    # Model info
    model_name: str
    model_id: str
    model_type: str  # "transformer", "cnn", "ssm"
    parameter_count: int

    # Timing
    timestamp: float = field(default_factory=time.time)

    # Results per backend
    eager_result: Optional[BackendResult] = None
    inductor_result: Optional[BackendResult] = None
    mock_result: Optional[BackendResult] = None

    # Guard/Graph analysis
    graph_count: int = 0
    graph_break_count: int = 0
    break_reasons: List[str] = field(default_factory=list)

    # KernelDiff results (mock vs eager)
    kerneldiff_passed: bool = True
    max_absolute_error: float = 0.0
    mean_absolute_error: float = 0.0

    # Device info
    device: str = "cpu"

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 70,
            f"Benchmark Results: {self.model_name}",
            "=" * 70,
            f"Model ID: {self.model_id}",
            f"Type: {self.model_type}",
            f"Parameters: {self.parameter_count:,}",
            f"Device: {self.device}",
            "",
            "Compilation Times:",
        ]

        if self.eager_result:
            lines.append(f"  Eager:    {self.eager_result.compile_time:.3f}s (no compilation)")
        if self.inductor_result:
            status = "✓" if self.inductor_result.success else "✗"
            lines.append(f"  Inductor: {self.inductor_result.compile_time:.3f}s {status}")
        if self.mock_result:
            status = "✓" if self.mock_result.success else "✗"
            lines.append(f"  Mock:     {self.mock_result.compile_time:.3f}s {status}")

        lines.append("")
        lines.append("Inference Times (avg ± std):")

        if self.eager_result and self.eager_result.success:
            lines.append(f"  Eager:    {self.eager_result.avg_inference_time*1000:.2f} ± {self.eager_result.std_inference_time*1000:.2f} ms")
        if self.inductor_result and self.inductor_result.success:
            lines.append(f"  Inductor: {self.inductor_result.avg_inference_time*1000:.2f} ± {self.inductor_result.std_inference_time*1000:.2f} ms")
        if self.mock_result and self.mock_result.success:
            lines.append(f"  Mock:     {self.mock_result.avg_inference_time*1000:.2f} ± {self.mock_result.std_inference_time*1000:.2f} ms")

        lines.append("")
        lines.append("Graph Analysis:")
        lines.append(f"  Graph Count: {self.graph_count}")
        lines.append(f"  Graph Breaks: {self.graph_break_count}")

        if self.break_reasons:
            lines.append("  Break Reasons:")
            for reason in self.break_reasons[:3]:
                lines.append(f"    - {reason[:60]}...")

        if self.mock_result and self.mock_result.constraint_warnings:
            lines.append("")
            lines.append(f"Constraint Warnings: {len(self.mock_result.constraint_warnings)}")
            for warning in self.mock_result.constraint_warnings[:5]:
                lines.append(f"  - {warning[:60]}...")

        lines.append("")
        lines.append("KernelDiff (Mock vs Eager):")
        status = "PASSED ✓" if self.kerneldiff_passed else "FAILED ✗"
        lines.append(f"  Status: {status}")
        lines.append(f"  Max Error: {self.max_absolute_error:.2e}")
        lines.append(f"  Mean Error: {self.mean_absolute_error:.2e}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "parameter_count": self.parameter_count,
            "timestamp": self.timestamp,
            "device": self.device,
            "graph_count": self.graph_count,
            "graph_break_count": self.graph_break_count,
            "break_reasons": self.break_reasons,
            "kerneldiff_passed": self.kerneldiff_passed,
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
        }

        for backend_name in ["eager", "inductor", "mock"]:
            backend_result = getattr(self, f"{backend_name}_result")
            if backend_result:
                result[f"{backend_name}_backend"] = {
                    "compile_time": backend_result.compile_time,
                    "avg_inference_time": backend_result.avg_inference_time,
                    "std_inference_time": backend_result.std_inference_time,
                    "success": backend_result.success,
                    "error_message": backend_result.error_message,
                    "constraint_warnings": backend_result.constraint_warnings if backend_name == "mock" else [],
                }

        return result

    def save(self, output_dir: str = "benchmarks/results"):
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.model_name.lower().replace(' ', '_')}_{int(self.timestamp)}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"[Benchmark] Results saved to {filepath}")
        return filepath


class WarningCapture:
    """Capture mock backend warnings during compilation."""

    def __init__(self):
        self.warnings: List[str] = []
        self._original_print = None

    def __enter__(self):
        import builtins
        self._original_print = builtins.print

        def capturing_print(*args, **kwargs):
            message = " ".join(str(a) for a in args)
            if "[MockBackend] WARNING:" in message:
                self.warnings.append(message.replace("[MockBackend] WARNING: ", ""))
            self._original_print(*args, **kwargs)

        builtins.print = capturing_print
        return self

    def __exit__(self, *args):
        import builtins
        builtins.print = self._original_print


class BaseBenchmark(ABC):
    """Abstract base class for model benchmarks."""

    def __init__(
        self,
        num_warmup: int = 2,
        num_inference: int = 5,
        device: str = "cpu",
    ):
        self.num_warmup = num_warmup
        self.num_inference = num_inference
        self.device = device

        # Will be set by subclasses
        self.model = None
        self.example_inputs = None
        self.model_name = "Unknown"
        self.model_id = "unknown"
        self.model_type = "unknown"

    @abstractmethod
    def setup(self):
        """Load model and prepare example inputs. Must set self.model and self.example_inputs."""
        pass

    def get_parameter_count(self) -> int:
        """Count model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def _run_inference(self, model_fn: Callable, inputs: Any) -> Any:
        """Run a single inference."""
        with torch.no_grad():
            if isinstance(inputs, dict):
                return model_fn(**inputs)
            elif isinstance(inputs, (tuple, list)):
                return model_fn(*inputs)
            else:
                return model_fn(inputs)

    def _benchmark_backend(
        self,
        name: str,
        compile_fn: Optional[Callable] = None,
    ) -> BackendResult:
        """Benchmark a single backend."""
        result = BackendResult(name=name)

        try:
            # Compilation
            print(f"  [{name}] Compiling...")
            start = time.time()

            if compile_fn is None:
                # Eager mode - no compilation
                compiled_model = self.model
            else:
                compiled_model = compile_fn(self.model)

            result.compile_time = time.time() - start
            print(f"  [{name}] Compile time: {result.compile_time:.3f}s")

            # Warmup
            print(f"  [{name}] Warming up ({self.num_warmup} runs)...")
            for _ in range(self.num_warmup):
                _ = self._run_inference(compiled_model, self.example_inputs)

            # Timed inference
            print(f"  [{name}] Running inference ({self.num_inference} runs)...")
            for i in range(self.num_inference):
                start = time.time()
                _ = self._run_inference(compiled_model, self.example_inputs)
                elapsed = time.time() - start
                result.inference_times.append(elapsed)

            result.compute_stats()
            print(f"  [{name}] Avg inference: {result.avg_inference_time*1000:.2f}ms")
            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            print(f"  [{name}] FAILED: {e}")
            traceback.print_exc()

        return result

    def run(self) -> BenchmarkResult:
        """Run the complete benchmark."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {self.model_name}")
        print(f"{'='*60}\n")

        # Setup
        print("[Setup] Loading model...")
        self.setup()
        print(f"[Setup] Model loaded: {self.get_parameter_count():,} parameters")
        print(f"[Setup] Device: {self.device}")

        # Create result object
        result = BenchmarkResult(
            model_name=self.model_name,
            model_id=self.model_id,
            model_type=self.model_type,
            parameter_count=self.get_parameter_count(),
            device=self.device,
        )

        # Import here to avoid circular imports
        from debug_module import mock_backend

        # 1. Eager baseline
        print("\n[Benchmark] Eager (baseline)...")
        result.eager_result = self._benchmark_backend("Eager", compile_fn=None)

        # Store eager output for later comparison
        eager_output = None
        if result.eager_result.success:
            with torch.no_grad():
                eager_output = self._run_inference(self.model, self.example_inputs)

        # 2. Inductor backend
        print("\n[Benchmark] Inductor...")
        result.inductor_result = self._benchmark_backend(
            "Inductor",
            compile_fn=lambda m: torch.compile(m, backend="inductor")
        )

        # 3. Mock backend (with warning capture)
        print("\n[Benchmark] Mock Backend...")
        with WarningCapture() as capture:
            result.mock_result = self._benchmark_backend(
                "Mock",
                compile_fn=lambda m: torch.compile(m, backend=mock_backend)
            )
            result.mock_result.constraint_warnings = capture.warnings

        # 4. Guard/Graph analysis
        print("\n[Analysis] Running guard inspection...")
        try:
            from debug_module.guards.inspector import GuardInspector
            inspector = GuardInspector(self.model)

            # Prepare inputs for inspector (needs dict)
            if isinstance(self.example_inputs, dict):
                inspect_inputs = self.example_inputs
            else:
                # Wrap in a dict with generic name
                inspect_inputs = {"x": self.example_inputs}

            report = inspector.inspect(inspect_inputs)
            result.graph_count = report.get("graph_count", 0)
            result.graph_break_count = report.get("graph_break_count", 0)
            result.break_reasons = report.get("break_reasons", [])
            print(f"  Graphs: {result.graph_count}, Breaks: {result.graph_break_count}")
        except Exception as e:
            print(f"  Guard inspection failed: {e}")

        # 5. KernelDiff comparison (mock vs eager)
        print("\n[Analysis] Running KernelDiff comparison...")
        if result.eager_result.success and result.mock_result.success:
            try:
                from debug_module.diff import KernelDiffHarness, ComparisonConfig

                config = ComparisonConfig(
                    atol=1e-4,
                    rtol=1e-3,
                    max_mismatch_percentage=1.0,
                    store_error_tensor=False,  # Save memory
                )

                harness = KernelDiffHarness(
                    model=self.model,
                    example_inputs=self.example_inputs,
                    model_name=self.model_name,
                    reference_backend="eager",
                    comparison_config=config,
                )

                diff_report = harness.compare(
                    generate_visualizations=False,
                    save_report=False,
                )

                result.kerneldiff_passed = diff_report.overall_passed
                if diff_report.tensor_results:
                    result.max_absolute_error = max(r.max_absolute_error for r in diff_report.tensor_results)
                    result.mean_absolute_error = sum(r.mean_absolute_error for r in diff_report.tensor_results) / len(diff_report.tensor_results)

                status = "PASSED" if result.kerneldiff_passed else "FAILED"
                print(f"  KernelDiff: {status} (max error: {result.max_absolute_error:.2e})")

            except Exception as e:
                print(f"  KernelDiff failed: {e}")
                traceback.print_exc()
        else:
            print("  Skipping KernelDiff (backend failed)")

        print("\n" + result.summary())
        return result
