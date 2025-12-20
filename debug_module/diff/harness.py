"""
KernelDiff Harness for comparing model outputs between backends.

This module provides the main KernelDiffHarness class that runs a model
on both a reference backend (e.g., inductor/GPU) and a test backend
(e.g., mock backend), then compares the outputs.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import time
import json
import os

from .metrics import (
    TensorComparisonResult,
    ComparisonConfig,
    compare_tensors,
)
from .visualization import (
    VisualizationConfig,
    generate_error_heatmap,
    generate_comparison_summary_plot,
    HAS_MATPLOTLIB,
)


@dataclass
class DiffReport:
    """Complete report from a KernelDiff comparison."""

    # Metadata
    model_name: str
    timestamp: float
    reference_backend: str
    test_backend: str

    # Timing
    reference_compile_time: float = 0.0
    reference_run_time: float = 0.0
    test_compile_time: float = 0.0
    test_run_time: float = 0.0

    # Results
    tensor_results: List[TensorComparisonResult] = field(default_factory=list)
    overall_passed: bool = True
    error_message: Optional[str] = None

    # Visualization paths
    visualization_paths: List[str] = field(default_factory=list)

    @property
    def total_tensors(self) -> int:
        return len(self.tensor_results)

    @property
    def passed_tensors(self) -> int:
        return sum(1 for r in self.tensor_results if r.passed)

    @property
    def failed_tensors(self) -> int:
        return sum(1 for r in self.tensor_results if not r.passed)

    def summary(self) -> str:
        """Return a human-readable summary of the diff report."""
        status = "PASSED" if self.overall_passed else "FAILED"
        lines = [
            "=" * 60,
            f"KernelDiff Report: {self.model_name}",
            "=" * 60,
            f"Status: {status}",
            f"Reference Backend: {self.reference_backend}",
            f"Test Backend: {self.test_backend}",
            "",
            "Timing:",
            f"  Reference: {self.reference_compile_time:.3f}s compile, "
            f"{self.reference_run_time:.3f}s run",
            f"  Test: {self.test_compile_time:.3f}s compile, "
            f"{self.test_run_time:.3f}s run",
            "",
            f"Tensors Compared: {self.total_tensors}",
            f"  Passed: {self.passed_tensors}",
            f"  Failed: {self.failed_tensors}",
            "",
        ]

        if self.error_message:
            lines.append(f"Error: {self.error_message}")
            lines.append("")

        # Show details for failed tensors
        failed = [r for r in self.tensor_results if not r.passed]
        if failed:
            lines.append("Failed Tensor Details:")
            lines.append("-" * 40)
            for result in failed[:5]:  # Limit to first 5
                lines.append(result.summary())
                lines.append("")
            if len(failed) > 5:
                lines.append(f"... and {len(failed) - 5} more failed tensors")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "reference_backend": self.reference_backend,
            "test_backend": self.test_backend,
            "timing": {
                "reference_compile_time": self.reference_compile_time,
                "reference_run_time": self.reference_run_time,
                "test_compile_time": self.test_compile_time,
                "test_run_time": self.test_run_time,
            },
            "overall_passed": self.overall_passed,
            "total_tensors": self.total_tensors,
            "passed_tensors": self.passed_tensors,
            "failed_tensors": self.failed_tensors,
            "error_message": self.error_message,
            "tensor_results": [
                {
                    "name": r.name,
                    "shape": list(r.shape),
                    "dtype": str(r.dtype),
                    "passed": r.passed,
                    "max_absolute_error": r.max_absolute_error,
                    "mean_absolute_error": r.mean_absolute_error,
                    "rmse": r.rmse,
                    "mismatch_percentage": r.mismatch_percentage,
                }
                for r in self.tensor_results
            ],
            "visualization_paths": self.visualization_paths,
        }

    def save_json(self, filepath: str):
        """Save report to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[KernelDiff] Saved report to {filepath}")


def _flatten_outputs(
    output: Any,
    prefix: str = "output"
) -> List[Tuple[str, torch.Tensor]]:
    """
    Flatten nested output structures into a list of (name, tensor) pairs.

    Handles:
    - Single tensors
    - Tuples/lists of tensors
    - Dicts of tensors
    - Nested combinations
    - HuggingFace model outputs (have .keys() or are dataclass-like)
    """
    results = []

    if isinstance(output, torch.Tensor):
        results.append((prefix, output))

    elif isinstance(output, (tuple, list)):
        for i, item in enumerate(output):
            results.extend(_flatten_outputs(item, f"{prefix}[{i}]"))

    elif isinstance(output, dict):
        for key, value in output.items():
            results.extend(_flatten_outputs(value, f"{prefix}.{key}"))

    elif hasattr(output, '__dict__'):
        # Handle dataclass-like objects (e.g., HuggingFace ModelOutput)
        for key, value in output.__dict__.items():
            if not key.startswith('_'):
                results.extend(_flatten_outputs(value, f"{prefix}.{key}"))

    elif hasattr(output, 'keys'):
        # Handle dict-like objects
        for key in output.keys():
            results.extend(_flatten_outputs(output[key], f"{prefix}.{key}"))

    # Skip None values and non-tensor types
    return results


def _clone_inputs(inputs: Any) -> Any:
    """
    Deep clone inputs to ensure no aliasing between runs.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.clone().detach()
    elif isinstance(inputs, dict):
        return {k: _clone_inputs(v) for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        cloned = [_clone_inputs(item) for item in inputs]
        return type(inputs)(cloned)
    else:
        return inputs


class KernelDiffHarness:
    """
    Harness for comparing model outputs between reference and test backends.

    Usage:
        harness = KernelDiffHarness(model, example_inputs)
        report = harness.compare()
        print(report.summary())
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable],
        example_inputs: Union[torch.Tensor, Tuple, Dict],
        model_name: Optional[str] = None,
        reference_backend: str = "inductor",
        test_backend: Optional[Callable] = None,
        comparison_config: Optional[ComparisonConfig] = None,
        visualization_config: Optional[VisualizationConfig] = None,
    ):
        """
        Initialize the KernelDiff harness.

        Args:
            model: The model or function to test
            example_inputs: Example inputs for the model (tensor, tuple, or dict)
            model_name: Optional name for the model (for reporting)
            reference_backend: Backend to use as reference ("inductor", "eager", etc.)
            test_backend: Backend function to test (e.g., mock_backend)
            comparison_config: Configuration for tensor comparison
            visualization_config: Configuration for visualizations
        """
        self.model = model
        self.example_inputs = example_inputs
        self.model_name = model_name or self._infer_model_name(model)
        self.reference_backend = reference_backend
        self.test_backend = test_backend
        self.test_backend_name = (
            getattr(test_backend, "__name__", "mock_backend")
            if test_backend is not None
            else "mock_backend"
        )
        self.comparison_config = comparison_config or ComparisonConfig()
        self.visualization_config = visualization_config or VisualizationConfig()

        # Compiled models (lazily initialized)
        self._reference_compiled = None
        self._test_compiled = None

        # Timing storage
        self._ref_compile_time = 0.0
        self._test_compile_time = 0.0

    def _infer_model_name(self, model: Any) -> str:
        """Infer a reasonable name for the model."""
        if hasattr(model, '__name__'):
            return model.__name__
        elif hasattr(model, '__class__'):
            return model.__class__.__name__
        else:
            return "unknown_model"

    def _prepare_inputs(self) -> Tuple[Any, Any]:
        """Prepare two copies of inputs (one for each backend)."""
        return _clone_inputs(self.example_inputs), _clone_inputs(self.example_inputs)

    def _compile_reference(self) -> Callable:
        """Compile the model with the reference backend."""
        if self._reference_compiled is not None:
            return self._reference_compiled

        print(f"[KernelDiff] Compiling with reference backend: {self.reference_backend}")
        start = time.time()

        if self.reference_backend == "eager":
            # No compilation, just use the model directly
            self._reference_compiled = self.model
        else:
            self._reference_compiled = torch.compile(
                self.model,
                backend=self.reference_backend
            )

        self._ref_compile_time = time.time() - start
        print(f"[KernelDiff] Reference compilation took {self._ref_compile_time:.3f}s")
        return self._reference_compiled

    def _compile_test(self) -> Callable:
        """Compile the model with the test backend."""
        if self._test_compiled is not None:
            return self._test_compiled

        if self.test_backend is None:
            # Import mock_backend as default
            from ..backend.mock import mock_backend
            self.test_backend = mock_backend
            self.test_backend_name = "mock_backend"

        print(f"[KernelDiff] Compiling with test backend: {self.test_backend_name}")
        start = time.time()

        self._test_compiled = torch.compile(
            self.model,
            backend=self.test_backend
        )

        self._test_compile_time = time.time() - start
        print(f"[KernelDiff] Test compilation took {self._test_compile_time:.3f}s")
        return self._test_compiled

    def run_reference(self) -> Tuple[Any, float]:
        """
        Run the model with the reference backend.

        Returns:
            Tuple of (output, run_time_seconds)
        """
        compiled = self._compile_reference()
        ref_inputs, _ = self._prepare_inputs()

        print(f"[KernelDiff] Running reference model...")
        start = time.time()

        with torch.no_grad():
            if isinstance(ref_inputs, dict):
                output = compiled(**ref_inputs)
            elif isinstance(ref_inputs, (tuple, list)):
                output = compiled(*ref_inputs)
            else:
                output = compiled(ref_inputs)

        run_time = time.time() - start
        print(f"[KernelDiff] Reference run took {run_time:.3f}s")
        return output, run_time

    def run_test(self) -> Tuple[Any, float]:
        """
        Run the model with the test backend.

        Returns:
            Tuple of (output, run_time_seconds)
        """
        compiled = self._compile_test()
        _, test_inputs = self._prepare_inputs()

        print(f"[KernelDiff] Running test model...")
        start = time.time()

        with torch.no_grad():
            if isinstance(test_inputs, dict):
                output = compiled(**test_inputs)
            elif isinstance(test_inputs, (tuple, list)):
                output = compiled(*test_inputs)
            else:
                output = compiled(test_inputs)

        run_time = time.time() - start
        print(f"[KernelDiff] Test run took {run_time:.3f}s")
        return output, run_time

    def compare(
        self,
        generate_visualizations: bool = True,
        save_report: bool = True,
        report_dir: str = "debug_artifacts/reports"
    ) -> DiffReport:
        """
        Run both backends and compare their outputs.

        Args:
            generate_visualizations: Whether to generate heatmap visualizations
            save_report: Whether to save the JSON report
            report_dir: Directory to save reports

        Returns:
            DiffReport with all comparison results
        """
        report = DiffReport(
            model_name=self.model_name,
            timestamp=time.time(),
            reference_backend=self.reference_backend,
            test_backend=self.test_backend_name,
        )

        try:
            # Run both backends
            ref_output, ref_run_time = self.run_reference()
            test_output, test_run_time = self.run_test()

            report.reference_compile_time = self._ref_compile_time
            report.reference_run_time = ref_run_time
            report.test_compile_time = self._test_compile_time
            report.test_run_time = test_run_time

            # Flatten outputs for comparison
            ref_tensors = _flatten_outputs(ref_output, "output")
            test_tensors = _flatten_outputs(test_output, "output")

            print(f"[KernelDiff] Comparing {len(ref_tensors)} output tensors...")

            # Create a mapping for test tensors
            test_tensor_map = {name: tensor for name, tensor in test_tensors}

            # Compare each tensor
            for name, ref_tensor in ref_tensors:
                if name not in test_tensor_map:
                    # Missing tensor in test output
                    result = TensorComparisonResult(
                        name=name,
                        shape=tuple(ref_tensor.shape),
                        dtype=ref_tensor.dtype,
                        max_absolute_error=float('inf'),
                        mean_absolute_error=float('inf'),
                        max_relative_error=float('inf'),
                        mean_relative_error=float('inf'),
                        rmse=float('inf'),
                        total_elements=ref_tensor.numel(),
                        mismatched_elements=ref_tensor.numel(),
                        mismatch_percentage=100.0,
                        passed_atol=False,
                        passed_rtol=False,
                    )
                    report.tensor_results.append(result)
                    report.overall_passed = False
                    continue

                test_tensor = test_tensor_map[name]

                try:
                    result = compare_tensors(
                        ref_tensor,
                        test_tensor,
                        name=name,
                        config=self.comparison_config
                    )
                    report.tensor_results.append(result)

                    if not result.passed:
                        report.overall_passed = False

                except Exception as e:
                    # Comparison failed (e.g., shape mismatch)
                    result = TensorComparisonResult(
                        name=name,
                        shape=tuple(ref_tensor.shape),
                        dtype=ref_tensor.dtype,
                        max_absolute_error=float('inf'),
                        mean_absolute_error=float('inf'),
                        max_relative_error=float('inf'),
                        mean_relative_error=float('inf'),
                        rmse=float('inf'),
                        total_elements=ref_tensor.numel(),
                        mismatched_elements=ref_tensor.numel(),
                        mismatch_percentage=100.0,
                        passed_atol=False,
                        passed_rtol=False,
                    )
                    report.tensor_results.append(result)
                    report.overall_passed = False
                    print(f"[KernelDiff] Warning: Comparison failed for {name}: {e}")

        except Exception as e:
            report.overall_passed = False
            report.error_message = str(e)
            print(f"[KernelDiff] Error during comparison: {e}")
            import traceback
            traceback.print_exc()

        # Generate visualizations
        if generate_visualizations and HAS_MATPLOTLIB:
            report.visualization_paths = self._generate_visualizations(report)

        # Save report
        if save_report:
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(
                report_dir,
                f"diff_report_{self.model_name}_{int(report.timestamp)}.json"
            )
            report.save_json(report_path)

        return report

    def _generate_visualizations(self, report: DiffReport) -> List[str]:
        """Generate visualization files for the report."""
        paths = []

        # Generate heatmaps for failed tensors (or all if few enough)
        tensors_to_visualize = [
            r for r in report.tensor_results
            if not r.passed or len(report.tensor_results) <= 5
        ]

        for result in tensors_to_visualize[:10]:  # Limit to 10 visualizations
            if result.error_tensor is not None:
                try:
                    path = generate_error_heatmap(
                        result,
                        config=self.visualization_config,
                        save=True,
                        show=False
                    )
                    if path:
                        paths.append(path)
                except Exception as e:
                    print(f"[KernelDiff] Warning: Failed to generate heatmap for {result.name}: {e}")

        # Generate summary plot
        if len(report.tensor_results) > 1:
            try:
                summary_path = generate_comparison_summary_plot(
                    report.tensor_results,
                    config=self.visualization_config,
                    save=True,
                    show=False
                )
                if summary_path:
                    paths.append(summary_path)
            except Exception as e:
                print(f"[KernelDiff] Warning: Failed to generate summary plot: {e}")

        return paths

    def quick_check(self, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
        """
        Quick pass/fail check without full report generation.

        Returns:
            True if all outputs match within tolerance, False otherwise
        """
        try:
            ref_output, _ = self.run_reference()
            test_output, _ = self.run_test()

            ref_tensors = _flatten_outputs(ref_output)
            test_tensors = _flatten_outputs(test_output)

            if len(ref_tensors) != len(test_tensors):
                return False

            for (name, ref), (_, test) in zip(ref_tensors, test_tensors):
                if not torch.allclose(ref, test, atol=atol, rtol=rtol):
                    return False

            return True

        except Exception:
            return False
