"""
KernelDiff module for comparing model outputs between backends.

This module provides tools for:
- Running models on reference and test backends
- Comparing tensor outputs with detailed metrics
- Generating visualizations (heatmaps, summary plots)
- Creating comprehensive diff reports

Example usage:
    from debug_module.diff import KernelDiffHarness

    harness = KernelDiffHarness(model, example_inputs)
    report = harness.compare()
    print(report.summary())
"""

from .harness import KernelDiffHarness, DiffReport
from .metrics import (
    TensorComparisonResult,
    ComparisonConfig,
    compare_tensors,
    tensors_are_close,
)
from .visualization import (
    VisualizationConfig,
    generate_error_heatmap,
    generate_comparison_summary_plot,
)

__all__ = [
    # Main harness
    "KernelDiffHarness",
    "DiffReport",
    # Metrics
    "TensorComparisonResult",
    "ComparisonConfig",
    "compare_tensors",
    "tensors_are_close",
    # Visualization
    "VisualizationConfig",
    "generate_error_heatmap",
    "generate_comparison_summary_plot",
]
