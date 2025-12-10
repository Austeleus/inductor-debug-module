"""
Visualization utilities for KernelDiff Harness.

This module provides functions to generate visual representations of
tensor comparison results, including error heatmaps.
"""

import torch
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .metrics import TensorComparisonResult


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""

    # Output settings
    output_dir: str = "debug_artifacts/visualizations"
    file_format: str = "png"  # png, svg, pdf
    dpi: int = 150

    # Heatmap settings
    colormap: str = "hot"  # hot, viridis, plasma, etc.
    figsize: Tuple[int, int] = (10, 8)

    # For 3D+ tensors, which slice to visualize
    slice_dims: Optional[Tuple[int, ...]] = None  # None = auto-select


def ensure_matplotlib():
    """Raise an error if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def generate_error_heatmap(
    result: TensorComparisonResult,
    config: Optional[VisualizationConfig] = None,
    save: bool = True,
    show: bool = False
) -> Optional[str]:
    """
    Generate a heatmap visualization of the error tensor.

    Args:
        result: TensorComparisonResult containing the error tensor
        config: Visualization configuration
        save: Whether to save the figure to disk
        show: Whether to display the figure (blocks execution)

    Returns:
        Path to saved figure, or None if not saved
    """
    ensure_matplotlib()

    if result.error_tensor is None:
        print(f"[Visualization] No error tensor available for '{result.name}'")
        return None

    if config is None:
        config = VisualizationConfig()

    error_tensor = result.error_tensor

    # Handle different tensor dimensions
    if error_tensor.dim() == 0:
        # Scalar - nothing to visualize
        print(f"[Visualization] Scalar tensor, skipping heatmap for '{result.name}'")
        return None

    elif error_tensor.dim() == 1:
        # 1D tensor - show as bar chart
        return _generate_1d_plot(result, config, save, show)

    elif error_tensor.dim() == 2:
        # 2D tensor - direct heatmap
        return _generate_2d_heatmap(result, config, save, show)

    else:
        # 3D+ tensor - take a 2D slice
        return _generate_nd_heatmap(result, config, save, show)


def _generate_1d_plot(
    result: TensorComparisonResult,
    config: VisualizationConfig,
    save: bool,
    show: bool
) -> Optional[str]:
    """Generate a 1D error plot."""
    error_data = result.error_tensor.numpy()

    fig, ax = plt.subplots(figsize=config.figsize)

    ax.bar(range(len(error_data)), error_data, color='red', alpha=0.7)
    ax.set_xlabel('Index')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f"Error Distribution: {result.name}\n"
                 f"Max: {result.max_absolute_error:.2e}, "
                 f"Mean: {result.mean_absolute_error:.2e}")

    # Add a horizontal line for the tolerance threshold
    ax.axhline(y=result.max_absolute_error * 0.1, color='orange',
               linestyle='--', label='10% of max error')
    ax.legend()

    plt.tight_layout()

    filepath = None
    if save:
        filepath = _save_figure(fig, result.name, config)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return filepath


def _generate_2d_heatmap(
    result: TensorComparisonResult,
    config: VisualizationConfig,
    save: bool,
    show: bool
) -> Optional[str]:
    """Generate a 2D error heatmap."""
    error_data = result.error_tensor.numpy()

    fig, ax = plt.subplots(figsize=config.figsize)

    # Use log scale if there's a large dynamic range
    vmin = error_data.min()
    vmax = error_data.max()

    if vmax > 0 and vmax / (vmin + 1e-10) > 1000:
        # Use log scale for large dynamic range
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
    else:
        norm = None

    im = ax.imshow(error_data, cmap=config.colormap, aspect='auto', norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Absolute Error')

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f"Error Heatmap: {result.name}\n"
                 f"Shape: {result.shape}, Max: {result.max_absolute_error:.2e}")

    # Mark the location of maximum error
    if result.max_error_indices is not None and len(result.max_error_indices) == 2:
        row, col = result.max_error_indices
        ax.plot(col, row, 'wo', markersize=10, markeredgecolor='black',
                markeredgewidth=2, label='Max error')
        ax.legend()

    plt.tight_layout()

    filepath = None
    if save:
        filepath = _save_figure(fig, result.name, config)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return filepath


def _generate_nd_heatmap(
    result: TensorComparisonResult,
    config: VisualizationConfig,
    save: bool,
    show: bool
) -> Optional[str]:
    """Generate a heatmap for N-dimensional tensor (takes 2D slice)."""
    error_tensor = result.error_tensor

    # For N-D tensors, we'll take slices and create a grid of heatmaps
    # or just visualize the first 2D slice

    # Flatten extra dimensions by taking first element
    while error_tensor.dim() > 2:
        error_tensor = error_tensor[0]

    # Now we have a 2D tensor
    error_data = error_tensor.numpy()

    fig, ax = plt.subplots(figsize=config.figsize)

    im = ax.imshow(error_data, cmap=config.colormap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Absolute Error')

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f"Error Heatmap: {result.name} (first 2D slice)\n"
                 f"Original shape: {result.shape}, Max: {result.max_absolute_error:.2e}")

    plt.tight_layout()

    filepath = None
    if save:
        filepath = _save_figure(fig, result.name, config)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return filepath


def _save_figure(fig, name: str, config: VisualizationConfig) -> str:
    """Save figure to disk."""
    os.makedirs(config.output_dir, exist_ok=True)

    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filepath = os.path.join(config.output_dir, f"error_{safe_name}.{config.file_format}")

    fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
    print(f"[Visualization] Saved heatmap to {filepath}")

    return filepath


def generate_comparison_summary_plot(
    results: List[TensorComparisonResult],
    config: Optional[VisualizationConfig] = None,
    save: bool = True,
    show: bool = False
) -> Optional[str]:
    """
    Generate a summary bar chart comparing errors across multiple tensors.

    Args:
        results: List of TensorComparisonResult objects
        config: Visualization configuration
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        Path to saved figure, or None if not saved
    """
    ensure_matplotlib()

    if not results:
        return None

    if config is None:
        config = VisualizationConfig()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r.name[:20] for r in results]  # Truncate long names
    max_errors = [r.max_absolute_error for r in results]
    mean_errors = [r.mean_absolute_error for r in results]
    rmse_values = [r.rmse for r in results]
    mismatch_pcts = [r.mismatch_percentage for r in results]

    # Colors based on pass/fail
    colors = ['green' if r.passed else 'red' for r in results]

    # Max Absolute Error
    ax1 = axes[0, 0]
    ax1.barh(names, max_errors, color=colors, alpha=0.7)
    ax1.set_xlabel('Max Absolute Error')
    ax1.set_title('Maximum Absolute Error')
    if any(e > 0 for e in max_errors):
        ax1.set_xscale('log')

    # Mean Absolute Error
    ax2 = axes[0, 1]
    ax2.barh(names, mean_errors, color=colors, alpha=0.7)
    ax2.set_xlabel('Mean Absolute Error')
    ax2.set_title('Mean Absolute Error')
    if any(e > 0 for e in mean_errors):
        ax2.set_xscale('log')

    # RMSE
    ax3 = axes[1, 0]
    ax3.barh(names, rmse_values, color=colors, alpha=0.7)
    ax3.set_xlabel('RMSE')
    ax3.set_title('Root Mean Square Error')
    if any(e > 0 for e in rmse_values):
        ax3.set_xscale('log')

    # Mismatch Percentage
    ax4 = axes[1, 1]
    ax4.barh(names, mismatch_pcts, color=colors, alpha=0.7)
    ax4.set_xlabel('Mismatch %')
    ax4.set_title('Percentage of Mismatched Elements')

    plt.suptitle('KernelDiff Comparison Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = None
    if save:
        os.makedirs(config.output_dir, exist_ok=True)
        filepath = os.path.join(config.output_dir, f"comparison_summary.{config.file_format}")
        fig.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        print(f"[Visualization] Saved summary to {filepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return filepath
