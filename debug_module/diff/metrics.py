"""
Comparison metrics for KernelDiff Harness.

This module provides functions to compare tensors and compute error metrics
between reference (GPU) and test (mock backend) outputs.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math


@dataclass
class TensorComparisonResult:
    """Results from comparing two tensors."""

    # Basic info
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype

    # Error metrics
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    mean_relative_error: float
    rmse: float  # Root Mean Square Error

    # Element-wise analysis
    total_elements: int
    mismatched_elements: int  # Elements exceeding tolerance
    mismatch_percentage: float

    # Location of worst errors
    max_error_indices: Optional[Tuple[int, ...]] = None
    max_error_ref_value: Optional[float] = None
    max_error_test_value: Optional[float] = None

    # Tolerance check results
    passed_atol: bool = True  # Passed absolute tolerance
    passed_rtol: bool = True  # Passed relative tolerance

    # Raw error tensor (for visualization)
    error_tensor: Optional[torch.Tensor] = None

    @property
    def passed(self) -> bool:
        """Overall pass/fail status."""
        return self.passed_atol and self.passed_rtol

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.name}: {status}\n"
            f"  Shape: {self.shape}, Dtype: {self.dtype}\n"
            f"  Max Abs Error: {self.max_absolute_error:.6e}\n"
            f"  Mean Abs Error: {self.mean_absolute_error:.6e}\n"
            f"  RMSE: {self.rmse:.6e}\n"
            f"  Mismatched: {self.mismatched_elements}/{self.total_elements} "
            f"({self.mismatch_percentage:.4f}%)"
        )


@dataclass
class ComparisonConfig:
    """Configuration for tensor comparison."""

    # Tolerance levels
    atol: float = 1e-5  # Absolute tolerance
    rtol: float = 1e-4  # Relative tolerance

    # What percentage of elements can exceed tolerance and still pass
    max_mismatch_percentage: float = 0.01  # 0.01% by default

    # Whether to store the full error tensor (for visualization)
    store_error_tensor: bool = True

    # Whether to compute relative errors (can be slow for large tensors)
    compute_relative_errors: bool = True


def compare_tensors(
    ref: torch.Tensor,
    test: torch.Tensor,
    name: str = "tensor",
    config: Optional[ComparisonConfig] = None
) -> TensorComparisonResult:
    """
    Compare two tensors and compute comprehensive error metrics.

    Args:
        ref: Reference tensor (e.g., from GPU/inductor backend)
        test: Test tensor (e.g., from mock backend)
        name: Name for this comparison (for reporting)
        config: Comparison configuration

    Returns:
        TensorComparisonResult with all metrics
    """
    if config is None:
        config = ComparisonConfig()

    # Validate shapes match
    if ref.shape != test.shape:
        raise ValueError(
            f"Shape mismatch for '{name}': ref={ref.shape}, test={test.shape}"
        )

    # Validate dtypes are compatible (allow some flexibility)
    if ref.dtype != test.dtype:
        # Convert to common dtype for comparison
        common_dtype = torch.promote_types(ref.dtype, test.dtype)
        ref = ref.to(common_dtype)
        test = test.to(common_dtype)

    # Move to CPU for analysis
    ref_cpu = ref.detach().cpu().float()
    test_cpu = test.detach().cpu().float()

    # Compute absolute errors
    abs_error = torch.abs(ref_cpu - test_cpu)

    # Basic metrics
    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()

    # RMSE
    mse = torch.mean((ref_cpu - test_cpu) ** 2).item()
    rmse = math.sqrt(mse)

    # Relative errors (handle division by zero)
    if config.compute_relative_errors:
        # Add small epsilon to avoid division by zero
        ref_abs = torch.abs(ref_cpu) + 1e-10
        rel_error = abs_error / ref_abs
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()
    else:
        max_rel_error = 0.0
        mean_rel_error = 0.0

    # Find mismatched elements (exceeding tolerance)
    # Using the same logic as torch.allclose:
    # |ref - test| <= atol + rtol * |ref|
    tolerance_threshold = config.atol + config.rtol * torch.abs(ref_cpu)
    mismatched_mask = abs_error > tolerance_threshold
    mismatched_count = mismatched_mask.sum().item()
    total_elements = ref_cpu.numel()
    mismatch_pct = (mismatched_count / total_elements) * 100 if total_elements > 0 else 0.0

    # Find location of maximum error
    max_error_flat_idx = abs_error.argmax().item()
    max_error_indices = None
    max_error_ref_val = None
    max_error_test_val = None

    if total_elements > 0:
        # Convert flat index to multi-dimensional index
        max_error_indices = tuple(
            torch.tensor(max_error_flat_idx).reshape(1).item()
            if ref_cpu.dim() == 0
            else tuple(x.item() for x in torch.unravel_index(
                torch.tensor(max_error_flat_idx), ref_cpu.shape
            ))
        )
        if isinstance(max_error_indices, int):
            max_error_indices = (max_error_indices,)

        try:
            max_error_ref_val = ref_cpu.flatten()[max_error_flat_idx].item()
            max_error_test_val = test_cpu.flatten()[max_error_flat_idx].item()
        except:
            pass

    # Tolerance checks
    passed_atol = max_abs_error <= config.atol
    passed_rtol = mismatch_pct <= config.max_mismatch_percentage

    # Store error tensor if requested
    error_tensor = abs_error if config.store_error_tensor else None

    return TensorComparisonResult(
        name=name,
        shape=tuple(ref.shape),
        dtype=ref.dtype,
        max_absolute_error=max_abs_error,
        mean_absolute_error=mean_abs_error,
        max_relative_error=max_rel_error,
        mean_relative_error=mean_rel_error,
        rmse=rmse,
        total_elements=total_elements,
        mismatched_elements=int(mismatched_count),
        mismatch_percentage=mismatch_pct,
        max_error_indices=max_error_indices,
        max_error_ref_value=max_error_ref_val,
        max_error_test_value=max_error_test_val,
        passed_atol=passed_atol,
        passed_rtol=passed_rtol,
        error_tensor=error_tensor,
    )


def tensors_are_close(
    ref: torch.Tensor,
    test: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4
) -> bool:
    """
    Quick check if two tensors are close (without full metrics).

    This is a thin wrapper around torch.allclose for convenience.
    """
    if ref.shape != test.shape:
        return False

    return torch.allclose(ref, test, atol=atol, rtol=rtol)
