"""
Mock Accelerator Adapter Implementation

This module provides a mock accelerator adapter that simulates
custom hardware constraints for testing and demonstration purposes.
It uses the existing constraint system but wraps it in the
AcceleratorAdapter interface.
"""

import os
from typing import Callable, List, Optional

import torch
import torch.fx

from .base import (
    AcceleratorAdapter,
    AcceleratorCapabilities,
    CompilationResult,
    CompilationStatus,
    ConstraintViolation,
)


class MockAcceleratorAdapter(AcceleratorAdapter):
    """
    Mock accelerator adapter for testing the debug module.

    This adapter simulates a custom accelerator with configurable
    constraints. It's useful for:
    - Testing the debug module's constraint checking
    - Demonstrating the adapter interface
    - Simulating various hardware limitations

    The constraints can be configured via environment variables:
        MOCK_STRICT: "1" for strict mode, "0" for warnings only
        MOCK_ALIGNMENT: Shape alignment requirement (default: 1)
        MOCK_MAX_MEMORY: Max memory in bytes (default: 16GB)

    Example:
        adapter = MockAcceleratorAdapter(strict=True)
        compiled = torch.compile(model, backend=adapter)
    """

    def __init__(
        self,
        strict: bool = True,
        verbose: bool = True,
        name: str = "Mock Accelerator",
        supported_dtypes: Optional[set] = None,
        unsupported_ops: Optional[set] = None,
        alignment: Optional[int] = None,
        max_memory: Optional[int] = None,
        require_contiguous: bool = True,
    ):
        """
        Initialize the mock accelerator adapter.

        Args:
            strict: If True, raise on violations. If False, warn and continue.
            verbose: If True, print detailed constraint checking info.
            name: Display name for this mock accelerator.
            supported_dtypes: Set of supported dtypes (default: float32, int64, bool).
            unsupported_ops: Set of operation names to reject.
            alignment: Shape alignment requirement (default from env or 1).
            max_memory: Max memory in bytes (default from env or 16GB).
            require_contiguous: Whether to require contiguous tensors.
        """
        # Check environment for defaults
        env_strict = os.environ.get("MOCK_STRICT", "1")
        env_alignment = os.environ.get("MOCK_ALIGNMENT", "1")
        env_max_memory = os.environ.get("MOCK_MAX_MEMORY", str(16 * 1024**3))

        strict = strict if strict is not None else (env_strict == "1")
        super().__init__(strict=strict, verbose=verbose)

        self.name = name
        self._supported_dtypes = supported_dtypes or {
            torch.float32,
            torch.int64,
            torch.bool,
        }
        self._unsupported_ops = unsupported_ops or set()
        self._alignment = alignment if alignment is not None else int(env_alignment)
        self._max_memory = max_memory if max_memory is not None else int(env_max_memory)
        self._require_contiguous = require_contiguous

    def get_capabilities(self) -> AcceleratorCapabilities:
        """Return the mock accelerator's capabilities."""
        return AcceleratorCapabilities(
            name=self.name,
            supported_dtypes=self._supported_dtypes,
            unsupported_ops=self._unsupported_ops,
            max_memory_bytes=self._max_memory,
            requires_contiguous=self._require_contiguous,
            alignment_requirement=self._alignment,
            max_tensor_dims=8,
            supports_dynamic_shapes=True,
        )

    def compile(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> CompilationResult:
        """
        Compile the graph, checking against mock constraints.

        In strict mode, returns failure on any violation.
        In non-strict mode, returns success with warnings.
        Always falls back to eager execution (since this is a mock).

        Args:
            gm: The FX GraphModule to compile
            example_inputs: Example input tensors

        Returns:
            CompilationResult with violations and fallback function
        """
        if self.verbose:
            print(f"[{self.name}] Compiling graph with {len(list(gm.graph.nodes))} nodes...")

        # Check constraints
        violations = self.check_constraints(gm, example_inputs)

        # Convert to warnings list for backward compatibility
        warnings = [str(v) for v in violations if v.severity == 'warning']

        if violations and self.strict:
            return CompilationResult(
                status=CompilationStatus.FAILED,
                compiled_fn=None,
                violations=violations,
                warnings=warnings,
                fallback_fn=self.get_fallback_fn(gm),
                metadata={
                    'node_count': len(list(gm.graph.nodes)),
                    'violation_count': len(violations),
                },
            )

        if violations and self.verbose:
            for v in violations:
                print(f"[{self.name}] WARNING: {v}")

        # Mock accelerator always falls back to eager
        # A real accelerator would do actual compilation here
        return CompilationResult(
            status=CompilationStatus.SUCCESS if not violations else CompilationStatus.PARTIAL,
            compiled_fn=gm.forward,
            violations=violations,
            warnings=warnings,
            metadata={
                'node_count': len(list(gm.graph.nodes)),
                'warning_count': len(warnings),
                'used_fallback': True,
            },
        )

    def get_fallback_fn(self, gm: torch.fx.GraphModule) -> Callable:
        """Return eager execution as fallback."""
        return gm.forward


def create_mock_backend(
    strict: bool = True,
    alignment: int = 1,
    max_memory: Optional[int] = None,
    unsupported_ops: Optional[set] = None,
) -> MockAcceleratorAdapter:
    """
    Factory function to create a mock backend adapter.

    This is a convenience function for creating mock adapters with
    common configurations.

    Args:
        strict: Whether to raise on violations
        alignment: Shape alignment requirement
        max_memory: Maximum memory in bytes
        unsupported_ops: Set of ops to reject

    Returns:
        Configured MockAcceleratorAdapter

    Example:
        backend = create_mock_backend(strict=False, alignment=16)
        compiled = torch.compile(model, backend=backend)
    """
    return MockAcceleratorAdapter(
        strict=strict,
        alignment=alignment,
        max_memory=max_memory,
        unsupported_ops=unsupported_ops,
    )
