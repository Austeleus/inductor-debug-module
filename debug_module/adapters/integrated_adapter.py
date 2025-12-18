"""
Integrated Adapter - Wraps the existing mock_backend with the Adapter Interface

This shows how the existing mock_backend functionality can be exposed
through the AcceleratorAdapter interface, providing the best of both:
- All existing functionality (artifacts, repros, constraint registry)
- Clean abstract interface for extensibility
"""

import os
from typing import Callable, List, Optional, Set

import torch
import torch.fx

from .base import (
    AcceleratorAdapter,
    AcceleratorCapabilities,
    CompilationResult,
    CompilationStatus,
    ConstraintViolation,
)

# Import the existing constraint system
from ..constraints.registry import (
    DEFAULT_CONSTRAINTS,
    DtypeConstraint,
    LayoutConstraint,
    ShapeConstraint,
    MemoryConstraint,
    UnsupportedOpsConstraint,
    deny_ops,
)
from ..backend.compiler import save_artifact, _maybe_generate_repro


class IntegratedMockAdapter(AcceleratorAdapter):
    """
    Adapter that wraps the existing mock_backend functionality.

    This provides the AcceleratorAdapter interface while using all the
    existing constraint checking, artifact capture, and repro generation
    from the original mock_backend.

    Usage:
        adapter = IntegratedMockAdapter(strict=True, alignment=16)
        compiled = torch.compile(model, backend=adapter)
    """

    def __init__(
        self,
        strict: bool = None,
        verbose: bool = True,
        alignment: int = None,
        max_memory: int = None,
    ):
        # Load from environment if not specified
        if strict is None:
            strict = os.environ.get("MOCK_STRICT", "1") == "1"
        if alignment is None:
            alignment = int(os.environ.get("MOCK_ALIGNMENT", "1"))
        if max_memory is None:
            max_memory = int(os.environ.get("MOCK_MAX_MEMORY", str(16 * 1024**3)))

        super().__init__(strict=strict, verbose=verbose)
        self._alignment = alignment
        self._max_memory = max_memory

        # Build constraints using the existing registry classes
        self._constraints = [
            UnsupportedOpsConstraint(deny_ops),
            DtypeConstraint(),
            LayoutConstraint(),
            ShapeConstraint(alignment=alignment),
            MemoryConstraint(max_memory_bytes=max_memory),
        ]

    def get_capabilities(self) -> AcceleratorCapabilities:
        """Return capabilities based on the constraint configuration."""
        return AcceleratorCapabilities(
            name="Mock Accelerator (Integrated)",
            supported_dtypes={torch.float32, torch.int64, torch.bool},
            unsupported_ops=set(str(op) for op in deny_ops),
            max_memory_bytes=self._max_memory,
            requires_contiguous=True,
            alignment_requirement=self._alignment,
        )

    def compile(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> CompilationResult:
        """
        Compile using the existing constraint system.

        This method:
        1. Saves artifacts (like original mock_backend)
        2. Checks constraints using the registry classes
        3. Generates repro scripts on failure (like original)
        4. Returns structured CompilationResult
        """
        if self.verbose:
            print(f"[MockBackend] Compiling graph with {len(list(gm.graph.nodes))} nodes...")

        # Save artifact (existing functionality)
        save_artifact(gm)

        if self.verbose:
            print(f"[MockBackend] Checking constraints (Strict={self.strict}, Alignment={self._alignment})...")

        # Check constraints using existing registry classes
        violations = []
        for node in gm.graph.nodes:
            for constraint in self._constraints:
                if not constraint.check(node):
                    msg = constraint.message(node)
                    violations.append(ConstraintViolation(
                        node_name=node.name,
                        constraint_type=constraint.__class__.__name__.replace('Constraint', '').lower(),
                        message=msg,
                    ))

                    if self.strict:
                        # Generate repro script (existing functionality)
                        repro_path = _maybe_generate_repro(gm, example_inputs, msg)

                        return CompilationResult(
                            status=CompilationStatus.FAILED,
                            violations=violations,
                            fallback_fn=gm.forward,
                            metadata={
                                'repro_path': repro_path,
                                'error_message': msg,
                            },
                        )
                    else:
                        if self.verbose:
                            print(f"[MockBackend] WARNING: {msg}")

        # Success case
        if violations:
            if self.verbose:
                print(f"[MockBackend] Constraints checked (Warnings only). Total warnings: {len(violations)}")
            status = CompilationStatus.PARTIAL
        else:
            if self.verbose:
                print("[MockBackend] Constraints passed. Returning eager execution.")
            status = CompilationStatus.SUCCESS

        return CompilationResult(
            status=status,
            compiled_fn=gm.forward,
            violations=violations,
            warnings=[str(v) for v in violations],
        )


def integrated_mock_backend(strict: bool = None, alignment: int = None):
    """
    Factory function that returns an IntegratedMockAdapter.

    This can be used as a drop-in replacement for mock_backend:

        # Old way
        compiled = torch.compile(model, backend=mock_backend)

        # New way with adapter interface
        compiled = torch.compile(model, backend=integrated_mock_backend())

        # Or with custom settings
        compiled = torch.compile(model, backend=integrated_mock_backend(strict=False, alignment=16))
    """
    return IntegratedMockAdapter(strict=strict, alignment=alignment)
