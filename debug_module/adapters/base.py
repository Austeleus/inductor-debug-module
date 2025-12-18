"""
Base Adapter Interface for Custom Accelerator Backends

This module defines the abstract interface that custom accelerator backends
must implement to integrate with the TorchInductor Debug Module.

Example Usage:
    class MyCustomAccelerator(AcceleratorAdapter):
        def get_capabilities(self) -> AcceleratorCapabilities:
            return AcceleratorCapabilities(
                supported_dtypes={torch.float32, torch.float16},
                supported_ops=self._get_supported_ops(),
                max_memory_bytes=8 * 1024**3,  # 8GB
                requires_contiguous=True,
                alignment_requirement=64,
            )

        def compile(self, gm, example_inputs):
            # Custom compilation logic
            return CompilationResult(success=True, compiled_fn=gm.forward)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.fx


class CompilationStatus(Enum):
    """Status of a compilation attempt."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some subgraphs compiled, others fell back to eager


@dataclass
class AcceleratorCapabilities:
    """
    Defines the capabilities and constraints of a custom accelerator.

    This dataclass specifies what operations, data types, and tensor layouts
    the accelerator supports, enabling the debug module to identify
    compatibility issues before deployment.

    Attributes:
        name: Human-readable name for the accelerator
        supported_dtypes: Set of PyTorch dtypes the accelerator can handle
        unsupported_ops: Set of operation names (aten ops) that are NOT supported
        supported_ops: Optional set of ops that ARE supported (if specified,
                       only these ops are allowed)
        max_memory_bytes: Maximum memory available on the accelerator
        requires_contiguous: Whether tensors must be contiguous in memory
        alignment_requirement: Memory alignment requirement in bytes (1 = no requirement)
        max_tensor_dims: Maximum number of dimensions supported
        supports_dynamic_shapes: Whether dynamic shapes are supported
        custom_constraints: Additional custom constraint functions
    """
    name: str = "Custom Accelerator"
    supported_dtypes: Set[torch.dtype] = field(default_factory=lambda: {
        torch.float32, torch.float16, torch.bfloat16,
        torch.int32, torch.int64, torch.bool
    })
    unsupported_ops: Set[str] = field(default_factory=set)
    supported_ops: Optional[Set[str]] = None  # None means all ops except unsupported
    max_memory_bytes: int = 16 * 1024**3  # 16GB default
    requires_contiguous: bool = False
    alignment_requirement: int = 1  # 1 means no alignment requirement
    max_tensor_dims: int = 8
    supports_dynamic_shapes: bool = True
    custom_constraints: List[Callable[[torch.fx.Node], Optional[str]]] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to a dictionary for serialization."""
        return {
            'name': self.name,
            'supported_dtypes': [str(d) for d in self.supported_dtypes],
            'unsupported_ops': list(self.unsupported_ops),
            'supported_ops': list(self.supported_ops) if self.supported_ops else None,
            'max_memory_bytes': self.max_memory_bytes,
            'requires_contiguous': self.requires_contiguous,
            'alignment_requirement': self.alignment_requirement,
            'max_tensor_dims': self.max_tensor_dims,
            'supports_dynamic_shapes': self.supports_dynamic_shapes,
        }


@dataclass
class ConstraintViolation:
    """Represents a single constraint violation."""
    node_name: str
    constraint_type: str  # 'dtype', 'layout', 'shape', 'memory', 'op', 'custom'
    message: str
    severity: str = "error"  # 'error' or 'warning'
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.constraint_type.upper()}] {self.node_name}: {self.message}"


@dataclass
class CompilationResult:
    """
    Result of a compilation attempt.

    Attributes:
        status: Whether compilation succeeded, failed, or was partial
        compiled_fn: The compiled callable (if successful)
        violations: List of constraint violations found
        warnings: Non-fatal issues detected during compilation
        metadata: Additional compilation metadata (timing, graph stats, etc.)
        fallback_fn: Fallback function to use if compilation failed
    """
    status: CompilationStatus = CompilationStatus.FAILED
    compiled_fn: Optional[Callable] = None
    violations: List[ConstraintViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_fn: Optional[Callable] = None

    @property
    def success(self) -> bool:
        """Check if compilation was successful."""
        return self.status == CompilationStatus.SUCCESS

    @property
    def has_violations(self) -> bool:
        """Check if there are any constraint violations."""
        return len(self.violations) > 0

    def get_callable(self) -> Callable:
        """Get the callable to use (compiled or fallback)."""
        if self.compiled_fn is not None:
            return self.compiled_fn
        if self.fallback_fn is not None:
            return self.fallback_fn
        raise RuntimeError("No compiled function or fallback available")


class AcceleratorAdapter(ABC):
    """
    Abstract base class for custom accelerator backend adapters.

    Implement this interface to integrate a custom accelerator with the
    TorchInductor Debug Module. The adapter handles:
    1. Defining accelerator capabilities and constraints
    2. Checking graphs for compatibility
    3. Compiling compatible graphs
    4. Providing fallback behavior for incompatible operations

    Example:
        class TPUAdapter(AcceleratorAdapter):
            def get_capabilities(self) -> AcceleratorCapabilities:
                return AcceleratorCapabilities(
                    name="Google TPU v4",
                    supported_dtypes={torch.float32, torch.bfloat16},
                    requires_contiguous=True,
                    alignment_requirement=128,
                )

            def compile(self, gm, example_inputs) -> CompilationResult:
                # TPU-specific compilation logic
                ...
    """

    def __init__(self, strict: bool = True, verbose: bool = False):
        """
        Initialize the adapter.

        Args:
            strict: If True, raise errors on constraint violations.
                    If False, emit warnings and use fallback.
            verbose: If True, print detailed information during compilation.
        """
        self.strict = strict
        self.verbose = verbose
        self._capabilities: Optional[AcceleratorCapabilities] = None

    @property
    def capabilities(self) -> AcceleratorCapabilities:
        """Get accelerator capabilities (cached)."""
        if self._capabilities is None:
            self._capabilities = self.get_capabilities()
        return self._capabilities

    @abstractmethod
    def get_capabilities(self) -> AcceleratorCapabilities:
        """
        Return the capabilities and constraints of this accelerator.

        Must be implemented by subclasses to define:
        - Supported data types
        - Unsupported operations
        - Memory limits
        - Layout requirements
        - Any custom constraints

        Returns:
            AcceleratorCapabilities describing what this accelerator supports.
        """
        pass

    @abstractmethod
    def compile(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> CompilationResult:
        """
        Compile an FX graph for this accelerator.

        This method should:
        1. Check the graph against accelerator constraints
        2. If compatible, compile and return the compiled function
        3. If not compatible, return violations and optionally a fallback

        Args:
            gm: The FX GraphModule to compile
            example_inputs: Example input tensors for shape inference

        Returns:
            CompilationResult with the compiled function or violations
        """
        pass

    def check_constraints(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> List[ConstraintViolation]:
        """
        Check a graph against accelerator constraints without compiling.

        This is useful for analysis without triggering compilation.

        Args:
            gm: The FX GraphModule to check
            example_inputs: Example input tensors

        Returns:
            List of constraint violations found
        """
        violations = []
        caps = self.capabilities

        for node in gm.graph.nodes:
            # Check dtype constraints
            dtype_violation = self._check_dtype(node, caps)
            if dtype_violation:
                violations.append(dtype_violation)

            # Check layout constraints
            layout_violation = self._check_layout(node, caps)
            if layout_violation:
                violations.append(layout_violation)

            # Check shape constraints
            shape_violation = self._check_shape(node, caps)
            if shape_violation:
                violations.append(shape_violation)

            # Check memory constraints
            memory_violation = self._check_memory(node, caps)
            if memory_violation:
                violations.append(memory_violation)

            # Check operation support
            op_violation = self._check_op_support(node, caps)
            if op_violation:
                violations.append(op_violation)

            # Run custom constraints
            for custom_check in caps.custom_constraints:
                msg = custom_check(node)
                if msg:
                    violations.append(ConstraintViolation(
                        node_name=node.name,
                        constraint_type='custom',
                        message=msg,
                    ))

        return violations

    def _check_dtype(
        self,
        node: torch.fx.Node,
        caps: AcceleratorCapabilities,
    ) -> Optional[ConstraintViolation]:
        """Check if node's dtype is supported."""
        val = node.meta.get('val') or node.meta.get('example_value')
        if val is None:
            return None

        if hasattr(val, 'dtype') and val.dtype not in caps.supported_dtypes:
            return ConstraintViolation(
                node_name=node.name,
                constraint_type='dtype',
                message=f"Unsupported dtype {val.dtype}. "
                        f"Supported: {caps.supported_dtypes}",
                suggestion=f"Cast tensor to one of: {caps.supported_dtypes}",
            )
        return None

    def _check_layout(
        self,
        node: torch.fx.Node,
        caps: AcceleratorCapabilities,
    ) -> Optional[ConstraintViolation]:
        """Check if node produces contiguous output (if required)."""
        if not caps.requires_contiguous:
            return None

        val = node.meta.get('val') or node.meta.get('example_value')
        if val is None:
            return None

        if hasattr(val, 'is_contiguous') and not val.is_contiguous():
            return ConstraintViolation(
                node_name=node.name,
                constraint_type='layout',
                message="Produces non-contiguous tensor. "
                        f"Accelerator '{caps.name}' requires contiguous memory.",
                suggestion="Add .contiguous() call after this operation",
            )
        return None

    def _check_shape(
        self,
        node: torch.fx.Node,
        caps: AcceleratorCapabilities,
    ) -> Optional[ConstraintViolation]:
        """Check shape alignment and dimension constraints."""
        val = node.meta.get('val') or node.meta.get('example_value')
        if val is None:
            return None

        if not hasattr(val, 'shape'):
            return None

        shape = val.shape

        # Check dimension count
        if len(shape) > caps.max_tensor_dims:
            return ConstraintViolation(
                node_name=node.name,
                constraint_type='shape',
                message=f"Tensor has {len(shape)} dimensions, "
                        f"but accelerator supports max {caps.max_tensor_dims}.",
            )

        # Check alignment
        if caps.alignment_requirement > 1:
            for dim_size in shape:
                if dim_size % caps.alignment_requirement != 0:
                    return ConstraintViolation(
                        node_name=node.name,
                        constraint_type='shape',
                        message=f"Shape {tuple(shape)} violates alignment "
                                f"requirement of {caps.alignment_requirement}.",
                        suggestion=f"Pad tensors to multiples of {caps.alignment_requirement}",
                    )

        return None

    def _check_memory(
        self,
        node: torch.fx.Node,
        caps: AcceleratorCapabilities,
    ) -> Optional[ConstraintViolation]:
        """Check if tensor size exceeds memory limit."""
        val = node.meta.get('val') or node.meta.get('example_value')
        if val is None:
            return None

        if hasattr(val, 'nelement') and hasattr(val, 'element_size'):
            size_bytes = val.nelement() * val.element_size()
            if size_bytes > caps.max_memory_bytes:
                return ConstraintViolation(
                    node_name=node.name,
                    constraint_type='memory',
                    message=f"Tensor size {size_bytes / 1024**2:.1f}MB exceeds "
                            f"limit {caps.max_memory_bytes / 1024**2:.1f}MB.",
                    suggestion="Reduce batch size or use gradient checkpointing",
                )

        return None

    def _check_op_support(
        self,
        node: torch.fx.Node,
        caps: AcceleratorCapabilities,
    ) -> Optional[ConstraintViolation]:
        """Check if operation is supported."""
        if node.op != 'call_function':
            return None

        op_name = str(node.target)

        # Check against unsupported ops
        if op_name in caps.unsupported_ops:
            return ConstraintViolation(
                node_name=node.name,
                constraint_type='op',
                message=f"Operation '{op_name}' is not supported.",
                suggestion="Replace with a supported alternative or use eager fallback",
            )

        # If supported_ops is specified, check allowlist
        if caps.supported_ops is not None and op_name not in caps.supported_ops:
            return ConstraintViolation(
                node_name=node.name,
                constraint_type='op',
                message=f"Operation '{op_name}' is not in the supported ops list.",
            )

        return None

    def get_fallback_fn(
        self,
        gm: torch.fx.GraphModule,
    ) -> Callable:
        """
        Get a fallback function for when compilation fails.

        Default implementation returns eager execution.
        Override for custom fallback behavior.

        Args:
            gm: The GraphModule that failed to compile

        Returns:
            A callable that can execute the graph
        """
        return gm.forward

    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
    ) -> Callable:
        """
        Make the adapter callable as a torch.compile backend.

        This allows using the adapter directly with torch.compile:
            compiled = torch.compile(model, backend=my_adapter)

        Args:
            gm: The FX GraphModule to compile
            example_inputs: Example inputs for compilation

        Returns:
            The compiled function or fallback
        """
        result = self.compile(gm, example_inputs)

        if result.has_violations and self.strict:
            violation_msgs = '\n'.join(str(v) for v in result.violations)
            raise RuntimeError(
                f"Accelerator '{self.capabilities.name}' constraint violations:\n"
                f"{violation_msgs}"
            )

        if result.has_violations and self.verbose:
            for v in result.violations:
                print(f"[WARNING] {v}")

        return result.get_callable()
