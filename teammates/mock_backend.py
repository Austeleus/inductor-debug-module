"""
Mock Accelerator Backend for TorchInductor Debugging

A standalone, black-box backend that simulates custom accelerator constraints.
Just import and use with torch.compile() - no internal knowledge required.

Usage:
    from mock_backend import mock_backend, limited_precision_backend
    
    model = torch.compile(my_model, backend=mock_backend)
    output = model(input)

Author: TorchInductor Debug Project Team
License: MIT
"""

import torch
import torch.fx as fx
from typing import Callable, List, Dict, Any, Optional, Set
import logging
import json
from dataclasses import dataclass, field
from enum import Enum

__version__ = "0.1.0"
__all__ = [
    'mock_backend',
    'limited_precision_backend', 
    'no_complex_ops_backend',
    'strict_layout_backend',
    'custom_backend',
    'get_last_report',
    'export_last_report',
    'get_backend_constraints',
    'list_unsupported_ops',
    'list_supported_dtypes',
]

logger = logging.getLogger(__name__)


class LayoutConstraint(Enum):
    STRIDED = "strided"
    CHANNELS_LAST = "channels_last"
    CHANNELS_LAST_3D = "channels_last_3d"


@dataclass
class BackendConstraints:
    """Configuration for mock accelerator limitations."""
    
    supported_dtypes: Set[torch.dtype] = field(default_factory=lambda: {
        torch.float32, torch.float16, torch.int32, torch.int64, torch.bool
    })
    unsupported_ops: Set[str] = field(default_factory=lambda: {
        'aten.complex', 'aten.polar', 'aten.fft', 'aten.stft',
        'aten.tensordot', 'aten.einsum', 'aten.cdist', 'aten.index_put',
    })
    supported_layouts: Set[LayoutConstraint] = field(
        default_factory=lambda: {LayoutConstraint.STRIDED}
    )
    max_tensor_dims: int = 6
    supports_dynamic_shapes: bool = False
    supports_inplace_ops: bool = True


@dataclass
class CompilationReport:
    """Report of compilation analysis results."""
    unsupported_ops_found: List[Dict[str, Any]] = field(default_factory=list)
    dtype_violations: List[Dict[str, Any]] = field(default_factory=list)
    layout_violations: List[Dict[str, Any]] = field(default_factory=list)
    fallback_count: int = 0
    total_ops: int = 0
    supported_ops: int = 0
    
    def to_json(self) -> str:
        return json.dumps({
            'unsupported_ops_found': self.unsupported_ops_found,
            'dtype_violations': self.dtype_violations,
            'layout_violations': self.layout_violations,
            'fallback_count': self.fallback_count,
            'total_ops': self.total_ops,
            'supported_ops': self.supported_ops,
        }, indent=2)


class _BackendState:
    """Internal state storage for compilation reports."""
    last_report: Optional[CompilationReport] = None
    all_reports: List[CompilationReport] = []
    last_constraints: Optional[BackendConstraints] = None


def _check_dtype_support(node: fx.Node, constraints: BackendConstraints) -> List[Dict[str, Any]]:
    """Check if node uses unsupported dtypes."""
    violations = []
    if hasattr(node, 'meta') and 'val' in node.meta:
        val = node.meta['val']
        if isinstance(val, torch.Tensor):
            if val.dtype not in constraints.supported_dtypes:
                violations.append({
                    'node': str(node),
                    'op': str(node.target),
                    'dtype': str(val.dtype),
                    'reason': f'Dtype {val.dtype} not supported'
                })
    return violations


def _check_op_support(node: fx.Node, constraints: BackendConstraints) -> Optional[Dict[str, Any]]:
    """Check if operation is supported."""
    if node.op == 'call_function':
        op_name = str(node.target)
        for unsupported in constraints.unsupported_ops:
            if unsupported in op_name:
                return {
                    'node': str(node),
                    'op': op_name,
                    'reason': f'Operation not supported',
                }
    elif node.op == 'call_method':
        method_name = f"aten.{node.target}"
        if method_name in constraints.unsupported_ops:
            return {
                'node': str(node),
                'op': method_name,
                'reason': f'Method not supported',
            }
    return None


def _check_layout_support(node: fx.Node, constraints: BackendConstraints) -> List[Dict[str, Any]]:
    """Check if tensor layouts are supported."""
    violations = []
    if hasattr(node, 'meta') and 'val' in node.meta:
        val = node.meta['val']
        if isinstance(val, torch.Tensor):
            if (val.is_contiguous(memory_format=torch.channels_last) and
                LayoutConstraint.CHANNELS_LAST not in constraints.supported_layouts):
                violations.append({
                    'node': str(node),
                    'op': str(node.target),
                    'layout': 'channels_last',
                    'reason': 'Channels-last layout not supported'
                })
    return violations


def _analyze_graph(gm: fx.GraphModule, constraints: BackendConstraints) -> CompilationReport:
    """Analyze FX graph for compatibility issues."""
    unsupported_ops = []
    dtype_violations = []
    layout_violations = []
    total_ops = 0
    
    for node in gm.graph.nodes:
        if node.op in ('call_function', 'call_method', 'call_module'):
            total_ops += 1
            
            op_issue = _check_op_support(node, constraints)
            if op_issue:
                unsupported_ops.append(op_issue)
            
            dtype_issues = _check_dtype_support(node, constraints)
            dtype_violations.extend(dtype_issues)
            
            layout_issues = _check_layout_support(node, constraints)
            layout_violations.extend(layout_issues)
    
    supported_ops = total_ops - len(unsupported_ops)
    
    return CompilationReport(
        unsupported_ops_found=unsupported_ops,
        dtype_violations=dtype_violations,
        layout_violations=layout_violations,
        fallback_count=len(unsupported_ops),
        total_ops=total_ops,
        supported_ops=supported_ops
    )


def _create_backend(constraints: BackendConstraints) -> Callable:
    """Internal factory to create backend compiler function."""
    
    def compiler_fn(gm: fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
        # Store constraints for inspection
        _BackendState.last_constraints = constraints
        
        # Analyze graph
        report = _analyze_graph(gm, constraints)
        
        # Store report
        _BackendState.last_report = report
        _BackendState.all_reports.append(report)
        
        # Log summary
        logger.info(f"Mock Backend: {report.supported_ops}/{report.total_ops} ops supported")
        if report.unsupported_ops_found:
            logger.warning(f"Found {len(report.unsupported_ops_found)} unsupported ops")
        if report.dtype_violations:
            logger.warning(f"Found {len(report.dtype_violations)} dtype violations")
        if report.layout_violations:
            logger.warning(f"Found {len(report.layout_violations)} layout violations")
        
        # Return forward function (fallback to eager if violations exist)
        return gm.forward
    
    return compiler_fn


# ============================================================================
# PUBLIC API - These are the only functions users need to know about
# ============================================================================

def mock_backend(gm: fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """
    Default mock backend with standard accelerator constraints.
    
    Use with torch.compile:
        model = torch.compile(my_model, backend=mock_backend)
    
    Simulates an accelerator with:
    - FP32, FP16, INT32, INT64, BOOL support only
    - No complex ops, FFT, einsum, etc.
    - Strided layout only
    """
    return _create_backend(BackendConstraints())(gm, example_inputs)


def limited_precision_backend(gm: fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """
    Backend that only supports FP32 and FP16 (no FP64, BF16, etc).
    
    Use with torch.compile:
        model = torch.compile(my_model, backend=limited_precision_backend)
    """
    constraints = BackendConstraints(
        supported_dtypes={torch.float32, torch.float16, torch.int32, torch.bool}
    )
    return _create_backend(constraints)(gm, example_inputs)


def no_complex_ops_backend(gm: fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """
    Backend without complex numbers or advanced math operations.
    
    Use with torch.compile:
        model = torch.compile(my_model, backend=no_complex_ops_backend)
    """
    constraints = BackendConstraints(
        unsupported_ops={
            'aten.complex', 'aten.polar', 'aten.angle',
            'aten.fft', 'aten.stft', 'aten.ifft',
            'aten.tensordot', 'aten.einsum',
            'aten.linalg_eigh', 'aten.linalg_svd',
            'aten.cdist', 'aten.pdist'
        }
    )
    return _create_backend(constraints)(gm, example_inputs)


def strict_layout_backend(gm: fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
    """
    Backend that requires strided layout only (no channels-last).
    
    Use with torch.compile:
        model = torch.compile(my_model, backend=strict_layout_backend)
    """
    constraints = BackendConstraints(
        supported_layouts={LayoutConstraint.STRIDED}
    )
    return _create_backend(constraints)(gm, example_inputs)


def custom_backend(
    supported_dtypes: Optional[Set[torch.dtype]] = None,
    unsupported_ops: Optional[Set[str]] = None,
    max_tensor_dims: int = 6,
) -> Callable:
    """
    Create a custom backend with your own constraints.
    
    Args:
        supported_dtypes: Set of allowed torch dtypes (default: FP32, FP16, INT32, INT64, BOOL)
        unsupported_ops: Set of ATen op names to block (default: complex ops, FFT, einsum, etc)
        max_tensor_dims: Maximum tensor dimensions allowed (default: 6)
    
    Returns:
        Backend function for torch.compile
    
    Example:
        backend = custom_backend(
            supported_dtypes={torch.float32},
            unsupported_ops={'aten.einsum', 'aten.matmul'}
        )
        model = torch.compile(my_model, backend=backend)
    """
    constraints = BackendConstraints(
        max_tensor_dims=max_tensor_dims
    )
    if supported_dtypes is not None:
        constraints.supported_dtypes = supported_dtypes
    if unsupported_ops is not None:
        constraints.unsupported_ops = unsupported_ops
    
    return _create_backend(constraints)


def get_last_report() -> Optional[CompilationReport]:
    """
    Get the most recent compilation report.
    
    Returns:
        CompilationReport with details about last compilation, or None if no compilations yet
    
    Example:
        model = torch.compile(my_model, backend=mock_backend)
        output = model(input)
        report = get_last_report()
        print(report.to_json())
    """
    return _BackendState.last_report


def export_last_report(filepath: str = "mock_backend_report.json") -> None:
    """
    Export the last compilation report to a JSON file.
    
    Args:
        filepath: Where to save the report (default: "mock_backend_report.json")
    
    Example:
        model = torch.compile(my_model, backend=mock_backend)
        output = model(input)
        export_last_report("my_report.json")
    """
    report = _BackendState.last_report
    if report:
        with open(filepath, 'w') as f:
            f.write(report.to_json())
        logger.info(f"Report exported to {filepath}")
    else:
        logger.warning("No compilation report available to export")


def get_backend_constraints() -> Optional[Dict[str, Any]]:
    """
    Get the constraints of the most recently used backend.
    
    Returns:
        Dictionary containing backend limitations (dtypes, ops, layouts, etc.)
        or None if no backend has been used yet
    
    Example:
        model = torch.compile(my_model, backend=mock_backend)
        output = model(input)
        constraints = get_backend_constraints()
        print(f"Supported dtypes: {constraints['supported_dtypes']}")
        print(f"Unsupported ops: {constraints['unsupported_ops']}")
    """
    if _BackendState.last_constraints:
        return {
            'supported_dtypes': [str(dt) for dt in _BackendState.last_constraints.supported_dtypes],
            'unsupported_ops': sorted(list(_BackendState.last_constraints.unsupported_ops)),
            'supported_layouts': [l.value for l in _BackendState.last_constraints.supported_layouts],
            'max_tensor_dims': _BackendState.last_constraints.max_tensor_dims,
            'supports_dynamic_shapes': _BackendState.last_constraints.supports_dynamic_shapes,
            'supports_inplace_ops': _BackendState.last_constraints.supports_inplace_ops,
        }
    return None


def list_unsupported_ops() -> Optional[List[str]]:
    """
    Get list of operations blocked by the current backend.
    
    Returns:
        List of ATen operation names that are not supported, or None if no backend used
    
    Example:
        model = torch.compile(my_model, backend=limited_precision_backend)
        output = model(input)
        blocked_ops = list_unsupported_ops()
        print(f"Backend blocks: {blocked_ops}")
    """
    if _BackendState.last_constraints:
        return sorted(list(_BackendState.last_constraints.unsupported_ops))
    return None


def list_supported_dtypes() -> Optional[List[str]]:
    """
    Get list of data types supported by the current backend.
    
    Returns:
        List of dtype strings supported by the backend, or None if no backend used
    
    Example:
        model = torch.compile(my_model, backend=limited_precision_backend)
        output = model(input)
        dtypes = list_supported_dtypes()
        print(f"Backend supports: {dtypes}")
    """
    if _BackendState.last_constraints:
        return [str(dt) for dt in _BackendState.last_constraints.supported_dtypes]
    return None


# ============================================================================
# DEMO CODE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    print(f"Mock Accelerator Backend v{__version__}")
    print("=" * 60)
    print("\nThis is a black-box backend for torch.compile testing.")
    print("Just import and use - no need to understand internals!\n")
    
    # Demo model
    class DemoModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(20, 10)
        
        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))
    
    print("EXAMPLE 1: Default mock backend")
    print("-" * 60)
    model = DemoModel()
    compiled = torch.compile(model, backend=mock_backend)
    x = torch.randn(5, 10)
    output = compiled(x)
    print(f"✓ Compiled and ran successfully!")
    print(f"✓ Output shape: {output.shape}")
    
    report = get_last_report()
    if report:
        print(f"✓ Operations: {report.supported_ops}/{report.total_ops} supported")
        if report.unsupported_ops_found:
            print(f"✗ Unsupported ops: {len(report.unsupported_ops_found)}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Limited precision backend")
    print("-" * 60)
    compiled2 = torch.compile(model, backend=limited_precision_backend)
    output2 = compiled2(x)
    print(f"✓ Compiled with limited precision backend")
    
    report2 = get_last_report()
    if report2:
        print(f"✓ Operations: {report2.supported_ops}/{report2.total_ops} supported")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom backend")
    print("-" * 60)
    custom = custom_backend(
        supported_dtypes={torch.float32},
        unsupported_ops={'aten.relu'}
    )
    compiled3 = torch.compile(model, backend=custom)
    output3 = compiled3(x)
    
    report3 = get_last_report()
    if report3:
        print(f"✓ Operations: {report3.supported_ops}/{report3.total_ops} supported")
        print(f"✗ Unsupported: {len(report3.unsupported_ops_found)} ops blocked by custom rules")
    
    # Export report
    export_last_report("demo_report.json")
    print(f"\n✓ Report exported to demo_report.json")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Inspecting backend constraints")
    print("-" * 60)
    
    # Check what the backend actually blocks
    constraints = get_backend_constraints()
    if constraints:
        print(f"✓ Supported dtypes: {constraints['supported_dtypes']}")
        print(f"✓ Max tensor dims: {constraints['max_tensor_dims']}")
        print(f"✓ Dynamic shapes: {constraints['supports_dynamic_shapes']}")
    
    blocked = list_unsupported_ops()
    if blocked:
        print(f"\n✗ Backend blocks {len(blocked)} operations:")
        for op in blocked[:5]:
            print(f"   - {op}")
        if len(blocked) > 5:
            print(f"   ... and {len(blocked) - 5} more")
    
    dtypes = list_supported_dtypes()
    if dtypes:
        print(f"\n✓ Supported dtypes: {', '.join(dtypes)}")
    
    print("\n" + "=" * 60)
    print("Usage Summary:")
    print("-" * 60)
    print("1. Import: from mock_backend import mock_backend")
    print("2. Compile: model = torch.compile(model, backend=mock_backend)")
    print("3. Run: output = model(input)")
    print("4. Check report: report = get_last_report()")
    print("5. Check constraints: constraints = get_backend_constraints()")
    print("6. List blocked ops: ops = list_unsupported_ops()")
    print("=" * 60)
