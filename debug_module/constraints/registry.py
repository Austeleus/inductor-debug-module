import torch
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import SymInt
from typing import Set, List, Type
from .base import Constraint


def resolve_aten(op_names: List[str]) -> Set[str]:
    """
    Resolve a list of operator names to their full aten qualified names.
    Returns a set of op names prefixed with 'aten.' for matching.
    """
    return {f"aten.{op_name}.default" for op_name in op_names}

class LayoutConstraint(Constraint):
    def __init__(self, strict_contiguous: bool = True):
        self.strict_contiguous = strict_contiguous

    def check(self, node: torch.fx.Node) -> bool:
        # Only check nodes that produce tensors
        val = node.meta.get('val')
        if val is None:
             val = node.meta.get('example_value')
        
        if isinstance(val, torch.Tensor):
            try:
                is_contig = bool(val.is_contiguous())
            except Exception:
                # FakeTensor case: assume unknown but safe
                return True

            if self.strict_contiguous and not is_contig:
                return False
        return True

    def message(self, node: torch.fx.Node) -> str:
        return f"Node {node.name} produces a non-contiguous tensor, which is not supported."

class ShapeConstraint(Constraint):
    def __init__(self, alignment: int = 1):
        self.alignment = alignment

    def check(self, node: torch.fx.Node) -> bool:
        val = node.meta.get('val')
        if val is None:
             val = node.meta.get('example_value')

        if isinstance(val, torch.Tensor):
            for dim in val.shape:
                # Case 1: symbolic dimensions -> cannot check
                if isinstance(dim, SymInt):
                    continue

                # Case 2: Python int
                if isinstance(dim, int):
                    if dim % self.alignment != 0:
                        return False
                    continue

                # Case 3: FakeTensor shapes appear as tensors
                try:
                    dim_int = int(dim)
                    if dim_int % self.alignment != 0:
                        return False
                except Exception:
                    # Can't evaluate shape — skip
                    continue
        return True

    def message(self, node: torch.fx.Node) -> str:
        val = node.meta.get('val') or node.meta.get('example_value')
        return f"Node {node.name} has shape {val.shape}, which violates alignment {self.alignment}."

class MemoryConstraint(Constraint):
    def __init__(self, max_memory_bytes: int = float('inf')):
        self.max_memory_bytes = max_memory_bytes
        self.current_memory = 0

    def check(self, node: torch.fx.Node) -> bool:
        # This is a simplified memory check (accumulating output sizes)
        # In reality, we'd need liveness analysis. 
        # For now, we'll just check if any SINGLE tensor exceeds the limit
        # or if we want to track peak, we'd need a stateful pass.
        # Let's implement "Single Tensor Limit" for simplicity in this pass.
        val = node.meta.get('val')
        if val is None:
             val = node.meta.get('example_value')

        if isinstance(val, torch.Tensor):
            numel = val.numel()
            element_size = val.element_size()
            tensor_size = numel * element_size
            if tensor_size > self.max_memory_bytes:
                return False
        return True

    def message(self, node: torch.fx.Node) -> str:
        val = node.meta.get('val') or node.meta.get('example_value')
        size = val.numel() * val.element_size()
        return f"Node {node.name} produces a tensor of size {size} bytes, exceeding limit {self.max_memory_bytes}."

class UnsupportedOpsConstraint(Constraint):
    def __init__(self, unsupported_ops: Set[str] = frozenset()):
        """
        unsupported_ops: Set of operator names (e.g., 'aten.sin.default')
        """
        self.unsupported_ops = unsupported_ops

    def check(self, node: torch.fx.Node) -> bool:
        if node.op != 'call_function':
            return True
        
        target = node.target

        # Must be OpOverload
        try:
            op_name = f"{target._schema.name}.{target.overload}"
        except Exception:
            # Fallback
            op_name = str(target)

        return bool(op_name not in self.unsupported_ops)

    def message(self, node: torch.fx.Node) -> str:
        return f"Operator {node.target} is not supported by this accelerator."


class DtypeConstraint(Constraint):
    def __init__(self, allowed_dtypes: Set[torch.dtype] = None):
        if allowed_dtypes is None:
            self.allowed_dtypes = {torch.float32, torch.int64, torch.bool}
        else:
            self.allowed_dtypes = allowed_dtypes

    def check(self, node: torch.fx.Node) -> bool:
        # We check the 'val' metadata which contains the FakeTensor
        # Fallback to 'example_value' if 'val' is not present
        val = node.meta.get('val', node.meta.get('example_value'))
        
        if val is None:
            return True # Skip nodes without type info (or handle strictly)
        
        # Handle tuples/lists of tensors (e.g. return values)
        if isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, torch.Tensor):
                     if v.dtype not in self.allowed_dtypes:
                         return False
            return True

        if isinstance(val, torch.Tensor):
            try:
                allowed = bool(val.dtype in self.allowed_dtypes)
            except Exception:
                return True
            if not allowed:
                return False
        
        return True

    def message(self, node: torch.fx.Node) -> str:
        val = node.meta.get('val', node.meta.get('example_value'))
        dtype = "unknown"
        if isinstance(val, torch.Tensor):
            dtype = val.dtype
        return f"Dtype {dtype} is not allowed. Allowed: {self.allowed_dtypes}"


# Unsupported Ops
lu_like = resolve_aten([
    "_lu_with_info",
    "linalg_lu",
    "linalg_lu_factor",
    "linalg_lu_factor_ex",
    "linalg_lu_solve",
    "lu_solve",
    "lu_unpack",
])

conv_like = resolve_aten([
    "convolution", "convolution_backward",
    "_convolution", "_convolution_mode", "_convolution_double_backward",
])

inplace_acts = resolve_aten([
    "relu_", "silu_", "gelu_", "elu_", "celu_", "leaky_relu_", "rrelu_", "selu_", "relu6_",
])

prelu_like = resolve_aten([
    "prelu", "_prelu_kernel", "_prelu_kernel_backward",
])

value_sel = resolve_aten(["kthvalue", "value_selecting_reduction_backward"])

deny_ops = set().union(lu_like, conv_like, inplace_acts, prelu_like, value_sel)


# Default configuration for the Mock Backend
# We can expand this later or make it configurable via env vars
DEFAULT_CONSTRAINTS: List[Constraint] = [
    # Example: Let's ban float64 to simulate a lower-precision accelerator
    DtypeConstraint({torch.float32, torch.float16, torch.int64, torch.int32, torch.bool}),
    
    # Unsupported ops constraint with deny_ops set
    UnsupportedOpsConstraint(deny_ops)
]
