import torch
import torch.fx
from typing import Set, List, Type
from .base import Constraint

class LayoutConstraint(Constraint):
    def __init__(self, strict_contiguous: bool = True):
        self.strict_contiguous = strict_contiguous

    def check(self, node: torch.fx.Node) -> bool:
        # Only check nodes that produce tensors
        val = node.meta.get('val')
        if val is None:
             val = node.meta.get('example_value')
        
        if isinstance(val, torch.Tensor):
            if self.strict_contiguous and not val.is_contiguous():
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
                if dim % self.alignment != 0:
                    return False
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
        
        # Get the qualified name of the target
        if hasattr(node.target, '__name__'):
             # This handles simple functions
             # For aten ops, it might be complex, so we might need a better stringifier
             # But for now, let's try to match against the string representation or name
             pass
        
        # A more robust way for torch.ops is often checking the packet or the op itself
        # For this mock, let's check if the str(node.target) is in the set
        # Example: node.target might be <OpOverload(op='aten.sin', overload='default')>
        
        op_name = str(node.target)
        return op_name not in self.unsupported_ops

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
            return val.dtype in self.allowed_dtypes
        
        return True

    def message(self, node: torch.fx.Node) -> str:
        val = node.meta.get('val', node.meta.get('example_value'))
        dtype = "unknown"
        if isinstance(val, torch.Tensor):
            dtype = val.dtype
        return f"Dtype {dtype} is not allowed. Allowed: {self.allowed_dtypes}"


# Default configuration for the Mock Backend
# We can expand this later or make it configurable via env vars
DEFAULT_CONSTRAINTS: List[Constraint] = [
    # Example: Let's ban float64 to simulate a lower-precision accelerator
    DtypeConstraint({torch.float32, torch.float16, torch.int64, torch.int32, torch.bool}),
    
    # Example: Let's ban a random op to test the mechanism (e.g. 'aten.sin.default' if we wanted)
    # For now, empty list of unsupported ops
    UnsupportedOpsConstraint(set())
]
