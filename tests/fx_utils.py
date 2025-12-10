import torch
from torch.fx import Interpreter
from torch.fx.experimental.proxy_tensor import make_fx


class MetaCaptureInterpreter(Interpreter):
    """Interpreter that stores real tensor values for metadata-driven passes."""

    def run_node(self, node):
        result = super().run_node(node)
        node.meta["example_value"] = result
        return result


def trace_with_metadata(model: torch.nn.Module, *inputs):
    """Trace a module with make_fx and attach concrete example values to nodes."""
    gm = make_fx(model)(*inputs)
    MetaCaptureInterpreter(gm).run(*inputs)
    return gm
