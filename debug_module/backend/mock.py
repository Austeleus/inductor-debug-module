import torch
from .compiler import mock_compile

def mock_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Entry point for torch.compile(backend=mock_backend).
    """
    return mock_compile(gm, example_inputs)
