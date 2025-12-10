import torch
from .compiler import compile_graph_with_aot

def mock_backend(strict=None, constraints=None):
    """
    Factory that returns a backend function configured with optional
    constraint overrides and strict/warning mode.

    Usage:
        backend = mock_backend(strict=True)
        torch.compile(model, backend=backend)

        backend = mock_backend(
            strict=False,
            constraints=[ShapeConstraint(16), DtypeConstraint({...})],
        )
    """
    
    def backend_fn(gm: torch.fx.GraphModule, example_inputs):
        return compile_graph_with_aot(
            gm,
            example_inputs,
            constraints=constraints,
            strict=strict,
        )
    
    return backend_fn