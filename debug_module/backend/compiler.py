import torch
import torch.fx
from typing import List, Optional
from ..constraints.base import Constraint
import os
from ..constraints.registry import (
    DEFAULT_CONSTRAINTS,
    UnsupportedOpsConstraint,
    DtypeConstraint,
    LayoutConstraint,
    ShapeConstraint,
    MemoryConstraint,
    deny_ops,
)
from ..minifier.minifier import Minifier
from ..utils import BackendCompilerFailed

import time
import hashlib

def save_artifact(gm: torch.fx.GraphModule):
    """
    Saves the FX graph to a debug artifact file.
    """
    artifact_dir = "debug_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Generate a unique filename
    timestamp = int(time.time())
    graph_str = str(gm.graph)
    graph_hash = hashlib.md5(graph_str.encode('utf-8')).hexdigest()[:8]
    filename = f"{artifact_dir}/graph_{timestamp}_{graph_hash}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Hash: {graph_hash}\n")
        f.write("-" * 40 + "\n")
        f.write("Graph Code:\n")
        f.write(gm.print_readable(print_output=False))
        f.write("\n" + "-" * 40 + "\n")
        f.write("Graph Nodes:\n")
        for node in gm.graph.nodes:
            f.write(f"{node.name} ({node.op}) target={node.target} args={node.args} kwargs={node.kwargs}\n")
            if 'val' in node.meta:
                 f.write(f"  val: {node.meta['val']}\n")
            elif 'example_value' in node.meta:
                 f.write(f"  example_value: {node.meta['example_value']}\n")
    
    print(f"[MockBackend] Saved artifact to {filename}")


def _maybe_generate_repro(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], error_message: str
) -> Optional[str]:
    repro_dir = os.path.join("debug_artifacts", "repros")
    os.makedirs(repro_dir, exist_ok=True)

    timestamp = int(time.time())
    graph_hash = hashlib.md5(str(gm.graph).encode("utf-8")).hexdigest()[:8]
    repro_path = os.path.join(repro_dir, f"repro_{timestamp}_{graph_hash}.py")

    try:
        minifier = Minifier(gm, tuple(example_inputs), RuntimeError(error_message))
        minifier.minify()
        minifier.generate_repro_script(repro_path)
        print(f"[MockBackend] Saved repro script to {repro_path}")
        return repro_path
    except Exception as exc:
        print(f"[MockBackend] Failed to generate repro script: {exc}")
        return None

def mock_compile(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], constraints: List[Constraint] = None):
    """
    The core compilation function for the Mock Backend.
    
    1. Captures artifacts.
    2. Validates the graph against constraints.
    3. If valid, returns the executable (using eager PyTorch execution).
    """
    if constraints is None:
        constraints = DEFAULT_CONSTRAINTS
    # The 'constraints' argument is now ignored as constraints are configured via env vars.

    print(f"[MockBackend] Compiling graph with {len(gm.graph.nodes)} nodes...")
    
    # 0. Artifact Capture
    save_artifact(gm)
    
    # Load Configuration from Env
    strict_mode = os.environ.get("MOCK_STRICT", "1") == "1"
    alignment = int(os.environ.get("MOCK_ALIGNMENT", "1"))
    max_memory = int(os.environ.get("MOCK_MAX_MEMORY", str(1024**3 * 16))) # 16GB default

    # Initialize Constraints
    constraints = [
        UnsupportedOpsConstraint(deny_ops),
        DtypeConstraint(),
        LayoutConstraint(),
        ShapeConstraint(alignment=alignment),
        MemoryConstraint(max_memory_bytes=max_memory)
    ]

    # 2. Check Constraints
    print(f"[MockBackend] Checking constraints (Strict={strict_mode}, Alignment={alignment})...")
    
    # Collect all errors, even in non-strict mode, to potentially report them later
    all_errors = [] 
    for node in gm.graph.nodes:
        for constraint in constraints:
            if not constraint.check(node):
                msg = constraint.message(node)
                all_errors.append(msg)
                if strict_mode:
                    repro_path = _maybe_generate_repro(gm, example_inputs, msg)
                    extra = f"\nRepro script saved to: {repro_path}" if repro_path else ""
                    raise BackendCompilerFailed(f"Mock Backend Compilation Failed:\n{msg}{extra}")
                else:
                    print(f"[MockBackend] WARNING: {msg}")
    
    if strict_mode:
        print("[MockBackend] Constraints passed. Returning eager execution.")
    else:
        if all_errors:
            # If not strict mode, but there were warnings, we still return eager execution
            # but indicate that there were issues.
            print(f"[MockBackend] Constraints checked (Warnings only). Returning eager execution. Total warnings: {len(all_errors)}")
        else:
            print("[MockBackend] Constraints passed. Returning eager execution.")
    
    # 2. Execution (Eager Fallback)
    # We return the forward method of the GraphModule, which executes the graph using PyTorch
    return gm.forward
