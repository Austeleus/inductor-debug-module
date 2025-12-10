import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

import os
import time
import hashlib
import json

def save_svg_graph(gm: torch.fx.GraphModule, filename: str):
    """
    Save an SVG visualization of an FX graph.
    """
    artifact_dir = "debug_artifacts/visualizations"
    os.makedirs(artifact_dir, exist_ok=True)

    drawer = FxGraphDrawer(gm, "AOT Graph")
    svg_data = drawer.get_dot_graph().create_svg()

    filepath = os.path.join(artifact_dir, filename)
    with open(filepath, "wb") as f:
        f.write(svg_data)

    print(f"[MockBackend] Saved SVG visualization: {filepath}")
    
def save_pre_aot_artifact(gm: torch.fx.GraphModule):
    """
    Save the pre-AOT FX graph (the original graph passed to the backend).
    """
    artifact_dir = "debug_artifacts/fx_graphs"
    os.makedirs(artifact_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{artifact_dir}/fx_pre_{timestamp}.txt"
    save_svg_graph(gm, f"fx_pre_{timestamp}.svg")

    with open(filename, "w") as f:
        f.write(gm.print_readable(print_output=False))

    print(f"[MockBackend] Saved pre-AOT FX graph: {filename}")


def save_post_aot_forward(aot_gm: torch.fx.GraphModule):
    """
    Save the forward AOT graph produced by AOTAutograd.
    """
    artifact_dir = "debug_artifacts/aot_graphs"
    os.makedirs(artifact_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{artifact_dir}/aot_fwd_{timestamp}.txt"
    save_svg_graph(aot_gm, f"aot_fwd_{timestamp}.svg")

    with open(filename, "w") as f:
        f.write(aot_gm.print_readable(print_output=False))

    print(f"[MockBackend] Saved AOT Forward graph: {filename}")

def save_post_aot_backward(aot_gm: torch.fx.GraphModule):
    """
    Save the backward AOT graph produced by AOTAutograd.
    """
    artifact_dir = "debug_artifacts/aot_graphs"
    os.makedirs(artifact_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{artifact_dir}/aot_bwd_{timestamp}.txt"
    save_svg_graph(aot_gm, f"aot_bwd_{timestamp}.svg")

    with open(filename, "w") as f:
        f.write(aot_gm.print_readable(print_output=False))

    print(f"[MockBackend] Saved AOT Backward graph: {filename}")

def save_graph_statistics(gm: torch.fx.GraphModule, direction: str):
    """
    Compute and save statistics about the AOT graph.
    """
    artifact_dir = "debug_artifacts/statistics"
    os.makedirs(artifact_dir, exist_ok=True)

    stats = {
        "total_nodes": 0,
        "placeholder_count": 0,
        "output_nodes": 0,
        "call_function": 0,
        "call_method": 0,
        "call_module": 0,
        "dtype_histogram": {},
        "op_histogram": {},
    }

    for node in gm.graph.nodes:
        stats["total_nodes"] += 1
        node_op = node.op

        # Count op type
        stats["op_histogram"][node_op] = stats["op_histogram"].get(node_op, 0) + 1

        if node_op == "placeholder":
            stats["placeholder_count"] += 1
        if node_op == "output":
            stats["output_nodes"] += 1
        if node_op == "call_function":
            stats["call_function"] += 1
        if node_op == "call_method":
            stats["call_method"] += 1
        if node_op == "call_module":
            stats["call_module"] += 1

        # dtype histogram
        if "val" in node.meta and hasattr(node.meta["val"], "dtype"):
            dtype = str(node.meta["val"].dtype)
            stats["dtype_histogram"][dtype] = stats["dtype_histogram"].get(dtype, 0) + 1

    timestamp = int(time.time())
    filename = f"{artifact_dir}/stats_{direction}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[MockBackend] Saved graph statistics: {filename}")

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