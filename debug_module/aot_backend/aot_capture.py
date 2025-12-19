import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

import os
import time
import hashlib
import json
import shutil
import glob
from ..minifier.minifier import Minifier

def save_svg_graph(gm: torch.fx.GraphModule, filename: str):
    """
    Save an SVG visualization of an FX graph.

    Requires pydot and graphviz to be installed. Gracefully skips if unavailable.
    """
    artifact_dir = "debug_artifacts/visualizations"
    os.makedirs(artifact_dir, exist_ok=True)

    try:
        drawer = FxGraphDrawer(gm, "AOT Graph")
        svg_data = drawer.get_dot_graph().create_svg()
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        # pydot or graphviz not available; skip SVG capture gracefully
        # RuntimeError: "FXGraphDrawer requires the pydot package"
        # FileNotFoundError: graphviz `dot` binary not found
        # ImportError: pydot module not installed
        print(f"[Viz] Skipped SVG generation: {exc}")
        return
    except Exception:
        # Catch any other visualization errors silently
        return

    filepath = os.path.join(artifact_dir, filename)
    with open(filepath, "wb") as f:
        f.write(svg_data)

    print(f"[AOT MockBackend] Saved SVG visualization: {filepath}")
    
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

    print(f"[AOT MockBackend] Saved pre-AOT FX graph: {filename}")


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

    print(f"[AOT MockBackend] Saved AOT Forward graph: {filename}")

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

    print(f"[AOT MockBackend] Saved AOT Backward graph: {filename}")

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

    print(f"[AOT MockBackend] Saved graph statistics: {filename}")

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
    
    print(f"[AOT MockBackend] Saved artifact to {filename}")

_INDUCTOR_DEBUG_ENABLED = False

def _enable_inductor_debug():
    """
    Force TorchInductor to write debug artifacts to disk, in a stable repo-local folder.
    Must be called BEFORE the first compile_fx() that you want debug for.
    """
    global _INDUCTOR_DEBUG_ENABLED
    if _INDUCTOR_DEBUG_ENABLED:
        return

    debug_root = os.path.abspath("torch_compile_debug")
    os.makedirs(debug_root, exist_ok=True)

    # Force a stable debug dir inside the repo (works on many torch versions).
    # If your torch ignores TORCH_COMPILE_DEBUG_DIR, it will still write elsewhere,
    # and dump_inductor_artifacts() will fall back to searching.
    os.environ.setdefault("TORCH_COMPILE_DEBUG_DIR", debug_root)

    # Inductor debug + save files (best effort across torch versions)
    os.environ.setdefault("TORCHINDUCTOR_DEBUG", "1")
    os.environ.setdefault("TORCHINDUCTOR_SAVE_DEBUG_FILES", "1")

    # Useful logs (don’t require, but helps)
    os.environ.setdefault("TORCH_LOGS", "+inductor")

    # Best-effort config toggles (safe if fields don’t exist)
    try:
        import torch._inductor.config as inductor_config
        inductor_config.debug = True
        if hasattr(inductor_config, "save_debug_files"):
            inductor_config.save_debug_files = True
    except Exception:
        pass

    _INDUCTOR_DEBUG_ENABLED = True

def dump_inductor_artifacts(tag: str = "inductor"):
    """
    Copy TorchInductor-generated debug files into:
      debug_artifacts/inductor_ir/<ts>/...
      debug_artifacts/inductor_kernels/<ts>/...

    Searches common locations:
      ./torch_compile_debug/run_*/torchinductor/
      $TORCH_COMPILE_DEBUG_DIR/run_*/torchinductor/
      /tmp/torchinductor_*   (fallback)
      /var/tmp/torchinductor_* (fallback)
    """
    base_dir = "debug_artifacts"
    ts = int(time.time())

    ir_dir = os.path.join(base_dir, "inductor_ir", str(ts))
    kernel_dir = os.path.join(base_dir, "inductor_kernels", str(ts))
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(kernel_dir, exist_ok=True)

    # Extensions we treat as "lowered IR" vs "generated kernels/code"
    IR_EXTS = {".ttir", ".llir", ".mlir"}
    KERNEL_EXTS = {".py", ".cpp", ".cu", ".c", ".h", ".s"}

    # Candidate roots
    candidates = []

    # 1) Repo-local torch_compile_debug (most common with torch>=2.1)
    candidates += glob.glob(os.path.join(os.getcwd(), "torch_compile_debug", "run_*", "torchinductor"))

    # 2) Respect env override if present
    debug_dir = os.environ.get("TORCH_COMPILE_DEBUG_DIR")
    if debug_dir:
        candidates += glob.glob(os.path.join(debug_dir, "run_*", "torchinductor"))

    # 3) Fallback older/alternate locations
    for pat in ("/tmp/torchinductor_*", "/var/tmp/torchinductor_*"):
        candidates += glob.glob(pat)

    # Keep only directories, newest first
    candidates = [p for p in candidates if os.path.isdir(p)]
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    if not candidates:
        # print("[AOT MockBackend] No Inductor debug directories found (nothing to copy)")
        return

    copied_ir = 0
    copied_kernels = 0

    # Copy from the newest candidate that contains anything interesting
    for root in candidates[:5]:
        # Walk all files
        for src in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if os.path.isdir(src):
                continue

            ext = os.path.splitext(src)[1]
            rel = os.path.relpath(src, root)

            try:
                if ext in IR_EXTS:
                    dst = os.path.join(ir_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    copied_ir += 1
                elif ext in KERNEL_EXTS:
                    dst = os.path.join(kernel_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    copied_kernels += 1
            except Exception:
                pass

        # If we copied anything from this root, stop
        if copied_ir or copied_kernels:
            break

    if copied_ir:
        print(f"[AOT MockBackend] Saved Inductor lowered IR ({copied_ir} files) → {ir_dir}")
    if copied_kernels:
        print(f"[AOT MockBackend] Saved Inductor kernels ({copied_kernels} files) → {kernel_dir}")

    if not copied_ir and not copied_kernels:
        print("[AOT MockBackend] Found Inductor debug dir but no matching IR/kernel files to copy")

def _generate_repro_aot(gm, example_inputs, error_message):
    """
    Best-effort repro generation for AOT graphs.

    NOTE:
    Some AOT graphs (especially backward graphs) may contain scalar-extraction
    or control-flow artifacts (e.g. aten._local_scalar_dense) that cannot be
    serialized or executed standalone. In these cases, repro generation is
    intentionally skipped and the backend falls back to full graph artifacts.
    """
    repro_dir = os.path.join("debug_artifacts", "repros")
    os.makedirs(repro_dir, exist_ok=True)

    timestamp = int(time.time())
    graph_hash = hashlib.md5(str(gm.graph).encode()).hexdigest()[:8]
    repro_path = os.path.join(repro_dir, f"repro_{timestamp}_{graph_hash}.py")

    try:
        minifier = Minifier(gm, tuple(example_inputs), RuntimeError(error_message))
        minifier.minify()
        minifier.generate_repro_script(repro_path)

        print(f"[AOT MockBackend] Saved repro script: {repro_path}")
        return repro_path

    except Exception as exc:
        unminifiable_op_hints = (
            "aten._local_scalar_dense",
            "local_scalar",
        )

        if any(hint in str(exc) for hint in unminifiable_op_hints):
            print(
                "[AOT MockBackend] Repro generation skipped: "
                "AOT graph contains scalar or non-tensor ops "
                "(e.g. aten._local_scalar_dense) that cannot be "
                "minified or serialized safely."
            )
        else:
            print(
                "[AOT MockBackend] Repro generation failed unexpectedly: "
                f"{exc}"
            )

        # Intentionally return None: repro is best-effort only
        return None