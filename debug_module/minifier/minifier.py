import base64
import os
import pickle
import textwrap
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
import torch.fx
from torch.fx.node import map_arg

from ..constraints.registry import (
    DtypeConstraint,
    LayoutConstraint,
    MemoryConstraint,
    ShapeConstraint,
    UnsupportedOpsConstraint,
)


class Minifier:
    """
    Utility that reduces a failing FX graph down to the smallest subgraph that still
    violates backend constraints and then emits a self-contained reproduction script.
    """

    def __init__(self, gm, example_inputs, error):
        self.gm = gm
        self.inputs = example_inputs
        self.error = error
        self.minified_gm: Optional[torch.fx.GraphModule] = None
        self.failing_node: Optional[torch.fx.Node] = None
        self.failure_message: str = str(error) if error is not None else ""

    def minify(self) -> torch.fx.GraphModule:
        """Reduce graph to minimal failing case using a dependency-aware binary search."""
        data_nodes = [n for n in self.gm.graph.nodes if n.op not in ("placeholder", "output")]
        if not data_nodes:
            self.minified_gm = self.gm
            return self.gm

        failing_idx, failing_node, failure_message = self._binary_search_failure(data_nodes)
        if failing_idx is None or failing_node is None:
            # Fallback to original graph if we cannot isolate a failing prefix.
            self.minified_gm = self.gm
            return self.gm

        needed_nodes = self._collect_required_nodes({failing_node})
        minified = self._build_minified_graph(needed_nodes)

        self.minified_gm = minified
        self.failing_node = failing_node
        if failure_message:
            self.failure_message = failure_message
        return minified

    def generate_repro_script(self, output_path: str) -> str:
        """Generate standalone Python script for reproducing the failure."""
        gm_to_use = self.minified_gm or self.gm
        serialized_gm = self._serialize_graph_module(gm_to_use)
        example_inputs_literal = self._format_value(self.inputs)
        comment_block, runtime_text = self._format_failure_comment()

        script = self._build_script(serialized_gm, example_inputs_literal, comment_block, runtime_text)

        script_dir = os.path.dirname(output_path)
        if script_dir:
            os.makedirs(script_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script)
        return output_path

    # -------------------------------------------------------------------------
    # Minification helpers
    # -------------------------------------------------------------------------
	# O(n log n) complexity
    def _binary_search_failure(
        self, nodes: Sequence[torch.fx.Node]
    ) -> Tuple[Optional[int], Optional[torch.fx.Node], Optional[str]]:
        left, right = 1, len(nodes)
        best_idx: Optional[int] = None
        failing_node: Optional[torch.fx.Node] = None
        failure_message: Optional[str] = None

        while left <= right:
            mid = (left + right) // 2
            fails, info = self._prefix_fails(nodes, mid)
            if fails:
                best_idx = mid
                failing_node = info.get("node") if info else None
                failure_message = info.get("message") if info else None
                right = mid - 1
            else:
                left = mid + 1

        return best_idx, failing_node, failure_message

    def _prefix_fails(
        self, nodes: Sequence[torch.fx.Node], count: int
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        constraints = self._constraint_instances()
        for node in nodes[:count]:
            for constraint in constraints:
                if not constraint.check(node):
                    return True, {"node": node, "message": constraint.message(node)}
        return False, None

    def _constraint_instances(self):
        alignment = int(os.environ.get("MOCK_ALIGNMENT", "1"))
        max_memory = int(os.environ.get("MOCK_MAX_MEMORY", str(16 * 1024**3)))
        return [
            UnsupportedOpsConstraint(),
            DtypeConstraint(),
            LayoutConstraint(),
            ShapeConstraint(alignment=alignment),
            MemoryConstraint(max_memory_bytes=max_memory),
        ]

    def _collect_required_nodes(self, seeds: Set[torch.fx.Node]) -> Set[torch.fx.Node]:
        required: Set[torch.fx.Node] = set()

        def visit(node: torch.fx.Node):
            if node in required:
                return
            required.add(node)

            def _recurse(arg):
                if isinstance(arg, torch.fx.Node):
                    visit(arg)

            map_arg(node.args, _recurse)
            map_arg(node.kwargs, _recurse)

        for node in seeds:
            visit(node)
        return required

    def _build_minified_graph(self, required_nodes: Set[torch.fx.Node]) -> torch.fx.GraphModule:
        new_graph = torch.fx.Graph()
        env: Dict[torch.fx.Node, torch.fx.Node] = {} # map origianl FX node->node in minified graph
        output_node: Optional[torch.fx.Node] = None

        for node in self.gm.graph.nodes:
            if node.op == "output":
                output_node = node
                continue
            if node not in required_nodes and node.op not in ("placeholder", "get_attr"):
                continue
            if node.op == "placeholder":
                if node not in required_nodes:
                    continue
                new_node = new_graph.placeholder(node.target)
            elif node.op == "get_attr":
                if node not in required_nodes:
                    continue
                new_node = new_graph.get_attr(node.target)
            else:
                new_node = new_graph.node_copy(node, lambda n: env[n])
            new_node.meta = dict(node.meta)
            env[node] = new_node

        if not env:
            raise RuntimeError("Failed to construct minified graph.")

        new_output = None
        if output_node is not None:
            try:
                new_output = map_arg(output_node.args, lambda n: env[n])
            except KeyError:
                new_output = None

        if new_output is None:
            data_nodes = [env[n] for n in required_nodes if n in env and n.op not in ("placeholder", "get_attr")]
            new_output = data_nodes[0] if len(data_nodes) == 1 else tuple(data_nodes)

        new_graph.output(new_output)
        return torch.fx.GraphModule(self.gm, new_graph, class_name="MinifiedGraph")

    # -------------------------------------------------------------------------
    # Script generation helpers
    # -------------------------------------------------------------------------

    def _serialize_graph_module(self, gm: torch.fx.GraphModule) -> str:
        payload = pickle.dumps(gm)
        return base64.b64encode(payload).decode("ascii")

    def _format_failure_comment(self) -> Tuple[str, str]:
        message = self.failure_message or "Unknown error"
        suspect = f"Suspect node: {self.failing_node.name}" if self.failing_node else "Suspect node: <unresolved>"
        comment_lines = [
            "# Failure context:",
            f"#   {message}",
            f"#   {suspect}",
        ]
        runtime_lines = [
            "Failure context:",
            f"  {message}",
            f"  {suspect}",
        ]
        return "\n".join(comment_lines), "\n".join(runtime_lines)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            dtype_name = str(tensor.dtype).split(".")[-1]
            data_repr = repr(tensor.tolist())
            result = f"torch.tensor({data_repr}, dtype=torch.{dtype_name})"
            if value.requires_grad:
                result = f"{result}.requires_grad_(True)"
            device = value.device
            if device.type != "cpu":
                result = f"{result}.to('{device.type}')"
            return result
        if isinstance(value, (list, tuple)):
            inner = ", ".join(self._format_value(v) for v in value)
            if isinstance(value, tuple):
                if len(value) == 1:
                    inner += ","
                return f"({inner})"
            return f"[{inner}]"
        if isinstance(value, dict):
            items = ", ".join(f"{repr(k)}: {self._format_value(v)}" for k, v in value.items())
            return f"{{{items}}}"
        if isinstance(value, (int, float, bool)):
            return repr(value)
        if value is None:
            return "None"
        return repr(value)

    def _build_script(
        self, serialized_gm: str, inputs_literal: str, comment_block: str, runtime_text: str
    ) -> str:
        inputs_section = textwrap.dedent(
            f"""\
            def get_example_inputs():
                return {inputs_literal}
            """
        )

        script = f"""#!/usr/bin/env python3
\"\"\"Auto-generated TorchInductor reproduction script.\"\"\"

import base64
import pickle
import torch

from debug_module.backend.compiler import mock_compile

SERIALIZED_GRAPH = \"\"\"{serialized_gm}\"\"\" 

{comment_block}


def load_module():
    data = base64.b64decode(SERIALIZED_GRAPH.encode("utf-8"))
    return pickle.loads(data)


{inputs_section}

def run_repro():
    module = load_module()
    example_inputs = get_example_inputs()
    if isinstance(example_inputs, dict):
        args, kwargs = (), example_inputs
    elif isinstance(example_inputs, (list, tuple)):
        args, kwargs = example_inputs, {{}}
    else:
        args, kwargs = (example_inputs,), {{}}

    if isinstance(example_inputs, dict):
        tensor_inputs = list(example_inputs.values())
    elif isinstance(example_inputs, (list, tuple)):
        tensor_inputs = list(example_inputs)
    else:
        tensor_inputs = [example_inputs]

    print(\"\"\"{runtime_text}\"\"\")
    mock_compile(module, tensor_inputs)


if __name__ == "__main__":
{textwrap.indent(comment_block, '    ')}
    try:
        run_repro()
    except Exception as exc:
        print("Reproduction run raised:", exc)
        raise
"""
        return script
