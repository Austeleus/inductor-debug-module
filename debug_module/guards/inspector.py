import torch
import torch._dynamo
from typing import Dict, Any, Iterable


def _stringify_value(value: Any) -> Any:
    """Convert guard attributes into JSON friendly structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _stringify_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_stringify_value(v) for v in value]
    return str(value)


def _guard_to_dict(guard: Any) -> Dict[str, Any]:
    """Capture structured fields from a Guard object, falling back to repr."""
    info: Dict[str, Any] = {"text": str(guard)}
    guard_dict = {}

    if hasattr(guard, "_asdict"):
        guard_dict = guard._asdict()
    elif hasattr(guard, "__dict__"):
        guard_dict = {k: v for k, v in guard.__dict__.items() if not k.startswith("_")}

    if guard_dict:
        for key, val in guard_dict.items():
            info[key] = _stringify_value(val)

    # Some guard implementations expose properties instead of dict fields.
    for attr in ("name", "source", "expr"):
        if attr not in info and hasattr(guard, attr):
            info[attr] = _stringify_value(getattr(guard, attr))
    return info


class GuardInspector:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def inspect(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs torch._dynamo.explain and extracts guard information.
        """
        # Use the new API style: explain(f)(*args, **kwargs)
        explanation = torch._dynamo.explain(self.model)(**inputs)

        report = {
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": [],
            "graphs": []
        }

        # Extract break reasons
        for break_reason in getattr(explanation, "break_reasons", []):
            report["break_reasons"].append(str(break_reason))

        # Extract graphs
        graph_lookup = {}
        for index, graph in enumerate(getattr(explanation, "graphs", [])):
            graph_id = getattr(graph, "name", f"graph_{index}")
            graph_info = {
                "id": graph_id,
                "index": index,
                "guards": []
            }
            graph_lookup[graph_id] = index
            report["graphs"].append(graph_info)

        self._attach_guards(report, explanation, graph_lookup)
        return report

    def _attach_guards(self, report: Dict[str, Any], explanation: Any, graph_lookup: Dict[str, int]) -> None:
        """Populate guard info per-graph using whatever metadata Dynamo exposes."""
        graphs = report["graphs"]

        def assign_guards(idx: int, guards: Iterable[Any]) -> None:
            if 0 <= idx < len(graphs):
                graphs[idx]["guards"] = [_guard_to_dict(g) for g in guards]

        assigned = False
        graph_guards = getattr(explanation, "graph_guards", None)
        if graph_guards:
            assigned = True
            if isinstance(graph_guards, dict):
                for key, guards in graph_guards.items():
                    idx = None
                    if isinstance(key, int) and key < len(graphs):
                        idx = key
                    elif isinstance(key, str):
                        idx = graph_lookup.get(key)
                        if idx is None and key.isdigit():
                            parsed = int(key)
                            if parsed < len(graphs):
                                idx = parsed
                    if idx is not None:
                        assign_guards(idx, guards)
            else:
                for idx, guards in enumerate(graph_guards):
                    assign_guards(idx, guards)

        if not assigned:
            out_guards = getattr(explanation, "out_guards", None) or getattr(explanation, "guards", None)
            if out_guards and graphs:
                assign_guards(0, out_guards)

    def print_report(self, report: Dict[str, Any]):
        print(f"\n=== Guard Inspector Report ===")
        print(f"Total Graphs: {report['graph_count']}")
        print(f"Graph Breaks: {report['graph_break_count']}")

        if report['break_reasons']:
            print("\n--- Graph Breaks ---")
            for reason in report['break_reasons']:
                print(f"- {reason}")

        for graph in report['graphs']:
            print(f"\n--- Graph {graph['id']} ---")
            guard_count = len(graph['guards'])
            print(f"Guards: {guard_count}")
            for guard in graph['guards'][:10]:  # Limit to first 10
                print(f"  - {guard['text']}")
            if guard_count > 10:
                print(f"  ... and {guard_count - 10} more")
