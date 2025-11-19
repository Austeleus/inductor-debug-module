import torch
import torch._dynamo
from typing import List, Dict, Any

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
        
        # Extract Break Reasons
        for break_reason in explanation.break_reasons:
            report["break_reasons"].append(str(break_reason))
            
        # Extract Graphs
        for i, graph in enumerate(explanation.graphs):
            graph_info = {
                "id": i,
                "guards": []
            }
            report["graphs"].append(graph_info)

        # Extract Guards (Global list in this version)
        if hasattr(explanation, 'out_guards'):
            # In this version, out_guards might be a list of Guard objects
            # We'll add them to the first graph for visualization purposes
            if report["graphs"]:
                for guard in explanation.out_guards:
                    report["graphs"][0]["guards"].append(str(guard))
            
        return report

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
            print(f"Guards: {len(graph['guards'])}")
            for guard in graph['guards'][:10]: # Limit to first 10
                print(f"  - {guard}")
            if len(graph['guards']) > 10:
                print(f"  ... and {len(graph['guards']) - 10} more")
