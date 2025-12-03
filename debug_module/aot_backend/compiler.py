import os
import torch
import torch.fx

from debug_module.constraints.registry import DEFAULT_CONSTRAINTS

class ConstraintChecker:
    """
    Runs a list of constraints on an FX GraphModule and determines if
    the backend should accept or reject the graph.
    """

    def __init__(self, constraints=None, strict=None):
        self.constraints = constraints or DEFAULT_CONSTRAINTS

        # STRICT MODE FROM ENV
        env = os.getenv("MOCK_STRICT", "0").strip()
        if strict is None:
            self.strict = env == "1"
        else:
            self.strict = strict

        self.violations = []

    def check_graph(self, gm: torch.fx.GraphModule):
        """
        Run all constraints on all nodes.
        """
        for node in gm.graph.nodes:
            for constraint in self.constraints:
                if not constraint.check(node):
                    msg = constraint.message(node)
                    self.violations.append((node, msg))

        # STRICT MODE = FAIL
        if self.strict and self.violations:
            msgs = "\n".join([f"- {msg}" for (_, msg) in self.violations])
            raise RuntimeError(
                f"[MockBackend-STRICT] Constraint violations detected:\n{msgs}"
            )

        # WARNING MODE = PRINT ONLY
        if self.violations:
            print("[MockBackend-WARNING] Constraint violations:")
            for (_, msg) in self.violations:
                print("   ", msg)

        return True