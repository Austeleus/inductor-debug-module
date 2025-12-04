import os
import torch
import torch.fx
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.compile_fx import compile_fx
from functorch.compile import make_boxed_func

from ..constraints.registry import DEFAULT_CONSTRAINTS
from .aot_capture import save_pre_aot_artifact, save_post_aot_forward, save_post_aot_backward, save_graph_statistics

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
    
def compile_graph_with_aot(gm: torch.fx.GraphModule, example_inputs, constraints, strict):
    """
    The single entry point used by mock_backend.
    Handles:
        • saving pre-AOT artifacts
        • running fw/bw AOTAutograd passes
        • constraint checking
        • compile_fx lowering
    """
    # Save pre-AOT FX graph before AOTAutograd touches it
    save_pre_aot_artifact(gm)

    # Instantiate constraint checker
    checker = ConstraintChecker(constraints=constraints, strict=strict)

    # Define AOT forward compiler hook
    def fw_compiler(aot_gm, aot_inputs):
        save_post_aot_forward(aot_gm)
        save_graph_statistics(aot_gm, "fwd")

        checker.check_graph(aot_gm)
        
        return make_boxed_func(compile_fx(aot_gm, aot_inputs))
    
    # Define AOT backward compiler hook
    def bw_compiler(aot_gm, aot_inputs):
        save_post_aot_backward(aot_gm)
        save_graph_statistics(aot_gm, "bwd")

        checker.check_graph(aot_gm)

        return make_boxed_func(compile_fx(aot_gm, aot_inputs))

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )