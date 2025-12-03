import torch
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.compile_fx import compile_fx
from functorch.compile import make_boxed_func
from .aot_capture import save_pre_aot_artifact, save_post_aot_forward, save_post_aot_backward, save_graph_statistics

def mock_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Entry point used by torch.compile(backend=mock_backend).

    Responsibilities:
    - Save pre-AOT FX graph
    - Run AOTAutograd, capturing post-AOT graphs
    - Capture post-AOT graphs
    - Capture statistics
    - Then run mock_compile() which does constraint checking only
    - Return eager execution
    """
    # 1. Save pre-AOT FX graph before AOTAutograd touches it
    save_pre_aot_artifact(gm)

    # 2. Define AOT forward compiler hook
    def fw_compiler(aot_gm, aot_inputs):
        save_post_aot_forward(aot_gm)
        save_graph_statistics(aot_gm, "fwd")
        
        return make_boxed_func(compile_fx(aot_gm, aot_inputs))
    
    # 3. Define AOT backward compiler hook
    def bw_compiler(aot_gm, aot_inputs):
        save_post_aot_backward(aot_gm)
        save_graph_statistics(aot_gm, "bwd")

        return make_boxed_func(compile_fx(aot_gm, aot_inputs))

    # 4. Compile using AOTAutograd (but return eager execution)
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )