#!/usr/bin/env python3
"""
Phase 1: AOTAutograd + TorchInductor Integration Test Suite
===========================================================

Validates:

  ✔ FX pre-AOT graph capture
  ✔ AOT Forward graph capture
  ✔ AOT Backward graph capture
  ✔ SVG visualizations created
  ✔ Graph statistics saved
  ✔ torch.compile executes correctly
  ✔ Tested on: simple, module, dynamic-shape-ish models
  ✔ CPU-only compatibility

Run:
    python test_aotbackend.py
"""

import os
import sys
import time
import glob
import torch

# Add parent directory to imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.aot_backend.mock import mock_backend


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
class C:
    BOLD = "\033[1m"
    RED  = "\033[91m"
    GRN  = "\033[92m"
    YLW  = "\033[93m"
    BLU  = "\033[94m"
    END  = "\033[0m"


def banner(title):
    print(f"\n{C.BLU}{C.BOLD}{'='*70}\n{title:^70}\n{'='*70}{C.END}\n")


def ok(msg):
    print(f"{C.GRN}[PASS]{C.END} {msg}")


def fail(msg):
    print(f"{C.RED}[FAIL]{C.END} {msg}")
    raise SystemExit(1)


def info(msg):
    print(f"{C.YLW}[INFO]{C.END} {msg}")


# ---------------------------------------------------------------------------
# File-check helper
# ---------------------------------------------------------------------------
def expect(pattern, count, label):
    files = glob.glob(pattern)
    if len(files) >= count:
        ok(f"{label} (found {len(files)})")
    else:
        fail(f"Missing {label}: expected ≥ {count}, found {len(files)}")


# ---------------------------------------------------------------------------
# Test 1: baseline AOT compilation
# ---------------------------------------------------------------------------
def test_basic_aot():
    banner("TEST 1 — Basic AOT Autograd Integration")

    # Reset debug directory
    if os.path.exists("debug_artifacts"):
        import shutil
        shutil.rmtree("debug_artifacts")

    def model(x, y):
        return (x + y).sin().relu()

    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(4, 4, requires_grad=True)

    info("Compiling simple model under AOT backend...")
    compiled = torch.compile(model, backend=mock_backend)
    out = compiled(x, y)
    out.sum().backward()

    # Check artifacts
    expect("debug_artifacts/fx_graphs/fx_pre_*.txt",   1, "FX Pre-AOT Graph")
    expect("debug_artifacts/aot_graphs/aot_fwd_*.txt", 1, "AOT Forward Graph")
    expect("debug_artifacts/aot_graphs/aot_bwd_*.txt", 1, "AOT Backward Graph")
    expect("debug_artifacts/visualizations/*.svg",     3, "SVG Graph Visualizations")
    expect("debug_artifacts/statistics/stats_*.json",  2, "Graph Statistics")


# ---------------------------------------------------------------------------
# Test 2: nn.Module with parameters
# ---------------------------------------------------------------------------
def test_module_model():
    banner("TEST 2 — nn.Module Compilation")

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x):
            return torch.sin(self.lin(x))

    m = MyModule()
    x = torch.randn(2, 8)

    info("Compiling nn.Module through AOT backend...")
    compiled = torch.compile(m, backend=mock_backend)
    out = compiled(x)

    if out.shape == (2, 8):
        ok("Module executed correctly")
    else:
        fail("Incorrect module output shape")


# ---------------------------------------------------------------------------
# Test 3: Dynamic-shape simulation
# ---------------------------------------------------------------------------
def test_dynamic_proxy():
    banner("TEST 3 — Dynamic-ish Shape Behavior")

    def model(x):
        return (x * 2).cos()

    compiled = torch.compile(model, backend=mock_backend)

    for shape in [(3,3), (10,10), (4,7)]:
        info(f"Running shape {shape}...")
        x = torch.randn(*shape)
        out = compiled(x)

        if out.shape == shape:
            ok(f"Run succeeded for shape {shape}")
        else:
            fail(f"Unexpected output shape for {shape}")


# ---------------------------------------------------------------------------
# Test 4: TorchInductor debug directory on GPU
# ---------------------------------------------------------------------------
def test_inductor():
    banner("TEST 4 — TorchInductor IR (GPU Only)")

    if torch.version.cuda is None:
        info("CPU-only environment — Inductor IR test skipped")
        return

    def model(x):
        return (x * 3).relu()

    x = torch.randn(32, 32, device="cuda")

    compiled = torch.compile(model, backend=mock_backend)
    compiled(x)

    dirs = glob.glob("torch_compile_debug/*")
    if dirs:
        ok("Inductor IR generated")
    else:
        fail("Inductor did NOT generate IR!")


# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    banner("PHASE 1 — AOT Autograd / Inductor Integration Test Suite")

    start = time.time()

    test_basic_aot()
    test_module_model()
    test_dynamic_proxy()
    test_inductor()

    banner("ALL TESTS COMPLETED")
    print(f"{C.GRN}{C.BOLD}Total time: {time.time() - start:.2f}s{C.END}")