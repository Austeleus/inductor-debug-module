"""
Test Suite for AOTAutograd + Mock Backend Constraint System
===========================================================

Validates:

  ✔ ConstraintChecker applies constraints to AOT forward & backward graphs
  ✔ aot_mock_backend(strict=X, constraints=[...]) works properly
  ✔ DtypeConstraint
  ✔ ShapeConstraint
  ✔ UnsupportedOpsConstraint
  ✔ MemoryConstraint
  ✔ Strict mode → raises RuntimeError
  ✔ Warning mode → prints but runs
  ✔ Artifact generation still occurs

Run:
    python test_constraints_backend.py
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.aot_backend.mock import aot_mock_backend
from debug_module.constraints.registry import (
    DtypeConstraint,
    ShapeConstraint,
    UnsupportedOpsConstraint,
    MemoryConstraint,
    resolve_aten,
)

# -----------------------------------------------------------------------------
# Pretty Colors
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def reset_artifacts():
    import shutil
    if os.path.exists("debug_artifacts"):
        shutil.rmtree("debug_artifacts")


def _compile_and_run(backend, model, *inputs):
    """
    Helper: Compiles using torch.compile(model, backend=backend) and checks success.
    Note: Named with underscore prefix to avoid pytest auto-discovery.
    """
    try:
        compiled = torch.compile(model, backend=backend)
        out = compiled(*inputs)
        out.sum().backward()
        return True
    except Exception as e:
        print(e)
        return False


# -----------------------------------------------------------------------------
# TEST 1 — DTYPE CONSTRAINT
# -----------------------------------------------------------------------------
def test_dtype_constraint_strict_fail():
    banner("TEST 1 — DtypeConstraint strict fail")

    reset_artifacts()

    # Only allow float32
    constraints = [DtypeConstraint({torch.float32})]
    backend = aot_mock_backend(strict=True, constraints=constraints)

    def model(x):
        return x.sin().cos()

    x = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)

    info("Running model with float64 input (should fail)...")

    try:
        compiled = torch.compile(model, backend=backend)
        compiled(x)
        fail("DtypeConstraint should fail in strict mode")
    except RuntimeError:
        ok("Strict DtypeConstraint correctly rejected float64")
    except Exception as e:
        fail(f"Unexpected error: {e}")


def test_dtype_constraint_warning():
    banner("TEST 2 — DtypeConstraint warning")

    reset_artifacts()

    constraints = [DtypeConstraint({torch.float32})]
    backend = aot_mock_backend(strict=False, constraints=constraints)

    def model(x):
        return x * 2

    x = torch.randn(2, 2, dtype=torch.float64)

    info("Running model with float64 in warning mode (should print warning)...")
    try:
        compiled = torch.compile(model, backend=backend)
        compiled(x)
        ok("Warning mode allowed float64 with warnings")
    except Exception as e:
        fail(f"Warning mode should NOT fail: {e}")


# -----------------------------------------------------------------------------
# TEST 3 — SHAPE CONSTRAINT
# -----------------------------------------------------------------------------
def test_shape_constraint_strict():
    banner("TEST 3 — ShapeConstraint strict")

    reset_artifacts()

    constraints = [ShapeConstraint(alignment=8)]
    backend = aot_mock_backend(strict=True, constraints=constraints)

    def model(x):
        return x + 1

    x = torch.randn(3, 3)  # shape not divisible by 8

    info("Running shape-misaligned input (should fail)...")
    try:
        compiled = torch.compile(model, backend=backend)
        compiled(x)
        fail("ShapeConstraint strict mode should fail")
    except RuntimeError:
        ok("ShapeConstraint strict mode caught alignment violation")


def test_shape_constraint_warning():
    banner("TEST 4 — ShapeConstraint warning")

    reset_artifacts()

    constraints = [ShapeConstraint(alignment=8)]
    backend = aot_mock_backend(strict=False, constraints=constraints)

    def model(x):
        return x.relu()

    x = torch.randn(5, 5)

    info("Running misaligned input in warning mode...")
    try:
        compiled = torch.compile(model, backend=backend, dynamic=True)
        compiled(x)
        ok("ShapeConstraint warning allowed execution")
    except Exception as e:
        fail(f"Warning mode should allow execution: {e}")


# -----------------------------------------------------------------------------
# TEST 5 — UNSUPPORTED OP CONSTRAINT
# -----------------------------------------------------------------------------
def test_unsupported_op():
    banner("TEST 5 — UnsupportedOpsConstraint")

    reset_artifacts()

    sin_op = resolve_aten(["sin"])

    constraints = [UnsupportedOpsConstraint(sin_op)]
    backend = aot_mock_backend(strict=True, constraints=constraints)

    def model(x):
        return torch.sin(x)   # forbidden op

    x = torch.randn(4, 4)

    info("Running model with unsupported operator aten.sin (should fail)...")
    try:
        compiled = torch.compile(model, backend=backend)
        compiled(x)
        fail("Unsupported operator should fail in strict mode")
    except RuntimeError:
        ok("UnsupportedOpsConstraint correctly rejected aten.sin")


# -----------------------------------------------------------------------------
# TEST 6 — MEMORY CONSTRAINT
# -----------------------------------------------------------------------------
def test_memory_constraint():
    banner("TEST 6 — MemoryConstraint")

    reset_artifacts()

    constraints = [MemoryConstraint(max_memory_bytes=1024)]  # 1 KB
    backend = aot_mock_backend(strict=True, constraints=constraints)

    def model(x):
        return x * 2  # output > 1 KB

    x = torch.randn(128, 128)  # ~64 KB tensor

    info("Running model that exceeds memory limit (should fail)...")
    try:
        compiled = torch.compile(model, backend=backend)
        compiled(x)
        fail("MemoryConstraint strict should fail")
    except RuntimeError:
        ok("MemoryConstraint strict correctly rejected large tensors")


# -----------------------------------------------------------------------------
# TEST 7 — MULTIPLE CONSTRAINTS
# -----------------------------------------------------------------------------
def test_multiple_constraints():
    banner("TEST 7 — Multiple Constraints Working Together")

    reset_artifacts()

    constraints = [
        DtypeConstraint({torch.float32}),
        ShapeConstraint(alignment=4),
        UnsupportedOpsConstraint(resolve_aten(["cos"])),
    ]

    backend = aot_mock_backend(strict=True, constraints=constraints)

    def model(x):
        return torch.cos(x + 1.0)  # forbidden + misaligned shape + dtype mismatch possible

    x = torch.randn(3, 3, dtype=torch.float64)

    info("Running model with multiple constraint violations...")

    try:
        compiled = torch.compile(model, backend=backend)
        compiled(x)
        fail("Multiple constraints should cause strict failure")
    except RuntimeError:
        ok("Multiple constraints correctly triggered strict failure")

# -----------------------------------------------------------------------------
# TEST 8 — FORCE INDUCTOR IR + KERNEL GENERATION
# -----------------------------------------------------------------------------
def test_inductor_ir_and_kernels_dump():
    banner("TEST 8 — Inductor Lowered IR + Kernel Dump")

    reset_artifacts()

    backend = aot_mock_backend(strict=False, constraints=[])

    def model(x, y):
        # forces matmul + elementwise + reduction
        for _ in range(3):
            x = x @ y
            x = torch.relu(x)
        return x.sum()

    # Big enough to avoid eager fallback
    x = torch.randn(512, 512, requires_grad=True)
    y = torch.randn(512, 512, requires_grad=True)

    info("Compiling with Inductor and executing kernels...")

    compiled = torch.compile(
        model,
        backend=backend,
        mode="max-autotune",   # IMPORTANT
    )

    out = compiled(x, y)
    out.backward()

    # Verify artifacts exist
    ir_root = "debug_artifacts/inductor_ir"
    kernel_root = "debug_artifacts/inductor_kernels"

    def has_any_ext(root, exts):
        if not os.path.isdir(root):
            return False
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if os.path.splitext(fn)[1] in exts:
                    return True
        return False

    has_ir = has_any_ext(ir_root, {".ttir", ".llir", ".mlir"})
    has_kernels = has_any_ext(kernel_root, {".py", ".cpp", ".cu", ".c", ".h", ".s"})

    if not (has_ir or has_kernels):
        fail("No Inductor IR or kernel artifacts found")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    banner("MOCK BACKEND CONSTRAINT TEST SUITE")

    tests = [
        ("Dtype strict fail",       test_dtype_constraint_strict_fail),
        ("Dtype warning",           test_dtype_constraint_warning),
        ("Shape strict",            test_shape_constraint_strict),
        ("Shape warning",           test_shape_constraint_warning),
        ("Unsupported op",          test_unsupported_op),
        ("Memory constraint",       test_memory_constraint),
        ("Multiple constraints",    test_multiple_constraints),
        ("Inductor IR + kernels",   test_inductor_ir_and_kernels_dump),
    ]

    passed = 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(e)

    print(f"\n{C.BOLD}Passed {passed}/{len(tests)} tests{C.END}")

    if passed == len(tests):
        print(f"{C.GRN}{C.BOLD}ALL TESTS PASSED!{C.END}")
        return 0
    else:
        print(f"{C.RED}{C.BOLD}SOME TESTS FAILED{C.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())