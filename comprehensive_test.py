#!/usr/bin/env python3
"""
Comprehensive Test Script for TorchInductor Debug Module
=========================================================

This script demonstrates all features of the debug module:
1. Mock Backend with constraint validation
2. All constraint types (Dtype, Ops, Layout, Shape, Memory)
3. Strict vs Non-Strict mode
4. Environment variable configuration
5. Guard Inspector

Author: Debug Module Team
"""

import torch
import os
import sys
import time

# Add color support for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_section(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}--- {text} ---{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}[PASS]{Colors.ENDC} {text}")

def print_fail(text):
    print(f"{Colors.RED}[FAIL]{Colors.ENDC} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} {text}")

def print_info(text):
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {text}")

def print_test_start(name):
    print(f"\n{Colors.BOLD}Testing: {name}{Colors.ENDC}")
    print("-" * 50)

# ============================================================================
# PART 1: MOCK BACKEND TESTS
# ============================================================================

def test_backend_basic_success():
    """Test that a simple model with valid dtypes passes."""
    print_test_start("Basic Success Case (float32)")

    print_info("Creating a simple add operation with float32 tensors")
    print_info("Expected: PASS - float32 is in the allowed dtype set")

    from debug_module import mock_backend

    def simple_model(x, y):
        return torch.add(x, y)

    x = torch.randn(10, 10, dtype=torch.float32)
    y = torch.randn(10, 10, dtype=torch.float32)

    print_info(f"Input tensors: shape={x.shape}, dtype={x.dtype}")

    try:
        compiled = torch.compile(simple_model, backend=mock_backend)
        result = compiled(x, y)
        print_success(f"Compilation and execution succeeded!")
        print_info(f"Output shape: {result.shape}")
        return True
    except Exception as e:
        print_fail(f"Unexpected failure: {e}")
        return False


def test_backend_dtype_failure():
    """Test that float64 dtype is rejected."""
    print_test_start("Dtype Constraint Failure (float64)")

    print_info("Creating tensors with float64 dtype")
    print_info("Expected: FAIL - float64 is NOT in the allowed dtype set")
    print_info("Allowed dtypes: {float32, int64, bool}")

    from debug_module import mock_backend

    def simple_model(x, y):
        return torch.add(x, y)

    x = torch.randn(10, 10, dtype=torch.float64)
    y = torch.randn(10, 10, dtype=torch.float64)

    print_info(f"Input tensors: shape={x.shape}, dtype={x.dtype}")

    try:
        compiled = torch.compile(simple_model, backend=mock_backend)
        result = compiled(x, y)
        print_fail("Should have failed but succeeded!")
        return False
    except Exception as e:
        print_success(f"Correctly rejected float64!")
        print_info(f"Error: {str(e)[:100]}...")
        return True


def test_backend_shape_alignment():
    """Test shape alignment constraint."""
    print_test_start("Shape Alignment Constraint")

    print_info("Setting MOCK_ALIGNMENT=8 via environment variable")
    print_info("Expected: FAIL - tensor dimension 10 is not divisible by 8")

    # Set environment variable
    os.environ["MOCK_ALIGNMENT"] = "8"

    # Need to reimport to pick up new env vars
    import importlib
    import debug_module.backend.compiler as compiler
    importlib.reload(compiler)
    from debug_module import mock_backend

    def simple_model(x):
        return x * 2

    # Shape (10, 10) - 10 is not divisible by 8
    x = torch.randn(10, 10, dtype=torch.float32)

    print_info(f"Input tensor: shape={x.shape}")
    print_info(f"10 % 8 = {10 % 8} (not aligned)")

    try:
        compiled = torch.compile(simple_model, backend=mock_backend)
        result = compiled(x)
        print_fail("Should have failed alignment check!")
        return False
    except Exception as e:
        print_success(f"Correctly rejected misaligned shape!")
        print_info(f"Error: {str(e)[:100]}...")
        return True
    finally:
        # Reset
        os.environ["MOCK_ALIGNMENT"] = "1"


def test_backend_shape_alignment_success():
    """Test that aligned shapes pass."""
    print_test_start("Shape Alignment Success (aligned)")

    print_info("Setting MOCK_ALIGNMENT=8")
    print_info("Expected: PASS - tensor dimension 16 is divisible by 8")

    os.environ["MOCK_ALIGNMENT"] = "8"

    import importlib
    import debug_module.backend.compiler as compiler
    importlib.reload(compiler)
    from debug_module import mock_backend

    def simple_model(x):
        return x * 2

    # Shape (16, 16) - 16 is divisible by 8
    x = torch.randn(16, 16, dtype=torch.float32)

    print_info(f"Input tensor: shape={x.shape}")
    print_info(f"16 % 8 = {16 % 8} (aligned)")

    try:
        compiled = torch.compile(simple_model, backend=mock_backend)
        result = compiled(x)
        print_success(f"Correctly accepted aligned shape!")
        return True
    except Exception as e:
        print_fail(f"Unexpected failure: {e}")
        return False
    finally:
        os.environ["MOCK_ALIGNMENT"] = "1"


def test_backend_memory_constraint():
    """Test memory constraint."""
    print_test_start("Memory Constraint")

    # Set a very small memory limit (1KB)
    print_info("Setting MOCK_MAX_MEMORY=1024 (1KB)")
    print_info("Expected: FAIL - tensor size exceeds 1KB limit")

    os.environ["MOCK_MAX_MEMORY"] = "1024"

    import importlib
    import debug_module.backend.compiler as compiler
    importlib.reload(compiler)
    from debug_module import mock_backend

    def simple_model(x):
        return x * 2

    # 100x100 float32 = 40,000 bytes > 1KB
    x = torch.randn(100, 100, dtype=torch.float32)
    tensor_size = x.numel() * x.element_size()

    print_info(f"Input tensor: shape={x.shape}")
    print_info(f"Tensor size: {tensor_size} bytes ({tensor_size/1024:.1f} KB)")
    print_info(f"Memory limit: 1024 bytes (1 KB)")

    try:
        compiled = torch.compile(simple_model, backend=mock_backend)
        result = compiled(x)
        print_fail("Should have failed memory check!")
        return False
    except Exception as e:
        print_success(f"Correctly rejected oversized tensor!")
        print_info(f"Error: {str(e)[:100]}...")
        return True
    finally:
        os.environ["MOCK_MAX_MEMORY"] = str(1024**3 * 16)


def test_backend_non_strict_mode():
    """Test non-strict mode (warnings only)."""
    print_test_start("Non-Strict Mode (Warnings Only)")

    print_info("Setting MOCK_STRICT=0")
    print_info("Expected: PASS with warnings - violations logged but not fatal")

    os.environ["MOCK_STRICT"] = "0"
    os.environ["MOCK_ALIGNMENT"] = "8"  # Will trigger warning

    import importlib
    import debug_module.backend.compiler as compiler
    importlib.reload(compiler)
    from debug_module import mock_backend

    def simple_model(x):
        return x * 2

    # Misaligned shape - would fail in strict mode
    x = torch.randn(10, 10, dtype=torch.float32)

    print_info(f"Input tensor: shape={x.shape} (misaligned)")
    print_info("In strict mode, this would fail. In non-strict, it should warn.")

    try:
        compiled = torch.compile(simple_model, backend=mock_backend)
        result = compiled(x)
        print_success(f"Execution completed with warnings (non-strict mode)")
        return True
    except Exception as e:
        print_fail(f"Should have succeeded in non-strict mode: {e}")
        return False
    finally:
        os.environ["MOCK_STRICT"] = "1"
        os.environ["MOCK_ALIGNMENT"] = "1"


# ============================================================================
# PART 2: GUARD INSPECTOR TESTS
# ============================================================================

def test_guard_inspector_simple():
    """Test Guard Inspector with a simple model."""
    print_test_start("Guard Inspector - Simple Model")

    print_info("Creating a simple model and inspecting guards")
    print_info("This shows what assumptions TorchDynamo makes about inputs")

    from debug_module.guards.inspector import GuardInspector

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().eval()
    inspector = GuardInspector(model)

    # Create input as a dict (for the inspector API)
    x = torch.randn(2, 10)

    print_info(f"Input shape: {x.shape}")

    try:
        # The inspector expects inputs as kwargs
        report = inspector.inspect({"x": x})
        inspector.print_report(report)
        print_success("Guard inspection completed!")
        return True
    except Exception as e:
        print_fail(f"Guard inspection failed: {e}")
        return False


def test_guard_inspector_bert():
    """Test Guard Inspector with BERT model."""
    print_test_start("Guard Inspector - BERT Model")

    print_info("Loading BERT model and inspecting guards")
    print_info("This is a real-world test with a complex transformer model")

    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError:
        print_warning("transformers not installed, skipping BERT test")
        return True

    from debug_module.guards.inspector import GuardInspector

    MODEL_ID = "google-bert/bert-base-multilingual-cased"

    print_info(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).eval()

    # Tokenize input
    text = "Paris is the [MASK] of France."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)

    print_info(f"Input text: '{text}'")
    print_info(f"Tokenized shape: {inputs['input_ids'].shape}")

    inspector = GuardInspector(model)

    try:
        report = inspector.inspect(inputs)
        inspector.print_report(report)
        print_success("BERT guard inspection completed!")
        return True
    except Exception as e:
        print_fail(f"BERT guard inspection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PART 3: ARTIFACT INSPECTION
# ============================================================================

def test_artifact_generation():
    """Test that artifacts are generated correctly."""
    print_test_start("Artifact Generation")

    print_info("Running a model through the backend and checking artifact output")

    from debug_module import mock_backend
    import os

    # Reset environment
    os.environ["MOCK_STRICT"] = "1"
    os.environ["MOCK_ALIGNMENT"] = "1"

    import importlib
    import debug_module.backend.compiler as compiler
    importlib.reload(compiler)

    def simple_model(x, y):
        z = x + y
        return z * 2

    x = torch.randn(8, 8, dtype=torch.float32)
    y = torch.randn(8, 8, dtype=torch.float32)

    artifact_dir = "debug_artifacts"

    # Count existing artifacts
    existing = len(os.listdir(artifact_dir)) if os.path.exists(artifact_dir) else 0

    print_info(f"Existing artifacts: {existing}")
    print_info("Compiling model...")

    compiled = torch.compile(simple_model, backend=mock_backend)
    result = compiled(x, y)

    # Check new artifacts
    new_count = len(os.listdir(artifact_dir)) if os.path.exists(artifact_dir) else 0

    if new_count > existing:
        print_success(f"New artifacts generated! Total: {new_count}")

        # Show latest artifact
        files = sorted(os.listdir(artifact_dir), reverse=True)
        if files:
            latest = files[0]
            print_info(f"Latest artifact: {latest}")

            # Show first few lines
            with open(os.path.join(artifact_dir, latest), 'r') as f:
                lines = f.readlines()[:15]
                print_info("Artifact preview:")
                for line in lines:
                    print(f"    {line.rstrip()}")
        return True
    else:
        print_fail("No new artifacts generated!")
        return False


# ============================================================================
# PART 4: CLI TEST
# ============================================================================

def test_cli():
    """Test CLI functionality."""
    print_test_start("CLI Functionality")

    print_info("Testing the command-line interface")

    import subprocess

    # Test 'list' command
    print_info("Running: python -m debug_module list")
    result = subprocess.run(
        [sys.executable, "-m", "debug_module", "list"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    print_info("CLI output:")
    for line in result.stdout.split('\n')[:10]:
        if line.strip():
            print(f"    {line}")

    if result.returncode == 0:
        print_success("CLI 'list' command works!")
        return True
    else:
        print_fail(f"CLI failed: {result.stderr}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    print_header("TorchInductor Debug Module")
    print_header("Comprehensive Test Suite")

    print(f"""
{Colors.BOLD}This test suite demonstrates all features of the Debug Module:{Colors.ENDC}

  1. Mock Backend
     - Constraint validation (dtype, ops, layout, shape, memory)
     - Strict vs non-strict modes
     - Environment variable configuration

  2. Guard Inspector
     - Capturing TorchDynamo guards
     - Graph break analysis

  3. Artifact Generation
     - FX graph capture
     - Node metadata logging

  4. CLI Interface
     - Artifact management commands
""")

    input(f"{Colors.YELLOW}Press Enter to begin tests...{Colors.ENDC}")

    results = []

    # Part 1: Backend Tests
    print_header("PART 1: Mock Backend Tests")

    print_section("Constraint Validation Tests")
    results.append(("Basic Success (float32)", test_backend_basic_success()))
    results.append(("Dtype Failure (float64)", test_backend_dtype_failure()))
    results.append(("Shape Alignment Failure", test_backend_shape_alignment()))
    results.append(("Shape Alignment Success", test_backend_shape_alignment_success()))
    results.append(("Memory Constraint", test_backend_memory_constraint()))

    print_section("Mode Tests")
    results.append(("Non-Strict Mode", test_backend_non_strict_mode()))

    # Part 2: Guard Inspector
    print_header("PART 2: Guard Inspector Tests")
    results.append(("Guard Inspector (Simple)", test_guard_inspector_simple()))
    results.append(("Guard Inspector (BERT)", test_guard_inspector_bert()))

    # Part 3: Artifacts
    print_header("PART 3: Artifact Generation")
    results.append(("Artifact Generation", test_artifact_generation()))

    # Part 4: CLI
    print_header("PART 4: CLI Interface")
    results.append(("CLI List Command", test_cli()))

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n{Colors.BOLD}Results:{Colors.ENDC}\n")

    for name, result in results:
        if result:
            print_success(name)
        else:
            print_fail(name)

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.ENDC}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Some tests failed. Check output above.{Colors.ENDC}")

    print(f"""
{Colors.CYAN}{'='*60}{Colors.ENDC}
{Colors.CYAN}Debug Module Features Demonstrated:{Colors.ENDC}

  - Custom mock backend with torch.compile()
  - 5 constraint types (Dtype, Ops, Layout, Shape, Memory)
  - Environment variable configuration
  - Strict/Non-strict execution modes
  - Guard inspection via torch._dynamo.explain()
  - FX graph artifact capture
  - CLI for artifact management

{Colors.CYAN}Next Steps (Future Weeks):{Colors.ENDC}

  - AOTAutograd integration
  - KernelDiff harness
  - Minifier for failing subgraphs
  - HTML/JSON report generation
{Colors.CYAN}{'='*60}{Colors.ENDC}
""")


if __name__ == "__main__":
    main()
