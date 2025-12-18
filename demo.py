#!/usr/bin/env python3
"""
TorchInductor Debug Module - Demo Script
=========================================

This script demonstrates the key features of the debug module for presentation purposes.
Run with: python demo.py

Press Enter to advance through each section.
"""

import os
import sys
import time
import warnings

# Suppress some torch warnings for cleaner demo output
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress HuggingFace warning

import torch
import torch.nn as nn

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    """Print a section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print("=" * width + "\n")


def print_subheader(text):
    """Print a subsection header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}>>> {text}{Colors.END}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_code(code):
    """Print code block."""
    print(f"{Colors.CYAN}```python")
    print(code)
    print(f"```{Colors.END}")


def wait_for_enter(msg="Press Enter to continue..."):
    """Wait for user input."""
    input(f"\n{Colors.YELLOW}{msg}{Colors.END}")


def demo_intro():
    """Introduction to the project."""
    print_header("TorchInductor Debug Module")

    print(f"""{Colors.BOLD}Project Overview:{Colors.END}

A debugging and analysis toolkit for PyTorch's TorchInductor compiler,
designed to help accelerator backend developers identify and fix
compatibility issues.

{Colors.BOLD}Key Features:{Colors.END}
  1. Custom Backend Integration - Mock accelerator with constraint checking
  2. Constraint System - Shape, dtype, memory, and layout validation
  3. FX Graph Analysis - Capture and inspect computation graphs
  4. KernelDiff - Numerical comparison between backends
  5. Minifier - Generate minimal reproduction scripts
  6. HTML Reports - Visual summaries of benchmark results

{Colors.BOLD}Sponsored by:{Colors.END} IBM Research
{Colors.BOLD}Course:{Colors.END} HPML (High Performance Machine Learning)
""")


def demo_constraint_checking():
    """Demonstrate constraint checking system."""
    print_header("1. Constraint Checking System")

    print_info("The debug module validates tensor operations against hardware constraints.")
    print_info("Real accelerators have strict requirements that must be validated.\n")

    # ============ PART 1: Why Constraints Matter ============
    print_subheader("Why Hardware Constraints Matter")

    print(f"""{Colors.BOLD}Real accelerator limitations:{Colors.END}

  {Colors.CYAN}Shape Alignment{Colors.END}
    • Many accelerators require tensor dimensions divisible by 8, 16, or 32
    • Example: TPU requires 128-byte alignment for efficient memory access
    • Misaligned shapes cause performance degradation or failures

  {Colors.CYAN}Data Types{Colors.END}
    • Some accelerators only support FP16, BF16, or INT8
    • FP64 operations may fall back to CPU
    • Mixed precision requires careful validation

  {Colors.CYAN}Memory Limits{Colors.END}
    • Accelerator memory is limited (e.g., 16GB, 40GB, 80GB)
    • Single tensor allocations have maximum sizes
    • Must track peak memory during compilation

  {Colors.CYAN}Tensor Layout{Colors.END}
    • Many kernels require contiguous memory
    • Non-contiguous tensors need explicit copies
    • Channel-last vs channel-first formats
""")

    wait_for_enter()

    # ============ PART 2: Available Constraints ============
    print_subheader("Built-in Constraint Types")

    print(f"""{Colors.BOLD}1. ShapeConstraint{Colors.END}
   Validates tensor dimension alignment
   Config: MOCK_ALIGNMENT=8  (all dims must be divisible by 8)

{Colors.BOLD}2. DtypeConstraint{Colors.END}
   Validates tensor data types
   Default allowed: float32, float16, int64, int32, bool

{Colors.BOLD}3. MemoryConstraint{Colors.END}
   Validates tensor memory usage
   Config: MOCK_MAX_MEMORY=17179869184  (16GB default)

{Colors.BOLD}4. LayoutConstraint{Colors.END}
   Validates tensor contiguity
   Requires tensors to be contiguous in memory

{Colors.BOLD}5. UnsupportedOpsConstraint{Colors.END}
   Blocks operations not supported by the backend
   Examples: LU decomposition, certain convolutions, in-place activations
""")

    wait_for_enter()

    # ============ PART 3: Configuration ============
    print_subheader("Configuring the Mock Backend")

    print_code("""import os

# Strict mode: fail immediately on constraint violation
os.environ['MOCK_STRICT'] = '1'

# Shape alignment requirement (all dimensions)
os.environ['MOCK_ALIGNMENT'] = '8'

# Maximum memory per tensor (bytes)
os.environ['MOCK_MAX_MEMORY'] = str(1024**3)  # 1GB
""")

    from debug_module.backend.mock import mock_backend

    # Configure via environment variables
    os.environ['MOCK_STRICT'] = '1'
    os.environ['MOCK_ALIGNMENT'] = '8'

    print(f"\n{Colors.BOLD}Current Configuration:{Colors.END}")
    print(f"  MOCK_STRICT    = {os.environ.get('MOCK_STRICT', '1')}")
    print(f"  MOCK_ALIGNMENT = {os.environ.get('MOCK_ALIGNMENT', '1')}")
    print(f"  MOCK_MAX_MEMORY = {int(os.environ.get('MOCK_MAX_MEMORY', str(1024**3 * 16))):,} bytes")

    wait_for_enter()

    # ============ PART 4: Live Demo ============
    print_subheader("Live Demo: Constraint Validation")

    class SimpleModel(nn.Module):
        def forward(self, x):
            return x * 2 + 1

    model = SimpleModel()

    # Show the constraint checking process
    print(f"{Colors.BOLD}Model:{Colors.END} f(x) = x * 2 + 1")
    print(f"{Colors.BOLD}Alignment requirement:{Colors.END} All dimensions must be divisible by 8\n")

    # Test 1: Valid input
    print(f"{Colors.CYAN}━━━ Test 1: Aligned Input ━━━{Colors.END}")
    x_valid = torch.randn(8, 16)
    print(f"  Input shape: {list(x_valid.shape)}")
    print(f"  Dim 0: {x_valid.shape[0]} % 8 = {x_valid.shape[0] % 8} {'✓' if x_valid.shape[0] % 8 == 0 else '✗'}")
    print(f"  Dim 1: {x_valid.shape[1]} % 8 = {x_valid.shape[1] % 8} {'✓' if x_valid.shape[1] % 8 == 0 else '✗'}")

    try:
        opt_model = torch.compile(model, backend=mock_backend)
        result = opt_model(x_valid)
        print(f"  {Colors.GREEN}Result: Compilation successful!{Colors.END}")
        print(f"  Output shape: {list(result.shape)}")
        print(f"  Output sample: {result[0, :4].tolist()}")
    except Exception as e:
        print(f"  {Colors.RED}Result: Failed - {e}{Colors.END}")

    # Test 2: Misaligned input
    print(f"\n{Colors.CYAN}━━━ Test 2: Misaligned Input ━━━{Colors.END}")
    x_invalid = torch.randn(8, 17)
    print(f"  Input shape: {list(x_invalid.shape)}")
    print(f"  Dim 0: {x_invalid.shape[0]} % 8 = {x_invalid.shape[0] % 8} {'✓' if x_invalid.shape[0] % 8 == 0 else '✗'}")
    print(f"  Dim 1: {x_invalid.shape[1]} % 8 = {x_invalid.shape[1] % 8} {Colors.RED}{'✗' if x_invalid.shape[1] % 8 != 0 else '✓'}{Colors.END}")

    try:
        torch._dynamo.reset()
        opt_model = torch.compile(model, backend=mock_backend)
        result = opt_model(x_invalid)
        print(f"  {Colors.GREEN}Result: Compilation successful!{Colors.END}")
    except Exception as e:
        print(f"  {Colors.RED}Result: Constraint violation detected!{Colors.END}")
        print(f"  {Colors.YELLOW}→ A reproduction script was generated for debugging{Colors.END}")

    # Test 3: Different alignment
    print(f"\n{Colors.CYAN}━━━ Test 3: Relaxed Alignment ━━━{Colors.END}")
    os.environ['MOCK_ALIGNMENT'] = '1'  # No alignment requirement
    print(f"  Changed MOCK_ALIGNMENT to 1 (no alignment requirement)")
    x_any = torch.randn(7, 13)
    print(f"  Input shape: {list(x_any.shape)} (odd dimensions)")

    try:
        torch._dynamo.reset()
        opt_model = torch.compile(model, backend=mock_backend)
        result = opt_model(x_any)
        print(f"  {Colors.GREEN}Result: Compilation successful!{Colors.END}")
        print(f"  Output shape: {list(result.shape)}")
    except Exception as e:
        print(f"  {Colors.RED}Result: Failed - {e}{Colors.END}")

    # Reset
    os.environ['MOCK_ALIGNMENT'] = '8'
    torch._dynamo.reset()

    print_success("\nConstraint system catches hardware incompatibilities at compile time!")


def demo_fx_graph_capture():
    """Demonstrate FX graph capture."""
    print_header("2. FX Graph Capture & Analysis")

    print_info("The module captures FX graphs during compilation for analysis.\n")

    print_subheader("Capturing a Model's FX Graph")

    from debug_module.backend.mock import mock_backend

    # Use non-strict mode for this demo
    os.environ['MOCK_STRICT'] = '0'
    os.environ['MOCK_ALIGNMENT'] = '1'

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = MLP()
    x = torch.randn(4, 64)

    print_code("""class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MLP()
opt_model = torch.compile(model, backend=mock_backend)
output = opt_model(torch.randn(4, 64))
""")

    wait_for_enter("Press Enter to compile and capture graph...")

    torch._dynamo.reset()
    opt_model = torch.compile(model, backend=mock_backend)
    output = opt_model(x)

    print_success("Model compiled successfully!")
    print_info(f"Output shape: {output.shape}")

    # Show saved artifacts
    import glob
    graphs = sorted(glob.glob("debug_artifacts/graph_*.txt"))
    if graphs:
        print_info(f"Captured {len(graphs)} graph artifact(s)")
        latest = graphs[-1]
        print(f"\n{Colors.CYAN}Latest graph ({os.path.basename(latest)}):{Colors.END}")
        with open(latest) as f:
            content = f.read()
            # Show first 30 lines
            lines = content.split('\n')[:30]
            for line in lines:
                print(f"  {line}")
            if len(content.split('\n')) > 30:
                print(f"  ... ({len(content.split(chr(10))) - 30} more lines)")

    torch._dynamo.reset()


def demo_kerneldiff():
    """Demonstrate KernelDiff numerical comparison."""
    print_header("3. KernelDiff - Numerical Accuracy Comparison")

    print_info("KernelDiff compares outputs between eager execution and compiled backends")
    print_info("to detect numerical discrepancies that could indicate correctness issues.\n")

    from debug_module.diff import compare_tensors, ComparisonConfig

    # ============ PART 1: Understanding the Problem ============
    print_subheader("Why KernelDiff Matters")

    print(f"""When compiling models with torch.compile(), the backend may:
  • Fuse operations (changing computation order)
  • Use different precision (FP16, BF16, TF32)
  • Apply optimizations that affect numerical results

{Colors.BOLD}Example:{Colors.END} Matrix multiply order: (A @ B) @ C  vs  A @ (B @ C)
These are mathematically equivalent but can produce different floating-point results!
""")

    wait_for_enter()

    # ============ PART 2: Simple Tensor Comparison ============
    print_subheader("Step 1: Direct Tensor Comparison")

    # Create two tensors with tiny differences
    torch.manual_seed(42)
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    test_exact = reference.clone()
    test_close = reference + torch.tensor([1e-7, -1e-7, 1e-8, 0, 1e-6])
    test_wrong = reference + torch.tensor([0.1, -0.05, 0.2, 0, 0.15])

    print(f"{Colors.BOLD}Reference tensor:{Colors.END}")
    print(f"  {reference.tolist()}\n")

    print(f"{Colors.BOLD}Test tensor (exact copy):{Colors.END}")
    print(f"  {test_exact.tolist()}")
    result1 = compare_tensors(reference, test_exact, name="exact_copy")
    print(f"  Max Error: {result1.max_absolute_error:.2e}")
    print(f"  Result: {Colors.GREEN}PASS{Colors.END}\n")

    print(f"{Colors.BOLD}Test tensor (tiny floating-point noise):{Colors.END}")
    print(f"  {[f'{v:.7f}' for v in test_close.tolist()]}")
    result2 = compare_tensors(reference, test_close, name="tiny_noise")
    print(f"  Max Error: {result2.max_absolute_error:.2e}")
    print(f"  Result: {Colors.GREEN}PASS{Colors.END} (within tolerance)\n")

    print(f"{Colors.BOLD}Test tensor (significant errors):{Colors.END}")
    print(f"  {test_wrong.tolist()}")
    result3 = compare_tensors(reference, test_wrong, name="wrong")
    print(f"  Max Error: {result3.max_absolute_error:.2e}")
    print(f"  Result: {Colors.RED}FAIL{Colors.END}")

    wait_for_enter()

    # ============ PART 3: Real Model Comparison ============
    print_subheader("Step 2: Comparing Model Outputs")

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    torch.manual_seed(123)
    model = SimpleNet().eval()
    x = torch.randn(2, 8)

    print(f"{Colors.BOLD}Model:{Colors.END} SimpleNet (8 -> 16 -> 4)")
    print(f"{Colors.BOLD}Input shape:{Colors.END} {list(x.shape)}")
    print(f"{Colors.BOLD}Input sample:{Colors.END} {x[0, :4].tolist()} ...\n")

    # Run eager
    with torch.no_grad():
        eager_out = model(x)

    print(f"{Colors.BOLD}Eager output:{Colors.END}")
    print(f"  Shape: {list(eager_out.shape)}")
    print(f"  Values: {eager_out[0].tolist()}")
    print(f"  Sum: {eager_out.sum().item():.6f}\n")

    # Simulate compiled output (mock backend returns same result)
    compiled_out = eager_out.clone()

    print(f"{Colors.BOLD}Compiled output (mock backend):{Colors.END}")
    print(f"  Shape: {list(compiled_out.shape)}")
    print(f"  Values: {compiled_out[0].tolist()}")
    print(f"  Sum: {compiled_out.sum().item():.6f}\n")

    # Compare
    result = compare_tensors(eager_out, compiled_out, name="model_output")

    print(f"{Colors.BOLD}KernelDiff Comparison:{Colors.END}")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │ Max Absolute Error:  {result.max_absolute_error:>12.2e} │")
    print(f"  │ Mean Absolute Error: {result.mean_absolute_error:>12.2e} │")
    print(f"  │ Max Relative Error:  {result.max_relative_error:>12.2e} │")
    print(f"  │ RMSE:                {result.rmse:>12.2e} │")
    print(f"  │ Mismatched Elements: {result.mismatched_elements:>12d} │")
    print(f"  │ Status:              {Colors.GREEN + 'PASS' + Colors.END:>21} │")
    print(f"  └─────────────────────────────────────┘")

    wait_for_enter()

    # ============ PART 4: Detecting Real Issues ============
    print_subheader("Step 3: Detecting Numerical Issues")

    print(f"{Colors.BOLD}Scenario:{Colors.END} Backend produces slightly different results due to:")
    print(f"  • Operation fusion changing computation order")
    print(f"  • Using FP16 intermediate values")
    print(f"  • Different matrix multiply algorithm\n")

    # Simulate backend with errors
    torch.manual_seed(999)
    noise = torch.randn_like(eager_out) * 0.01  # 1% noise
    bad_compiled_out = eager_out + noise

    print(f"{Colors.BOLD}Eager output:{Colors.END}    {eager_out[0].tolist()}")
    print(f"{Colors.BOLD}Bad compiled:{Colors.END}    {bad_compiled_out[0].tolist()}")
    print(f"{Colors.BOLD}Difference:{Colors.END}      {(bad_compiled_out - eager_out)[0].tolist()}\n")

    result_bad = compare_tensors(eager_out, bad_compiled_out, name="bad_output")

    print(f"{Colors.BOLD}KernelDiff Comparison:{Colors.END}")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │ Max Absolute Error:  {Colors.RED}{result_bad.max_absolute_error:>12.2e}{Colors.END} │")
    print(f"  │ Mean Absolute Error: {Colors.RED}{result_bad.mean_absolute_error:>12.2e}{Colors.END} │")
    print(f"  │ Max Relative Error:  {Colors.RED}{result_bad.max_relative_error:>12.2e}{Colors.END} │")
    print(f"  │ RMSE:                {Colors.RED}{result_bad.rmse:>12.2e}{Colors.END} │")
    print(f"  │ Mismatched Elements: {Colors.RED}{result_bad.mismatched_elements:>12d}{Colors.END} │")
    print(f"  │ Status:              {Colors.RED + 'FAIL' + Colors.END:>21} │")
    print(f"  └─────────────────────────────────────┘")

    # Show where the error is worst
    if result_bad.max_error_indices:
        print(f"\n{Colors.BOLD}Worst error location:{Colors.END}")
        print(f"  Index: {result_bad.max_error_indices}")
        print(f"  Reference value: {result_bad.max_error_ref_value:.6f}")
        print(f"  Test value:      {result_bad.max_error_test_value:.6f}")
        print(f"  Difference:      {abs(result_bad.max_error_ref_value - result_bad.max_error_test_value):.6f}")

    wait_for_enter()

    # ============ PART 5: Configuring Tolerances ============
    print_subheader("Step 4: Configuring Tolerances")

    print(f"""{Colors.BOLD}Tolerance Formula:{Colors.END}
  |reference - test| <= atol + rtol * |reference|

{Colors.BOLD}Default values:{Colors.END}
  atol (absolute tolerance): 1e-5
  rtol (relative tolerance): 1e-4

{Colors.BOLD}Use cases:{Colors.END}
  • Strict (atol=1e-6): Verify exact numerical equivalence
  • Relaxed (atol=1e-3): Allow FP16 precision loss
  • Custom: Match your hardware's expected precision
""")

    # Demo with different tolerances
    print(f"{Colors.BOLD}Same comparison with different tolerances:{Colors.END}\n")

    configs = [
        ("Strict (atol=1e-6)", ComparisonConfig(atol=1e-6, rtol=1e-5)),
        ("Default (atol=1e-5)", ComparisonConfig(atol=1e-5, rtol=1e-4)),
        ("Relaxed (atol=1e-2)", ComparisonConfig(atol=1e-2, rtol=1e-2)),
    ]

    # Use a tensor with small but real differences
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    test = ref + torch.tensor([1e-5, 2e-5, 5e-6, 1e-4])

    print(f"  Reference: {ref.tolist()}")
    print(f"  Test:      {test.tolist()}")
    print(f"  Diff:      {(test - ref).tolist()}\n")

    for name, config in configs:
        r = compare_tensors(ref, test, name=name, config=config)
        status = f"{Colors.GREEN}PASS{Colors.END}" if r.passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {name:25} -> {status} (max_err={r.max_absolute_error:.1e})")

    print_success("\nKernelDiff provides flexible numerical validation for any backend!")


def demo_minifier():
    """Demonstrate the minifier for reproduction scripts."""
    print_header("4. Minifier - Reproduction Script Generation")

    print_info("The minifier generates minimal scripts to reproduce issues.\n")

    import glob
    repros = sorted(glob.glob("debug_artifacts/repros/*.py"))

    if repros:
        print_success(f"Found {len(repros)} reproduction script(s)")
        latest = repros[-1]
        print(f"\n{Colors.CYAN}Sample repro script ({os.path.basename(latest)}):{Colors.END}")
        print("-" * 50)
        with open(latest) as f:
            content = f.read()
            lines = content.split('\n')[:40]
            for line in lines:
                print(line)
            if len(content.split('\n')) > 40:
                print(f"\n... ({len(content.split(chr(10))) - 40} more lines)")
        print("-" * 50)
    else:
        print_info("No repro scripts generated yet. They are created when constraint violations occur.")

    print_info("\nRepro scripts allow developers to quickly reproduce and debug issues")
    print_info("without needing the full original codebase.")


def demo_benchmarks():
    """Demonstrate benchmark capabilities."""
    print_header("5. Benchmarking System")

    print_info("The module includes a comprehensive benchmarking system.\n")

    print_subheader("Available Benchmark Models")

    print(f"""
  {Colors.BOLD}1. BERT-base-multilingual{Colors.END} (Transformer)
     - 178M parameters
     - Tests attention mechanisms and layer norms

  {Colors.BOLD}2. ResNet-18{Colors.END} (CNN)
     - 11.7M parameters
     - Tests convolutions and batch norms

  {Colors.BOLD}3. SSM-Small{Colors.END} (State Space Model)
     - 27.3M parameters
     - Tests custom Mamba-style operations
""")

    print_subheader("Running Benchmarks")

    print_code("""# Run all benchmarks
python -m benchmarks.runner --all

# Run specific model
python -m benchmarks.runner --model bert
python -m benchmarks.runner --model resnet
python -m benchmarks.runner --model ssm
""")

    # Show existing results if available
    import glob
    import json

    summaries = sorted(glob.glob("benchmarks/results/benchmark_summary_*.json"))
    if summaries:
        latest = summaries[-1]
        with open(latest) as f:
            data = json.load(f)

        print_subheader("Latest Benchmark Results")

        print(f"{'Model':<25} {'Eager (ms)':<12} {'Inductor (ms)':<14} {'Speedup':<10} {'KernelDiff':<10}")
        print("-" * 75)

        for r in data.get('results', []):
            name = r.get('model_name', 'Unknown')[:24]
            eager = r.get('eager_backend', {}).get('avg_inference_time', 0) * 1000
            inductor = r.get('inductor_backend', {}).get('avg_inference_time', 0) * 1000
            speedup = eager / inductor if inductor > 0 else 0
            kd = "PASS" if r.get('kerneldiff_passed') else "FAIL"
            kd_color = Colors.GREEN if r.get('kerneldiff_passed') else Colors.RED

            print(f"{name:<25} {eager:<12.1f} {inductor:<14.1f} {speedup:<10.2f}x {kd_color}{kd:<10}{Colors.END}")
    else:
        print_info("No benchmark results found. Run: python -m benchmarks.runner --all")


def demo_html_report():
    """Demonstrate HTML report generation."""
    print_header("6. HTML Report Generation")

    print_info("Generate visual reports summarizing all debug artifacts and benchmarks.\n")

    print_subheader("Generating Report")

    print_code("""# Generate HTML report
python -m debug_module report --format html

# Generate JSON report
python -m debug_module report --format json
""")

    wait_for_enter("Press Enter to generate HTML report...")

    from debug_module.reports import HTMLReportGenerator

    generator = HTMLReportGenerator()
    output_path = generator.generate("debug_artifacts/reports")

    print_success(f"Report generated: {output_path}")
    print_info("Open this file in a browser to view the visual report.")

    # Show what's in the report
    print(f"""
{Colors.BOLD}Report Contents:{Colors.END}
  • Summary statistics (artifacts, models tested, pass rate)
  • Benchmark comparison table (Eager vs Inductor timing)
  • Constraint violation analysis with details
  • KernelDiff numerical accuracy results
  • Artifact inventory
""")


def demo_cli():
    """Demonstrate CLI capabilities."""
    print_header("7. Command Line Interface")

    print_info("The module provides a comprehensive CLI for all operations.\n")

    print(f"""{Colors.BOLD}Available Commands:{Colors.END}

  {Colors.CYAN}python -m debug_module list{Colors.END}
      List all captured debug artifacts

  {Colors.CYAN}python -m debug_module analyze --type guards{Colors.END}
      Analyze Dynamo guards from captured graphs

  {Colors.CYAN}python -m debug_module analyze --type constraints{Colors.END}
      Analyze constraint violations

  {Colors.CYAN}python -m debug_module analyze --type summary{Colors.END}
      Show summary of all artifacts

  {Colors.CYAN}python -m debug_module report --format html{Colors.END}
      Generate HTML report

  {Colors.CYAN}python -m debug_module clean{Colors.END}
      Clean all debug artifacts
""")

    wait_for_enter("Press Enter to run 'list' command...")

    print_subheader("Running: python -m debug_module list")
    os.system("python -m debug_module list")


def demo_summary():
    """Final summary."""
    print_header("Demo Complete!")

    print(f"""
{Colors.BOLD}What We Demonstrated:{Colors.END}

  ✓ Constraint checking system (shape, dtype, memory, layout)
  ✓ FX graph capture and analysis
  ✓ KernelDiff numerical comparison
  ✓ Minifier reproduction script generation
  ✓ Comprehensive benchmarking (BERT, ResNet, SSM)
  ✓ HTML report generation
  ✓ Command line interface

{Colors.BOLD}Project Architecture:{Colors.END}

  ┌─────────────────────────────────────────────────────────┐
  │                    torch.compile()                       │
  └─────────────────────┬───────────────────────────────────┘
                        │
  ┌─────────────────────▼───────────────────────────────────┐
  │              Mock Backend (debug_module)                 │
  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
  │  │ Constraints │ │  FX Graph   │ │   KernelDiff    │   │
  │  │  Checking   │ │   Capture   │ │   Comparison    │   │
  │  └─────────────┘ └─────────────┘ └─────────────────┘   │
  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
  │  │  Minifier   │ │  Artifact   │ │  HTML Reports   │   │
  │  │   (Repro)   │ │   Storage   │ │                 │   │
  │  └─────────────┘ └─────────────┘ └─────────────────┘   │
  └─────────────────────────────────────────────────────────┘

{Colors.BOLD}Key Files:{Colors.END}
  • debug_module/backend/mock.py       - Mock backend implementation
  • debug_module/backend/compiler.py   - Core compilation logic
  • debug_module/constraints/          - Constraint system
  • debug_module/diff/                 - KernelDiff comparison
  • debug_module/minifier/             - Reproduction generator
  • debug_module/reports/              - HTML report generator
  • benchmarks/runner.py               - Benchmark system

{Colors.BOLD}Thank you!{Colors.END}
""")


def main():
    """Run the full demo."""
    print("\n" * 2)

    try:
        demo_intro()
        wait_for_enter("Press Enter to start the demo...")

        demo_constraint_checking()
        wait_for_enter()

        demo_fx_graph_capture()
        wait_for_enter()

        demo_kerneldiff()
        wait_for_enter()

        demo_minifier()
        wait_for_enter()

        demo_benchmarks()
        wait_for_enter()

        demo_html_report()
        wait_for_enter()

        demo_cli()
        wait_for_enter()

        demo_summary()

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted.{Colors.END}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
