# TorchInductor Debug Module

A comprehensive debugging toolkit for developing custom accelerators with PyTorch’s TorchInductor compiler, supporting both eager graph validation and full AOTAutograd + kernel-level compilation. This project provides tools to simulate hardware constraints, compare outputs across backends, generate detailed reports, and benchmark model performance.

**Sponsored by:** IBM Research
**Course:** HPML (High Performance Machine Learning)

**Team Members**:
- Alena Chan (ac5477)
- Michael Chen (yc4131)
- Nikhil Mudumbi (nm3497)
- Srirag Tatavarti (sst2161)

*Note*: We have not used a Weights and Biases (W&B) Dashboard -- since we are building a debugging and inspection tool rather than a training or experimentation pipeline. Our mentor also agreed that W&B tracking did not make sense for our project.



## Features

### 1. Mock Backend Simulator (Eager Validation Mode)
Validates FX graphs before lowering, returning eager execution while providing fast failure, repro generation, and graph inspection.

**Constraints:**
- **Shape**: Enforce dimension alignment (e.g., multiples of 8, 16, 32)
- **Dtype**: Restrict supported data types (e.g., no `float64`)
- **Memory**: Enforce maximum memory usage limits per tensor
- **Layout**: Enforce contiguous memory layouts
- **Unsupported Ops**: Denylist specific operators (LU decomposition, certain convolutions, etc.)

**Modes:**
- **Strict Mode** (`MOCK_STRICT=1`): Hard failure on constraint violation with reproduction script generation
- **Warning Mode** (`MOCK_STRICT=0`): Log warnings but allow execution

### 2. AOTAutograd + TorchInductor Backend
End-to-end compilation backend that lowers models through AOTAutograd and TorchInductor to generate real kernels.

- **Pre-AOT FX capture**: Graph before AOTAutograd transformation
- **Forward / Backward graphs**: Separate graphs for training workloads
- **Constraint checking**: Applied independently to forward and backward graphs
- **Pre-AOT repro generation**: Minimal reproduction scripts are generated only from pre-AOT FX graphs
- **TorchInductor lowering**: Generates intermediate representations and Triton/C++ kernels
- **IR & kernel dumps**: Saves lowered IR and generated kernel code

### 3. KernelDiff Harness
Compare model outputs between reference (eager/Inductor) and mock backend to detect numerical discrepancies.

- **Comprehensive Metrics**: Max/mean absolute error, relative error, RMSE, mismatch percentage
- **Tolerance Configuration**: Configurable atol/rtol for different precision requirements
- **Error Localization**: Identifies exact location of worst errors
- **Visualization**: Error heatmaps, comparison summary plots
- **Complex Output Handling**: Supports nested dicts, tuples, HuggingFace outputs

### 4. HTML Report Generator
Generate visual HTML reports summarizing debug artifacts and benchmark results.

- **Summary Statistics**: Total artifacts, models tested, pass rates
- **Benchmark Comparison**: Side-by-side eager vs inductor timing with speedup calculation
- **Constraint Analysis**: Grouped warnings by model with categorization
- **KernelDiff Results**: Numerical accuracy status for each model

### 5. Backend Adapter Interface
Abstract interface for integrating custom accelerator backends.

- **AcceleratorAdapter**: Base class for custom backends
- **AcceleratorCapabilities**: Define supported dtypes, ops, memory limits
- **IntegratedMockAdapter**: Reference implementation bridging to existing mock backend

### 6. Guard Inspector
Analyzes specialization guards that cause graph breaks or recompilations.

- **Graph Analysis**: Count graphs and graph breaks
- **Break Reasons**: Identify why compilation failed
- **Guard Details**: Per-graph guard information

### 7. Benchmarking Suite
Benchmark models across multiple backends with detailed metrics.

**Supported Models:**
- BERT-base-multilingual (~178M parameters) - Transformer encoder
- ResNet-18 (~11M parameters) - Convolutional neural network
- Custom SSM (~27M parameters) - State space model (Mamba-style)

**Metrics Collected:**
- Compilation time per backend
- Inference latency (avg ± std)
- Graph count and breaks
- Constraint violations
- KernelDiff pass/fail status

### 8. Minifier
Automatically generates minimal reproduction scripts when constraint violations occur.

- **Serialized Graphs**: Captures the exact FX graph state
- **Example Inputs**: Includes the inputs that triggered the failure
- **Standalone Scripts**: Can be shared for debugging without the full codebase

### 9. CLI Tool
Command-line interface for managing the debug workflow.

```bash
# List captured artifacts
inductor-debug list

# Analyze artifacts
inductor-debug analyze --type guards      # Dynamo guard analysis
inductor-debug analyze --type constraints # Constraint violations
inductor-debug analyze --type summary     # Overall summary

# Generate reports
inductor-debug report --format html       # Visual HTML report
inductor-debug report --format json       # Machine-readable JSON

# Clean artifacts
inductor-debug clean

# The module entry point still works if you prefer `python -m ...`
python -m debug_module list
```

## Installation

### From source (recommended)

```bash
git clone https://github.com/Austeleus/inductor-debug-module.git
cd inductor-debug-module

# Create & activate a virtual environment (any manager works)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the library
pip install .
```

### Editable/dev install

```bash
# Include tests, linters, and CLI entry points
pip install -e .[dev]

# Optional extras
pip install -e .[benchmarks]    # Hugging Face + matplotlib
pip install -e .[visualization] # Only matplotlib for KernelDiff plots
```

PyTorch has hardware specific wheels. When targeting CUDA builds refer to
the [official installation selector](https://pytorch.org/get-started/locally/)
and install the wheel *before* installing this package, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
```

## Quick Start

### Interactive Demo

Run the interactive demo for a guided tour of all features:

```bash
python demo.py
```

Choose:

- Full tutorial: Guided steps through constraint checking, FX capture, AOTAutograd + TorchInductor, KernelDiff, minifier, benchmarks, HTML report, and CLI.
- Quick run: Small HF model with interactive knobs (batch/padding/tolerances/constraints), optional AOT backend for KernelDiff, heatmaps, and an optional forced failure to generate a repro.

### Using the Mock Backend

```python
import torch
import os
from debug_module import mock_backend

# Configure constraints
os.environ['MOCK_STRICT'] = '1'      # Fail on violations
os.environ['MOCK_ALIGNMENT'] = '8'   # Require 8-byte alignment

model = YourModel()
compiled_model = torch.compile(model, backend=mock_backend)
output = compiled_model(input_tensor)
```

### Using the AOTAutograd + TorchInductor Backend

```python
import torch
from debug_module.aot_backend.mock import aot_mock_backend
from debug_module.constraints import ShapeConstraint, DtypeConstraint

model = YourTrainingModel()

backend = aot_mock_backend(
    strict=True,  # Fail on constraint violations
    constraints=[
        ShapeConstraint(alignment=8),
        DtypeConstraint(allowed_dtypes={"float32", "float16"}),
    ],
)

compiled = torch.compile(model, backend=backend)

out = compiled(x, y)
out.backward()
```

### Running KernelDiff Comparison

```python
from debug_module.diff import compare_tensors, KernelDiffHarness

# Simple tensor comparison
result = compare_tensors(eager_output, compiled_output)
print(f"Match: {result.passed}")
print(f"Max Error: {result.max_absolute_error:.2e}")

# Full model comparison with visualization
harness = KernelDiffHarness(model, example_inputs)
report = harness.compare(generate_visualizations=True)
print(report.summary())
```

### Running Benchmarks

```bash
# Run all benchmarks
python -m benchmarks.runner --all

# Run specific models
python -m benchmarks.runner --model bert --model resnet

# With custom settings
python -m benchmarks.runner --all --warmup 3 --runs 10 --device cuda

# List available benchmarks
python -m benchmarks.runner --list
```

### Generating Reports

```bash
# Generate HTML report (after running benchmarks)
inductor-debug report --format html

# Open the report
# → debug_artifacts/reports/debug_report_<timestamp>.html
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MOCK_STRICT` | Enable strict mode (1) or warning mode (0) | `1` |
| `MOCK_ALIGNMENT` | Shape alignment requirement for all dimensions | `1` |
| `MOCK_MAX_MEMORY` | Maximum memory per tensor in bytes | `17179869184` (16GB) |

```bash
# Example: Strict mode with 8-byte alignment
export MOCK_STRICT=1
export MOCK_ALIGNMENT=8
export MOCK_MAX_MEMORY=1073741824  # 1 GB
python your_script.py
```

## Project Structure

```
inductor-debug-module/
├── debug_module/           # Core package
│   ├── adapters/           # Backend adapter interface
│   │   ├── base.py         # AcceleratorAdapter abstract class
│   │   ├── mock_adapter.py # Standalone mock implementation
│   │   └── integrated_adapter.py  # Bridges to existing mock backend
│   ├── aot_backend/        # AOTAutograd integration
│   │   ├── aot_capture.py  # Artifact capture + SVG + statistics
│   │   ├── compiler.py     # Core AOT compilation logic
│   │   └── mock.py         # AOTAutograd backend
│   ├── backend/            # Mock backend implementation
│   │   ├── compiler.py     # Core compilation logic
│   │   └── mock.py         # Backend entry point
│   ├── constraints/        # Constraint system
│   │   ├── base.py         # Abstract constraint class
│   │   └── registry.py     # Constraint implementations
│   ├── diff/               # KernelDiff harness
│   │   ├── harness.py      # Main comparison class
│   │   ├── metrics.py      # Error metrics
│   │   └── visualization.py # Heatmap generation
│   ├── guards/             # Guard inspection
│   │   └── inspector.py    # GuardInspector class
│   ├── minifier/           # Reproduction script generator
│   │   └── minifier.py     # Minifier class
│   ├── reports/            # Report generation
│   │   └── generator.py    # HTML/JSON report generator
│   └── cli.py              # Command-line interface
├── benchmarks/             # Benchmarking suite
│   ├── base.py             # BaseBenchmark class
│   ├── bert.py             # BERT benchmark
│   ├── resnet.py           # ResNet benchmark
│   ├── mamba.py            # SSM benchmark
│   ├── runner.py           # CLI runner
│   └── results/            # Output directory
├── frontend/               # Streamlit dashboard
│   └── app.py              # Web UI for interactive execution
├── tests/                  # Test suite (97 tests)
│   ├── test_adapters.py    # Backend adapter tests
│   ├── test_cli.py         # CLI tests
│   ├── test_reports.py     # Report generator tests
│   ├── test_constraints.py # Constraint tests
│   ├── test_kerneldiff.py  # KernelDiff tests
│   ├── test_guards.py      # Guard tests
│   └── test_aotbackend.py  # AOTAutograd tests
├── demo.py                 # Interactive presentation demo
├── debug_artifacts/        # Captured artifacts (auto-generated)
└── README.md
```

## Example Output

### HTML Report
The HTML report includes:
- Summary statistics (total artifacts, models tested, pass rate)
- Benchmark comparison table with timing data
- Constraint violation analysis grouped by model
- KernelDiff numerical accuracy results

### Benchmark Summary
```
================================================================================
                    BENCHMARK SUMMARY REPORT
================================================================================
Model                     Eager (ms)   Inductor (ms)  Speedup    KernelDiff
--------------------------------------------------------------------------------
BERT-base-multilingual    277.8        269.5          1.03x      PASS
ResNet-18                 41.6         53.8           0.77x      PASS
SSM-Small (Custom)        21.0         27.7           0.76x      PASS
--------------------------------------------------------------------------------
```

### Constraint Warnings
The mock backend identifies potential hardware compatibility issues:
- **BERT**: 49 warnings for non-contiguous attention tensors
- **SSM**: 24 warnings for non-contiguous state tensors
- **ResNet**: 0 warnings (CNN-friendly architecture)

## Running Tests

```bash
# Run all tests with pytest
pytest tests/ -v

# Run specific test files
pytest tests/test_kerneldiff.py -v
pytest tests/test_constraints.py -v
pytest tests/test_adapters.py -v

# Run with coverage
pytest tests/ --cov=debug_module
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with torch.compile support)
- torchvision (for ResNet benchmarks)
- transformers (for BERT benchmarks)
- matplotlib (for visualizations)

## Known Limitations

1. **CPU Inductor on macOS**: May have slower compilation. Use Linux or GPU for best performance.
2. **Memory tracking**: Current implementation checks per-tensor limits, not peak memory during execution.
3. **Graph breaks**: Some operations may cause graph breaks that bypass constraint checking.

## License

MIT License
