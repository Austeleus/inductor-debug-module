# TorchInductor Debug Module

A comprehensive debugging toolkit for developing custom accelerators with PyTorch's TorchInductor compiler. This project provides tools to simulate hardware constraints, compare outputs across backends, and benchmark model performance.

## Features

### 1. Mock Backend Simulator
Simulates a custom hardware accelerator with configurable constraints.

**Constraints:**
- **Dtype**: Restrict supported data types (e.g., no `float64`)
- **Layout**: Enforce contiguous memory layouts
- **Shape**: Enforce dimension alignment (e.g., multiples of 16)
- **Memory**: Enforce maximum memory usage limits
- **Unsupported Ops**: Denylist specific operators

**Modes:**
- **Strict Mode**: Hard failure on constraint violation
- **Warning Mode**: Log warnings but allow execution (soft fallback)

### 2. KernelDiff Harness
Compare model outputs between reference (GPU/Inductor) and mock backend.

- **Comprehensive Metrics**: Max/mean absolute error, RMSE, mismatch percentage
- **Visualization**: Error heatmaps, comparison summary plots
- **Complex Output Handling**: Supports nested dicts, tuples, HuggingFace outputs
- **JSON Reports**: Structured output for CI/CD integration

### 3. Guard Inspector
Analyzes specialization guards that cause graph breaks or recompilations.

- **Graph Analysis**: Count graphs and graph breaks
- **Break Reasons**: Identify why compilation failed
- **Guard Details**: Per-graph guard information

### 4. Benchmarking Suite
Benchmark models across multiple backends with detailed metrics.

**Supported Models:**
- BERT-base-multilingual (~178M parameters) - Transformer encoder
- ResNet-18 (~11M parameters) - Convolutional neural network
- Custom SSM (~27M parameters) - State space model

**Metrics Collected:**
- Compilation time per backend
- Inference latency (avg ± std)
- Graph count and breaks
- Constraint violations
- KernelDiff pass/fail status

### 5. Artifact Capture
Automatically captures intermediate artifacts from compilation.

- **FX Graphs**: Saves to `debug_artifacts/`
- **Metadata**: Node shapes, dtypes, stride information
- **Timestamped**: Unique filenames prevent overwrites

### 6. CLI Tool
Command-line interface for managing the debug workflow.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd inductor-debug-module

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision transformers matplotlib

# Optional: For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Using the Mock Backend

```python
import torch
from debug_module import mock_backend

model = YourModel()
compiled_model = torch.compile(model, backend=mock_backend)
output = compiled_model(input_tensor)
```

### Running KernelDiff Comparison

```python
from debug_module.diff import KernelDiffHarness

harness = KernelDiffHarness(
    model=model,
    example_inputs=inputs,
    reference_backend="eager",  # or "inductor" on GPU
)

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

### Using the CLI

```bash
# List captured artifacts
python -m debug_module list

# Clean old artifacts
python -m debug_module clean

# Analyze guards
python -m debug_module analyze --type guards
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MOCK_STRICT` | Enable strict mode (1) or warning mode (0) | `0` |
| `MOCK_ALIGNMENT` | Shape alignment requirement | `8` |
| `MOCK_MAX_MEMORY` | Maximum memory in bytes | `inf` |

```bash
# Example: Strict mode with 16-byte alignment
export MOCK_STRICT=1
export MOCK_ALIGNMENT=16
export MOCK_MAX_MEMORY=17179869184  # 16 GB
python your_script.py
```

## Project Structure

```
inductor-debug-module/
├── debug_module/           # Core package
│   ├── backend/            # Mock backend implementation
│   │   ├── compiler.py     # Core compilation logic
│   │   └── mock.py         # Backend entry point
│   ├── constraints/        # Constraint system
│   │   ├── base.py         # Abstract constraint class
│   │   └── registry.py     # Constraint implementations
│   ├── guards/             # Guard inspection
│   │   └── inspector.py    # GuardInspector class
│   ├── diff/               # KernelDiff harness
│   │   ├── harness.py      # Main comparison class
│   │   ├── metrics.py      # Error metrics
│   │   └── visualization.py # Heatmap generation
│   └── cli.py              # Command-line interface
├── benchmarks/             # Benchmarking suite
│   ├── base.py             # BaseBenchmark class
│   ├── bert.py             # BERT benchmark
│   ├── resnet.py           # ResNet benchmark
│   ├── mamba.py            # SSM benchmark
│   ├── runner.py           # CLI runner
│   └── results/            # Output directory
├── tests/                  # Test scripts
│   ├── test_kerneldiff.py  # KernelDiff tests
│   ├── test_kerneldiff_bert.py
│   ├── test_bert.py        # BERT verification
│   ├── test_constraints.py # Constraint tests
│   ├── test_guards.py      # Guard tests
│   └── comprehensive_test.py
├── context/                # Project documentation
│   └── project_description.txt
├── debug_artifacts/        # Captured artifacts (auto-generated)
├── PROJECT_PLAN.md         # Development roadmap
├── ROBUSTNESS_ANALYSIS.md  # Code quality analysis
└── README.md
```

## Example Output

### Benchmark Summary
```
================================================================================
TORCHINDUCTOR DEBUG MODULE - BENCHMARK SUMMARY REPORT
================================================================================
Models Tested: 3

OVERVIEW
--------------------------------------------------------------------------------
Model                     Type         Params       KernelDiff   Warnings
--------------------------------------------------------------------------------
BERT-base-multilingual    transformer  178.0M       PASS         49
ResNet-18                 cnn          11.7M        PASS         0
SSM-Small (Custom)        ssm          27.4M        PASS         24
--------------------------------------------------------------------------------

INFERENCE TIMES (milliseconds, avg ± std)
--------------------------------------------------------------------------------
Model                     Eager              Mock
--------------------------------------------------------------------------------
BERT-base-multilingual    45.2 ± 1.2         43.5 ± 0.9
ResNet-18                 53.9 ± 2.3         56.9 ± 2.2
SSM-Small (Custom)        46.6 ± 1.3         48.1 ± 1.4
--------------------------------------------------------------------------------
```

### Constraint Warnings
The mock backend identifies potential hardware compatibility issues:
- **BERT**: 49 warnings for non-contiguous attention tensors
- **SSM**: 24 warnings for non-contiguous conv1d outputs
- **ResNet**: 0 warnings (CNN-friendly architecture)

## Running Tests

```bash
# Run all KernelDiff tests
python tests/test_kerneldiff.py

# Run BERT integration test
python tests/test_kerneldiff_bert.py

# Run comprehensive test suite
python tests/comprehensive_test.py

# Run constraint tests
python tests/test_constraints.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with torch.compile support)
- torchvision (for ResNet benchmarks)
- transformers (for BERT/Mamba benchmarks)
- matplotlib (for visualizations)

## Known Limitations

1. **CPU Inductor on macOS**: May fail due to C++ header issues. Use Linux or GPU for full Inductor support.
2. **HuggingFace Mamba**: Has `torch.compile` compatibility issues. We provide a custom SSM implementation.
3. **Memory tracking**: Current implementation doesn't track peak memory during execution.

## Contributing

See `PROJECT_PLAN.md` for the development roadmap and `ROBUSTNESS_ANALYSIS.md` for code quality notes.

## License

MIT License
