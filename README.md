# TorchInductor Debug Module

A tool for debugging and developing custom accelerators with PyTorch Inductor.

## Features

### 1. Mock Backend Simulator
Simulates a custom hardware accelerator with configurable constraints.
-   **Constraints**:
    -   **Dtype**: Restrict supported data types (e.g., no `float64`).
    -   **Layout**: Enforce contiguous memory layouts.
    -   **Shape**: Enforce dimension alignment (e.g., multiples of 16).
    -   **Memory**: Enforce maximum memory usage limits.
    -   **Unsupported Ops**: Denylist specific operators.
-   **Modes**:
    -   **Strict**: Hard failure on constraint violation.
    -   **Warning**: Log warnings but allow execution (Soft Fallback).
-   **Configuration**: Controlled via environment variables (`MOCK_STRICT`, `MOCK_ALIGNMENT`, `MOCK_MAX_MEMORY`).

### 2. Artifact Capture
Automatically captures and saves intermediate artifacts from the compilation process.
-   **FX Graphs**: Saves the AOTAutograd graph to `debug_artifacts/`.
-   **Metadata**: Includes node shapes, dtypes, and stride information.

### 3. Guard Inspector
Analyzes specialization guards that cause graph breaks or recompilations.
-   **API**: `GuardInspector` class to inspect models programmatically.
-   **CLI**: `python -m debug_module analyze --type guards`.

### 4. CLI Tool
A command-line interface for managing the debug workflow.
-   `list`: View captured artifacts.
-   `clean`: Remove old artifacts.
-   `analyze`: Run analysis tools (like Guard Inspector).

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Mock Backend
```python
import torch
from debug_module import mock_backend

model = MyModel()
opt_model = torch.compile(model, backend=mock_backend)
opt_model(input)
```

### Configuration
```bash
export MOCK_STRICT=1
export MOCK_ALIGNMENT=16
export MOCK_MAX_MEMORY=17179869184  # 16 GB
python my_script.py
```

### CLI
```bash
python -m debug_module list
python -m debug_module clean
python -m debug_module analyze --type guards
```

## Project Structure
-   `debug_module/`: Core package.
    -   `backend/`: Mock backend implementation (`compiler.py`, `mock.py`).
    -   `constraints/`: Constraint logic (`registry.py`, `base.py`).
    -   `guards/`: Guard Inspector implementation.
    -   `cli.py`: CLI entry point.
-   `test_bert.py`: Verification script using Hugging Face BERT.
-   `test_constraints.py`: Unit tests for backend constraints.
-   `test_guards.py`: Verification script for Guard Inspector.
