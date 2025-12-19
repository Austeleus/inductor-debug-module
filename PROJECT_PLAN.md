# TorchInductor Debug Module - Remaining Work Plan

## Overview

This document outlines the remaining work to complete the project, organized into phases with specific tasks and deliverables.

---

## Phase 1: AOTAutograd/Inductor Integration (Priority: HIGH) ✅ COMPLETED

**Why:** Currently the backend bypasses TorchInductor entirely. To capture AOT graphs and IR, we need proper integration.

### Tasks:

#### 1.1 Integrate AOTAutograd ✅
- [x] Modify `mock_backend()` to use `aot_module_simplified`
- [x] Create separate compilers for forward and backward passes
- [x] Capture both FX graph (pre-AOT) and AOT graph (post-AOT)

```python
# Target implementation in backend/mock.py
from torch._functorch.aot_autograd import aot_module_simplified

def mock_backend(gm, example_inputs):
    # Capture pre-AOT FX graph
    save_artifact(gm, artifact_type="fx_graph")

    def fw_compiler(aot_gm, aot_inputs):
        # Capture post-AOT graph
        save_artifact(aot_gm, artifact_type="aot_graph")
        check_constraints(aot_gm)
        return aot_gm.forward

    return aot_module_simplified(gm, example_inputs, fw_compiler=fw_compiler)
```

#### 1.2 Enhanced Artifact Capture ✅
- [x] Save FX graph (pre-AOT)
- [x] Save AOT graph (post-AOT)
- [x] Save graph statistics (op counts, dtypes, memory estimates)
- [x] Optional: SVG visualization of graphs

#### 1.3 Update Artifact Directory Structure ✅
```
debug_artifacts/
├── fx_graphs/
├── aot_graphs/
├── inductor_ir/
├── inductor_kernels/
├── reports/
└── statistics/
```

### Test Coverage:
- `tests/test_aotbackend.py` - 8 unit tests (all passing)

**Deliverable:** Backend that properly uses TorchInductor pipeline and captures all intermediate representations. ✅ Complete

---

## Phase 2: KernelDiff Harness (Priority: HIGH) ✅ COMPLETED

**Status:** Implemented and tested.

### Completed Tasks:

#### 2.1 Create KernelDiff Module ✅
- [x] Created `debug_module/diff/` directory
- [x] Implemented `KernelDiffHarness` class
- [x] Implemented `DiffReport` dataclass for results

#### 2.2 Implement Comparison Metrics ✅
- [x] Max absolute error
- [x] Mean absolute error
- [x] Max/mean relative error
- [x] RMSE (Root Mean Square Error)
- [x] Percentage of mismatched elements
- [x] Configurable tolerances (atol, rtol, max_mismatch_percentage)

#### 2.3 Error Visualization ✅
- [x] Generate error heatmaps (1D, 2D, N-D tensors)
- [x] Identify indices of largest errors
- [x] Save visualizations to `debug_artifacts/visualizations/`
- [x] Summary comparison plots

#### 2.4 Handle Complex Outputs ✅
- [x] Support nested outputs (dicts, tuples, lists)
- [x] Support multiple output tensors
- [x] Handle HuggingFace model outputs (ModelOutput dataclass)
- [x] Dict inputs support

### Files Created:
```
debug_module/diff/
├── __init__.py          # Module exports
├── metrics.py           # TensorComparisonResult, compare_tensors()
├── visualization.py     # Heatmaps, summary plots
└── harness.py           # KernelDiffHarness, DiffReport
```

### Usage:
```python
from debug_module.diff import KernelDiffHarness

harness = KernelDiffHarness(model, inputs, reference_backend="inductor")
report = harness.compare(generate_visualizations=True)
print(report.summary())
```

### Test Coverage:
- `test_kerneldiff.py` - 14 unit tests (all passing)
- `test_kerneldiff_bert.py` - BERT integration test (passing)

**Deliverable:** ✅ Complete

---

### Phase 2 Next Steps (Future Improvements)

The current implementation is functional for the school project but has areas for improvement:

| Issue | Current State | Production Fix |
|-------|---------------|----------------|
| GPU support | CPU-only comparison | Add device handling, CUDA sync |
| Memory efficiency | Stores full error tensors | Streaming comparison for large tensors |
| Gradient comparison | Forward pass only | Add backward pass comparison |
| Error recovery | Basic exception handling | Comprehensive try/catch, partial results |
| Model caching | Recompiles every run | Cache compiled models |
| Statistical analysis | Single run | Multiple runs, variance calculation |
| NaN/Inf handling | Not handled | Add explicit checks and reporting |

**Priority:** Low - current implementation sufficient for project requirements.

---

## Phase 3: Minifier (Priority: MEDIUM)

**Why:** Auto-generate minimal reproduction scripts for debugging failures.

### Tasks:

#### 3.1 Create Minifier Module
- [ ] Create `debug_module/minifier/` directory
- [ ] Implement `Minifier` class

```python
# debug_module/minifier/minifier.py
class Minifier:
    def __init__(self, gm, example_inputs, error):
        self.gm = gm
        self.inputs = example_inputs
        self.error = error

    def minify(self):
        """Reduce graph to minimal failing case."""
        pass

    def generate_repro_script(self, output_path):
        """Generate standalone Python script."""
        pass
```

#### 3.2 Implement Minification Algorithm
- [ ] Binary search on nodes to find minimal set
- [ ] Preserve graph semantics during reduction
- [ ] Handle dependencies between nodes

#### 3.3 Script Generation
- [ ] Generate standalone Python script
- [ ] Include necessary imports
- [ ] Include example inputs that trigger the error
- [ ] Add comments explaining the failure

#### 3.4 Integration with Backend
- [ ] Hook minifier into constraint failure path
- [ ] Auto-generate repro when `BackendCompilerFailed` is raised
- [ ] Save repro scripts to `debug_artifacts/repros/`

**Deliverable:** Minifier that automatically generates minimal reproduction scripts for failures.

---

## Phase 4: Reporting System (Priority: MEDIUM)

**Why:** Project requires JSON/HTML reports summarizing all findings.

### Tasks:

#### 4.1 JSON Report Generation
- [ ] Create `debug_module/reports/` directory
- [ ] Implement `ReportGenerator` class
- [ ] Generate structured JSON with all findings

```python
# Report structure
{
    "model": "bert-base-multilingual-cased",
    "timestamp": "2024-01-15T10:30:00",
    "summary": {
        "total_graphs": 5,
        "graph_breaks": 2,
        "constraint_violations": 3,
        "kernel_diff_passed": true
    },
    "guards": [...],
    "constraints": [...],
    "kernel_diff": {...},
    "artifacts": [...]
}
```

#### 4.2 HTML Report Generation
- [ ] Create HTML template
- [ ] Include graphs and visualizations
- [ ] Collapsible sections for detailed info
- [ ] Syntax highlighting for code snippets

#### 4.3 CLI Integration
- [ ] Add `python -m debug_module report` command
- [ ] Options: `--format json|html`, `--output-dir`
- [ ] Generate reports from captured artifacts

**Deliverable:** Report generator that produces JSON and HTML reports with all debugging findings.

---

## Phase 5: Backend Adapter Interface (Priority: LOW)

**Why:** Clean API for future integration with real custom accelerators.

### Tasks:

#### 5.1 Define Abstract Interface
- [ ] Create `debug_module/adapters/base.py`
- [ ] Define `AcceleratorAdapter` abstract class

```python
# debug_module/adapters/base.py
from abc import ABC, abstractmethod

class AcceleratorAdapter(ABC):
    @abstractmethod
    def get_supported_dtypes(self) -> Set[torch.dtype]:
        pass

    @abstractmethod
    def get_supported_ops(self) -> Set[str]:
        pass

    @abstractmethod
    def get_memory_limit(self) -> int:
        pass

    @abstractmethod
    def compile(self, gm, inputs):
        pass
```

#### 5.2 Refactor Mock Backend
- [ ] Make `MockAcceleratorAdapter` implement the interface
- [ ] Move constraint configuration to adapter

#### 5.3 Documentation
- [ ] Document how to create custom adapters
- [ ] Provide example adapter implementation

**Deliverable:** Clean adapter interface that future accelerator teams can implement.

---

## Phase 6: Testing & Benchmarking (Priority: HIGH)

**Why:** Project requires testing on 2-3 benchmark models.

### Tasks:

#### 6.1 Benchmark Models
- [ ] BERT (already done)
- [ ] ResNet (CNN workload)
- [ ] Transformer block or small SSM

#### 6.2 Test Scripts
- [ ] Create `benchmarks/` directory
- [ ] `benchmark_bert.py`
- [ ] `benchmark_resnet.py`
- [ ] `benchmark_transformer.py`

#### 6.3 Performance Metrics
- [ ] Compilation time
- [ ] Inference latency
- [ ] Memory usage
- [ ] Number of graph breaks

#### 6.4 Case Studies
- [ ] Document findings for each model
- [ ] Identify common constraint violations
- [ ] Provide recommendations

**Deliverable:** Benchmark results and case studies for 3 models.

---

## Phase 7: Documentation & Final Polish (Priority: HIGH)

**Why:** Project requires final documentation and presentation.

### Tasks:

#### 7.1 Code Documentation
- [ ] Docstrings for all public functions/classes
- [ ] Type hints throughout
- [ ] Inline comments for complex logic

#### 7.2 User Documentation
- [ ] Update README.md with full usage guide
- [ ] Installation instructions
- [ ] Quick start guide
- [ ] API reference

#### 7.3 Final Presentation
- [ ] Prepare slides
- [ ] Demo script
- [ ] Record demo video (optional)

**Deliverable:** Complete documentation and presentation materials.

---

## Implementation Order (Recommended)

Based on dependencies and priorities:

### Week 1: AOTAutograd Integration
- Phase 1 (all tasks)
- This unblocks proper artifact capture

### Week 2: KernelDiff
- Phase 2 (all tasks)
- Core debugging feature

### Week 3: Reporting + Benchmarks
- Phase 4 (JSON/HTML reports)
- Phase 6.1-6.2 (benchmark scripts)

### Week 4: Minifier + Adapter
- Phase 3 (Minifier)
- Phase 5 (Adapter interface)

### Week 5: Testing + Documentation
- Phase 6.3-6.4 (metrics, case studies)
- Phase 7 (documentation, presentation)

---

## File Structure (Final)

```
inductor-debug-module/
├── debug_module/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── aot_backend/
│   │   ├── __init__.py
│   │   ├── mock.py
│   │   └── compiler.py
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── aot_capture.py
│   │   ├── mock.py
│   │   └── compiler.py
│   ├── constraints/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── registry.py
│   ├── guards/
│   │   ├── __init__.py
│   │   └── inspector.py
│   ├── diff/                    # NEW
│   │   ├── __init__.py
│   │   └── harness.py
│   ├── minifier/                # NEW
│   │   ├── __init__.py
│   │   └── minifier.py
│   ├── reports/                 # NEW
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── templates/
│   ├── adapters/                # NEW
│   │   ├── __init__.py
│   │   └── base.py
│   └── utils/
│       └── __init__.py
├── benchmarks/                  # NEW
│   ├── benchmark_bert.py
│   ├── benchmark_resnet.py
│   └── benchmark_transformer.py
├── tests/
│   ├── test_bert.py
│   ├── test_guards.py
│   ├── test_constraints.py
│   └── comprehensive_test.py
├── docs/                        # NEW
│   └── README.md
└── README.md
```

---

## Effort Estimates

| Phase | Effort | Complexity |
|-------|--------|------------|
| 1. AOTAutograd Integration | 4-6 hours | Medium |
| 2. KernelDiff | 6-8 hours | Medium |
| 3. Minifier | 8-10 hours | High |
| 4. Reporting | 4-6 hours | Medium |
| 5. Adapter Interface | 2-3 hours | Low |
| 6. Benchmarking | 4-6 hours | Medium |
| 7. Documentation | 4-6 hours | Low |

**Total: ~35-45 hours of work**

---

## Risk Areas

1. **AOTAutograd Integration** - May require debugging PyTorch internals
2. **Minifier** - Graph manipulation is complex, may need simplification
3. **KernelDiff with complex models** - Handling various output formats

---

## Success Criteria

Project is complete when:

- [ ] Backend uses AOTAutograd and captures FX + AOT graphs
- [ ] KernelDiff compares GPU vs mock with error metrics
- [ ] Minifier generates standalone repro scripts
- [ ] HTML/JSON reports summarize all findings
- [ ] Adapter interface is documented
- [ ] 3 benchmark models tested with case studies
- [ ] Full documentation and presentation ready
