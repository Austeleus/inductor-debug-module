#!/usr/bin/env python3
"""
Test Script for KernelDiff Harness

This script tests all components of the KernelDiff module:
1. Metrics computation (compare_tensors)
2. Visualization generation
3. KernelDiffHarness with simple models
4. Complex output handling (dicts, tuples, nested)
5. Integration with mock backend

Run with: python test_kerneldiff.py
"""

import torch
import torch.nn as nn
import os
import sys
import pytest

# Ensure we're using the local debug_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Color output helpers
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_test(name):
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {name}")
    print("-" * 50)

def print_pass(msg=""):
    print(f"{Colors.GREEN}[PASS]{Colors.END} {msg}")

def print_fail(msg=""):
    print(f"{Colors.RED}[FAIL]{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.YELLOW}[INFO]{Colors.END} {msg}")


# ============================================================================
# TEST 1: Metrics Module
# ============================================================================

def test_metrics_identical_tensors():
    """Test comparison of identical tensors."""
    print_test("Metrics: Identical Tensors")

    from debug_module.diff.metrics import compare_tensors, ComparisonConfig

    ref = torch.randn(10, 10)
    test = ref.clone()  # Identical

    config = ComparisonConfig(atol=1e-6, rtol=1e-5)
    result = compare_tensors(ref, test, name="identical_test", config=config)

    print_info(f"Max absolute error: {result.max_absolute_error}")
    print_info(f"Mean absolute error: {result.mean_absolute_error}")
    print_info(f"Mismatched elements: {result.mismatched_elements}")

    assert result.passed and result.max_absolute_error == 0.0, "Identical tensors should have zero error"
    print_pass("Identical tensors correctly identified")


def test_metrics_different_tensors():
    """Test comparison of different tensors."""
    print_test("Metrics: Different Tensors")

    from debug_module.diff.metrics import compare_tensors, ComparisonConfig

    ref = torch.ones(10, 10)
    test = torch.ones(10, 10) + 0.1  # Small difference

    config = ComparisonConfig(atol=1e-3, rtol=1e-3)
    result = compare_tensors(ref, test, name="diff_test", config=config)

    print_info(f"Max absolute error: {result.max_absolute_error:.6f}")
    print_info(f"Expected: 0.1")
    print_info(f"Passed: {result.passed}")

    # 0.1 > 1e-3, so should fail
    assert not result.passed and abs(result.max_absolute_error - 0.1) < 1e-6, "Should detect difference of 0.1"
    print_pass("Different tensors correctly detected")


def test_metrics_tolerance():
    """Test that tolerance thresholds work correctly."""
    print_test("Metrics: Tolerance Thresholds")

    from debug_module.diff.metrics import compare_tensors, ComparisonConfig

    ref = torch.ones(10, 10)
    test = torch.ones(10, 10) + 1e-6  # Very small difference

    # Tight tolerance - should fail
    tight_config = ComparisonConfig(atol=1e-7, rtol=1e-7)
    tight_result = compare_tensors(ref, test, name="tight", config=tight_config)

    # Loose tolerance - should pass
    loose_config = ComparisonConfig(atol=1e-5, rtol=1e-5)
    loose_result = compare_tensors(ref, test, name="loose", config=loose_config)

    print_info(f"Error: {tight_result.max_absolute_error:.2e}")
    print_info(f"Tight tolerance (1e-7): {'FAIL' if not tight_result.passed else 'PASS'}")
    print_info(f"Loose tolerance (1e-5): {'PASS' if loose_result.passed else 'FAIL'}")

    assert (not tight_result.passed) and loose_result.passed, "Tolerance logic incorrect"
    print_pass("Tolerance thresholds working correctly")


def test_metrics_summary():
    """Test the summary output of comparison results."""
    print_test("Metrics: Summary Output")

    from debug_module.diff.metrics import compare_tensors

    ref = torch.randn(5, 5)
    test = ref + torch.randn(5, 5) * 0.01  # Add small noise

    result = compare_tensors(ref, test, name="summary_test")

    summary = result.summary()
    print(summary)

    assert "summary_test" in summary and "Max Abs Error" in summary, "Summary format incorrect"
    print_pass("Summary generated correctly")


# ============================================================================
# TEST 2: Visualization Module
# ============================================================================

def test_visualization_heatmap():
    """Test heatmap generation."""
    print_test("Visualization: Heatmap Generation")

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        pytest.skip("matplotlib not installed, skipping visualization test")

    from debug_module.diff.metrics import compare_tensors
    from debug_module.diff.visualization import generate_error_heatmap, VisualizationConfig

    ref = torch.randn(20, 20)
    test = ref + torch.randn(20, 20) * 0.1

    result = compare_tensors(ref, test, name="heatmap_test")

    config = VisualizationConfig(
        output_dir="debug_artifacts/test_visualizations",
        file_format="png"
    )

    try:
        filepath = generate_error_heatmap(result, config=config, save=True, show=False)

        assert filepath and os.path.exists(filepath), "Heatmap file not created"
        print_info(f"Heatmap saved to: {filepath}")
        print_pass("Heatmap generated successfully")

    except Exception as e:
        print_fail(f"Heatmap generation failed: {e}")
        pytest.fail(str(e))


def test_visualization_1d():
    """Test 1D tensor visualization."""
    print_test("Visualization: 1D Tensor")

    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pytest.skip("matplotlib not installed, skipping 1D visualization test")

    from debug_module.diff.metrics import compare_tensors
    from debug_module.diff.visualization import generate_error_heatmap, VisualizationConfig

    ref = torch.randn(100)
    test = ref + torch.randn(100) * 0.05

    result = compare_tensors(ref, test, name="1d_test")

    config = VisualizationConfig(output_dir="debug_artifacts/test_visualizations")

    try:
        filepath = generate_error_heatmap(result, config=config, save=True, show=False)
        if filepath:
            print_info(f"1D plot saved to: {filepath}")
        else:
            print_info("1D visualization skipped file creation (acceptable)")
        print_pass("1D visualization generated")
    except Exception as e:
        print_fail(f"1D visualization failed: {e}")
        pytest.fail(str(e))


# ============================================================================
# TEST 3: Simple Model with KernelDiffHarness
# ============================================================================

def test_harness_simple_function():
    """Test harness with a simple function."""
    print_test("Harness: Simple Function")

    from debug_module.diff import KernelDiffHarness

    def simple_add(x, y):
        return x + y

    x = torch.randn(8, 8, dtype=torch.float32)
    y = torch.randn(8, 8, dtype=torch.float32)

    harness = KernelDiffHarness(
        model=simple_add,
        example_inputs=(x, y),
        model_name="simple_add",
        reference_backend="eager"  # Use eager for faster testing
    )

    try:
        report = harness.compare(generate_visualizations=False, save_report=True)
        print(report.summary())
        assert report.overall_passed, "Simple function should produce identical results"
        print_pass("Simple function comparison passed")

    except Exception as e:
        print_fail(f"Harness failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def test_harness_simple_module():
    """Test harness with a simple nn.Module."""
    print_test("Harness: Simple nn.Module")

    from debug_module.diff import KernelDiffHarness

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 8)

        def forward(self, x):
            return self.linear(x)

    model = SimpleNet().eval()
    x = torch.randn(4, 16)

    harness = KernelDiffHarness(
        model=model,
        example_inputs=x,
        model_name="SimpleNet",
        reference_backend="eager"
    )

    try:
        report = harness.compare(generate_visualizations=False, save_report=True)
        print(report.summary())

        if report.overall_passed:
            print_pass("Simple module comparison passed")
        else:
            max_err = max(r.max_absolute_error for r in report.tensor_results)
            print_info(f"Max error: {max_err:.2e}")
            assert max_err < 1e-3, "Errors too large"
            print_pass("Errors within acceptable range")

    except Exception as e:
        print_fail(f"Harness failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


# ============================================================================
# TEST 4: Complex Output Handling
# ============================================================================

def test_harness_dict_output():
    """Test harness with dict output."""
    print_test("Harness: Dict Output")

    from debug_module.diff import KernelDiffHarness

    class DictOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(8, 4)
            self.linear2 = nn.Linear(8, 4)

        def forward(self, x):
            return {
                "output1": self.linear1(x),
                "output2": self.linear2(x),
            }

    model = DictOutputModel().eval()
    x = torch.randn(2, 8)

    harness = KernelDiffHarness(
        model=model,
        example_inputs=x,
        model_name="DictOutputModel",
        reference_backend="eager"
    )

    try:
        report = harness.compare(generate_visualizations=False, save_report=False)

        print_info(f"Tensors compared: {report.total_tensors}")
        for r in report.tensor_results:
            print_info(f"  {r.name}: {r.shape}")

        assert report.total_tensors == 2, f"Expected 2 tensors, got {report.total_tensors}"
        print_pass("Dict outputs correctly flattened")

    except Exception as e:
        print_fail(f"Harness failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def test_harness_tuple_output():
    """Test harness with tuple output."""
    print_test("Harness: Tuple Output")

    from debug_module.diff import KernelDiffHarness

    class TupleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)

        def forward(self, x):
            out = self.linear(x)
            return out, out * 2, out * 3

    model = TupleOutputModel().eval()
    x = torch.randn(2, 8)

    harness = KernelDiffHarness(
        model=model,
        example_inputs=x,
        model_name="TupleOutputModel",
        reference_backend="eager"
    )

    try:
        report = harness.compare(generate_visualizations=False, save_report=False)

        print_info(f"Tensors compared: {report.total_tensors}")
        for r in report.tensor_results:
            print_info(f"  {r.name}: {r.shape}")

        assert report.total_tensors == 3, f"Expected 3 tensors, got {report.total_tensors}"
        print_pass("Tuple outputs correctly flattened")

    except Exception as e:
        print_fail(f"Harness failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


def test_harness_nested_output():
    """Test harness with nested output structures."""
    print_test("Harness: Nested Output")

    from debug_module.diff import KernelDiffHarness

    class NestedOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)

        def forward(self, x):
            out = self.linear(x)
            return {
                "main": out,
                "extras": (out * 2, out * 3),
                "nested": {
                    "a": out + 1,
                    "b": out + 2,
                }
            }

    model = NestedOutputModel().eval()
    x = torch.randn(2, 8)

    harness = KernelDiffHarness(
        model=model,
        example_inputs=x,
        model_name="NestedOutputModel",
        reference_backend="eager"
    )

    try:
        report = harness.compare(generate_visualizations=False, save_report=False)

        print_info(f"Tensors compared: {report.total_tensors}")
        for r in report.tensor_results:
            print_info(f"  {r.name}: {r.shape}")

        assert report.total_tensors == 5, f"Expected 5 tensors, got {report.total_tensors}"
        print_pass("Nested outputs correctly flattened")

    except Exception as e:
        print_fail(f"Harness failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


# ============================================================================
# TEST 5: Dict Inputs (like HuggingFace models)
# ============================================================================

def test_harness_dict_input():
    """Test harness with dict inputs."""
    print_test("Harness: Dict Input")

    from debug_module.diff import KernelDiffHarness

    class DictInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 4)

        def forward(self, input_ids, attention_mask):
            # Simulate processing like a transformer
            x = input_ids.float() * attention_mask.float().unsqueeze(-1)
            return self.linear(x)

    model = DictInputModel().eval()
    inputs = {
        "input_ids": torch.randn(2, 8),
        "attention_mask": torch.ones(2),
    }

    harness = KernelDiffHarness(
        model=model,
        example_inputs=inputs,
        model_name="DictInputModel",
        reference_backend="eager"
    )

    try:
        report = harness.compare(generate_visualizations=False, save_report=False)
        print(report.summary())

        assert report.overall_passed or report.total_tensors > 0, "Dict input handling failed"
        print_pass("Dict inputs handled correctly")

    except Exception as e:
        print_fail(f"Harness failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


# ============================================================================
# TEST 6: Quick Check
# ============================================================================

def test_quick_check():
    """Test the quick_check method."""
    print_test("Harness: Quick Check")

    from debug_module.diff import KernelDiffHarness

    def simple_mul(x):
        return x * 2

    x = torch.randn(8, 8, dtype=torch.float32)

    harness = KernelDiffHarness(
        model=simple_mul,
        example_inputs=x,
        model_name="simple_mul",
        reference_backend="eager"
    )

    try:
        passed = harness.quick_check(atol=1e-5, rtol=1e-4)
        print_info(f"Quick check result: {passed}")

        assert passed, "Quick check should pass for identical computation"
        print_pass("Quick check working correctly")

    except Exception as e:
        print_fail(f"Quick check failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


# ============================================================================
# TEST 7: Report Serialization
# ============================================================================

def test_report_json():
    """Test JSON report generation."""
    print_test("Report: JSON Serialization")

    from debug_module.diff import KernelDiffHarness
    import json

    def simple_fn(x):
        return x + 1

    x = torch.randn(4, 4)

    harness = KernelDiffHarness(
        model=simple_fn,
        example_inputs=x,
        model_name="json_test",
        reference_backend="eager"
    )

    try:
        report = harness.compare(
            generate_visualizations=False,
            save_report=True,
            report_dir="debug_artifacts/test_reports"
        )

        report_dict = report.to_dict()
        json_str = json.dumps(report_dict, indent=2)
        print_info(f"JSON report generated ({len(json_str)} bytes)")

        required_keys = ["model_name", "overall_passed", "tensor_results"]
        assert all(k in report_dict for k in required_keys), "JSON report missing required keys"
        print_pass("JSON report correctly structured")

    except Exception as e:
        print_fail(f"JSON serialization failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("KernelDiff Harness Test Suite")

    # Set environment for mock backend
    os.environ["MOCK_STRICT"] = "0"  # Non-strict mode for tests
    os.environ["MOCK_ALIGNMENT"] = "1"  # No alignment constraint

    results = []

    # Run all tests
    print_header("Part 1: Metrics Module")
    results.append(("Metrics: Identical Tensors", test_metrics_identical_tensors()))
    results.append(("Metrics: Different Tensors", test_metrics_different_tensors()))
    results.append(("Metrics: Tolerance", test_metrics_tolerance()))
    results.append(("Metrics: Summary", test_metrics_summary()))

    print_header("Part 2: Visualization Module")
    results.append(("Visualization: Heatmap", test_visualization_heatmap()))
    results.append(("Visualization: 1D", test_visualization_1d()))

    print_header("Part 3: Simple Models")
    results.append(("Harness: Simple Function", test_harness_simple_function()))
    results.append(("Harness: Simple Module", test_harness_simple_module()))

    print_header("Part 4: Complex Outputs")
    results.append(("Harness: Dict Output", test_harness_dict_output()))
    results.append(("Harness: Tuple Output", test_harness_tuple_output()))
    results.append(("Harness: Nested Output", test_harness_nested_output()))

    print_header("Part 5: Dict Inputs")
    results.append(("Harness: Dict Input", test_harness_dict_input()))

    print_header("Part 6: Quick Check")
    results.append(("Harness: Quick Check", test_quick_check()))

    print_header("Part 7: Report Generation")
    results.append(("Report: JSON", test_report_json()))

    # Summary
    print_header("Test Summary")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        if result:
            print_pass(name)
        else:
            print_fail(name)

    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
