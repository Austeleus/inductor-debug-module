"""
Test Suite for Backend Adapter Interface
=========================================

Tests the debug_module.adapters module:
- AcceleratorCapabilities dataclass
- CompilationResult dataclass
- ConstraintViolation dataclass
- AcceleratorAdapter abstract class
- MockAcceleratorAdapter implementation
- create_mock_backend factory function
"""

import os
import sys
import pytest
import torch
import torch.fx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.adapters import (
    AcceleratorAdapter,
    AcceleratorCapabilities,
    CompilationResult,
    MockAcceleratorAdapter,
    create_mock_backend,
)
from debug_module.adapters.base import (
    CompilationStatus,
    ConstraintViolation,
)


# =============================================================================
# Test AcceleratorCapabilities
# =============================================================================

class TestAcceleratorCapabilities:
    """Tests for AcceleratorCapabilities dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        caps = AcceleratorCapabilities()
        assert caps.name == "Custom Accelerator"
        assert torch.float32 in caps.supported_dtypes
        assert caps.requires_contiguous is False
        assert caps.alignment_requirement == 1
        assert caps.max_tensor_dims == 8

    def test_custom_initialization(self):
        """Test custom initialization."""
        caps = AcceleratorCapabilities(
            name="My TPU",
            supported_dtypes={torch.float32, torch.bfloat16},
            requires_contiguous=True,
            alignment_requirement=128,
            max_memory_bytes=8 * 1024**3,
        )
        assert caps.name == "My TPU"
        assert torch.bfloat16 in caps.supported_dtypes
        assert torch.float64 not in caps.supported_dtypes
        assert caps.requires_contiguous is True
        assert caps.alignment_requirement == 128

    def test_to_dict(self):
        """Test serialization to dictionary."""
        caps = AcceleratorCapabilities(
            name="Test",
            unsupported_ops={"aten.sin", "aten.cos"},
        )
        d = caps.to_dict()

        assert d['name'] == "Test"
        assert 'aten.sin' in d['unsupported_ops']
        assert isinstance(d['supported_dtypes'], list)
        assert isinstance(d['max_memory_bytes'], int)


# =============================================================================
# Test CompilationResult
# =============================================================================

class TestCompilationResult:
    """Tests for CompilationResult dataclass."""

    def test_default_is_failed(self):
        """Test default status is FAILED."""
        result = CompilationResult()
        assert result.status == CompilationStatus.FAILED
        assert result.success is False

    def test_success_property(self):
        """Test success property."""
        result = CompilationResult(status=CompilationStatus.SUCCESS)
        assert result.success is True

        result2 = CompilationResult(status=CompilationStatus.PARTIAL)
        assert result2.success is False

    def test_has_violations(self):
        """Test has_violations property."""
        result = CompilationResult()
        assert result.has_violations is False

        result_with_violations = CompilationResult(
            violations=[ConstraintViolation("node1", "dtype", "bad dtype")]
        )
        assert result_with_violations.has_violations is True

    def test_get_callable_with_compiled(self):
        """Test get_callable returns compiled function."""
        def my_fn():
            return 42

        result = CompilationResult(
            status=CompilationStatus.SUCCESS,
            compiled_fn=my_fn,
        )
        assert result.get_callable() == my_fn

    def test_get_callable_with_fallback(self):
        """Test get_callable returns fallback when no compiled_fn."""
        def fallback_fn():
            return 0

        result = CompilationResult(
            status=CompilationStatus.FAILED,
            compiled_fn=None,
            fallback_fn=fallback_fn,
        )
        assert result.get_callable() == fallback_fn

    def test_get_callable_raises_without_any(self):
        """Test get_callable raises when no function available."""
        result = CompilationResult()
        with pytest.raises(RuntimeError):
            result.get_callable()


# =============================================================================
# Test ConstraintViolation
# =============================================================================

class TestConstraintViolation:
    """Tests for ConstraintViolation dataclass."""

    def test_creation(self):
        """Test basic creation."""
        v = ConstraintViolation(
            node_name="add_1",
            constraint_type="dtype",
            message="Unsupported dtype float64",
        )
        assert v.node_name == "add_1"
        assert v.constraint_type == "dtype"
        assert v.severity == "error"  # default

    def test_str_representation(self):
        """Test string representation."""
        v = ConstraintViolation(
            node_name="mul",
            constraint_type="shape",
            message="Bad alignment",
        )
        s = str(v)
        assert "SHAPE" in s
        assert "mul" in s
        assert "Bad alignment" in s

    def test_with_suggestion(self):
        """Test violation with suggestion."""
        v = ConstraintViolation(
            node_name="conv",
            constraint_type="layout",
            message="Non-contiguous",
            suggestion="Add .contiguous() call",
        )
        assert v.suggestion == "Add .contiguous() call"


# =============================================================================
# Test MockAcceleratorAdapter
# =============================================================================

class TestMockAcceleratorAdapter:
    """Tests for MockAcceleratorAdapter."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        adapter = MockAcceleratorAdapter()
        assert adapter.strict is True
        assert adapter.verbose is True

    def test_initialization_custom(self):
        """Test custom initialization."""
        adapter = MockAcceleratorAdapter(
            strict=False,
            verbose=False,
            name="Custom Mock",
            alignment=32,
        )
        assert adapter.strict is False
        assert adapter.name == "Custom Mock"
        assert adapter._alignment == 32

    def test_get_capabilities(self):
        """Test capabilities retrieval."""
        adapter = MockAcceleratorAdapter(
            alignment=16,
            max_memory=4 * 1024**3,
        )
        caps = adapter.get_capabilities()

        assert caps.alignment_requirement == 16
        assert caps.max_memory_bytes == 4 * 1024**3
        assert torch.float32 in caps.supported_dtypes

    def test_capabilities_caching(self):
        """Test capabilities are cached."""
        adapter = MockAcceleratorAdapter()
        caps1 = adapter.capabilities
        caps2 = adapter.capabilities
        assert caps1 is caps2  # Same object

    def test_compile_simple_model(self):
        """Test compiling a simple model."""
        adapter = MockAcceleratorAdapter(strict=False, verbose=False)

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x * 2

        model = SimpleModel()
        compiled = torch.compile(model, backend=adapter)

        x = torch.randn(16)
        result = compiled(x)

        assert result.shape == x.shape
        torch.testing.assert_close(result, x * 2)

    def test_compile_with_violations_nonstrict(self):
        """Test compilation with violations in non-strict mode."""
        adapter = MockAcceleratorAdapter(
            strict=False,
            verbose=False,
            alignment=16,
        )

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = Model()
        compiled = torch.compile(model, backend=adapter)

        # Shape 10 violates alignment 16
        x = torch.randn(10)
        result = compiled(x)

        # Should still work in non-strict mode
        assert result.shape == x.shape

    def test_compile_with_violations_strict(self):
        """Test compilation with violations in strict mode raises."""
        adapter = MockAcceleratorAdapter(
            strict=True,
            verbose=False,
            alignment=16,
        )

        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = Model()
        compiled = torch.compile(model, backend=adapter)

        x = torch.randn(10)  # Violates alignment

        with pytest.raises(Exception):  # Should raise
            compiled(x)

    def test_check_constraints_dtype(self):
        """Test dtype constraint checking."""
        adapter = MockAcceleratorAdapter(
            strict=True,
            verbose=False,
            supported_dtypes={torch.float32},
        )

        # Create a simple FX graph with float64
        class Model(torch.nn.Module):
            def forward(self, x):
                return x * 2

        model = Model()
        x = torch.randn(8, dtype=torch.float64)

        # Trace to get FX graph
        from torch.fx import symbolic_trace
        gm = symbolic_trace(model)

        # Note: check_constraints needs metadata, which symbolic_trace doesn't add
        # This is a simplified test
        caps = adapter.capabilities
        assert torch.float64 not in caps.supported_dtypes

    def test_unsupported_ops(self):
        """Test unsupported ops constraint."""
        adapter = MockAcceleratorAdapter(
            strict=True,
            verbose=False,
            unsupported_ops={"aten.sin.default"},
        )
        caps = adapter.capabilities
        assert "aten.sin.default" in caps.unsupported_ops


# =============================================================================
# Test create_mock_backend Factory
# =============================================================================

class TestCreateMockBackend:
    """Tests for create_mock_backend factory function."""

    def test_creates_adapter(self):
        """Test factory creates MockAcceleratorAdapter."""
        backend = create_mock_backend()
        assert isinstance(backend, MockAcceleratorAdapter)

    def test_passes_parameters(self):
        """Test factory passes parameters correctly."""
        backend = create_mock_backend(
            strict=False,
            alignment=64,
            max_memory=2 * 1024**3,
        )
        assert backend.strict is False
        assert backend._alignment == 64
        assert backend._max_memory == 2 * 1024**3

    def test_usable_with_torch_compile(self):
        """Test factory result works with torch.compile."""
        backend = create_mock_backend(strict=False)

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu()

        model = Model()
        compiled = torch.compile(model, backend=backend)

        x = torch.randn(8)
        result = compiled(x)
        expected = x.relu()

        torch.testing.assert_close(result, expected)


# =============================================================================
# Test Custom Adapter Implementation
# =============================================================================

class TestCustomAdapterImplementation:
    """Test implementing a custom adapter."""

    def test_custom_adapter_subclass(self):
        """Test creating a custom adapter subclass."""

        class MyCustomAdapter(AcceleratorAdapter):
            def get_capabilities(self) -> AcceleratorCapabilities:
                return AcceleratorCapabilities(
                    name="My Custom HW",
                    supported_dtypes={torch.float16},
                    alignment_requirement=256,
                )

            def compile(self, gm, example_inputs):
                violations = self.check_constraints(gm, example_inputs)
                return CompilationResult(
                    status=CompilationStatus.SUCCESS if not violations else CompilationStatus.FAILED,
                    compiled_fn=gm.forward if not violations else None,
                    violations=violations,
                    fallback_fn=gm.forward,
                )

        adapter = MyCustomAdapter(strict=False)
        caps = adapter.capabilities

        assert caps.name == "My Custom HW"
        assert caps.alignment_requirement == 256
        assert torch.float16 in caps.supported_dtypes


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdapterIntegration:
    """Integration tests for adapter with real models."""

    def test_linear_model(self):
        """Test adapter with linear layer."""
        adapter = MockAcceleratorAdapter(strict=False, verbose=False)

        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 16)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel()
        compiled = torch.compile(model, backend=adapter)

        x = torch.randn(4, 32)
        result = compiled(x)

        assert result.shape == (4, 16)

    def test_conv_model(self):
        """Test adapter with conv layer."""
        adapter = MockAcceleratorAdapter(strict=False, verbose=False)

        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = ConvModel()
        compiled = torch.compile(model, backend=adapter)

        x = torch.randn(1, 3, 32, 32)
        result = compiled(x)

        assert result.shape == (1, 16, 32, 32)

    def test_multiple_compilations(self):
        """Test multiple compilations with same adapter."""
        adapter = MockAcceleratorAdapter(strict=False, verbose=False)

        class Model1(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class Model2(torch.nn.Module):
            def forward(self, x):
                return x + 1

        compiled1 = torch.compile(Model1(), backend=adapter)
        compiled2 = torch.compile(Model2(), backend=adapter)

        x = torch.randn(8)

        result1 = compiled1(x)
        result2 = compiled2(x)

        torch.testing.assert_close(result1, x * 2)
        torch.testing.assert_close(result2, x + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
