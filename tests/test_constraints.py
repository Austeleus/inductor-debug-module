import torch
import torch._dynamo.exc
import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.backend.mock import mock_backend

# PyTorch wraps backend exceptions in torch._dynamo.exc.BackendCompilerFailed
DynamoBackendError = torch._dynamo.exc.BackendCompilerFailed


def test_shape_constraint():
    """Test that shape alignment constraint rejects misaligned tensors in strict mode."""
    print("\n=== Testing Shape Constraint (Alignment=16) ===")
    # Set env var for alignment
    os.environ["MOCK_ALIGNMENT"] = "16"
    os.environ["MOCK_STRICT"] = "1"

    # Create a model with odd shape (10)
    class OddModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = OddModel()
    x = torch.randn(10)  # Not divisible by 16

    with pytest.raises(DynamoBackendError) as excinfo:
        opt_model = torch.compile(model, backend=mock_backend)
        opt_model(x)

    # Verify the error message mentions alignment
    assert "alignment" in str(excinfo.value).lower() or "shape" in str(excinfo.value).lower()
    print(f"PASS: Caught expected BackendCompilerFailed with alignment violation")

def test_layout_constraint():
    """Test that layout constraint rejects non-contiguous tensors in strict mode."""
    print("\n=== Testing Layout Constraint ===")
    os.environ["MOCK_ALIGNMENT"] = "1"  # Reset alignment
    os.environ["MOCK_STRICT"] = "1"

    class TransposeModel(torch.nn.Module):
        def forward(self, x):
            # Transpose creates non-contiguous tensor
            y = x.transpose(0, 1)
            return y + 1

    model = TransposeModel()
    x = torch.randn(10, 10)

    with pytest.raises(DynamoBackendError) as excinfo:
        opt_model = torch.compile(model, backend=mock_backend)
        opt_model(x)

    # Verify the error message mentions layout/contiguous
    error_msg = str(excinfo.value).lower()
    assert "contiguous" in error_msg or "layout" in error_msg
    print(f"PASS: Caught expected BackendCompilerFailed with layout violation")


def test_warning_mode():
    """Test that warning mode (non-strict) allows execution despite violations."""
    print("\n=== Testing Warning Mode (Strict=0) ===")
    os.environ["MOCK_ALIGNMENT"] = "16"
    os.environ["MOCK_STRICT"] = "0"  # Turn off strict mode

    class OddModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = OddModel()
    x = torch.randn(10)  # Violation - not aligned to 16

    # Should NOT raise an exception in warning mode
    opt_model = torch.compile(model, backend=mock_backend)
    result = opt_model(x)

    # Verify we got a valid result
    assert result is not None
    assert result.shape == x.shape
    print("PASS: Ran successfully despite violation (Warning Mode)")


def test_dtype_constraint():
    """Test that dtype constraint rejects unsupported dtypes in strict mode."""
    print("\n=== Testing Dtype Constraint (float64) ===")
    os.environ["MOCK_ALIGNMENT"] = "1"
    os.environ["MOCK_STRICT"] = "1"

    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = SimpleModel()
    x = torch.randn(8, 8, dtype=torch.float64)  # float64 not supported

    with pytest.raises(DynamoBackendError) as excinfo:
        opt_model = torch.compile(model, backend=mock_backend)
        opt_model(x)

    # Verify the error message mentions dtype
    error_msg = str(excinfo.value).lower()
    assert "dtype" in error_msg or "float64" in error_msg
    print(f"PASS: Caught expected BackendCompilerFailed with dtype violation")


def test_memory_constraint():
    """Test that memory constraint rejects tensors exceeding memory limit."""
    print("\n=== Testing Memory Constraint ===")
    os.environ["MOCK_ALIGNMENT"] = "1"
    os.environ["MOCK_STRICT"] = "1"
    os.environ["MOCK_MAX_MEMORY"] = "1024"  # Very small: 1KB

    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = SimpleModel()
    # 100x100 float32 = 40,000 bytes > 1KB limit
    x = torch.randn(100, 100, dtype=torch.float32)

    with pytest.raises(DynamoBackendError) as excinfo:
        opt_model = torch.compile(model, backend=mock_backend)
        opt_model(x)

    # Verify the error message mentions memory/size/limit
    error_msg = str(excinfo.value).lower()
    assert "memory" in error_msg or "exceeding limit" in error_msg or "bytes" in error_msg
    print(f"PASS: Caught expected BackendCompilerFailed with memory violation")

    # Reset memory limit
    os.environ["MOCK_MAX_MEMORY"] = str(16 * 1024**3)


if __name__ == "__main__":
    test_shape_constraint()
    test_layout_constraint()
    test_warning_mode()
    test_dtype_constraint()
    test_memory_constraint()
