import torch
import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.backend.mock import mock_backend
from debug_module.utils import BackendCompilerFailed

def test_shape_constraint():
    print("\n=== Testing Shape Constraint (Alignment=16) ===")
    # Set env var for alignment
    os.environ["MOCK_ALIGNMENT"] = "16"
    os.environ["MOCK_STRICT"] = "1"
    
    # Create a model with odd shape (10)
    class OddModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = OddModel()
    x = torch.randn(10) # Not divisible by 16
    
    with pytest.raises(BackendCompilerFailed) as excinfo:
        opt_model = torch.compile(model, backend=mock_backend)
        opt_model(x)
    print(f"PASS: Caught expected BackendCompilerFailed: {excinfo.value}")

def test_layout_constraint():
    print("\n=== Testing Layout Constraint ===")
    os.environ["MOCK_ALIGNMENT"] = "1" # Reset alignment
    os.environ["MOCK_STRICT"] = "1"
    
    class TransposeModel(torch.nn.Module):
        def forward(self, x):
            # Transpose creates non-contiguous tensor
            y = x.transpose(0, 1)
            return y + 1

    model = TransposeModel()
    x = torch.randn(10, 10)
    
    with pytest.raises(BackendCompilerFailed) as excinfo:
        opt_model = torch.compile(model, backend=mock_backend)
        opt_model(x)
    print(f"PASS: Caught expected BackendCompilerFailed: {excinfo.value}")

def test_warning_mode():
    print("\n=== Testing Warning Mode (Strict=0) ===")
    os.environ["MOCK_ALIGNMENT"] = "16"
    os.environ["MOCK_STRICT"] = "0" # Turn off strict mode
    
    class OddModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = OddModel()
    x = torch.randn(10) # Violation
    
    try:
        opt_model = torch.compile(model, backend=mock_backend)
        opt_model(x)
        print("PASS: Ran successfully despite violation (Warning Mode)")
    except BackendCompilerFailed as e:
        pytest.fail(f"FAIL: Should NOT have raised BackendCompilerFailed: {e}")
    except Exception as e:
        pytest.fail(f"FAIL: Unexpected error type: {e}")

if __name__ == "__main__":
    test_shape_constraint()
    # test_layout_constraint() # Layout is tricky with FX, let's focus on Shape/Strict first
    test_warning_mode()
