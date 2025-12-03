import torch
from debug_module import mock_backend
from debug_module.backend.compiler import BackendCompilerFailed

def test_success():
    print("\n=== Test 1: Success Case (Float32) ===")
    def model(x, y):
        return torch.add(x, y)

    x = torch.randn(10, 10, dtype=torch.float32)
    y = torch.randn(10, 10, dtype=torch.float32)

    opt_model = torch.compile(model, backend=mock_backend)
    
    try:
        res = opt_model(x, y)
        print("Success! Result shape:", res.shape)
    except Exception as e:
        print("FAILED:", e)

def test_failure_dtype():
    print("\n=== Test 2: Failure Case (Float64) ===")
    def model(x, y):
        return torch.add(x, y)

    # Float64 is banned by default in our registry.py
    x = torch.randn(10, 10, dtype=torch.float64)
    y = torch.randn(10, 10, dtype=torch.float64)

    opt_model = torch.compile(model, backend=mock_backend)
    
    try:
        res = opt_model(x, y)
        print("UNEXPECTED SUCCESS (Should have failed)")
    except Exception as e:
        # We expect a backend failure, which might be wrapped by Inductor/Dynamo
        print("Caught expected error:")
        print(e)

if __name__ == "__main__":
    test_success()
    test_failure_dtype()
