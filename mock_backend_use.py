from mock_backend import (
    mock_backend, 
    get_last_report, 
    get_backend_constraints,
    list_unsupported_ops
)

# Compile model
compiled_model = torch.compile(model, backend=mock_backend)
output = compiled_model(input_tensor)

# Get what happened
report = get_last_report()

# Get what the backend can/can't do
constraints = get_backend_constraints()
blocked_ops = list_unsupported_ops()

# Now your debugger can:
# 1. Show which ops in the model are blocked by the backend
# 2. Check if dtypes match backend requirements
# 3. Explain WHY something failed (check against constraints)
# 4. Generate recommendations ("Backend doesn't support aten.einsum, consider alternative")

print(f"Backend blocks {len(blocked_ops)} operations")
print(f"Your model tried to use {len(report.unsupported_ops_found)} of them")
