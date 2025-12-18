import os
import torch
from debug_module.backend.mock import mock_backend

os.environ["MOCK_STRICT"] = "1"
os.environ["MOCK_ALIGNMENT"] = "8"

class BadModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :17]  # violates alignment

model = BadModel()
compiled = torch.compile(model, backend=mock_backend)
compiled(torch.randn(2, 32))
