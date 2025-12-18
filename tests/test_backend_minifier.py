import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.backend.compiler import mock_compile
from debug_module.utils import BackendCompilerFailed
from tests.fx_utils import trace_with_metadata


class BackendMinifierIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.repro_dir = os.path.join("debug_artifacts", "repros")
        os.makedirs(self.repro_dir, exist_ok=True)
        self._preexisting = set(os.listdir(self.repro_dir))
        os.environ["MOCK_STRICT"] = "1"
        os.environ["MOCK_ALIGNMENT"] = "1"

    def tearDown(self):
        after = set(os.listdir(self.repro_dir))
        for filename in after - self._preexisting:
            try:
                os.remove(os.path.join(self.repro_dir, filename))
            except OSError:
                pass

    def test_repro_generated_on_backend_failure(self):
        class DoubleLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(16, 16)

            def forward(self, x):
                return torch.relu(self.proj(x))

        model = DoubleLinear().double()
        inputs = (torch.randn(2, 16, dtype=torch.float64),)
        gm = trace_with_metadata(model, *inputs)

        with self.assertRaises(BackendCompilerFailed) as ctx:
            mock_compile(gm, list(inputs))

        message = str(ctx.exception)
        self.assertIn("Mock Backend Compilation Failed", message)
        self.assertIn("Repro script saved to:", message)

        new_files = set(os.listdir(self.repro_dir)) - self._preexisting
        self.assertTrue(new_files, "Expected a new repro script to be created")
