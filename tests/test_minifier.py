import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.minifier.minifier import Minifier
from tests.fx_utils import trace_with_metadata


class MinifierFailureTests(unittest.TestCase):
    def setUp(self):
        self._env_backup = {
            "MOCK_ALIGNMENT": os.environ.get("MOCK_ALIGNMENT"),
            "MOCK_MAX_MEMORY": os.environ.get("MOCK_MAX_MEMORY"),
        }
        os.environ["MOCK_ALIGNMENT"] = "1"
        os.environ["MOCK_MAX_MEMORY"] = str(2 * 1024**3)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _run_minifier(self, model, inputs, error_message):
        gm = trace_with_metadata(model, *inputs)
        minifier = Minifier(gm, inputs, RuntimeError(error_message))
        minified = minifier.minify()
        self.assertIsNotNone(minifier.failing_node, "Minifier failed to record offending node")
        return gm, minified, minifier

    def test_cnn_convolution_failure(self):
        """CNN conv (unsupported op) should be isolated."""

        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, kernel_size=3)

            def forward(self, x):
                return self.conv(x)

        inputs = (torch.randn(1, 3, 8, 8),)
        _, minified, minifier = self._run_minifier(SimpleCNN(), inputs, "CNN convolution failure")

        self.assertIn("convolution", str(minifier.failing_node.target))
        call_targets = [n.target for n in minified.graph.nodes if n.op == "call_function"]
        self.assertEqual(len(call_targets), 1)

    def test_mamba_alignment_failure(self):
        """State-space style block with misaligned length triggers shape constraint."""

        class SimpleSSM(torch.nn.Module):
            def forward(self, x):
                seq = torch.cumsum(x, dim=1)
                return seq[:, :17, :]

        os.environ["MOCK_ALIGNMENT"] = "8"
        inputs = (torch.randn(1, 20, 16),)
        _, _, minifier = self._run_minifier(SimpleSSM(), inputs, "Mamba alignment failure")
        self.assertIn("violates alignment", minifier.failure_message)

    def test_transformer_dtype_failure(self):
        """Transformer-style FFN running in float64 should hit dtype constraint."""

        class TinyTransformerFFN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ffn = torch.nn.Linear(32, 32)

            def forward(self, x):
                return torch.nn.functional.relu(self.ffn(x))

        model = TinyTransformerFFN().double()
        inputs = (torch.randn(2, 8, 32, dtype=torch.float64),)
        _, _, minifier = self._run_minifier(model, inputs, "Transformer dtype failure")
        self.assertIn("Dtype", minifier.failure_message)

    def test_relu_inplace_failure(self):
        """In-place nonlinearities are denied operations; minifier should flag them."""

        class InplaceActivation(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                return torch.relu_(y)

        inputs = (torch.randn(4, 4),)
        _, _, minifier = self._run_minifier(InplaceActivation(), inputs, "In-place relu failure")
        self.assertIn("relu_", str(minifier.failing_node.target))

    def test_int8_cast_dtype_failure(self):
        """Casts to unsupported int8 dtype should require minification."""

        class Int8Caster(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.int8)

        inputs = (torch.ones(3, 3),)
        _, _, minifier = self._run_minifier(Int8Caster(), inputs, "Int8 dtype failure")
        self.assertIn("int8", minifier.failure_message)


if __name__ == "__main__":
    unittest.main()
