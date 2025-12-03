"""
Mamba SSM Benchmark for TorchInductor Debug Module.

Tests the debug module with Mamba-130M (~129M parameters),
a state space model for sequence modeling.

Mamba is a selective structured state space model (SSM) that provides
an alternative to Transformers for sequence modeling with linear
scaling in sequence length.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.base import BaseBenchmark, BenchmarkResult


class MambaBenchmark(BaseBenchmark):
    """Benchmark for Mamba-130M State Space Model."""

    MODEL_ID = "state-spaces/mamba-130m-hf"

    def __init__(
        self,
        num_warmup: int = 2,
        num_inference: int = 5,
        device: str = "cpu",
        batch_size: int = 2,
        seq_length: int = 64,
    ):
        super().__init__(num_warmup, num_inference, device)

        self.model_name = "Mamba-130M"
        self.model_id = self.MODEL_ID
        self.model_type = "ssm"
        self.batch_size = batch_size
        self.seq_length = seq_length

    def setup(self):
        """Load Mamba model and prepare tokenized inputs."""
        from transformers import AutoTokenizer, MambaForCausalLM

        print(f"  Loading tokenizer: {self.MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        print(f"  Loading model: {self.MODEL_ID}")
        # Note: Will show warning about fast path not available, that's expected
        self.model = MambaForCausalLM.from_pretrained(self.MODEL_ID)
        self.model.eval()
        self.model.to(self.device)

        # Create example inputs
        sample_texts = [
            "The state space model processes sequences",
            "Mamba is an efficient alternative to transformers",
        ]

        # Extend to batch_size if needed
        while len(sample_texts) < self.batch_size:
            sample_texts.extend(sample_texts)
        sample_texts = sample_texts[:self.batch_size]

        print(f"  Tokenizing {len(sample_texts)} samples (max_length={self.seq_length})")

        # Mamba tokenizer might not have pad_token, use eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.example_inputs = self.tokenizer(
            sample_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
        )

        # Move to device
        self.example_inputs = {k: v.to(self.device) for k, v in self.example_inputs.items()}

        print(f"  Input shape: {self.example_inputs['input_ids'].shape}")


class MambaSmallBenchmark(BaseBenchmark):
    """
    Benchmark using a custom small Mamba configuration.

    Uses a pure PyTorch SSM-like model that is compatible with torch.compile.
    The HuggingFace Mamba implementation has torch.compile issues due to
    `mark_static_address` in the cache initialization.
    """

    def __init__(
        self,
        num_warmup: int = 2,
        num_inference: int = 5,
        device: str = "cpu",
        batch_size: int = 2,
        seq_length: int = 64,
        hidden_size: int = 256,
        num_layers: int = 4,
    ):
        super().__init__(num_warmup, num_inference, device)

        self.model_name = "SSM-Small (Custom)"
        self.model_id = "custom-ssm-small"
        self.model_type = "ssm"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def setup(self):
        """Create a small SSM-like model from scratch that works with torch.compile."""
        import torch.nn as nn

        class SimpleSSMBlock(nn.Module):
            """A simplified SSM-inspired block for benchmarking."""

            def __init__(self, hidden_size, state_size=16, expand=2):
                super().__init__()
                inner_size = hidden_size * expand

                # Input projection
                self.in_proj = nn.Linear(hidden_size, inner_size * 2)

                # SSM parameters (simplified)
                self.conv1d = nn.Conv1d(inner_size, inner_size, kernel_size=4, padding=3, groups=inner_size)

                # State space parameters
                self.x_proj = nn.Linear(inner_size, state_size * 2)
                self.dt_proj = nn.Linear(state_size, inner_size)

                # Output projection
                self.out_proj = nn.Linear(inner_size, hidden_size)

                self.layer_norm = nn.LayerNorm(hidden_size)

            def forward(self, x):
                residual = x
                x = self.layer_norm(x)

                # Project and split
                xz = self.in_proj(x)
                x, z = xz.chunk(2, dim=-1)

                # Conv (transpose for conv1d)
                x = x.transpose(1, 2)
                x = self.conv1d(x)[:, :, :x.shape[2]]
                x = x.transpose(1, 2)

                # Simplified SSM-like operation
                x = x * torch.sigmoid(z)

                # Output
                x = self.out_proj(x)
                return x + residual

        class SimpleSSMModel(nn.Module):
            """A simple SSM-like model for benchmarking."""

            def __init__(self, vocab_size, hidden_size, num_layers, state_size=16):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    SimpleSSMBlock(hidden_size, state_size) for _ in range(num_layers)
                ])
                self.norm = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                x = self.norm(x)
                logits = self.lm_head(x)
                return logits

        print(f"  Creating custom SSM model:")
        print(f"    hidden_size={self.hidden_size}")
        print(f"    num_layers={self.num_layers}")

        vocab_size = 50280
        self.model = SimpleSSMModel(
            vocab_size=vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        self.model.eval()
        self.model.to(self.device)

        # Create random input_ids
        print(f"  Creating input tensor: [{self.batch_size}, {self.seq_length}]")
        self.example_inputs = torch.randint(
            0, vocab_size,
            (self.batch_size, self.seq_length),
            dtype=torch.long,
            device=self.device,
        )

        print(f"  Input shape: {self.example_inputs.shape}")


def run_mamba_benchmark(
    num_warmup: int = 2,
    num_inference: int = 5,
    device: str = "cpu",
    use_pretrained: bool = True,
    save_results: bool = True,
) -> BenchmarkResult:
    """Run the Mamba benchmark."""

    # Set mock backend to non-strict mode
    os.environ["MOCK_STRICT"] = "0"
    os.environ["MOCK_ALIGNMENT"] = "1"

    if use_pretrained:
        benchmark = MambaBenchmark(
            num_warmup=num_warmup,
            num_inference=num_inference,
            device=device,
        )
    else:
        benchmark = MambaSmallBenchmark(
            num_warmup=num_warmup,
            num_inference=num_inference,
            device=device,
        )

    result = benchmark.run()

    if save_results:
        result.save()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Mamba SSM benchmark")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs")
    parser.add_argument("--small", action="store_true", help="Use small custom model instead of pretrained")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    result = run_mamba_benchmark(
        num_warmup=args.warmup,
        num_inference=args.runs,
        device=args.device,
        use_pretrained=not args.small,
        save_results=not args.no_save,
    )
