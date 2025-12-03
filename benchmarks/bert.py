"""
BERT Benchmark for TorchInductor Debug Module.

Tests the debug module with BERT-base-multilingual-cased (~178M parameters),
a transformer encoder model for NLP tasks.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.base import BaseBenchmark, BenchmarkResult


class BERTBenchmark(BaseBenchmark):
    """Benchmark for BERT-base-multilingual-cased."""

    MODEL_ID = "google-bert/bert-base-multilingual-cased"

    def __init__(
        self,
        num_warmup: int = 2,
        num_inference: int = 5,
        device: str = "cpu",
        batch_size: int = 2,
        seq_length: int = 32,
    ):
        super().__init__(num_warmup, num_inference, device)

        self.model_name = "BERT-base-multilingual"
        self.model_id = self.MODEL_ID
        self.model_type = "transformer"
        self.batch_size = batch_size
        self.seq_length = seq_length

    def setup(self):
        """Load BERT model and prepare tokenized inputs."""
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        print(f"  Loading tokenizer: {self.MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        print(f"  Loading model: {self.MODEL_ID}")
        self.model = AutoModelForMaskedLM.from_pretrained(self.MODEL_ID)
        self.model.eval()
        self.model.to(self.device)

        # Create example inputs
        sample_texts = [
            "Paris is the [MASK] of France.",
            "The quick brown [MASK] jumps over the lazy dog.",
        ]

        # Extend to batch_size if needed
        while len(sample_texts) < self.batch_size:
            sample_texts.extend(sample_texts)
        sample_texts = sample_texts[:self.batch_size]

        print(f"  Tokenizing {len(sample_texts)} samples (max_length={self.seq_length})")
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


def run_bert_benchmark(
    num_warmup: int = 2,
    num_inference: int = 5,
    device: str = "cpu",
    save_results: bool = True,
) -> BenchmarkResult:
    """Run the BERT benchmark."""

    # Set mock backend to non-strict mode
    os.environ["MOCK_STRICT"] = "0"
    os.environ["MOCK_ALIGNMENT"] = "1"

    benchmark = BERTBenchmark(
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

    parser = argparse.ArgumentParser(description="Run BERT benchmark")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    result = run_bert_benchmark(
        num_warmup=args.warmup,
        num_inference=args.runs,
        device=args.device,
        save_results=not args.no_save,
    )
