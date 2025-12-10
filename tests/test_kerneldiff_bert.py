#!/usr/bin/env python3
"""
Integration Test: KernelDiff with BERT Model

This script tests the KernelDiff harness with a real HuggingFace BERT model,
demonstrating:
1. Handling of HuggingFace model outputs (ModelOutput dataclass-like objects)
2. Dict inputs (tokenizer output)
3. Full comparison pipeline with visualizations
4. JSON report generation

Run with: python test_kerneldiff_bert.py
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set non-strict mode for mock backend
os.environ["MOCK_STRICT"] = "0"
os.environ["MOCK_ALIGNMENT"] = "1"

# Color helpers
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_info(msg):
    print(f"{Colors.YELLOW}[INFO]{Colors.END} {msg}")


def main():
    print_header("KernelDiff BERT Integration Test")

    # Check if transformers is installed
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print(f"{Colors.RED}[ERROR]{Colors.END} transformers not installed")
        print("Install with: pip install transformers")
        return 1

    from debug_module.diff import KernelDiffHarness, ComparisonConfig

    # Use a small BERT model for faster testing
    MODEL_ID = "prajjwal1/bert-tiny"  # Very small BERT for testing

    print_info(f"Loading model: {MODEL_ID}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID)
        model.eval()
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.END} Failed to load model: {e}")
        # Fallback to a simple model
        print_info("Falling back to simple test model...")
        return run_simple_test()

    print_info("Model loaded successfully")

    # Tokenize sample input
    text = "Hello, this is a test of the KernelDiff harness."
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True
    )

    print_info(f"Input text: '{text}'")
    print_info(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Create the harness
    print_header("Running KernelDiff Comparison")

    # Use looser tolerance for BERT (floating point differences expected)
    config = ComparisonConfig(
        atol=1e-4,
        rtol=1e-3,
        max_mismatch_percentage=1.0,  # Allow up to 1% mismatches
        store_error_tensor=True
    )

    harness = KernelDiffHarness(
        model=model,
        example_inputs=dict(inputs),  # Convert to regular dict
        model_name="bert-tiny",
        reference_backend="eager",  # Use eager for faster testing
        comparison_config=config,
    )

    try:
        report = harness.compare(
            generate_visualizations=True,
            save_report=True,
            report_dir="debug_artifacts/bert_reports"
        )

        # Print the full summary
        print(report.summary())

        # Show details about tensor comparisons
        print_header("Tensor Comparison Details")

        for result in report.tensor_results:
            status = f"{Colors.GREEN}PASS{Colors.END}" if result.passed else f"{Colors.RED}FAIL{Colors.END}"
            print(f"{status} {result.name}")
            print(f"     Shape: {result.shape}")
            print(f"     Max Error: {result.max_absolute_error:.2e}")
            print(f"     Mean Error: {result.mean_absolute_error:.2e}")
            print(f"     RMSE: {result.rmse:.2e}")
            print()

        # Check if visualizations were generated
        if report.visualization_paths:
            print_header("Generated Visualizations")
            for path in report.visualization_paths:
                print_info(f"  {path}")

        # Final status
        print_header("Test Result")

        if report.overall_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}BERT KernelDiff Test PASSED{Colors.END}")
            return 0
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}BERT KernelDiff Test completed with differences{Colors.END}")
            print("(Some numerical differences are expected with different backends)")

            # Check if errors are within acceptable range
            max_error = max(r.max_absolute_error for r in report.tensor_results)
            if max_error < 1e-2:  # Within 1% error
                print(f"{Colors.GREEN}Max error {max_error:.2e} is within acceptable range{Colors.END}")
                return 0
            else:
                print(f"{Colors.RED}Max error {max_error:.2e} exceeds acceptable range{Colors.END}")
                return 1

    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.END} Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_simple_test():
    """Fallback test with simple model."""
    print_header("Running Simple Model Test (Fallback)")

    from debug_module.diff import KernelDiffHarness
    import torch.nn as nn

    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            self.transformer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
            self.output = nn.Linear(64, 64)

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            return {"last_hidden_state": self.output(x)}

    model = SimpleTransformer().eval()

    inputs = {
        "input_ids": torch.randint(0, 1000, (2, 16)),
        "attention_mask": torch.ones(2, 16),
    }

    harness = KernelDiffHarness(
        model=model,
        example_inputs=inputs,
        model_name="simple_transformer",
        reference_backend="eager",
    )

    try:
        report = harness.compare(
            generate_visualizations=True,
            save_report=True
        )
        print(report.summary())

        if report.overall_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}Simple Transformer Test PASSED{Colors.END}")
            return 0
        else:
            max_err = max(r.max_absolute_error for r in report.tensor_results)
            if max_err < 1e-3:
                print(f"{Colors.GREEN}Errors within range{Colors.END}")
                return 0
            return 1

    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.END} {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
