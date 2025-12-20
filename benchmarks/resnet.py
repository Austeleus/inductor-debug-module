"""
ResNet Benchmark for TorchInductor Debug Module.

Tests the debug module with ResNet-18 (~11M parameters),
a convolutional neural network for image classification.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.base import BaseBenchmark, BenchmarkResult


class ResNetBenchmark(BaseBenchmark):
    """Benchmark for ResNet-18."""

    def __init__(
        self,
        num_warmup: int = 2,
        num_inference: int = 5,
        device: str = "cpu",
        batch_size: int = 4,
        image_size: int = 224,
        variant: str = "resnet18",  # resnet18, resnet34, resnet50
    ):
        super().__init__(num_warmup, num_inference, device)

        self.model_name = f"ResNet-{variant.replace('resnet', '')}"
        self.model_id = f"torchvision.models.{variant}"
        self.model_type = "cnn"
        self.batch_size = batch_size
        self.image_size = image_size
        self.variant = variant

    def setup(self):
        """Load ResNet model and prepare image inputs."""
        import torchvision.models as models

        # Get the model constructor
        model_fn = getattr(models, self.variant, None)
        if model_fn is None:
            raise ValueError(f"Unknown ResNet variant: {self.variant}")

        print(f"  Loading model: {self.variant} (pretrained=False)")
        # Use pretrained=False for faster loading during benchmarks
        self.model = model_fn(pretrained=False)
        self.model.eval()
        self.model.to(self.device)

        # Create random image inputs (simulating ImageNet-like data)
        # Shape: [batch_size, 3, image_size, image_size]
        print(f"  Creating input tensor: [{self.batch_size}, 3, {self.image_size}, {self.image_size}]")
        self.example_inputs = torch.randn(
            self.batch_size, 3, self.image_size, self.image_size,
            dtype=torch.float32,
            device=self.device,
        )

        print(f"  Input shape: {self.example_inputs.shape}")


class ResNet18Benchmark(ResNetBenchmark):
    """Convenience class for ResNet-18."""

    def __init__(self, **kwargs):
        kwargs.setdefault("variant", "resnet18")
        super().__init__(**kwargs)


class ResNet50Benchmark(ResNetBenchmark):
    """Convenience class for ResNet-50."""

    def __init__(self, **kwargs):
        kwargs.setdefault("variant", "resnet50")
        super().__init__(**kwargs)


def run_resnet_benchmark(
    num_warmup: int = 2,
    num_inference: int = 5,
    device: str = "cpu",
    variant: str = "resnet18",
    backend: str = "mock",
    save_results: bool = True,
) -> BenchmarkResult:
    """Run the ResNet benchmark."""

    # Set mock backend to non-strict mode
    os.environ["MOCK_STRICT"] = "0"
    os.environ["MOCK_ALIGNMENT"] = "1"

    benchmark = ResNetBenchmark(
        num_warmup=num_warmup,
        num_inference=num_inference,
        device=device,
        variant=variant,
    )

    result = benchmark.run(test_backend=backend)

    if save_results:
        result.save()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ResNet benchmark")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs")
    parser.add_argument("--variant", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    result = run_resnet_benchmark(
        num_warmup=args.warmup,
        num_inference=args.runs,
        device=args.device,
        variant=args.variant,
        save_results=not args.no_save,
    )
