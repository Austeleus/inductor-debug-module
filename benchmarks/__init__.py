"""
Benchmarking module for TorchInductor Debug Module.

This module provides benchmarks for testing the debug module against
various model architectures:
- BERT (Transformer/NLP)
- ResNet (CNN/Vision)
- Mamba (State Space Model)

Usage:
    python -m benchmarks.runner --all
    python -m benchmarks.runner --model bert
"""

from .base import BenchmarkResult, BaseBenchmark

__all__ = ["BenchmarkResult", "BaseBenchmark"]
