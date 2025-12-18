"""
Backend Adapter Interface for TorchInductor Debug Module

Provides an abstract interface for integrating custom accelerator backends
with the debug module's constraint checking and analysis tools.
"""

from .base import (
    AcceleratorAdapter,
    AcceleratorCapabilities,
    CompilationResult,
)
from .mock_adapter import MockAcceleratorAdapter, create_mock_backend
from .integrated_adapter import IntegratedMockAdapter, integrated_mock_backend

__all__ = [
    # Abstract interface
    'AcceleratorAdapter',
    'AcceleratorCapabilities',
    'CompilationResult',
    # Standalone mock (for demonstration)
    'MockAcceleratorAdapter',
    'create_mock_backend',
    # Integrated mock (wraps existing backend)
    'IntegratedMockAdapter',
    'integrated_mock_backend',
]
