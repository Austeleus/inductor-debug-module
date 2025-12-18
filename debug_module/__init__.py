from .backend.mock import mock_backend
from .guards.inspector import GuardInspector
from .diff.harness import KernelDiffHarness
from .adapters.base import AcceleratorAdapter, AcceleratorCapabilities, CompilationResult
from .adapters.mock_adapter import MockAcceleratorAdapter, create_mock_backend
from .reports.generator import HTMLReportGenerator, ReportData

__all__ = [
    # Core backend
    "mock_backend",
    # Analysis tools
    "GuardInspector",
    "KernelDiffHarness",
    # Adapter interface
    "AcceleratorAdapter",
    "AcceleratorCapabilities",
    "CompilationResult",
    "MockAcceleratorAdapter",
    "create_mock_backend",
    # Reports
    "HTMLReportGenerator",
    "ReportData",
]
