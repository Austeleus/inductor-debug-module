"""
Test Suite for HTML Report Generator
=====================================

Tests the debug_module.reports module:
- ReportData dataclass
- HTMLReportGenerator class
- Report content validation
"""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_module.reports import HTMLReportGenerator, ReportData


class TestReportData:
    """Tests for ReportData dataclass."""

    def test_default_initialization(self):
        """Test ReportData initializes with defaults."""
        data = ReportData()
        assert data.timestamp == ""
        assert data.artifacts == {}
        assert data.benchmarks == []
        assert data.constraint_warnings == {}
        assert data.kerneldiff_results == []
        assert data.visualizations == []

    def test_custom_initialization(self):
        """Test ReportData with custom values."""
        data = ReportData(
            timestamp="2024-01-01 12:00:00",
            artifacts={'graphs': 5, 'repros': 2},
            benchmarks=[{'model': 'test', 'passed': True}],
        )
        assert data.timestamp == "2024-01-01 12:00:00"
        assert data.artifacts['graphs'] == 5
        assert len(data.benchmarks) == 1


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator class."""

    def test_initialization(self):
        """Test generator initializes correctly."""
        generator = HTMLReportGenerator()
        assert generator.artifact_dir == "debug_artifacts"
        assert generator.benchmark_dir == "benchmarks/results"

    def test_custom_directories(self):
        """Test generator with custom directories."""
        generator = HTMLReportGenerator(
            artifact_dir="/custom/artifacts",
            benchmark_dir="/custom/benchmarks",
        )
        assert generator.artifact_dir == "/custom/artifacts"
        assert generator.benchmark_dir == "/custom/benchmarks"

    def test_collect_data(self):
        """Test data collection returns ReportData."""
        generator = HTMLReportGenerator()
        data = generator.collect_data()

        assert isinstance(data, ReportData)
        assert data.timestamp != ""  # Should be populated
        assert isinstance(data.artifacts, dict)

    def test_generate_creates_file(self):
        """Test generate() creates an HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HTMLReportGenerator()
            filepath = generator.generate(tmpdir)

            assert os.path.exists(filepath)
            assert filepath.endswith('.html')

            # Read and validate content
            with open(filepath, 'r') as f:
                content = f.read()

            assert '<!DOCTYPE html>' in content
            assert '<html' in content
            assert '</html>' in content

    def test_html_contains_required_sections(self):
        """Test generated HTML contains all required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HTMLReportGenerator()
            filepath = generator.generate(tmpdir)

            with open(filepath, 'r') as f:
                content = f.read()

            # Check for main sections
            assert 'TorchInductor Debug' in content
            assert 'Summary' in content
            assert 'Artifacts' in content

    def test_html_has_valid_structure(self):
        """Test HTML has proper structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HTMLReportGenerator()
            filepath = generator.generate(tmpdir)

            with open(filepath, 'r') as f:
                content = f.read()

            # Check for proper HTML structure
            assert '<head>' in content
            assert '</head>' in content
            assert '<body>' in content
            assert '</body>' in content
            assert '<style>' in content  # CSS should be embedded

    def test_html_has_styling(self):
        """Test HTML includes CSS styling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HTMLReportGenerator()
            filepath = generator.generate(tmpdir)

            with open(filepath, 'r') as f:
                content = f.read()

            # Check for CSS variables and styling
            assert '--primary' in content or 'color:' in content
            assert '.card' in content or 'padding' in content


class TestReportDataIntegration:
    """Integration tests for report data collection."""

    def test_artifacts_counted(self):
        """Test that artifacts are counted correctly."""
        generator = HTMLReportGenerator()
        data = generator.collect_data()

        # artifacts should be a dict with string keys and int values
        for key, value in data.artifacts.items():
            assert isinstance(key, str)
            assert isinstance(value, int)
            assert value >= 0

    def test_benchmarks_loaded(self):
        """Test that benchmarks are loaded if available."""
        generator = HTMLReportGenerator()
        data = generator.collect_data()

        # benchmarks is a list (may be empty if no benchmarks run)
        assert isinstance(data.benchmarks, list)

    def test_visualizations_found(self):
        """Test that visualizations list is populated."""
        generator = HTMLReportGenerator()
        data = generator.collect_data()

        assert isinstance(data.visualizations, list)
        # All items should be file paths if any exist
        for viz in data.visualizations:
            assert isinstance(viz, str)


class TestReportEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_artifact_dir(self):
        """Test handling of empty/nonexistent artifact directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HTMLReportGenerator(
                artifact_dir=os.path.join(tmpdir, "nonexistent"),
                benchmark_dir=os.path.join(tmpdir, "nonexistent2"),
            )
            data = generator.collect_data()

            # Should not crash, just have empty counts
            assert all(v == 0 for v in data.artifacts.values())

    def test_generate_creates_output_dir(self):
        """Test that generate() creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "new", "nested", "dir")
            generator = HTMLReportGenerator()

            filepath = generator.generate(output_dir)

            assert os.path.exists(output_dir)
            assert os.path.exists(filepath)

    def test_multiple_reports(self):
        """Test generating multiple reports doesn't overwrite."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = HTMLReportGenerator()

            filepath1 = generator.generate(tmpdir)
            time.sleep(1.1)  # Ensure different timestamp
            filepath2 = generator.generate(tmpdir)

            # Should create different files
            assert filepath1 != filepath2
            assert os.path.exists(filepath1)
            assert os.path.exists(filepath2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
