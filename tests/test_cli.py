"""
Test Suite for CLI Commands
============================

Tests the debug_module CLI commands:
- list: List artifacts
- analyze: Run analysis (guards, constraints, graphs, summary)
- report: Generate reports (HTML, JSON)
- clean: Clean artifacts
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import pytest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCLIList:
    """Tests for the 'list' command."""

    def test_list_command_runs(self):
        """Test that list command executes without error."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "list"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        # Should not error even if no artifacts exist
        assert result.returncode == 0

    def test_list_shows_artifacts(self):
        """Test that list command shows artifact information."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "list"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        # Should contain header or artifact info
        output = result.stdout + result.stderr
        assert "Artifact" in output or "Total" in output or "not exist" in output


class TestCLIAnalyze:
    """Tests for the 'analyze' command."""

    def test_analyze_guards(self):
        """Test guard analysis runs and produces output."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "analyze", "--type", "guards"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=60,
        )
        output = result.stdout + result.stderr
        assert "Guard Analysis" in output
        # Should show graph count
        assert "Graph Count" in output or "guard" in output.lower()

    def test_analyze_summary(self):
        """Test summary analysis runs."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "analyze", "--type", "summary"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        output = result.stdout + result.stderr
        assert "Analysis Summary" in output or "Artifact" in output

    def test_analyze_graphs(self):
        """Test graph analysis runs."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "analyze", "--type", "graphs"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        output = result.stdout + result.stderr
        assert "Graph Analysis" in output

    def test_analyze_constraints(self):
        """Test constraint analysis runs."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "analyze", "--type", "constraints"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        output = result.stdout + result.stderr
        assert "Constraint" in output


class TestCLIReport:
    """Tests for the 'report' command."""

    def test_report_html_generation(self):
        """Test HTML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-m", "debug_module", "report",
                 "--format", "html", "--output", tmpdir],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            output = result.stdout + result.stderr
            assert "HTML report saved" in output or "html" in output.lower()

            # Check that an HTML file was created
            html_files = [f for f in os.listdir(tmpdir) if f.endswith('.html')]
            assert len(html_files) > 0, "No HTML file generated"

    def test_report_json_generation(self):
        """Test JSON report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-m", "debug_module", "report",
                 "--format", "json", "--output", tmpdir],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            output = result.stdout + result.stderr
            assert "JSON report saved" in output

            # Check that a JSON file was created and is valid
            json_files = [f for f in os.listdir(tmpdir) if f.endswith('.json')]
            assert len(json_files) > 0, "No JSON file generated"

            # Validate JSON
            with open(os.path.join(tmpdir, json_files[0]), 'r') as f:
                data = json.load(f)
            assert 'timestamp' in data
            assert 'artifacts' in data


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self):
        """Test main help displays correctly."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "list" in result.stdout
        assert "analyze" in result.stdout
        assert "report" in result.stdout
        assert "clean" in result.stdout

    def test_analyze_help(self):
        """Test analyze subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "analyze", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "guards" in result.stdout
        assert "summary" in result.stdout

    def test_report_help(self):
        """Test report subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "debug_module", "report", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "html" in result.stdout
        assert "json" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
