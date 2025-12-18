"""
TorchInductor Debug Module CLI

Provides commands for:
- Listing and managing artifacts
- Running analysis (guards, constraints)
- Generating reports (JSON, HTML)
"""

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Optional

ARTIFACT_DIR = "debug_artifacts"


# =============================================================================
# Utility Functions
# =============================================================================

def _find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Find the most recently modified file matching pattern."""
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_json_report(filepath: str) -> dict:
    """Load a JSON report file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def _print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}\n")


# =============================================================================
# List Command
# =============================================================================

def list_artifacts(args):
    """List all captured artifacts with details."""
    if not os.path.exists(ARTIFACT_DIR):
        print(f"Artifact directory '{ARTIFACT_DIR}' does not exist.")
        return

    _print_section("Artifact Listing")

    # Walk through all directories
    total_files = 0
    total_size = 0

    for root, dirs, files in os.walk(ARTIFACT_DIR):
        rel_path = os.path.relpath(root, ARTIFACT_DIR)
        if rel_path == ".":
            rel_path = ""

        for f in sorted(files):
            filepath = os.path.join(root, f)
            stat = os.stat(filepath)
            size = stat.st_size
            total_size += size
            total_files += 1

            size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
            created_str = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

            display_path = os.path.join(rel_path, f) if rel_path else f
            print(f"  {display_path:<50} {size_str:>10}  {created_str}")

    print(f"\n{'-' * 60}")
    print(f"Total: {total_files} files, {total_size / 1024:.1f} KB")


# =============================================================================
# Clean Command
# =============================================================================

def clean_artifacts(args):
    """Remove all captured artifacts."""
    if not os.path.exists(ARTIFACT_DIR):
        print(f"Artifact directory '{ARTIFACT_DIR}' does not exist.")
        return

    if not args.force:
        confirm = input(f"Delete all files in '{ARTIFACT_DIR}'? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    shutil.rmtree(ARTIFACT_DIR)
    os.makedirs(ARTIFACT_DIR)
    print(f"Cleaned '{ARTIFACT_DIR}'.")


# =============================================================================
# Analyze Command
# =============================================================================

def analyze_artifacts(args):
    """Analyze artifacts and display results."""

    if args.type == 'guards':
        _analyze_guards(args)
    elif args.type == 'constraints':
        _analyze_constraints(args)
    elif args.type == 'graphs':
        _analyze_graphs(args)
    elif args.type == 'summary':
        _analyze_summary(args)
    else:
        print("Please specify analysis type: --type [guards|constraints|graphs|summary]")


def _analyze_guards(args):
    """Analyze guards using a demo model or provided script."""
    _print_section("Guard Analysis")

    print("Running guard inspection on demo model...\n")

    try:
        import torch
        from debug_module.guards.inspector import GuardInspector

        # Create a simple demo model for analysis
        class DemoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 32)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        model = DemoModel().eval()
        inspector = GuardInspector(model)

        # Run inspection
        x = torch.randn(4, 64)
        report = inspector.inspect({"x": x})

        # Display results
        print(f"Graph Count: {report['graph_count']}")
        print(f"Graph Breaks: {report['graph_break_count']}")

        if report['break_reasons']:
            print(f"\nBreak Reasons:")
            for reason in report['break_reasons']:
                print(f"  - {reason}")

        for graph in report['graphs']:
            print(f"\nGraph '{graph['id']}':")
            print(f"  Guards: {len(graph['guards'])}")
            for guard in graph['guards'][:5]:
                print(f"    - {guard['text'][:70]}...")
            if len(graph['guards']) > 5:
                print(f"    ... and {len(graph['guards']) - 5} more")

        # Save JSON report
        report_dir = os.path.join(ARTIFACT_DIR, "reports")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"guard_analysis_{int(datetime.now().timestamp())}.json")

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[Saved] JSON report: {report_path}")

    except Exception as e:
        print(f"Error during guard analysis: {e}")
        print("\nTo analyze guards for a custom model, use the Python API:")
        print("  from debug_module import GuardInspector")
        print("  inspector = GuardInspector(model)")
        print("  report = inspector.inspect(inputs)")


def _analyze_constraints(args):
    """Analyze constraint violations from captured artifacts."""
    _print_section("Constraint Analysis")

    # Find benchmark results
    results_dir = "benchmarks/results"
    if not os.path.exists(results_dir):
        print("No benchmark results found. Run benchmarks first:")
        print("  python -m benchmarks.runner --all")
        return

    # Load latest summary
    summary_file = _find_latest_file(results_dir, "benchmark_summary_*.json")
    if not summary_file:
        print("No benchmark summary found.")
        return

    data = _load_json_report(summary_file)

    print(f"Loaded: {os.path.basename(summary_file)}\n")

    total_warnings = 0
    for result in data.get('results', []):
        model_name = result.get('model_name', 'Unknown')
        mock_result = result.get('mock_result', {})
        warnings = mock_result.get('constraint_warnings', [])

        print(f"{model_name}:")
        print(f"  Total warnings: {len(warnings)}")

        if warnings:
            # Group by type
            warning_types = {}
            for w in warnings:
                # Extract warning type from message
                if 'non-contiguous' in w.lower():
                    wtype = 'Layout (non-contiguous)'
                elif 'dtype' in w.lower():
                    wtype = 'Dtype'
                elif 'alignment' in w.lower() or 'shape' in w.lower():
                    wtype = 'Shape (alignment)'
                elif 'memory' in w.lower() or 'bytes' in w.lower():
                    wtype = 'Memory'
                elif 'unsupported' in w.lower():
                    wtype = 'Unsupported Op'
                else:
                    wtype = 'Other'
                warning_types[wtype] = warning_types.get(wtype, 0) + 1

            for wtype, count in sorted(warning_types.items(), key=lambda x: -x[1]):
                print(f"    - {wtype}: {count}")

        total_warnings += len(warnings)
        print()

    print(f"{'-' * 40}")
    print(f"Total constraint warnings: {total_warnings}")


def _analyze_graphs(args):
    """Analyze captured FX graphs."""
    _print_section("Graph Analysis")

    graph_files = glob.glob(os.path.join(ARTIFACT_DIR, "graph_*.txt"))
    fx_graphs = glob.glob(os.path.join(ARTIFACT_DIR, "fx_graphs", "*.txt"))
    aot_graphs = glob.glob(os.path.join(ARTIFACT_DIR, "aot_graphs", "*.txt"))

    print(f"Main artifacts: {len(graph_files)} graphs")
    print(f"FX graphs: {len(fx_graphs)}")
    print(f"AOT graphs: {len(aot_graphs)}")

    # Analyze a sample graph
    all_graphs = graph_files + fx_graphs + aot_graphs
    if all_graphs:
        latest = max(all_graphs, key=os.path.getmtime)
        print(f"\nLatest graph: {os.path.basename(latest)}")
        print(f"Size: {os.path.getsize(latest) / 1024:.1f} KB")

        # Read and count ops
        with open(latest, 'r') as f:
            content = f.read()

        # Simple op counting
        op_count = content.count('call_function')
        method_count = content.count('call_method')
        module_count = content.count('call_module')

        print(f"\nOperation counts (estimated):")
        print(f"  call_function: {op_count}")
        print(f"  call_method: {method_count}")
        print(f"  call_module: {module_count}")


def _analyze_summary(args):
    """Display overall analysis summary."""
    _print_section("Analysis Summary")

    # Count artifacts
    artifact_counts = {
        'graphs': len(glob.glob(os.path.join(ARTIFACT_DIR, "graph_*.txt"))),
        'fx_graphs': len(glob.glob(os.path.join(ARTIFACT_DIR, "fx_graphs", "*.txt"))),
        'aot_graphs': len(glob.glob(os.path.join(ARTIFACT_DIR, "aot_graphs", "*.txt"))),
        'visualizations': len(glob.glob(os.path.join(ARTIFACT_DIR, "visualizations", "*.png"))),
        'reports': len(glob.glob(os.path.join(ARTIFACT_DIR, "reports", "*.json"))),
        'repros': len(glob.glob(os.path.join(ARTIFACT_DIR, "repros", "*.py"))),
    }

    print("Artifact Counts:")
    for name, count in artifact_counts.items():
        print(f"  {name}: {count}")

    # Check benchmark results
    results_dir = "benchmarks/results"
    benchmark_count = len(glob.glob(os.path.join(results_dir, "*.json")))
    print(f"  benchmark_results: {benchmark_count}")

    # Load latest benchmark summary if available
    summary_file = _find_latest_file(results_dir, "benchmark_summary_*.json")
    if summary_file:
        data = _load_json_report(summary_file)
        results = data.get('results', [])

        print(f"\nLatest Benchmark Results ({len(results)} models):")
        for r in results:
            name = r.get('model_name', 'Unknown')
            kd_passed = r.get('kerneldiff_passed', False)
            status = "✓ PASS" if kd_passed else "✗ FAIL"
            print(f"  {name}: {status}")


# =============================================================================
# Report Command
# =============================================================================

def generate_report(args):
    """Generate HTML or JSON reports."""

    output_format = args.format or 'html'
    output_dir = args.output or os.path.join(ARTIFACT_DIR, "reports")

    os.makedirs(output_dir, exist_ok=True)

    if output_format == 'html':
        _generate_html_report(output_dir, args)
    else:
        _generate_json_report(output_dir, args)


def _generate_json_report(output_dir: str, args):
    """Generate a comprehensive JSON report."""
    _print_section("Generating JSON Report")

    report = {
        'timestamp': datetime.now().isoformat(),
        'artifacts': {},
        'benchmarks': None,
        'analysis': {}
    }

    # Count artifacts
    report['artifacts'] = {
        'graphs': len(glob.glob(os.path.join(ARTIFACT_DIR, "graph_*.txt"))),
        'fx_graphs': len(glob.glob(os.path.join(ARTIFACT_DIR, "fx_graphs", "*.txt"))),
        'aot_graphs': len(glob.glob(os.path.join(ARTIFACT_DIR, "aot_graphs", "*.txt"))),
        'visualizations': len(glob.glob(os.path.join(ARTIFACT_DIR, "visualizations", "*.png"))),
        'repros': len(glob.glob(os.path.join(ARTIFACT_DIR, "repros", "*.py"))),
    }

    # Load benchmark results
    summary_file = _find_latest_file("benchmarks/results", "benchmark_summary_*.json")
    if summary_file:
        report['benchmarks'] = _load_json_report(summary_file)

    # Save report
    timestamp = int(datetime.now().timestamp())
    report_path = os.path.join(output_dir, f"comprehensive_report_{timestamp}.json")

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"JSON report saved to: {report_path}")


def _generate_html_report(output_dir: str, args):
    """Generate an HTML report."""
    _print_section("Generating HTML Report")

    try:
        from debug_module.reports.generator import HTMLReportGenerator
        generator = HTMLReportGenerator()
        report_path = generator.generate(output_dir)
        print(f"HTML report saved to: {report_path}")
    except ImportError:
        print("HTML report generator not available.")
        print("Falling back to basic HTML generation...")
        _generate_basic_html_report(output_dir)


def _generate_basic_html_report(output_dir: str):
    """Generate a basic HTML report without the full generator."""

    # Load benchmark data
    summary_file = _find_latest_file("benchmarks/results", "benchmark_summary_*.json")
    benchmark_data = _load_json_report(summary_file) if summary_file else None

    # Count artifacts
    artifacts = {
        'graphs': len(glob.glob(os.path.join(ARTIFACT_DIR, "graph_*.txt"))),
        'visualizations': len(glob.glob(os.path.join(ARTIFACT_DIR, "visualizations", "*.png"))),
        'repros': len(glob.glob(os.path.join(ARTIFACT_DIR, "repros", "*.py"))),
    }

    # Generate HTML
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TorchInductor Debug Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .fail {{ color: #f44336; font-weight: bold; }}
        .metric {{ font-family: monospace; }}
        .summary-box {{ background: #e8f5e9; padding: 15px; border-radius: 4px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 TorchInductor Debug Module Report</h1>
        <p>Generated: {timestamp}</p>

        <div class="summary-box">
            <strong>Artifacts Summary:</strong>
            Graphs: {artifacts['graphs']} |
            Visualizations: {artifacts['visualizations']} |
            Repro Scripts: {artifacts['repros']}
        </div>
"""

    # Add benchmark results if available
    if benchmark_data and benchmark_data.get('results'):
        html += """
        <h2>📊 Benchmark Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Parameters</th>
                <th>KernelDiff</th>
                <th>Max Error</th>
                <th>Warnings</th>
            </tr>
"""
        for r in benchmark_data['results']:
            name = r.get('model_name', 'Unknown')
            mtype = r.get('model_type', '-')
            params = r.get('parameter_count', 0)
            params_str = f"{params/1e6:.1f}M" if params > 0 else '-'
            kd_passed = r.get('kerneldiff_passed', False)
            kd_class = 'pass' if kd_passed else 'fail'
            kd_text = '✓ PASS' if kd_passed else '✗ FAIL'
            max_err = r.get('max_absolute_error', 0)
            max_err_str = f"{max_err:.2e}"
            warnings = len(r.get('mock_result', {}).get('constraint_warnings', []))

            html += f"""
            <tr>
                <td>{name}</td>
                <td>{mtype}</td>
                <td class="metric">{params_str}</td>
                <td class="{kd_class}">{kd_text}</td>
                <td class="metric">{max_err_str}</td>
                <td>{warnings}</td>
            </tr>
"""
        html += "        </table>\n"

    html += """
        <h2>📁 Artifact Details</h2>
        <p>Use the CLI to explore artifacts:</p>
        <pre>python -m debug_module list</pre>

        <h2>🔍 Next Steps</h2>
        <ul>
            <li>Review constraint warnings to identify hardware compatibility issues</li>
            <li>Examine generated repro scripts for debugging failures</li>
            <li>Check visualizations for error heatmaps</li>
        </ul>

        <hr>
        <p style="color: #888; font-size: 12px;">
            Generated by TorchInductor Debug Module |
            <a href="https://github.com/pytorch/pytorch">PyTorch</a>
        </p>
    </div>
</body>
</html>
"""

    # Save HTML
    timestamp_int = int(datetime.now().timestamp())
    report_path = os.path.join(output_dir, f"debug_report_{timestamp_int}.html")

    with open(report_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {report_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TorchInductor Debug Module CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m debug_module list                    # List artifacts
    python -m debug_module analyze --type guards   # Run guard analysis
    python -m debug_module analyze --type summary  # Show summary
    python -m debug_module report --format html    # Generate HTML report
    python -m debug_module clean -f                # Clean artifacts
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    parser_list = subparsers.add_parser("list", help="List captured artifacts")
    parser_list.set_defaults(func=list_artifacts)

    # Clean command
    parser_clean = subparsers.add_parser("clean", help="Clean captured artifacts")
    parser_clean.add_argument("-f", "--force", action="store_true",
                              help="Force delete without confirmation")
    parser_clean.set_defaults(func=clean_artifacts)

    # Analyze command
    parser_analyze = subparsers.add_parser("analyze", help="Analyze artifacts")
    parser_analyze.add_argument("--type",
                                choices=['guards', 'constraints', 'graphs', 'summary'],
                                default='summary',
                                help="Type of analysis to run")
    parser_analyze.set_defaults(func=analyze_artifacts)

    # Report command
    parser_report = subparsers.add_parser("report", help="Generate reports")
    parser_report.add_argument("--format", choices=['html', 'json'], default='html',
                               help="Report format (default: html)")
    parser_report.add_argument("--output", "-o", help="Output directory")
    parser_report.set_defaults(func=generate_report)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
