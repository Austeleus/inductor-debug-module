"""
HTML Report Generator for TorchInductor Debug Module

Generates comprehensive HTML reports including:
- Benchmark results with comparison tables
- Constraint violation analysis
- KernelDiff results with error visualizations
- Graph statistics
- Artifact summaries
"""

import base64
import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

ARTIFACT_DIR = "debug_artifacts"
BENCHMARK_DIR = "benchmarks/results"


@dataclass
class ReportData:
    """Container for all report data."""
    timestamp: str = ""
    artifacts: Dict[str, int] = field(default_factory=dict)
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    constraint_warnings: Dict[str, List[str]] = field(default_factory=dict)
    kerneldiff_results: List[Dict[str, Any]] = field(default_factory=list)
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[str] = field(default_factory=list)


class HTMLReportGenerator:
    """Generate comprehensive HTML reports from debug artifacts."""

    def __init__(self, artifact_dir: str = ARTIFACT_DIR, benchmark_dir: str = BENCHMARK_DIR):
        self.artifact_dir = artifact_dir
        self.benchmark_dir = benchmark_dir

    def collect_data(self) -> ReportData:
        """Collect all data for the report."""
        data = ReportData()
        data.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Count artifacts
        data.artifacts = self._count_artifacts()

        # Load benchmark results
        data.benchmarks = self._load_benchmarks()

        # Extract constraint warnings
        data.constraint_warnings = self._extract_warnings(data.benchmarks)

        # Extract KernelDiff results
        data.kerneldiff_results = self._extract_kerneldiff(data.benchmarks)

        # Collect visualizations
        data.visualizations = self._find_visualizations()

        # Load graph statistics
        data.graph_stats = self._load_graph_stats()

        return data

    def _count_artifacts(self) -> Dict[str, int]:
        """Count artifacts by type."""
        return {
            'graphs': len(glob.glob(os.path.join(self.artifact_dir, "graph_*.txt"))),
            'fx_graphs': len(glob.glob(os.path.join(self.artifact_dir, "fx_graphs", "*.txt"))),
            'aot_graphs': len(glob.glob(os.path.join(self.artifact_dir, "aot_graphs", "*.txt"))),
            'visualizations': len(glob.glob(os.path.join(self.artifact_dir, "visualizations", "*.png"))),
            'reports': len(glob.glob(os.path.join(self.artifact_dir, "reports", "*.json"))),
            'repros': len(glob.glob(os.path.join(self.artifact_dir, "repros", "*.py"))),
            'statistics': len(glob.glob(os.path.join(self.artifact_dir, "statistics", "*.json"))),
        }

    def _load_benchmarks(self) -> List[Dict[str, Any]]:
        """Load benchmark results from the latest summary."""
        summary_files = glob.glob(os.path.join(self.benchmark_dir, "benchmark_summary_*.json"))
        if not summary_files:
            return []

        latest = max(summary_files, key=os.path.getmtime)
        with open(latest, 'r') as f:
            data = json.load(f)

        return data.get('results', [])

    def _extract_warnings(self, benchmarks: List[Dict]) -> Dict[str, List[str]]:
        """Extract constraint warnings grouped by model."""
        warnings = {}
        for b in benchmarks:
            name = b.get('model_name', 'Unknown')
            # Check all backends for warnings
            for backend_key in ['mock_backend', 'mock_result', 'inductor_backend', 'eager_backend']:
                backend = b.get(backend_key, {})
                warns = backend.get('constraint_warnings', [])
                if warns:
                    if name not in warnings:
                        warnings[name] = []
                    warnings[name].extend(warns)
        return warnings

    def _extract_kerneldiff(self, benchmarks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract KernelDiff results."""
        results = []
        for b in benchmarks:
            results.append({
                'model': b.get('model_name', 'Unknown'),
                'passed': b.get('kerneldiff_passed', False),
                'max_error': b.get('max_absolute_error', 0),
                'mean_error': b.get('mean_absolute_error', 0),
            })
        return results

    def _find_visualizations(self) -> List[str]:
        """Find visualization files."""
        viz_dir = os.path.join(self.artifact_dir, "visualizations")
        return glob.glob(os.path.join(viz_dir, "*.png"))

    def _load_graph_stats(self) -> Dict[str, Any]:
        """Load graph statistics from latest stats file."""
        stats_files = glob.glob(os.path.join(self.artifact_dir, "statistics", "*.json"))
        if not stats_files:
            return {}

        latest = max(stats_files, key=os.path.getmtime)
        with open(latest, 'r') as f:
            return json.load(f)

    def _embed_image(self, filepath: str) -> str:
        """Convert image to base64 data URI for embedding."""
        try:
            with open(filepath, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{data}"
        except Exception:
            return ""

    def generate(self, output_dir: str) -> str:
        """Generate the HTML report and return the filepath."""
        data = self.collect_data()
        html = self._render_html(data)

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(datetime.now().timestamp())
        filepath = os.path.join(output_dir, f"debug_report_{timestamp}.html")

        with open(filepath, 'w') as f:
            f.write(html)

        return filepath

    def _render_html(self, data: ReportData) -> str:
        """Render the complete HTML report."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TorchInductor Debug Report</title>
    {self._render_styles()}
</head>
<body>
    <div class="container">
        {self._render_header(data)}
        {self._render_summary(data)}
        {self._render_benchmark_results(data)}
        {self._render_constraint_analysis(data)}
        {self._render_kerneldiff_results(data)}
        {self._render_visualizations(data)}
        {self._render_artifacts(data)}
        {self._render_footer()}
    </div>
    {self._render_scripts()}
</body>
</html>"""

    def _render_styles(self) -> str:
        """Render CSS styles."""
        return """<style>
    :root {
        --primary: #4CAF50;
        --primary-dark: #388E3C;
        --danger: #f44336;
        --warning: #ff9800;
        --info: #2196F3;
        --bg: #f5f5f5;
        --card-bg: #ffffff;
        --text: #333333;
        --text-muted: #666666;
        --border: #e0e0e0;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.6;
    }

    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }

    header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    header h1 { font-size: 2em; margin-bottom: 10px; }
    header p { opacity: 0.9; }

    .card {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid var(--border);
    }

    .card h2 {
        color: var(--primary-dark);
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--primary);
        font-size: 1.4em;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .stat-box {
        background: var(--bg);
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid var(--primary);
    }

    .stat-box .value {
        font-size: 2.5em;
        font-weight: bold;
        color: var(--primary-dark);
    }

    .stat-box .label {
        color: var(--text-muted);
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
    }

    th, td {
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    th {
        background: var(--primary);
        color: white;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85em;
        letter-spacing: 0.5px;
    }

    tr:hover { background: #f9f9f9; }

    .pass { color: var(--primary); font-weight: bold; }
    .fail { color: var(--danger); font-weight: bold; }
    .warning-text { color: var(--warning); }

    .metric { font-family: 'Monaco', 'Consolas', monospace; font-size: 0.9em; }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }

    .badge-pass { background: #e8f5e9; color: var(--primary-dark); }
    .badge-fail { background: #ffebee; color: var(--danger); }
    .badge-warn { background: #fff3e0; color: #e65100; }

    .warning-list {
        max-height: 200px;
        overflow-y: auto;
        background: #fff8e1;
        padding: 12px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85em;
    }

    .warning-list li {
        padding: 4px 0;
        border-bottom: 1px solid #ffe082;
        list-style: none;
    }

    .collapsible {
        cursor: pointer;
        user-select: none;
    }

    .collapsible:after {
        content: ' ▼';
        font-size: 0.8em;
    }

    .collapsible.collapsed:after {
        content: ' ▶';
    }

    .collapse-content {
        display: block;
    }

    .collapse-content.hidden {
        display: none;
    }

    .viz-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 20px;
    }

    .viz-item {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    .viz-item img {
        width: 100%;
        height: auto;
        display: block;
    }

    .viz-item .caption {
        padding: 10px;
        background: var(--bg);
        font-size: 0.9em;
        color: var(--text-muted);
    }

    footer {
        text-align: center;
        padding: 20px;
        color: var(--text-muted);
        font-size: 0.85em;
        border-top: 1px solid var(--border);
        margin-top: 30px;
    }

    footer a { color: var(--primary); text-decoration: none; }
    footer a:hover { text-decoration: underline; }

    .progress-bar {
        height: 8px;
        background: var(--border);
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-bar .fill {
        height: 100%;
        background: var(--primary);
        transition: width 0.3s ease;
    }

    @media (max-width: 768px) {
        .container { padding: 10px; }
        header { padding: 20px; }
        .summary-grid { grid-template-columns: 1fr 1fr; }
        .viz-grid { grid-template-columns: 1fr; }
    }
</style>"""

    def _render_header(self, data: ReportData) -> str:
        """Render the header section."""
        return f"""<header>
    <h1>TorchInductor Debug Module Report</h1>
    <p>Generated: {data.timestamp}</p>
</header>"""

    def _render_summary(self, data: ReportData) -> str:
        """Render the summary section."""
        total_artifacts = sum(data.artifacts.values())
        total_warnings = sum(len(w) for w in data.constraint_warnings.values())
        passed_tests = sum(1 for r in data.kerneldiff_results if r['passed'])
        total_tests = len(data.kerneldiff_results)

        return f"""<div class="card">
    <h2>Summary</h2>
    <div class="summary-grid">
        <div class="stat-box">
            <div class="value">{total_artifacts}</div>
            <div class="label">Total Artifacts</div>
        </div>
        <div class="stat-box">
            <div class="value">{total_tests}</div>
            <div class="label">Models Tested</div>
        </div>
        <div class="stat-box">
            <div class="value">{passed_tests}/{total_tests}</div>
            <div class="label">KernelDiff Passed</div>
        </div>
        <div class="stat-box">
            <div class="value">{total_warnings}</div>
            <div class="label">Constraint Warnings</div>
        </div>
    </div>
</div>"""

    def _render_benchmark_results(self, data: ReportData) -> str:
        """Render benchmark results table."""
        if not data.benchmarks:
            return """<div class="card">
    <h2>Benchmark Results</h2>
    <p>No benchmark results found. Run benchmarks with:</p>
    <pre>python -m benchmarks.runner --all</pre>
</div>"""

        rows = ""
        for b in data.benchmarks:
            name = b.get('model_name', 'Unknown')
            mtype = b.get('model_type', '-')
            params = b.get('parameter_count', 0)
            params_str = f"{params/1e6:.1f}M" if params > 0 else '-'

            # Support both old format (eager_result) and new format (eager_backend)
            eager = b.get('eager_backend', b.get('eager_result', {}))
            inductor = b.get('inductor_backend', b.get('inductor_result', {}))
            mock = b.get('mock_backend', b.get('mock_result', {}))

            # Time can be in seconds (avg_inference_time) or milliseconds (avg_inference_time_ms)
            eager_time = eager.get('avg_inference_time', eager.get('avg_inference_time_ms', 0) / 1000)
            inductor_time = inductor.get('avg_inference_time', inductor.get('avg_inference_time_ms', 0) / 1000)

            # Convert to milliseconds for display
            eager_ms = eager_time * 1000
            inductor_ms = inductor_time * 1000

            speedup = eager_time / inductor_time if inductor_time > 0 else 0

            kd_passed = b.get('kerneldiff_passed', False)
            kd_class = 'pass' if kd_passed else 'fail'
            kd_badge = 'badge-pass' if kd_passed else 'badge-fail'
            kd_text = 'PASS' if kd_passed else 'FAIL'

            # Count warnings from all backends
            warnings = 0
            for backend in [eager, inductor, mock]:
                warnings += len(backend.get('constraint_warnings', []))
            warn_class = 'badge-warn' if warnings > 0 else ''

            rows += f"""<tr>
    <td><strong>{name}</strong></td>
    <td>{mtype}</td>
    <td class="metric">{params_str}</td>
    <td class="metric">{eager_ms:.1f} ms</td>
    <td class="metric">{inductor_ms:.1f} ms</td>
    <td class="metric">{speedup:.2f}x</td>
    <td><span class="badge {kd_badge}">{kd_text}</span></td>
    <td>{f'<span class="badge {warn_class}">{warnings}</span>' if warnings > 0 else '0'}</td>
</tr>"""

        return f"""<div class="card">
    <h2>Benchmark Results</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Params</th>
                <th>Eager</th>
                <th>Inductor</th>
                <th>Speedup</th>
                <th>KernelDiff</th>
                <th>Warnings</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</div>"""

    def _render_constraint_analysis(self, data: ReportData) -> str:
        """Render constraint warnings analysis."""
        if not data.constraint_warnings:
            return """<div class="card">
    <h2>Constraint Analysis</h2>
    <p class="pass">No constraint warnings detected.</p>
</div>"""

        sections = ""
        for model, warnings in data.constraint_warnings.items():
            # Group by type
            types = {}
            for w in warnings:
                if 'non-contiguous' in w.lower():
                    t = 'Layout'
                elif 'dtype' in w.lower():
                    t = 'Dtype'
                elif 'alignment' in w.lower():
                    t = 'Shape'
                elif 'memory' in w.lower():
                    t = 'Memory'
                else:
                    t = 'Other'
                types[t] = types.get(t, 0) + 1

            type_badges = ' '.join(f'<span class="badge badge-warn">{t}: {c}</span>' for t, c in types.items())

            warning_items = '\n'.join(f'<li>{w[:100]}...</li>' if len(w) > 100 else f'<li>{w}</li>'
                                       for w in warnings[:10])
            more = f'<li>... and {len(warnings) - 10} more</li>' if len(warnings) > 10 else ''

            sections += f"""<div style="margin-bottom: 20px;">
    <h3 class="collapsible" onclick="toggleCollapse(this)">{model} ({len(warnings)} warnings)</h3>
    <div class="collapse-content">
        <p style="margin: 10px 0;">{type_badges}</p>
        <ul class="warning-list">
            {warning_items}
            {more}
        </ul>
    </div>
</div>"""

        return f"""<div class="card">
    <h2>Constraint Analysis</h2>
    {sections}
</div>"""

    def _render_kerneldiff_results(self, data: ReportData) -> str:
        """Render KernelDiff comparison results."""
        if not data.kerneldiff_results:
            return ""

        rows = ""
        for r in data.kerneldiff_results:
            status_class = 'pass' if r['passed'] else 'fail'
            status_text = 'PASS' if r['passed'] else 'FAIL'

            rows += f"""<tr>
    <td>{r['model']}</td>
    <td class="{status_class}">{status_text}</td>
    <td class="metric">{r['max_error']:.2e}</td>
    <td class="metric">{r['mean_error']:.2e}</td>
</tr>"""

        return f"""<div class="card">
    <h2>KernelDiff Results</h2>
    <p>Numerical comparison between eager execution and mock backend.</p>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Max Error</th>
                <th>Mean Error</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</div>"""

    def _render_visualizations(self, data: ReportData) -> str:
        """Render visualization images."""
        if not data.visualizations:
            return ""

        items = ""
        for viz_path in data.visualizations[:6]:  # Limit to 6
            filename = os.path.basename(viz_path)
            data_uri = self._embed_image(viz_path)
            if data_uri:
                items += f"""<div class="viz-item">
    <img src="{data_uri}" alt="{filename}">
    <div class="caption">{filename}</div>
</div>"""

        return f"""<div class="card">
    <h2>Visualizations</h2>
    <div class="viz-grid">
        {items}
    </div>
</div>"""

    def _render_artifacts(self, data: ReportData) -> str:
        """Render artifact counts."""
        rows = ""
        for name, count in data.artifacts.items():
            if count > 0:
                rows += f"""<tr>
    <td>{name.replace('_', ' ').title()}</td>
    <td class="metric">{count}</td>
</tr>"""

        return f"""<div class="card">
    <h2>Artifacts</h2>
    <table>
        <thead>
            <tr>
                <th>Type</th>
                <th>Count</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    <p style="margin-top: 16px; color: var(--text-muted);">
        Use <code>python -m debug_module list</code> for detailed artifact listing.
    </p>
</div>"""

    def _render_footer(self) -> str:
        """Render the footer."""
        return """<footer>
    <p>
        Generated by <strong>TorchInductor Debug Module</strong> |
        <a href="https://pytorch.org">PyTorch</a> |
        <a href="https://github.com/pytorch/pytorch">GitHub</a>
    </p>
</footer>"""

    def _render_scripts(self) -> str:
        """Render JavaScript for interactivity."""
        return """<script>
function toggleCollapse(element) {
    element.classList.toggle('collapsed');
    const content = element.nextElementSibling;
    content.classList.toggle('hidden');
}
</script>"""
