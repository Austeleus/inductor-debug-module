# frontend/app.py

import os
import sys
import json
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_ARTIFACT_DIR = REPO_ROOT / "debug_artifacts"


def run_command(cmd, cwd=None):
    """Run a shell command and return stdout, stderr, returncode."""
    if cwd is None:
        cwd = REPO_ROOT
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    return proc.stdout, proc.stderr, proc.returncode


def strip_ansi(s: str) -> str:
    """Remove ANSI color codes from a string for cleaner display."""
    import re

    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", s)


# -------------------------------------------------------------------
# Benchmarks tab
# -------------------------------------------------------------------

def benchmarks_tab():
    st.header("Benchmarks")

    st.write(
        "Run the benchmarking suite (BERT, ResNet, SSM) using "
        "`benchmarks.runner`. This will collect metrics like compile "
        "time, inference latency, and KernelDiff status per model."
    )

    col1, col2 = st.columns(2)
    with col1:
        run_all = st.checkbox("Run all benchmarks", value=True)
    with col2:
        models = st.multiselect(
            "Or select specific models",
            options=["bert", "resnet", "mamba"],
            default=[],
            help="Ignored if 'Run all benchmarks' is checked.",
        )

    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    warmup = st.number_input("Warmup iterations", min_value=0, value=3, step=1)
    runs = st.number_input("Timed runs", min_value=1, value=10, step=1)

    if st.button("Run benchmarks"):
        cmd = [sys.executable, "-m", "benchmarks.runner"]
        if run_all:
            cmd.append("--all")
        else:
            for m in models:
                cmd.extend(["--model", m])

        cmd.extend(["--device", device])
        cmd.extend(["--warmup", str(warmup), "--runs", str(runs)])

        st.code(" ".join(cmd), language="bash")
        with st.spinner("Running benchmarks..."):
            out, err, code = run_command(cmd)

        st.subheader("Raw Output")
        st.text(strip_ansi(out))
        if err:
            st.subheader("Errors / stderr")
            st.text(strip_ansi(err))
        st.info(f"Return code: {code}")


# -------------------------------------------------------------------
# KernelDiff tab (improved)
# -------------------------------------------------------------------

def load_latest_bert_report():
    """Load the most recent KernelDiff JSON report for BERT, if any."""
    reports_dir = DEBUG_ARTIFACT_DIR / "bert_reports"
    if not reports_dir.exists():
        return None, None

    json_files = sorted(
        reports_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        return None, None

    latest = json_files[0]
    try:
        with latest.open("r") as f:
            data = json.load(f)
        return latest, data
    except Exception:
        return latest, None


def kerneldiff_tab():
    st.header("KernelDiff Harness")

    st.write(
        "Run the KernelDiff integration test for a tiny BERT model. "
        "This compares outputs between the reference backend (eager) "
        "and the mock backend, computes detailed error metrics, and "
        "generates visualizations (heatmaps and summary plots)."
    )

    if st.button("Run BERT KernelDiff Test"):
        cmd = [sys.executable, "tests/test_kerneldiff_bert.py"]
        st.code(" ".join(cmd), language="bash")

        with st.spinner("Running KernelDiff BERT integration test..."):
            out, err, code = run_command(cmd)

        # Show a compact status message
        if code == 0:
            st.success("KernelDiff test completed successfully (return code 0).")
        else:
            st.warning(
                f"KernelDiff test returned non-zero exit code: {code}. "
                "Check raw log for details."
            )

        # Raw log in an expander
        with st.expander("Show raw test log (stdout / stderr)"):
            st.subheader("stdout")
            st.text(strip_ansi(out))
            if err:
                st.subheader("stderr")
                st.text(strip_ansi(err))

    st.divider()
    st.subheader("Latest KernelDiff Report")

    latest_path, report = load_latest_bert_report()
    if latest_path is None:
        st.write(
            "No KernelDiff reports found yet. Click **'Run BERT KernelDiff Test'** above "
            "to generate one."
        )
        return

    if report is None:
        st.warning(f"Found report `{latest_path.name}` but failed to parse JSON.")
        return

    # High-level summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Model",
            report.get("model_name", "unknown"),
        )
    with col2:
        status_label = "PASSED" if report.get("overall_passed") else "FAILED"
        status_color = "✅" if report.get("overall_passed") else "❌"
        st.metric("Status", f"{status_color} {status_label}")
    with col3:
        st.metric(
            "Tensors Compared",
            f"{report.get('passed_tensors', 0)}/{report.get('total_tensors', 0)} passed",
        )

    st.caption(f"Report file: `{latest_path}`")

    # Backends
    ref_backend = report.get("reference_backend", "eager")
    test_backend = report.get("test_backend", "mock_backend")

    st.markdown(
        f"""
**Backends**

- Reference backend: `{ref_backend}`
- Test backend: `{test_backend}`
"""
    )

    # Timing metrics
    timing = report.get("timing", {})
    st.subheader("Timing")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("**Reference (eager)**")
        st.write(
            f"- Compile time: `{timing.get('reference_compile_time', 0.0):.3f}s`\n"
            f"- Run time: `{timing.get('reference_run_time', 0.0):.3f}s`"
        )
    with tcol2:
        st.markdown("**Test (mock backend)**")
        st.write(
            f"- Compile time: `{timing.get('test_compile_time', 0.0):.3f}s`\n"
            f"- Run time: `{timing.get('test_run_time', 0.0):.3f}s`"
        )

    # Tensor-level results as a table
    tensor_results = report.get("tensor_results", [])
    if tensor_results:
        st.subheader("Tensor Comparison Details")

        # Flatten into rows for a DataFrame
        rows = []
        for r in tensor_results:
            rows.append(
                {
                    "Tensor": r.get("name", ""),
                    "Shape": str(r.get("shape", "")),
                    "Dtype": r.get("dtype", ""),
                    "Passed": r.get("passed", False),
                    "Max Error": r.get("max_absolute_error", 0.0),
                    "Mean Error": r.get("mean_absolute_error", 0.0),
                    "RMSE": r.get("rmse", 0.0),
                    "Mismatch %": r.get("mismatch_percentage", 0.0),
                }
            )
        df = pd.DataFrame(rows)
        # Nice formatting
        df["Passed"] = df["Passed"].map(lambda x: "✅" if x else "❌")
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No tensor comparison results found in report.")

    # Visualizations
    vis_paths = report.get("visualization_paths", [])
    if vis_paths:
        st.subheader("Visualizations")

        for rel_path in vis_paths:
            img_path = REPO_ROOT / rel_path
            if img_path.exists() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(img_path), caption=rel_path)
            else:
                st.write(f"- `{rel_path}` (file not found or not an image)")
    else:
        st.write("No visualization paths recorded in report.")


# -------------------------------------------------------------------
# Artifacts tab
# -------------------------------------------------------------------

def artifacts_tab():
    st.header("Artifacts Browser")

    DEBUG_ARTIFACT_DIR.mkdir(exist_ok=True)
    files = sorted(
        DEBUG_ARTIFACT_DIR.iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        st.write("No artifacts found in `debug_artifacts/`.")
        return

    st.write(f"Found **{len(files)}** artifacts in `debug_artifacts/` (newest first).")
    selected = st.selectbox(
        "Select an artifact file",
        options=[f.name for f in files],
    )
    path = DEBUG_ARTIFACT_DIR / selected
    st.write(f"Selected: `{path}`")

    st.write("**File metadata:**")
    st.write(
        {
            "size_bytes": path.stat().st_size,
            "size_kb": path.stat().st_size / 1024.0,
            "modified": path.stat().st_mtime,
        }
    )

    if path.suffix in {".txt", ".log", ".json", ".html"}:
        try:
            content = path.read_text()
            st.subheader("Preview (first 5000 characters)")
            st.text(content[:5000])
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    else:
        st.info("Non-text artifact. Use your local file browser / VS Code to inspect it.")

    st.divider()
    if st.button("Clean all artifacts"):
        cmd = [sys.executable, "-m", "debug_module", "clean", "-f"]
        st.code(" ".join(cmd), language="bash")
        with st.spinner("Cleaning artifacts..."):
            out, err, code = run_command(cmd)
        st.text(strip_ansi(out))
        if err:
            st.text(strip_ansi(err))
        if code == 0:
            st.success("Artifacts cleaned. Refresh the page to update the list.")
        else:
            st.warning(f"Clean command returned code {code}.")


# -------------------------------------------------------------------
# Guards tab
# -------------------------------------------------------------------

def guards_tab():
    st.header("Guard Inspector")

    st.write(
        "Run guard inspection on built-in models. Currently this tab "
        "reuses the `tests/comprehensive_test.py` script, which includes "
        "both a simple model and a BERT guard inspector demo."
    )

    model_choice = st.selectbox("Model", options=["simple", "bert"], index=0)

    if st.button("Run Guard Inspector (via comprehensive_test.py)"):
        # For now, we always run the full comprehensive_test; it prints which parts run.
        cmd = [sys.executable, "tests/comprehensive_test.py"]
        note = (
            "This runs the full comprehensive test suite, including guard inspector "
            f"for the '{model_choice}' scenario."
        )

        st.caption(note)
        st.code(" ".join(cmd), language="bash")
        with st.spinner("Running comprehensive_test.py (this may take a while)..."):
            out, err, code = run_command(cmd)
        st.subheader("Output")
        st.text(strip_ansi(out))
        if err:
            st.subheader("Errors / stderr")
            st.text(strip_ansi(err))
        st.info(f"Return code: {code}")


# -------------------------------------------------------------------
# Constraints tab
# -------------------------------------------------------------------

def constraints_tab():
    st.header("Mock Backend Constraints")

    st.write("View and (temporarily) adjust constraint-related environment variables.")

    strict_current = os.environ.get("MOCK_STRICT", "1")
    alignment_current = os.environ.get("MOCK_ALIGNMENT", "1")
    max_mem_current = os.environ.get("MOCK_MAX_MEMORY", "inf")

    strict = st.selectbox(
        "MOCK_STRICT (1=strict, 0=warnings)",
        options=["1", "0"],
        index=0 if strict_current == "1" else 1,
    )

    alignment = st.text_input(
        "MOCK_ALIGNMENT (shape alignment requirement)",
        value=alignment_current,
    )

    max_mem = st.text_input(
        "MOCK_MAX_MEMORY (bytes; use 'inf' for no limit)",
        value=max_mem_current,
    )

    if st.button("Apply settings for this session"):
        os.environ["MOCK_STRICT"] = strict
        os.environ["MOCK_ALIGNMENT"] = alignment
        os.environ["MOCK_MAX_MEMORY"] = max_mem
        st.success("Environment variables updated for this Streamlit process.")

    st.write("**Current values (process environment):**")
    st.json(
        {
            "MOCK_STRICT": os.environ.get("MOCK_STRICT", "1"),
            "MOCK_ALIGNMENT": os.environ.get("MOCK_ALIGNMENT", "1"),
            "MOCK_MAX_MEMORY": os.environ.get("MOCK_MAX_MEMORY", "inf"),
        }
    )


# -------------------------------------------------------------------
# Full demo tab
# -------------------------------------------------------------------

def demo_tab():
    st.header("End-to-End Demo (Comprehensive Test)")

    st.write(
        "Run the `tests/comprehensive_test.py` script, which demonstrates:\n"
        "- Mock backend constraints\n"
        "- Strict vs non-strict mode\n"
        "- Guard inspector\n"
        "- Artifact generation\n"
        "- CLI interface\n"
    )

    if st.button("Run comprehensive_test.py"):
        cmd = [sys.executable, "tests/comprehensive_test.py"]
        st.code(" ".join(cmd), language="bash")
        with st.spinner("Running comprehensive_test.py (this may take a while)..."):
            out, err, code = run_command(cmd)
        st.subheader("Output")
        st.text(strip_ansi(out))
        if err:
            st.subheader("Errors / stderr")
            st.text(strip_ansi(err))
        st.info(f"Return code: {code}")


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="TorchInductor Debug Module UI", layout="wide")

    st.title("TorchInductor Debug Module – Front End")

    tabs = st.tabs(
        ["Benchmarks", "KernelDiff", "Artifacts", "Guards", "Constraints", "Full Demo"]
    )

    with tabs[0]:
        benchmarks_tab()
    with tabs[1]:
        kerneldiff_tab()
    with tabs[2]:
        artifacts_tab()
    with tabs[3]:
        guards_tab()
    with tabs[4]:
        constraints_tab()
    with tabs[5]:
        demo_tab()


if __name__ == "__main__":
    main()
