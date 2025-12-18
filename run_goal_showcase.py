#!/usr/bin/env python3
"""
Goal-driven runner for the TorchInductor Debug Module project.

This script executes the demos/tests/benchmarks that showcase each milestone:

1. CLI scaffolding + AOTAutograd artifact capture
2. Guard Inspector, recompilation analysis, and denylist enforcement
3. KernelDiff harness, metrics, and visualizations
4. Mock backend constraint simulation
5. Minifier + repro generation workflow
6. Final benchmark demos on Transformer/ResNet/SSM models

Usage:
    python run_goal_showcase.py                     # run every goal sequentially
    python run_goal_showcase.py --goal kernel       # run selected goals
    python run_goal_showcase.py --list              # list available goals
    python run_goal_showcase.py --dry-run           # only print planned commands
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


@dataclass
class CommandSpec:
    """Single runnable step tied to a goal."""

    name: str
    description: str
    argv: List[str]
    cwd: Path = REPO_ROOT
    env: Dict[str, str] = field(default_factory=dict)
    stdin: Optional[str] = None
    allow_failure: bool = False

    def render(self) -> str:
        return " ".join(self.argv)


@dataclass
class GoalSpec:
    name: str
    description: str
    commands: List[CommandSpec]

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-")


GOALS: List[GoalSpec] = [
    GoalSpec(
        name="Inductor CLI & Artifact Capture",
        description=(
            "Runs the AOTAutograd backend smoke tests to record FX/AOT artifacts "
            "and then exercises the debug_module CLI to list captured dumps."
        ),
        commands=[
            CommandSpec(
                name="AOTAutograd capture suite",
                description=(
                    "Executes tests/test_aotbackend.py to demonstrate FX, AOT, IR, "
                    "and kernel capture plus strict/relaxed constraint handling."
                ),
                argv=[PYTHON, "tests/test_aotbackend.py"],
            ),
            CommandSpec(
                name="CLI artifact listing",
                description="Invokes `python -m debug_module list` to enumerate captured dumps.",
                argv=[PYTHON, "-m", "debug_module", "list"],
            ),
        ],
    ),
    GoalSpec(
        name="Guard Inspector & Constraint Policy",
        description=(
            "Walk through tests/comprehensive_test.py to inspect Dynamo guards, "
            "graph breaks, CLI wiring, and policy-based denylist enforcement with "
            "an automatic prompt acknowledgement."
        ),
        commands=[
            CommandSpec(
                name="Comprehensive guard walkthrough",
                description="Runs the interactive comprehensive test suite (auto-confirm input).",
                argv=[PYTHON, "tests/comprehensive_test.py"],
                stdin="\n",
            ),
        ],
    ),
    GoalSpec(
        name="KernelDiff Harness & Metrics",
        description=(
            "Validates the KernelDiff harness, tensor comparison metrics, visualization "
            "pipeline, and report serialization."
        ),
        commands=[
            CommandSpec(
                name="KernelDiff unit suite",
                description="Runs tests/test_kerneldiff.py covering metrics, nested outputs, and plots.",
                argv=[PYTHON, "tests/test_kerneldiff.py"],
            ),
        ],
    ),
    GoalSpec(
        name="Mock Backend Constraint Simulation",
        description=(
            "Executes the constraint-focused pytest suite to showcase dtype/layout/mode restrictions "
            "and strict vs warning policy behavior."
        ),
        commands=[
            CommandSpec(
                name="Mock backend constraint tests",
                description="Runs pytest on tests/test_constraints.py for dtype/layout/alignment checks.",
                argv=[PYTHON, "-m", "pytest", "tests/test_constraints.py"],
            ),
        ],
    ),
    GoalSpec(
        name="Minifier & Repro Workflow",
        description=(
            "Runs the minifier unit tests plus backend integration test that emits standalone repro scripts."
        ),
        commands=[
            CommandSpec(
                name="Minifier unit tests",
                description="Executes tests/test_minifier.py to shrink CNN/SSM/Transformer failures.",
                argv=[PYTHON, "tests/test_minifier.py"],
            ),
            CommandSpec(
                name="Backend repro integration",
                description="Runs tests/test_backend_minifier.py to verify repro scripts on backend failure.",
                argv=[PYTHON, "tests/test_backend_minifier.py"],
            ),
        ],
    ),
    GoalSpec(
        name="Benchmark Demos (Transformer/ResNet/SSM)",
        description=(
            "Produces demo runs on the representative transformer block, ResNet-18 CNN, "
            "and custom SSM benchmark to collect reports/WIP documentation artifacts."
        ),
        commands=[
            CommandSpec(
                name="Transformer block KernelDiff demo",
                description=(
                    "Runs tests/test_kerneldiff_bert.py; falls back to the bundled SimpleTransformer "
                    "if HuggingFace weights are unavailable."
                ),
                argv=[PYTHON, "tests/test_kerneldiff_bert.py"],
                allow_failure=True,
            ),
            CommandSpec(
                name="ResNet benchmark (quick)",
                description="Benchmarks ResNet-18 with minimal warmup/runs and skips result persistence.",
                argv=[
                    PYTHON,
                    "benchmarks/resnet.py",
                    "--warmup",
                    "1",
                    "--runs",
                    "1",
                    "--variant",
                    "resnet18",
                    "--no-save",
                ],
                allow_failure=True,
            ),
            CommandSpec(
                name="SSM benchmark (custom small)",
                description="Runs the small custom SSM benchmark (no pretrained weights) with quick settings.",
                argv=[
                    PYTHON,
                    "benchmarks/mamba.py",
                    "--warmup",
                    "1",
                    "--runs",
                    "1",
                    "--small",
                    "--no-save",
                ],
            ),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project goal demos/tests.")
    parser.add_argument(
        "--goal",
        action="append",
        dest="goals",
        help="Filter goals by name/slug substring (can be repeated).",
    )
    parser.add_argument("--list", action="store_true", help="List available goals and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Abort as soon as a command fails.")
    return parser.parse_args()


def goal_selected(goal: GoalSpec, filters: Optional[List[str]]) -> bool:
    if not filters:
        return True
    name = goal.name.lower()
    slug = goal.slug
    return any(f.lower() in name or f.lower() in slug for f in filters)


def run_command(spec: CommandSpec) -> Dict[str, object]:
    env = os.environ.copy()
    if spec.env:
        env.update(spec.env)

    run_kwargs: Dict[str, object] = {}
    if spec.stdin is not None:
        run_kwargs["input"] = spec.stdin
        run_kwargs["text"] = True

    print(f"\n    → {spec.name}")
    wrapped = textwrap.fill(spec.description, width=76, subsequent_indent="      ")
    print(f"      {wrapped}")
    print(f"      $ {spec.render()}")

    status = "PASS"
    return_code = 0
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            spec.argv,
            cwd=spec.cwd,
            env=env,
            check=False,
            **run_kwargs,
        )
        return_code = completed.returncode
    except FileNotFoundError as exc:
        return_code = 127
        print(f"      [ERROR] {exc}")
    elapsed = time.perf_counter() - start

    if return_code != 0:
        status = "WARN" if spec.allow_failure else "FAIL"

    print(f"      Result: {status} (exit={return_code}, {elapsed:.1f}s)")
    return {
        "name": spec.name,
        "status": status,
        "return_code": return_code,
        "elapsed": elapsed,
        "command": spec.render(),
        "allow_failure": spec.allow_failure,
    }


def main() -> int:
    args = parse_args()

    if args.list:
        print("Available goals:\n")
        for goal in GOALS:
            print(f"  - {goal.name} ({goal.slug})")
        return 0

    selected = [goal for goal in GOALS if goal_selected(goal, args.goals)]
    if not selected:
        print("No goals matched the provided filter. Use --list to inspect options.")
        return 1

    summary: List[Dict[str, object]] = []
    overall_exit = 0

    for goal in selected:
        print("\n" + "=" * 80)
        print(f"{goal.name}")
        print("-" * 80)
        print(textwrap.fill(goal.description, width=80))

        for command in goal.commands:
            if args.dry_run:
                print(f"\n    → {command.name}")
                print(f"      {command.description}")
                print(f"      $ {command.render()}")
                continue

            result = run_command(command)
            summary.append({**result, "goal": goal.name})

            if result["status"] == "FAIL":
                overall_exit = 1
                if args.stop_on_failure:
                    print("\nStopping early due to failure (--stop-on-failure).")
                    return overall_exit

    if not args.dry_run:
        print("\n" + "=" * 80)
        print("Goal Showcase Summary")
        print("=" * 80)
        for item in summary:
            print(
                f"[{item['status']}] {item['goal']} :: {item['name']} "
                f"(exit={item['return_code']}, {item['elapsed']:.1f}s)"
            )

        failed = [item for item in summary if item["status"] == "FAIL"]
        if failed:
            print("\nFailures detected. Re-run with --goal to focus on specific sections.")

    return overall_exit


if __name__ == "__main__":
    raise SystemExit(main())
