import argparse
import os
import shutil
import sys
from datetime import datetime

ARTIFACT_DIR = "debug_artifacts"

def list_artifacts(args):
    """List all captured artifacts."""
    if not os.path.exists(ARTIFACT_DIR):
        print(f"Artifact directory '{ARTIFACT_DIR}' does not exist.")
        return

    files = os.listdir(ARTIFACT_DIR)
    if not files:
        print(f"No artifacts found in '{ARTIFACT_DIR}'.")
        return

    print(f"Found {len(files)} artifacts in '{ARTIFACT_DIR}':")
    print(f"{'Filename':<40} | {'Size':<10} | {'Created'}")
    print("-" * 70)
    
    for f in sorted(files):
        path = os.path.join(ARTIFACT_DIR, f)
        stat = os.stat(path)
        size_str = f"{stat.st_size / 1024:.1f} KB"
        created_str = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{f:<40} | {size_str:<10} | {created_str}")

def clean_artifacts(args):
    """Remove all captured artifacts."""
    if not os.path.exists(ARTIFACT_DIR):
        print(f"Artifact directory '{ARTIFACT_DIR}' does not exist.")
        return

    if not args.force:
        confirm = input(f"Are you sure you want to delete all files in '{ARTIFACT_DIR}'? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    shutil.rmtree(ARTIFACT_DIR)
    os.makedirs(ARTIFACT_DIR)
    print(f"Cleaned '{ARTIFACT_DIR}'.")

def analyze_artifacts(args):
    """Analyze artifacts and generate reports."""
    if args.type == 'guards':
        print("Running Guard Inspector...")
        # For now, we need a model to inspect. 
        # In a real scenario, we might load a captured graph or require a script.
        # Since the CLI is generic, we'll just print a placeholder or 
        # maybe run the test script if the user asks for a demo.
        print("To analyze guards for a specific model, please use the Python API:")
        print("  from debug_module import GuardInspector")
        print("  inspector = GuardInspector(model)")
        print("  inspector.inspect(inputs)")
        return

    print("Analysis feature is coming soon!")
    print("Available analysis types: guards")

def main():
    parser = argparse.ArgumentParser(description="TorchInductor Debug Module CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    parser_list = subparsers.add_parser("list", help="List captured artifacts")
    parser_list.set_defaults(func=list_artifacts)

    # Clean command
    parser_clean = subparsers.add_parser("clean", help="Clean captured artifacts")
    parser_clean.add_argument("-f", "--force", action="store_true", help="Force delete without confirmation")
    parser_clean.set_defaults(func=clean_artifacts)

    # Analyze command
    parser_analyze = subparsers.add_parser("analyze", help="Analyze artifacts and generate reports")
    parser_analyze.add_argument("--type", choices=['guards'], help="Type of analysis to run")
    parser_analyze.set_defaults(func=analyze_artifacts)

    args = parser.parse_args()

    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
