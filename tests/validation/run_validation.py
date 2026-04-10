# tests/validation/run_validation.py
"""
Entry point for the DSBF validation suite.

Usage:
    # Generate dataset(s) and validate their profiler output:
    python tests/validation/run_validation.py --datasets clean tiny

    # Validate against an already-generated report JSON:
    python tests/validation/run_validation.py --report path/to/report.json /
    --dataset clean

    # Run all available validations:
    python tests/validation/run_validation.py

Workflow per dataset:
    1. Generate CSV  (via tests/generators/dsbf_test_dataset_generator.py)
    2. Run DSBF profiler  (dsbf profile <csv>)
    3. Load the output report JSON
    4. Run the assertion module for that dataset
    5. Print pass/fail summary

After all assertions pass, the script prints a manual dashboard checklist
so you know exactly what to look at in the browser.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT: Path = Path(__file__).parent.parent.parent  # project root
DATASETS_DIR: Path = Path(__file__).parent.parent / "datasets"
GENERATOR_SCRIPT: Path = (
    Path(__file__).parent.parent / "generators" / "dsbf_test_dataset_generator.py"
)

# Maps dataset name → assertion module (relative to tests/validation/assertions/)
ASSERTION_MODULES: dict[str, str] = {
    "clean": "assertions.clean_dataset",
    "tiny": "assertions.tiny_dataset",
    # Add new datasets here as assertions are written:
    # "near_clean":       "assertions.near_clean_dataset",
    # "all_continuous":   "assertions.all_continuous_dataset",
    # "all_categorical":  "assertions.all_categorical_dataset",
    # "time_series":      "assertions.time_series_dataset",
}

# ── Manual dashboard checklist (printed after assertions pass) ─────────────────

DASHBOARD_CHECKLISTS: dict[str, list[str]] = {
    "clean": [
        "Overview: trust banner shows 'Looking Good' (green)",
        "Overview: DataHealthBar - all five traffic-light dots are green",
        "Overview: sample size adequacy metric is green (2,000 rows is adequate)",
        "Quality: trust banner shows 'Looking Good'",
        "Quality: all five dimension section headers show green dots",
        (
            "Quality: clicking each dimension header "
            "opens an 'All clear' body (no findings)"
        ),
        "Quality: Clean Columns section contains all 10 columns",
        (
            "Distributions: selecting each column shows "
            "no warning badges in section headers"
        ),
        "Distributions: Outlier Analysis shows 'Nothing flagged' for all columns",
        (
            "Distributions: Normality section badge shows"
            " 'Consistent with normal' for numeric cols"
        ),
        "Relationships: Summary card shows 0 collinearity and 0 leakage warnings",
        "ML Readiness: gate banner shows '✓ Ready for Modeling' (green)",
        "ML Readiness: all five dimension summary cards show 'All clear'",
    ],
    "tiny": [
        "Overview: sample size adequacy metric shows a warning (25 rows is too small)",
        "Overview: DataHealthBar renders without errors",
        "Quality: all sections render (including with potentially sparse findings)",
        (
            "Distributions: percentile table renders "
            "with 25-row data (some percentiles may duplicate)"
        ),
        "Distributions: selecting a column doesn't crash the detail panel",
        (
            "Relationships: association table either "
            "shows pairs or an appropriate 'no data' state"
        ),
        (
            "ML Readiness: no dimension shows an error-level "
            "finding triggered by tiny n alone"
        ),
        "No tab shows a blank white panel or uncaught error in the browser console",
    ],
}


# ── Core ──────────────────────────────────────────────────────────────────────


def generate_dataset(name: str) -> Path:
    """Generate a single dataset CSV via the generator script."""
    print(f"  Generating {name} dataset…")
    result: CompletedProcess[str] = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--only", name],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr)
        msg: str = f"Dataset generation failed for '{name}'"
        raise RuntimeError(msg)
    csv_path: Path = DATASETS_DIR / f"{name}.csv"
    assert csv_path.exists(), f"Expected {csv_path} to exist after generation"
    return csv_path


def find_report(name: str, override_dir: Path | None = None) -> Path | None:
    """
    Find the most recent DSBF report JSON for a dataset.

    DSBF saves runs under dsbf/outputs/<timestamp>/report.json.
    Searches all timestamped subdirectories and returns the most recently
    modified report.json.

    Args:
        name:         Dataset name - used to prefer name-matching runs when
                      multiple timestamped runs exist.
        override_dir: If provided, search this directory instead of the
                      default dsbf/outputs/ location.

    """
    outputs_dir: Path = override_dir or (ROOT / "dsbf" / "outputs")

    if not outputs_dir.exists():
        return None

    # Find all report.json files under timestamped subdirectories,
    # sorted most-recently-modified first
    candidates: list[Path] = sorted(
        outputs_dir.glob("*/report.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        return None

    # If only one run exists, use it regardless of name
    if len(candidates) == 1:
        return candidates[0]

    # Prefer a run whose parent directory name contains the dataset name
    name_matches: list[Path] = [p for p in candidates if name in p.parent.name]
    if name_matches:
        return name_matches[0]

    # Fall back to the most recently modified run
    return candidates[0]


def run_assertions(name: str, report_path: Path) -> bool:
    """Load report JSON and run the assertion module. Returns True on pass."""
    print(f"  Loading report from {report_path}…")
    with Path.open(report_path) as f:
        report = json.load(f)

    module_path: str = f"tests.validation.{ASSERTION_MODULES[name]}"
    try:
        module: ModuleType = importlib.import_module(module_path)
    except ImportError:
        # Try relative import if running from project root
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        module = importlib.import_module(f"tests.validation.{ASSERTION_MODULES[name]}")

    try:
        module.validate(report)
        return True
    except AssertionError as e:
        print(f"\n✗  ASSERTION FAILED - {name} dataset\n  {e}")
        return False


def print_dashboard_checklist(name: str) -> None:
    checklist: list[str] = DASHBOARD_CHECKLISTS.get(name, [])
    if not checklist:
        return
    print(f"\n{'─' * 60}")
    print(f"  Manual dashboard checklist - {name} dataset")
    print(f"{'─' * 60}")
    for item in checklist:
        print(f"  □  {item}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="DSBF validation suite")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(ASSERTION_MODULES),
        default=list(ASSERTION_MODULES),
        help="Which datasets to validate (default: all with assertion modules)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to a specific report JSON file (skips auto-detection)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to search for report JSONs (overrides default dsbf/outputs/)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip dataset generation (use existing CSVs in tests/datasets/)",
    )
    args: Namespace = parser.parse_args()

    results: dict[str, bool] = {}

    for name in args.datasets:
        print(f"\n{'═' * 60}")
        print(f"  Validating: {name}")
        print(f"{'═' * 60}")

        # Step 1: generate dataset
        if not args.skip_generate:
            try:
                csv_path: Path = generate_dataset(name)
                print(f"  CSV: {csv_path}")
            except RuntimeError as e:
                print(f"✗  {e}")
                results[name] = False
                continue
        else:
            csv_path = DATASETS_DIR / f"{name}.csv"
            if not csv_path.exists():
                print(f"✗  {csv_path} not found - run without --skip-generate first")
                results[name] = False
                continue

        # Step 2: locate report (user must run profiler separately)
        if args.report:
            report_path = args.report
        else:
            report_path: Path | None = find_report(name, override_dir=args.report_dir)
            if not report_path:
                outputs_dir: Path | Any = args.report_dir or (ROOT / "dsbf" / "outputs")
                print(f"\n  ⚠  No report found for '{name}'.")
                print(f"     Searched: {outputs_dir} (and timestamped subdirectories)")
                print("     Profile the dataset first:")
                print(f"       dsbf profile {csv_path}")
                print("     Then re-run with --skip-generate")
                results[name] = False
                continue
            print(f"  Report:  {report_path}")

        # Step 3: run assertions
        passed: bool = run_assertions(name, report_path)
        results[name] = passed

        # Step 4: print dashboard checklist
        if passed:
            print_dashboard_checklist(name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Summary")
    print(f"{'═' * 60}")
    for name, passed in results.items():
        icon: str = "✓" if passed else "✗"
        print(f"  {icon}  {name}")
    print()

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
