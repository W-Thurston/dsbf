# migrate_runs.py

"""
One-off script to backfill existing DSBF output directories into the SQLite database.

Run from the repo root:

    python migrate_runs.py

Optional arguments:

    python migrate_runs.py --outputs-dir path/to/dsbf/outputs
    python migrate_runs.py --db-path path/to/custom.db
    python migrate_runs.py --dry-run       # print what would be migrated, write nothing

The script is safe to re-run - runs already in the database are skipped.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Make sure the repo root is on sys.path so dsbf imports work when running
# this script directly from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dsbf.storage.schema import init_db  # noqa: E402
from dsbf.storage.writer import persist_run_from_dict  # noqa: E402

# Helpers


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}


def _parse_ran_at(run_key: str, metadata: dict) -> str | None:
    """
    Return an ISO-format datetime string for when this run executed.

    Priority:
      1. metadata["ran_at"] if present
      2. Parse the run_key itself ("20260303_075308" -> "2026-03-03 07:53:08")
    """
    if metadata.get("ran_at"):
        return metadata["ran_at"]
    try:
        return datetime.strptime(run_key, "%Y%m%d_%H%M%S").isoformat(
            sep=" "
        )  # noqa: DTZ007
    except ValueError:
        return None


def _extract_run_meta(report: dict, metadata: dict, run_key: str) -> dict:
    """Pull the fields we need for the runs table from report + metadata dicts."""
    results = report.get("results", report)

    quality_task = results.get("data_quality_scorer", {})
    quality_score = (
        quality_task.get("summary", {}).get("overall_score")
        if isinstance(quality_task, dict)
        else None
    )

    shape = results.get("summarize_dataset_shape", {})
    shape_data = shape.get("data", {}) if isinstance(shape, dict) else {}

    return {
        "profiling_depth": metadata.get("profiling_depth"),
        "inferred_stage": metadata.get("inferred_stage"),
        "config": metadata.get("config"),
        "row_count": shape_data.get("num_rows") or metadata.get("row_count"),
        "col_count": shape_data.get("num_columns") or metadata.get("col_count"),
        "quality_score": quality_score,
        "ran_at": _parse_ran_at(run_key, metadata),
    }


def _infer_dataset_name(metadata: dict, run_key: str) -> tuple[str, str | None]:
    """
    Tries metadata fields first, falls back to run_key.

    Every run gets recorded even if the original path is unknown.

    Args:
        metadata (dict): _description_
        run_key (str): _description_

    Returns:
        tuple[str, str | None]: (dataset_name, source_path).

    """
    source_path = metadata.get("dataset_path") or metadata.get("data_path")
    dataset_name = metadata.get("dataset_name")

    if not dataset_name and source_path:
        dataset_name = Path(source_path).stem

    if not dataset_name:
        dataset_name = f"unknown_{run_key}"

    return dataset_name, source_path


# Core migration logic
def migrate_directory(
    output_dir: Path,
    db_path: Path | None,
    dry_run: bool,
    source_path_override: str | None = None,
) -> bool:
    """
    Attempt to migrate a single output directory.

    Args:
        output_dir:           Path to the timestamped output directory.
        db_path:              Optional override for the SQLite database path.
        dry_run:              If True, print what would happen without writing.
        source_path_override: If provided, use this as the source_path for the
                              dataset record instead of whatever is in metadata.

    Returns:
        bool: True if the run was (or would be) migrated, False if skipped.

    """
    run_key = output_dir.name  # e.g. "20260303_075308"

    report_path = output_dir / "report.json"
    metadata_path = output_dir / "metadata_report.json"

    if not report_path.exists():
        logger.warning("  Skipping %s - no report.json found.", run_key)
        return False

    report = _load_json(report_path)
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}

    dataset_name, source_path = _infer_dataset_name(metadata, run_key)
    # CLI --source-path overrides whatever the metadata contained
    if source_path_override:
        source_path = source_path_override
        if not dataset_name or dataset_name.startswith("unknown_"):
            dataset_name = Path(source_path_override).stem
    run_meta = _extract_run_meta(report, metadata, run_key)
    results = report.get("results", report)

    logger.info(
        "  %s  dataset=%-20s  depth=%-10s  quality=%s",
        run_key,
        dataset_name,
        run_meta.get("profiling_depth") or "?",
        (
            f"{run_meta['quality_score']:.1f}"
            if run_meta.get("quality_score") is not None
            else "?"
        ),
    )

    if dry_run:
        return True

    try:
        persist_run_from_dict(
            run_key=run_key,
            dataset_name=dataset_name,
            source_path=source_path,
            run_meta=run_meta,
            results=results,
            db_path=db_path,
        )
        return True

    except Exception as exc:
        logger.error("  Failed to migrate %s: %s", run_key, exc)
        return False


# Entry point


def main():
    parser = argparse.ArgumentParser(
        description="Backfill existing DSBF output directories into SQLite.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("dsbf/outputs"),
        help="Path to the outputs directory (default: dsbf/outputs)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to the database file (default: dsbf/dsbf.db or DSBF_DB_PATH env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without writing anything",
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help=(
            "Path to the source data file (e.g. dsbf_test_dataset.csv). "
            "Overrides whatever is in metadata_report.json. "
            "Stored as-is so the API can resolve it relative to the repo root."
        ),
    )
    args = parser.parse_args()

    if not args.outputs_dir.is_dir():
        logger.error("Outputs directory not found: %s", args.outputs_dir)
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN - nothing will be written to the database.")
    else:
        db_path = init_db(args.db_path)
        logger.info("Database: %s", db_path)

    # Collect all subdirectories that look like timestamped run dirs
    run_dirs = sorted(
        d
        for d in args.outputs_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not run_dirs:
        logger.info("No run directories found in %s.", args.outputs_dir)
        sys.exit(0)

    logger.info(
        "Found %d run director%s in %s:\n",
        len(run_dirs),
        "y" if len(run_dirs) == 1 else "ies",
        args.outputs_dir,
    )

    migrated = 0
    skipped = 0

    for run_dir in run_dirs:
        success = migrate_directory(
            run_dir, args.db_path, args.dry_run, args.source_path
        )
        if success:
            migrated += 1
        else:
            skipped += 1

    label = "Would migrate" if args.dry_run else "Migrated"
    logger.info(
        "\n%s %d run%s.  Skipped %d.",
        label,
        migrated,
        "s" if migrated != 1 else "",
        skipped,
    )


if __name__ == "__main__":
    main()
