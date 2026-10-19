"""Child process for the engine runs: one ``dsbf.profile()`` call and what it cost.

A separate script in a fresh interpreter, like ``measure_one.py``, so the import time
and the peak memory belong to this run alone. It imports nothing from ``tests``.

The call covers loading the file, the profile pass, planning and executing every
adapted task, so ``total_seconds`` is what a library user waits. ``tasks`` holds each
check's wall time as the executor measured it.

Usage: python measure_engine.py <name> <csv> <out_dir> --workers {default,N}
                                [--time-index COLUMN]
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

START = time.perf_counter()
import dsbf  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - START
FAILED = {"error", "timeout"}


def peak_rss_mb() -> float:
    """Peak resident set size of this process (``ru_maxrss`` is KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("csv")
    parser.add_argument("out_dir")
    parser.add_argument("--workers", required=True)
    parser.add_argument("--time-index")
    args = parser.parse_args()

    started = time.perf_counter()
    run = dsbf.profile(
        Path(args.csv),
        name=args.name,
        depth="deep",
        time_index=args.time_index,
        workers=None if args.workers == "default" else int(args.workers),
    )
    total_seconds = time.perf_counter() - started

    result = {
        "shape": [run.dataset.n_rows, run.dataset.n_columns],
        "workers": args.workers,
        "n_checks": len(run.results),
        "statuses": run.health.counts.model_dump(),
        "failed_checks": sorted(
            check_id
            for check_id, result in run.results.items()
            if result.status.value in FAILED
        ),
        "import_seconds": round(IMPORT_SECONDS, 3),
        "total_seconds": round(total_seconds, 3),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "tasks": {
            check_id: round(result.timing.wall_seconds, 3)
            for check_id, result in run.results.items()
        },
        "run_bytes": len(run.model_dump_json()),
    }
    (Path(args.out_dir) / "perf_run.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
