"""Child process for the engine runs: one ``dsbf.profile()`` call and what it cost.

A separate script in a fresh interpreter, like ``measure_one.py``, so the import time
and the peak memory belong to this run alone. It imports nothing from ``tests``.

The call covers loading the file, the profile pass, planning and executing the
adapted tasks the depth selects, so ``total_seconds`` is what a library user waits.
``tasks`` holds each check's wall time as the executor measured it.

``--workers default`` leaves the choice to ``dsbf.profile()``; ``per-core`` is the
executor's own default, one per core.

Usage: python measure_engine.py <name> <csv> <out_dir> --workers {default,per-core,N}
                                --depth {quick,standard,deep} [--time-index COLUMN]
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

START = time.perf_counter()
import dsbf  # noqa: E402
from dsbf.engine.scheduler import default_workers  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - START
FAILED = {"error", "timeout"}


def peak_rss_mb() -> float:
    """Peak resident set size of this process.

    ``VmHWM`` where Linux has it: ``ru_maxrss`` survives ``exec``, so it would start
    at the harness's own peak at the moment it started this child. ``ru_maxrss``
    elsewhere (KiB on Linux, bytes on macOS).
    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _workers(value: str) -> int | None:
    if value == "default":
        return None
    if value == "per-core":
        return default_workers()
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("csv")
    parser.add_argument("out_dir")
    parser.add_argument("--workers", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--time-index")
    args = parser.parse_args()

    started = time.perf_counter()
    run = dsbf.profile(
        Path(args.csv),
        name=args.name,
        depth=args.depth,
        time_index=args.time_index,
        workers=_workers(args.workers),
    )
    total_seconds = time.perf_counter() - started

    result = {
        "shape": [run.dataset.n_rows, run.dataset.n_columns],
        "workers": args.workers,
        "depth": args.depth,
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
