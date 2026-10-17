"""Child process for the performance runs: profile one CSV and write what it cost.

This is a separate script, run in a fresh interpreter, so the import time, the peak
memory and the first-call costs belong to this run alone. It imports nothing from
``tests``.

Usage: python measure_one.py <name> <csv> <out_dir> [--time-series COLUMN] [--cprofile]
"""

import argparse
import cProfile
import json
import os
import pstats
import resource
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

START = time.perf_counter()
import dsbf.eda.profile_engine as pe  # noqa: E402
from dsbf.config import load_default_config  # noqa: E402
from dsbf.eda.graph import ExecutionGraph  # noqa: E402

IMPORT_SECONDS = time.perf_counter() - START

# Functions `ProfileEngine.run()` resolves from its module namespace, in run order.
TIMED_FUNCTIONS = [
    "data_sampling",
    "AnalysisContext",
    "load_task_group",
    "validate_config_and_graph",
    "infer_stage",
    "build_column_task_mappings",
    "render_user_report",
    "write_metadata_report",
    "persist_run",
]
STAGE_SECONDS: dict[str, float] = defaultdict(float)

# What the run view loads when a run is opened, one request per tab.
API_PATHS = [
    ("run", "/api/runs/{run_key}", {}),
    ("tasks", "/api/runs/{run_key}/tasks", {}),
    ("figures", "/api/runs/{run_key}/figures", {}),
    ("dq_status", "/api/runs/{run_key}/dq-status", {}),
    ("ml_readiness", "/api/runs/{run_key}/ml-readiness", {}),
    ("associations", "/api/runs/{run_key}/associations", {}),
    ("sample", "/api/runs/{run_key}/sample", {"n": 10}),
]


def peak_rss_mb() -> float:
    """Peak resident set size of this process (``ru_maxrss`` is KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def timed(name, fn):
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            STAGE_SECONDS[name] += time.perf_counter() - started

    return wrapper


def instrument() -> None:
    for name in TIMED_FUNCTIONS:
        setattr(pe, name, timed(name, getattr(pe, name)))
    engine = pe.ProfileEngine
    engine.__init__ = timed("ProfileEngine.__init__", engine.__init__)
    engine._load_data = timed("_load_data", engine._load_data)
    engine.build_graph = timed("build_graph", engine.build_graph)
    ExecutionGraph.run = timed("graph.run (all tasks)", ExecutionGraph.run)


def probe_api() -> dict:
    """Open the persisted run through the API the way the dashboard does."""
    try:
        from fastapi.testclient import TestClient

        from dsbf.api.main import app

        db_path = os.environ["DSBF_DB_PATH"]
        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT run_key FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return {"error": "no run was persisted"}
        # A server error becomes a recorded 500, not an exception.
        client = TestClient(app, raise_server_exceptions=False)
        probe = {}
        for label, template, params in API_PATHS:
            started = time.perf_counter()
            response = client.get(template.format(run_key=row[0]), params=params)
            probe[label] = {
                "status": response.status_code,
                "bytes": len(response.content),
                "milliseconds": round((time.perf_counter() - started) * 1000, 1),
            }
        return probe
    except Exception as exc:  # the probe must not lose the run's timings
        return {"error": f"{type(exc).__name__}: {exc}"}


def directory_bytes(directory: Path) -> int:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("csv")
    parser.add_argument("out_dir")
    parser.add_argument("--time-series")
    parser.add_argument("--cprofile", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    config = load_default_config()
    config["output_dir"] = str(out_dir)
    config["metadata"].update(
        dataset_name=args.name, dataset_path=os.path.abspath(args.csv)
    )
    config["time_series"].update(
        enabled=args.time_series is not None, datetime_index_column=args.time_series
    )

    instrument()
    profiler = cProfile.Profile() if args.cprofile else None
    started = time.perf_counter()
    engine = pe.ProfileEngine(config)
    if profiler:
        profiler.enable()
    engine.run()
    if profiler:
        profiler.disable()
    total_seconds = time.perf_counter() - started

    context = engine.context
    assert context is not None
    outcomes = context.metadata.get("task_outcomes", {})
    report = out_dir / "report.json"
    result = {
        "shape": list(context.data.shape),
        "inferred_stage": engine.inferred_stage,
        "task_outcomes": {status: len(names) for status, names in outcomes.items()},
        "failed_tasks": sorted(outcomes.get("failed", [])),
        "import_seconds": round(IMPORT_SECONDS, 3),
        "total_seconds": round(total_seconds, 3),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "stages": {k: round(v, 3) for k, v in STAGE_SECONDS.items()},
        "tasks": {
            k: round(v, 3)
            for k, v in context.metadata.get("task_durations", {}).items()
        },
        "output_bytes": directory_bytes(out_dir),
        "report_bytes": report.stat().st_size if report.exists() else None,
    }
    result["api"] = probe_api()
    (out_dir / "perf_run.json").write_text(json.dumps(result))

    if profiler:
        profiler.dump_stats(str(out_dir / "run.prof"))
        with open(out_dir / "run.prof.txt", "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("cumulative").print_stats(45)
            stats.sort_stats("tottime").print_stats(30)


if __name__ == "__main__":
    main()
