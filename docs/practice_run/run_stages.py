"""
Instrumented DSBF practice run.

Wraps every stage function that ProfileEngine.run() calls with a wall-time /
RSS timer, then profiles the whole run with cProfile. Fully isolated: callers
must set DSBF_DB_PATH, DSBF_AUTO_EXPORT_METADATA=0 and run from a scratch CWD.

Usage: python run_stages.py <label> <titanic|kitchen:N|adversarial> [--ts COL]
"""

import cProfile
import json
import os
import pstats
import sys
import time

import psutil

T0 = time.perf_counter()
import dsbf.eda.profile_engine as pe  # noqa: E402

IMPORT_S = time.perf_counter() - T0

from dsbf.config import load_default_config  # noqa: E402
from dsbf.eda.graph import ExecutionGraph  # noqa: E402

PROC = psutil.Process()
STAGES: list[dict] = []


def rss_mb() -> float:
    return PROC.memory_info().rss / 2**20


def timed(name, fn):
    def wrapper(*a, **kw):
        r0, t0 = rss_mb(), time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            STAGES.append(
                {
                    "stage": name,
                    "seconds": round(time.perf_counter() - t0, 3),
                    "rss_before_mb": round(r0, 1),
                    "rss_after_mb": round(rss_mb(), 1),
                }
            )

    return wrapper


# Patch the names ProfileEngine.run() resolves from its module namespace.
for fname in [
    "data_sampling",
    "AnalysisContext",
    "load_task_group",
    "validate_config_and_graph",
    "infer_stage",
    "build_column_task_mappings",
    "render_user_report",
    "write_metadata_report",
    "persist_run",
]:
    setattr(pe, fname, timed(fname, getattr(pe, fname)))
pe.ProfileEngine._load_data = timed("_load_data", pe.ProfileEngine._load_data)
pe.ProfileEngine.build_graph = timed("build_graph", pe.ProfileEngine.build_graph)
pe.ProfileEngine.record_run = timed("record_run", pe.ProfileEngine.record_run)
ExecutionGraph.run = timed("graph.run (all tasks)", ExecutionGraph.run)
ExecutionGraph.visualize = timed("graph.visualize (DAG png)", ExecutionGraph.visualize)


def main() -> None:
    label, dataset = sys.argv[1], sys.argv[2]
    ts_col = sys.argv[sys.argv.index("--ts") + 1] if "--ts" in sys.argv else None
    out_root = os.path.abspath("runs")
    os.makedirs(out_root, exist_ok=True)

    cfg = load_default_config()
    # Match committed defaults (working tree enables NFL-specific TS settings).
    cfg["time_series"].update(enabled=bool(ts_col), datetime_index_column=ts_col)
    cfg["tasks"]["decompose_time_series"]["seasonal_period"] = None
    cfg["metadata"]["message_verbosity"] = "stage"
    cfg["output_dir"] = os.path.join(out_root, label)

    if dataset == "titanic":
        cfg["metadata"].update(
            dataset_name="titanic", dataset_source="seaborn", dataset_path=None
        )
    else:
        cfg["metadata"].update(
            dataset_name=label, dataset_path=os.path.abspath(dataset)
        )

    c0 = time.perf_counter()
    r0, t0 = rss_mb(), time.perf_counter()
    engine = pe.ProfileEngine(cfg)
    STAGES.insert(
        0,
        {
            "stage": "ProfileEngine.__init__",
            "seconds": round(time.perf_counter() - t0, 3),
            "rss_before_mb": round(r0, 1),
            "rss_after_mb": round(rss_mb(), 1),
        },
    )
    prof = cProfile.Profile()
    prof.enable()
    engine.run()
    prof.disable()
    total = time.perf_counter() - c0

    prof_path = os.path.join(out_root, f"{label}.prof")
    prof.dump_stats(prof_path)

    md = engine.context.metadata
    durations = md.get("task_durations", {})
    statuses = {k: v.status for k, v in engine.results.items()}
    data_sizes = {
        k: len(json.dumps(v.to_dict().get("data"), default=str))
        for k, v in engine.results.items()
    }
    out_files = {}
    for root, _, files in os.walk(cfg["output_dir"]):
        for f in files:
            p = os.path.join(root, f)
            out_files[os.path.relpath(p, cfg["output_dir"])] = os.path.getsize(p)

    summary = {
        "label": label,
        "import_seconds": round(IMPORT_S, 2),
        "total_seconds": round(total, 2),
        "peak_rss_mb": round(rss_mb(), 1),
        "shape": list(engine.context.data.shape),
        "data_type": type(engine.context.data).__module__.split(".")[0],
        "inferred_stage": engine.inferred_stage,
        "stages": STAGES,
        "task_outcomes": md.get("task_outcomes"),
        "task_count": len(engine.results),
        "status_counts": {
            s: sum(1 for v in statuses.values() if v == s)
            for s in set(statuses.values())
        },
        "failed": {
            k: (v.error_metadata or {}).get("trace_summary")
            for k, v in engine.results.items()
            if v.status != "success"
        },
        "slowest_tasks": sorted(durations.items(), key=lambda kv: -kv[1])[:12],
        "sum_task_seconds": round(sum(durations.values()), 2),
        "largest_task_data_bytes": sorted(data_sizes.items(), key=lambda kv: -kv[1])[
            :8
        ],
        "output_files_bytes": dict(
            sorted(out_files.items(), key=lambda kv: -kv[1])[:10]
        ),
        "output_file_count": len(out_files),
        "column_task_map_keys_not_columns": sorted(
            set(md.get("column_task_map", {}))
            - set(map(str, engine.context.data.columns))
        )[:25],
        "reliability_flags_n_rows": (engine.context.reliability_flags or {}).get(
            "n_rows"
        ),
    }
    with open(os.path.join(out_root, f"{label}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    stats = pstats.Stats(prof_path)
    stats.sort_stats("cumulative")
    with open(os.path.join(out_root, f"{label}.prof.txt"), "w") as f:
        stats.stream = f
        stats.print_stats(45)
        stats.sort_stats("tottime")
        stats.print_stats(30)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
