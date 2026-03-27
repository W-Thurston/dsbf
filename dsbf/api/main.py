# dsbf/api/main.py

"""
DSBF FastAPI application.

Start locally:
    uvicorn dsbf.api.main:app --reload

The API is intentionally thin - it reads from the SQLite database and serves
figure files from disk.  All heavy computation stays in the profiling engine.

Endpoints
---------
  GET  /api/datasets                           list all datasets
  GET  /api/datasets/{name}                    dataset detail + run history
  GET  /api/datasets/{name}/runs               list runs for a dataset
  GET  /api/runs/{run_key}                     run detail (no task results)
  GET  /api/runs/{run_key}/tasks               all task results for a run
  GET  /api/runs/{run_key}/tasks/{task_name}   single task result
  GET  /api/runs/{run_key}/figures             figure index for a run
  GET  /api/figures/{figure_id}/file           serve the actual figure file
  GET  /api/runs/compare                       compare a task across run_keys
  GET  /api/runs/{run_key}/dq-status           data-health header bar summary
  GET  /api/runs/{run_key}/ml-readiness        ML readiness score and column report
  GET  /health                                 liveness check
"""

import json as _json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dsbf.api import db

#############
# App setup #
#############
app = FastAPI(
    title="DSBF API",
    description="Data Scientist's Best Friend - profiling run history and report data.",
    version="0.1.0",
)

# CORS - allow the Vue frontend (any localhost port during dev, configurable in prod)
_CORS_ORIGINS: list[str] = os.environ.get(
    "DSBF_CORS_ORIGINS",
    "http://localhost:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Optional db_path override - falls back to schema.py resolution if not set
_DB_PATH: str | None = os.environ.get("DSBF_DB_PATH") or None


##########
# Health #
##########
@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


############
# Datasets #
############
@app.get("/api/datasets", tags=["datasets"])
def list_datasets() -> list[dict[str, Any]]:
    """List all datasets with a summary of their run history."""
    return db.list_datasets(_DB_PATH)


@app.get("/api/datasets/{name}", tags=["datasets"])
def get_dataset(name: str) -> dict[str, Any]:
    """Get a single dataset by name."""
    dataset = db.get_dataset(name, _DB_PATH)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found.")
    return dataset


@app.get("/api/datasets/{name}/runs", tags=["datasets"])
def list_runs_for_dataset(name: str) -> list[dict[str, Any]]:
    """List all runs for a dataset, newest first."""
    dataset = db.get_dataset(name, _DB_PATH)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found.")
    return db.list_runs(name, _DB_PATH)


########
# Runs #
########
@app.get("/api/runs/compare", tags=["runs"])
def compare_runs(
    run_keys: Annotated[
        list[str],
        Query(description="Two or more run_keys to compare"),
    ],
    task_name: Annotated[
        str,
        Query(description="Task whose summary to compare across runs"),
    ],
) -> dict[str, Any]:
    """
    Compare a single task's summary across multiple runs.

    Example:
      GET /api/runs/compare?run_keys=20250715_080620&run_keys=20260303_075308&
          task_name=data_quality_scorer

    Args:
        run_keys (Annotated[ list[str], Query, optional): _description_.
        Defaults to "Two or more run_keys to compare") ].
        task_name (Annotated[ str, Query, optional): _description_.
        Defaults to "Task whose summary to compare across runs") ].

    Raises:
        HTTPException: _description_

    Returns:
        dict[str, Any]: _description_

    """
    if len(run_keys) < 2:  # noqa: PLR2004
        raise HTTPException(
            status_code=400,
            detail="Provide at least two run_keys to compare.",
        )
    return db.compare_runs(run_keys, task_name, _DB_PATH)


@app.get("/api/runs/{run_key}", tags=["runs"])
def get_run(run_key: str) -> dict[str, Any]:
    """Get run metadata by run_key (does not include task results)."""
    run = db.get_run(run_key, _DB_PATH)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_key}' not found.")
    return run


@app.get("/api/runs/{run_key}/tasks", tags=["runs"])
def get_run_tasks(run_key: str) -> dict[str, Any]:
    """
    Get all task results for a run.

    Returns a dict keyed by task_name - mirrors the structure of report.json.
    """
    run = db.get_run(run_key, _DB_PATH)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_key}' not found.")
    return db.get_run_tasks(run_key, _DB_PATH)


@app.get("/api/runs/{run_key}/tasks/{task_name}", tags=["runs"])
def get_task(run_key: str, task_name: str) -> dict[str, Any]:
    """Get a single task result for a run."""
    task = db.get_task(run_key, task_name, _DB_PATH)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_name}' not found for run '{run_key}'.",
        )
    return task


@app.get("/api/runs/{run_key}/correlations/{column}")
async def run_column_correlations(run_key: str, column: str, threshold: float = 0.0):
    """
    Return pairwise correlations for a single column.

    Reads from compute_pairwise_associations (the single source of truth for
    all column relationships). Extracts the numeric correlation value from the
    rich entry dict - for Pearson/Spearman pairs this is entry["metric"]; the
    metric_type is also returned for the frontend to display appropriately.
    """
    run: dict | None = db.get_run(run_key)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    task: dict | None = db.get_task(run_key, "compute_pairwise_associations")
    if not task or not task.get("data"):
        return {
            "column": column,
            "correlations": [],
            "unavailable": True,
            "reason": (
                "Correlation data not available. Run the profiler at full depth "
                "to enable this feature."
            ),
        }

    try:
        raw: dict | Any = (
            task["data"]
            if isinstance(task["data"], dict)
            else _json.loads(task["data"])
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "column": column,
            "correlations": [],
            "unavailable": True,
            "reason": f"Could not parse correlation data: {exc}",
        }

    results = []
    for pair_key, entry in raw.items():
        # Skip the sentinel correlation matrix key.
        if pair_key == "__correlation_matrix__":
            continue
        if "|" not in pair_key:
            continue
        parts = pair_key.split("|", 1)
        if column not in parts:
            continue

        other = parts[1] if parts[0] == column else parts[0]

        # entry is a rich dict with "metric", "metric_type", "strength", etc.
        # Fall back to treating entry as a raw float for any legacy data.
        if isinstance(entry, dict):
            try:
                corr = float(entry["metric"])
                metric_type = entry.get("metric_type", "pearson_r")
                strength = entry.get("strength", "")
            except (KeyError, TypeError, ValueError):
                continue
        else:
            try:
                corr = float(entry)
                metric_type = "pearson_r"
                strength = ""
            except (TypeError, ValueError):
                continue

        if abs(corr) >= threshold:
            results.append(
                {
                    "column": other,
                    "correlation": round(corr, 4),
                    "metric_type": metric_type,
                    "strength": strength,
                },
            )

    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {"column": column, "correlations": results, "unavailable": False}


@app.get("/api/runs/{run_key}/sample")
def read_run_sample(run_key: str, n: int = 10):
    result = db.get_run_sample(run_key, n=min(n, 50))
    if result is None:
        # source_path not recorded or file no longer on disk - return empty
        # payload rather than 404 so the frontend can show a friendly message
        return {"columns": [], "rows": [], "unavailable": True}
    return result


#################
# Data health   #
#################


@app.get("/api/runs/{run_key}/dq-status", tags=["runs"])
def get_dq_status(run_key: str) -> dict[str, Any]:
    """
    Return the data-health header bar summary for a run.

    Reads the data_quality_scorer task result and returns a compact dict
    with one entry per category, shaped for direct consumption by the
    Vue header bar component.  The full scorer output (with per-column
    findings) is still available via /tasks/data_quality_scorer.

    Response shape:
        {
            "available": true,
            "total_columns": 42,
            "categories": {
                "completeness": {
                    "level": "amber",
                    "affected_count": 3,
                    "pct_affected": 0.071
                },
                ...
            }
        }

    If the scorer has not run, returns { "available": false }.
    """
    run = db.get_run(run_key, _DB_PATH)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_key}' not found.")

    task = db.get_task(run_key, "data_quality_scorer", _DB_PATH)
    if not task or not task.get("data"):
        return {"available": False}

    raw_data = task["data"]
    categories_raw = raw_data.get("categories") or {}
    total_columns = raw_data.get("total_columns") or 0

    categories: dict[str, Any] = {}
    for name, block in categories_raw.items():
        categories[name] = {
            "level": block.get("level", "green"),
            "affected_count": block.get("affected_count", 0),
            "pct_affected": block.get("pct_affected", 0.0),
        }

    return {
        "available": True,
        "total_columns": total_columns,
        "categories": categories,
    }


@app.get("/api/runs/{run_key}/ml-readiness", tags=["runs"])
def get_ml_readiness(run_key: str) -> dict[str, Any]:
    """
    Return the ML readiness report for a run.

    Reads the ml_readiness_scorer task result.  Returns the full structured
    report including overall score, gate, per-column scores, and all ML
    guidance blurbs sorted by priority (worst columns first).

    Response shape:
        {
            "available": true,
            "overall_score":       int,
            "readiness_gate":      "ready"|"needs_work"|"not_ready",
            "total_columns":       int,
            "columns_ready":       int,
            "level_summary":       {level: count},
            "column_scores":       {col: int},
            "columns_by_priority": [{column, score, worst_level,
                                     blurb_count, task_count, blurbs}, ...]
        }

    If the scorer has not run, returns { "available": false }.
    """
    run = db.get_run(run_key, _DB_PATH)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_key}' not found.")

    task = db.get_task(run_key, "ml_readiness_scorer", _DB_PATH)
    if not task or not task.get("data"):
        return {"available": False}

    return {
        "available": True,
        **task["data"],
    }


###########
# Figures #
###########


@app.get("/api/runs/{run_key}/associations", tags=["relationships"])
def get_run_associations(run_key: str):
    """
    Return all pairwise association results from compute_pairwise_associations.

    The __correlation_matrix__ sentinel key is excluded from the pairs response
    since it is a derived artifact consumed by the plots endpoint, not a
    column-pair association entry.
    """
    run = db.get_run(run_key, _DB_PATH)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_key}' not found.")

    task = db.get_task(run_key, "compute_pairwise_associations", _DB_PATH)
    if not task or not task.get("data"):
        return {
            "source": "compute_pairwise_associations",
            "pairs": {},
            "summary": {"message": "Association data not available."},
            "unavailable": True,
        }

    # Strip the sentinel key before returning to the frontend.
    pairs = {k: v for k, v in task["data"].items() if k != "__correlation_matrix__"}

    return {
        "source": "compute_pairwise_associations",
        "pairs": pairs,
        "summary": task.get("summary", {}),
        "unavailable": False,
    }


def _pearson_strength(val: float) -> str:
    v = abs(val)
    if v >= 0.7:
        return "strong"
    if v >= 0.4:
        return "moderate"
    if v >= 0.2:
        return "weak"
    return "negligible"


@app.get("/api/runs/{run_key}/associations/{column}", tags=["relationships"])
def get_column_associations(run_key: str, column: str, min_strength: str = ""):
    """
    Return all pairwise associations for a single column, sorted by abs metric desc.
    Optional min_strength filter: "strong" | "moderate" | "weak" (inclusive upward).
    """
    all_data = get_run_associations(run_key)
    pairs = all_data.get("pairs", {})

    _strength_order = {"strong": 3, "moderate": 2, "weak": 1, "negligible": 0}
    min_rank = _strength_order.get(min_strength, -1)

    results = []
    for key, val in pairs.items():
        if "|" not in key:
            continue
        parts = key.split("|", 1)
        if column not in parts:
            continue
        other = parts[1] if parts[0] == column else parts[0]
        if _strength_order.get(val.get("strength", "negligible"), 0) < min_rank:
            continue
        results.append(
            {
                "column": other,
                **val,
            }
        )

    results.sort(key=lambda x: abs(x["metric"]), reverse=True)
    return {
        "column": column,
        "associations": results,
        "source": all_data.get("source"),
        "unavailable": len(pairs) == 0,
    }


@app.get("/api/runs/{run_key}/column-data", tags=["relationships"])
def get_column_data(
    run_key: str,
    cols: Annotated[list[str], Query(description="Column names to fetch")] = [],
    max_rows: int = 3000,
):
    """
    Return raw column values from the source dataset for pair plotting.
    Subsampled to max_rows if the dataset is larger.
    """
    if not cols:
        raise HTTPException(
            status_code=400, detail="Provide at least one column via ?cols="
        )
    if max_rows > 10_000:
        max_rows = 10_000

    result = db.get_column_data(run_key, cols, max_rows, _DB_PATH)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Source file not found. Ensure source_path is set for this dataset.",
        )
    return result


@app.get("/api/runs/{run_key}/figures", tags=["figures"])
def get_run_figures(run_key: str) -> list[dict[str, Any]]:
    """
    Get the figure index for a run.

    Returns a list of figure metadata records.  Use the id from each record
    to fetch the actual file via GET /api/figures/{id}/file.
    """
    run = db.get_run(run_key, _DB_PATH)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_key}' not found.")
    return db.get_run_figures(run_key, _DB_PATH)


@app.get("/api/figures/{figure_id}/file", tags=["figures"])
def get_figure_file(figure_id: int) -> FileResponse:
    """
    Serve the actual figure file (PNG or JSON) by figure id.

    The Content-Type is inferred from the file extension.
    """
    file_path_str = db.get_figure_path(figure_id, _DB_PATH)
    if not file_path_str:
        raise HTTPException(status_code=404, detail=f"Figure {figure_id} not found.")

    # Paths in the database are relative to the repo root
    repo_root = Path(__file__).resolve().parents[2]
    file_path = repo_root / file_path_str

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Figure file not found on disk: {file_path_str}",
        )

    # Determine media type from extension
    ext: str = file_path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
        ".svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")

    return FileResponse(path=str(file_path), media_type=media_type)
