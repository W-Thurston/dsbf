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
  GET  /health                                 liveness check
"""

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


@app.get("/api/runs/{run_key}/sample")
def read_run_sample(run_key: str, n: int = 10):
    result = db.get_run_sample(run_key, n=min(n, 50))
    if result is None:
        # source_path not recorded or file no longer on disk - return empty
        # payload rather than 404 so the frontend can show a friendly message
        return {"columns": [], "rows": [], "unavailable": True}
    return result


###########
# Figures #
###########
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
