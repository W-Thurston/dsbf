# dsbf/service/settings_routes.py
"""The settings endpoints of API v2: a dataset's saved settings and its rerun."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from dsbf.config import ProfileConfig
from dsbf.service.jobs import JobRunner
from dsbf.service.problems import Problem, ValidationProblem
from dsbf.service.routes import Repo
from dsbf.store import Job, JobRequest

settings_router = APIRouter(
    prefix="/v2/datasets",
    tags=["datasets"],
    responses={
        404: {"model": Problem, "description": "Not Found"},
        422: {"model": ValidationProblem, "description": "Unprocessable Content"},
    },
)


def _require_runs(repo: Repo, key: str) -> None:
    if not repo.list_runs(dataset_key=key, limit=1).total:
        raise HTTPException(404, f"no stored runs for dataset {key!r}")


def _validated(overlay: dict[str, Any]) -> dict[str, Any]:
    """The overlay when `ProfileConfig` accepts it, reduced to the fields it sets."""
    if "dataset" in overlay:
        raise HTTPException(
            422, "`dataset` is the key the settings belong to, not a setting"
        )
    try:
        config = ProfileConfig.model_validate(overlay)
    except ValidationError as error:
        raise HTTPException(422, str(error)) from error
    return config.model_dump(mode="json", exclude_unset=True)


@settings_router.get("/{key:path}/settings", operation_id="getDatasetSettings")
def get_settings(key: str, repo: Repo) -> dict[str, Any]:
    """The settings saved for a dataset: the fields it overrides, `{}` when it has
    none. The key is the dataset key, percent-encoded."""
    _require_runs(repo, key)
    return repo.get_settings_overlay(key) or {}


@settings_router.put("/{key:path}/settings", operation_id="putDatasetSettings")
def put_settings(key: str, overlay: dict[str, Any], repo: Repo) -> dict[str, Any]:
    """Save a dataset's settings, replacing any earlier ones. The body holds
    `ProfileConfig` fields; only those it sets are kept."""
    _require_runs(repo, key)
    saved = _validated(overlay)
    repo.set_settings_overlay(key, saved)
    return saved


@settings_router.delete(
    "/{key:path}/settings", operation_id="deleteDatasetSettings", status_code=204
)
def delete_settings(key: str, repo: Repo) -> None:
    """Forget a dataset's saved settings."""
    _require_runs(repo, key)
    repo.delete_settings_overlay(key)


@settings_router.post("/{key:path}/rerun", operation_id="rerunDataset", status_code=202)
def rerun(key: str, repo: Repo, request: Request) -> Job:
    """Queue a profiling run of the dataset's latest source with its saved
    settings. The source is read again, so a file that has gone fails the job."""
    _require_runs(repo, key)
    latest = repo.get_header(repo.list_runs(dataset_key=key, limit=1).items[0].run_id)
    source = latest.dataset.source.location
    if not source:
        raise HTTPException(
            422, "the latest run has no source location, so there is nothing to reread"
        )
    settings = ProfileConfig.model_validate(
        {**(repo.get_settings_overlay(key) or {}), "dataset": key}
    )
    runner: JobRunner = request.app.state.runner
    return runner.submit(
        JobRequest(source=source, name=latest.dataset.name, settings=settings)
    )
