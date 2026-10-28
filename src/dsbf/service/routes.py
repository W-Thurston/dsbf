# dsbf/service/routes.py
"""The read endpoints of API v2."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from dsbf.comparison import DEFAULT_TOLERANCE, compare
from dsbf.model import CheckResult, ColumnProfile, Dimension, Finding, Severity
from dsbf.model.diff import RunDiff
from dsbf.service.problems import Problem, ValidationProblem
from dsbf.service.views import (
    ArtifactRows,
    ArtifactValue,
    RunOverview,
    json_safe,
    table_rows,
)
from dsbf.store import (
    SNAPSHOT_REF,
    DatasetSummary,
    FindingOccurrence,
    Page,
    RunRepository,
    RunSummary,
)

MAX_LIMIT = 500
Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT, description="Items per page.")]
Offset = Annotated[int, Query(ge=0, description="Items to skip.")]

router = APIRouter(
    prefix="/v2",
    # Both phrases are fixed here: the standard library renamed 422's between Python
    # versions, which would change the exported document.
    responses={
        404: {"model": Problem, "description": "Not Found"},
        422: {"model": ValidationProblem, "description": "Unprocessable Content"},
    },
)


def repository(request: Request) -> RunRepository:
    return request.app.state.repository


Repo = Annotated[RunRepository, Depends(repository)]


@router.get("/datasets", operation_id="listDatasets", tags=["datasets"])
def list_datasets(
    repo: Repo, limit: Limit = 50, offset: Offset = 0
) -> Page[DatasetSummary]:
    """Datasets with stored runs, the most recently run first."""
    return repo.list_datasets(limit=limit, offset=offset)


@router.get(
    "/datasets/{key:path}/runs", operation_id="listDatasetRuns", tags=["datasets"]
)
def list_dataset_runs(
    key: str, repo: Repo, limit: Limit = 50, offset: Offset = 0
) -> Page[RunSummary]:
    """A dataset's runs, newest first. The key is the dataset key, percent-encoded."""
    page = repo.list_runs(dataset_key=key, limit=limit, offset=offset)
    if page.total == 0:
        raise HTTPException(404, f"no stored runs for dataset {key!r}")
    return page


@router.get(
    "/datasets/{key:path}/history", operation_id="getFindingHistory", tags=["datasets"]
)
def finding_history(
    key: str,
    repo: Repo,
    finding_id: Annotated[str | None, Query(description="One finding's id.")] = None,
    code: Annotated[
        str | None, Query(description="Every finding with this code.")
    ] = None,
    limit: Limit = 100,
    offset: Offset = 0,
) -> Page[FindingOccurrence]:
    """Where a finding (or a code) appeared over a dataset's runs, newest first."""
    if finding_id is None and code is None:
        raise HTTPException(422, "give finding_id or code")
    page = repo.finding_history(
        key, finding_id=finding_id, code=code, limit=limit, offset=offset
    )
    if page.total == 0 and not repo.list_runs(dataset_key=key, limit=1).total:
        raise HTTPException(404, f"no stored runs for dataset {key!r}")
    return page


@router.get("/runs", operation_id="listRuns", tags=["runs"])
def list_runs(repo: Repo, limit: Limit = 50, offset: Offset = 0) -> Page[RunSummary]:
    """Every stored run, newest first."""
    return repo.list_runs(limit=limit, offset=offset)


@router.get("/runs/compare", operation_id="compareRuns", tags=["runs"])
def compare_runs(
    repo: Repo,
    base: Annotated[UUID, Query(description="The run to compare against.")],
    head: Annotated[UUID, Query(description="The run to describe.")],
    tolerance: Annotated[
        float, Query(ge=0, le=1_000_000, description="Relative change that counts.")
    ] = DEFAULT_TOLERANCE,
) -> RunDiff:
    """What the head run shows that the base run did not. Any two runs."""
    return compare(repo.get(base), repo.get(head), tolerance=tolerance)


@router.get("/runs/{run_id}", operation_id="getRun", tags=["runs"])
def get_run(run_id: UUID, repo: Repo) -> RunOverview:
    """A run's identity, health, policy and finding counts, without its results."""
    counts = repo.finding_counts(run_id)
    return RunOverview(
        header=repo.get_header(run_id),
        finding_counts=counts,
        n_findings=sum(c.count for c in counts),
        artifacts=repo.artifacts(run_id),
    )


@router.get("/runs/{run_id}/findings", operation_id="listFindings", tags=["runs"])
def list_findings(
    run_id: UUID,
    repo: Repo,
    dimension: Dimension | None = None,
    min_severity: Severity | None = None,
    code: str | None = None,
    column: Annotated[
        str | None, Query(description="Findings about this column.")
    ] = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> Page[Finding]:
    """A run's findings, highest severity first."""
    repo.get_header(run_id)
    return repo.find_findings(
        run_id,
        dimension=dimension,
        min_severity=min_severity,
        code=code,
        column=column,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/runs/{run_id}/findings/{finding_id}", operation_id="getFinding", tags=["runs"]
)
def get_finding(run_id: UUID, finding_id: str, repo: Repo) -> Finding:
    """One finding."""
    return repo.get_finding(run_id, finding_id)


@router.get(
    "/runs/{run_id}/profile/{column:path}",
    operation_id="getColumnProfile",
    tags=["runs"],
)
def get_column_profile(run_id: UUID, column: str, repo: Repo) -> ColumnProfile:
    """One column's profile. The name is percent-encoded."""
    return repo.get_column_profile(run_id, column)


@router.get("/runs/{run_id}/profile", operation_id="findColumnProfile", tags=["runs"])
def find_column_profile(
    run_id: UUID,
    repo: Repo,
    column: Annotated[str, Query(description="The column's name.")],
) -> ColumnProfile:
    """One column's profile by query parameter, for the names a path cannot carry:
    clients resolve a path segment of ``.`` or ``..`` before sending it."""
    return repo.get_column_profile(run_id, column)


@router.get(
    "/runs/{run_id}/checks/{check_id}", operation_id="getCheckResult", tags=["runs"]
)
def get_check_result(run_id: UUID, check_id: str, repo: Repo) -> CheckResult:
    """One check's result; its artifacts are refs, fetched on demand."""
    return repo.get_result(run_id, check_id)


@router.get("/runs/{run_id}/snapshot", operation_id="getSnapshot", tags=["runs"])
def get_snapshot(
    run_id: UUID, repo: Repo, limit: Limit = 100, offset: Offset = 0
) -> ArtifactRows:
    """Rows of the data as profiled; 404 when the run kept no snapshot."""
    stored = {a.ref: a for a in repo.artifacts(run_id)}
    if SNAPSHOT_REF not in stored:
        raise HTTPException(404, f"run {run_id} kept no data snapshot")
    result = _artifact(run_id, SNAPSHOT_REF, repo, limit, offset)
    assert isinstance(result, ArtifactRows)  # the snapshot is a table
    return result


@router.get(
    "/runs/{run_id}/artifacts/{ref:path}", operation_id="getArtifact", tags=["runs"]
)
def get_artifact(
    run_id: UUID, ref: str, repo: Repo, limit: Limit = 100, offset: Offset = 0
) -> ArtifactRows | ArtifactValue:
    """A check's evidence: a page of rows for a table, the value for JSON."""
    return _artifact(run_id, ref, repo, limit, offset)


def _artifact(
    run_id: UUID, ref: str, repo: RunRepository, limit: int, offset: int
) -> ArtifactRows | ArtifactValue:
    stored = {a.ref: a for a in repo.artifacts(run_id)}
    if ref not in stored:
        raise HTTPException(404, f"run {run_id} has no stored artifact {ref!r}")
    artifact = stored[ref]
    if artifact.rows is None:
        return ArtifactValue(ref=ref, value=json_safe(repo.read_artifact(run_id, ref)))
    table = repo.read_artifact_rows(run_id, ref, offset, offset + limit)
    return ArtifactRows(
        ref=ref,
        columns=tuple(table.column_names),
        rows=table_rows(table),
        total=artifact.rows,
        offset=offset,
        limit=limit,
        sampling=artifact.sampling,
    )
