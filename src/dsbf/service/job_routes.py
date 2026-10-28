# dsbf/service/job_routes.py
"""The job endpoints of API v2: start a profiling run, follow it, cancel it."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from dsbf.service.jobs import JobFinishedError, JobRunner
from dsbf.service.problems import Problem
from dsbf.service.routes import Limit, Offset, Repo
from dsbf.service.views import JobProgress
from dsbf.store import Job, JobRequest, JobState, Page

EVENTS_LIMIT = 500

job_router = APIRouter(
    prefix="/v2/jobs",
    tags=["jobs"],
    responses={404: {"model": Problem, "description": "Not Found"}},
)


def runner(request: Request) -> JobRunner:
    return request.app.state.runner


@job_router.post("", operation_id="startJob", status_code=202)
def start_job(body: JobRequest, request: Request) -> Job:
    """Queue a profiling run of a local file. It runs after the jobs ahead of it,
    and its run is stored when it ends; follow it with `getJob`."""
    return runner(request).submit(body)


@job_router.get("", operation_id="listJobs")
def list_jobs(
    repo: Repo,
    state: Annotated[JobState | None, Query(description="Only jobs in it.")] = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> Page[Job]:
    """Jobs, the most recently requested first."""
    return repo.list_jobs(state=state, limit=limit, offset=offset)


@job_router.get("/{job_id}", operation_id="getJob")
def get_job(
    job_id: UUID,
    repo: Repo,
    after: Annotated[int, Query(ge=0, description="The last event already seen.")] = 0,
    limit: Annotated[
        int, Query(ge=1, le=EVENTS_LIMIT, description="Most events to return.")
    ] = EVENTS_LIMIT,
) -> JobProgress:
    """A job's state and its progress events numbered after `after`. The events end
    with a `job_finished` event that names the run, when there is one."""
    job = repo.get_job(job_id)
    events = repo.job_events(job_id, after=after, limit=limit)
    return JobProgress(
        job=job, events=events, next_after=events[-1].seq if events else after
    )


@job_router.post(
    "/{job_id}/cancel",
    operation_id="cancelJob",
    responses={409: {"model": Problem, "description": "Conflict"}},
)
def cancel_job(job_id: UUID, request: Request) -> Job:
    """Cancel a job. A queued job is cancelled at once; a running one stops before
    its next check, and is `running` until it does."""
    try:
        return runner(request).cancel(job_id)
    except JobFinishedError as error:
        raise HTTPException(409, str(error)) from error
