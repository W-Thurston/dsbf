# dsbf/service/jobs.py
"""Profiling jobs: queued in the store, run one at a time on a worker thread."""

import logging
import queue
import threading
from uuid import UUID

from dsbf.config import ConfigError
from dsbf.data import SamplingError, SourceError
from dsbf.engine import (
    CheckFinished,
    CheckStarted,
    Event,
    NoteRecorded,
    PlanError,
    RunCancelled,
)
from dsbf.library import profile_with
from dsbf.sinks import RunReadError
from dsbf.store import Job, JobRequest, JobState, RunRepository, StoreError

logger = logging.getLogger(__name__)

_STOP = None
_JOIN_SECONDS = 30.0
# What `dsbf check` reports as "error: <message>" rather than a traceback.
_SETUP_ERRORS = (
    PlanError,
    SourceError,
    ConfigError,
    SamplingError,
    RunReadError,
    StoreError,
)


class JobFinishedError(Exception):
    """The job is in a final state, so it cannot be cancelled."""


def failure_message(error: Exception) -> str:
    """The text of a failed job: what `dsbf check` prints for the same error."""
    if isinstance(error, PlanError):
        return "\n".join(error.problems)
    if isinstance(error, _SETUP_ERRORS):
        return str(error)
    return f"{type(error).__name__}: {error}"


class JobRunner:
    """Run the jobs of a repository one at a time, in the order they were queued.

    One worker thread takes a job, profiles its source with the settings it asked
    for and stores the run; a check still runs on its own thread or in its own
    process, as it does for ``dsbf check``. Starting the runner ends the jobs a
    stopped service left unfinished as ``interrupted``.
    """

    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository
        self._queue: queue.Queue[UUID | None] = queue.Queue()
        self._lock = threading.Lock()
        self._current: tuple[UUID, threading.Event] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._repository.interrupt_unfinished()
        self._thread = threading.Thread(
            target=self._work, name="dsbf-jobs", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Cancel the running job and wait for the worker to end it."""
        with self._lock:
            if self._current is not None:
                self._current[1].set()
        self._queue.put(_STOP)
        if self._thread is not None:
            self._thread.join(_JOIN_SECONDS)

    def submit(self, request: JobRequest) -> Job:
        job = self._repository.create_job(request)
        self._queue.put(job.job_id)
        return job

    def cancel(self, job_id: UUID) -> Job:
        """Cancel a queued job now, or ask the running one to stop before its next
        check (it is ``running`` until it does).

        Raises:
            JobNotFoundError: no such job.
            JobFinishedError: the job already ended.
        """
        with self._lock:
            job = self._repository.get_job(job_id)
            if job.state.final:
                raise JobFinishedError(f"job {job_id} already ended: {job.state.value}")
            if job.state is JobState.QUEUED:
                return self._repository.finish_job(
                    job_id, JobState.CANCELLED, error="cancelled before it started"
                )
            if self._current is not None and self._current[0] == job_id:
                self._current[1].set()
        return self._repository.get_job(job_id)

    # -- the worker ------------------------------------------------------------------

    def _work(self) -> None:
        while (job_id := self._queue.get()) is not _STOP:
            with self._lock:
                if self._repository.get_job(job_id).state is not JobState.QUEUED:
                    continue  # cancelled while it waited
                job = self._repository.start_job(job_id)
                cancel = threading.Event()
                self._current = (job_id, cancel)
            try:
                self._run(job, cancel)
            except Exception:
                logger.exception("job %s could not be recorded", job_id)
            finally:
                with self._lock:
                    self._current = None

    def _run(self, job: Job, cancel: threading.Event) -> None:
        request = job.request
        settings = request.settings

        def record(event: Event) -> None:
            self._record(job.job_id, event)

        try:
            run = profile_with(
                request.source,
                settings,
                name=request.name,
                hooks=[record],
                cancel=cancel,
            )
            if cancel.is_set():
                raise RunCancelled
            self._repository.save(run, snapshot_rows=settings.store.snapshot_rows)
        except RunCancelled:
            self._repository.finish_job(
                job.job_id, JobState.CANCELLED, error="cancelled while it ran"
            )
        except Exception as error:
            if not isinstance(error, _SETUP_ERRORS):
                logger.exception("job %s failed", job.job_id)
            self._repository.finish_job(
                job.job_id, JobState.FAILED, error=failure_message(error)
            )
        else:
            self._repository.finish_job(
                job.job_id, JobState.SUCCEEDED, run_id=run.run_id
            )

    def _record(self, job_id: UUID, event: Event) -> None:
        match event:
            case CheckStarted():
                self._repository.append_job_event(
                    job_id,
                    event.kind,
                    check_id=event.check_id,
                    position=event.position,
                    total=event.total,
                )
            case NoteRecorded():
                self._repository.append_job_event(
                    job_id,
                    event.kind,
                    check_id=event.check_id,
                    position=event.position,
                    total=event.total,
                    code=event.note.code,
                    message=event.note.message,
                )
            case CheckFinished():
                self._repository.append_job_event(
                    job_id,
                    event.kind,
                    check_id=event.check_id,
                    position=event.position,
                    total=event.total,
                    status=event.result.status,
                )
