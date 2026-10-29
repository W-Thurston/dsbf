# dsbf/store/job_rows.py
"""Job rows of the SQLite repository: the jobs table and each job's events."""

import json
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa

from dsbf.model import CheckStatus
from dsbf.store import tables as t
from dsbf.store.jobs import Job, JobEvent, JobRequest, JobState
from dsbf.store.repository import JobNotFoundError, Page

_INTERRUPTED = "the service stopped before the job finished"


def _text(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _job(row: Any) -> Job:
    return Job(
        job_id=UUID(row.job_id),
        state=JobState(row.state),
        request=JobRequest.model_validate_json(row.request_json),
        requested_at=datetime.fromisoformat(row.requested_at),
        started_at=row.started_at and datetime.fromisoformat(row.started_at),
        finished_at=row.finished_at and datetime.fromisoformat(row.finished_at),
        run_id=UUID(row.run_id) if row.run_id else None,
        error=row.error,
        last_seq=row.last_seq,
    )


class JobRows:
    """Mixed into the repository, which owns ``_engine`` and ``_write_lock``."""

    _engine: sa.Engine
    _write_lock: threading.Lock

    def create_job(self, request: JobRequest) -> Job:
        job_id = str(uuid4())
        with self._write_lock, self._engine.begin() as connection:
            connection.execute(
                sa.insert(t.jobs).values(
                    job_id=job_id,
                    state=JobState.QUEUED.value,
                    requested_at=_text(datetime.now(UTC)),
                    last_seq=0,
                    request_json=request.model_dump_json(),
                )
            )
        return self.get_job(UUID(job_id))

    def get_job(self, job_id: UUID) -> Job:
        with self._engine.connect() as connection:
            row = connection.execute(
                sa.select(t.jobs).where(t.jobs.c.job_id == str(job_id))
            ).first()
        if row is None:
            raise JobNotFoundError(f"no job {job_id}")
        return _job(row)

    def list_jobs(
        self, *, state: JobState | None = None, limit: int = 50, offset: int = 0
    ) -> Page[Job]:
        where = [] if state is None else [t.jobs.c.state == state.value]
        with self._engine.connect() as connection:
            total = connection.execute(
                sa.select(sa.func.count()).select_from(t.jobs).where(*where)
            ).scalar_one()
            rows = connection.execute(
                sa.select(t.jobs)
                .where(*where)
                .order_by(t.jobs.c.requested_at.desc(), t.jobs.c.job_id)
                .limit(limit)
                .offset(offset)
            ).all()
        return Page[Job](
            items=tuple(_job(row) for row in rows),
            total=total,
            limit=limit,
            offset=offset,
        )

    def start_job(self, job_id: UUID) -> Job:
        self._update_job(
            job_id, state=JobState.RUNNING.value, started_at=_text(datetime.now(UTC))
        )
        return self.get_job(job_id)

    def append_job_event(
        self,
        job_id: UUID,
        kind: str,
        *,
        check_id: str | None = None,
        position: int | None = None,
        total: int | None = None,
        status: CheckStatus | None = None,
        code: str | None = None,
        message: str | None = None,
    ) -> JobEvent:
        with self._write_lock, self._engine.begin() as connection:
            return self._append(
                connection,
                job_id,
                kind=kind,
                check_id=check_id,
                position=position,
                total=total,
                status=status,
                code=code,
                message=message,
                state=None,
                run_id=None,
            )

    def finish_job(
        self,
        job_id: UUID,
        state: JobState,
        *,
        run_id: UUID | None = None,
        error: str | None = None,
    ) -> Job:
        """End a job in a final ``state``, and record the closing event with it."""
        with self._write_lock, self._engine.begin() as connection:
            self._append(
                connection,
                job_id,
                kind="job_finished",
                check_id=None,
                position=None,
                total=None,
                status=None,
                code=None,
                message=error,
                state=state,
                run_id=run_id,
            )
            connection.execute(
                sa.update(t.jobs)
                .where(t.jobs.c.job_id == str(job_id))
                .values(
                    state=state.value,
                    finished_at=_text(datetime.now(UTC)),
                    run_id=None if run_id is None else str(run_id),
                    error=error,
                )
            )
        return self.get_job(job_id)

    def job_events(
        self, job_id: UUID, *, after: int = 0, limit: int = 500
    ) -> tuple[JobEvent, ...]:
        with self._engine.connect() as connection:
            self._require_job(connection, job_id)
            rows = connection.execute(
                sa.select(t.job_events.c.event_json)
                .where(t.job_events.c.job_id == str(job_id), t.job_events.c.seq > after)
                .order_by(t.job_events.c.seq)
                .limit(limit)
            ).all()
        return tuple(JobEvent.model_validate_json(row.event_json) for row in rows)

    def interrupt_unfinished(self) -> int:
        """End every queued or running job as ``interrupted``; how many there were."""
        with self._engine.connect() as connection:
            ids = (
                connection.execute(
                    sa.select(t.jobs.c.job_id).where(
                        t.jobs.c.state.in_(
                            [JobState.QUEUED.value, JobState.RUNNING.value]
                        )
                    )
                )
                .scalars()
                .all()
            )
        for job_id in ids:
            self.finish_job(UUID(job_id), JobState.INTERRUPTED, error=_INTERRUPTED)
        return len(ids)

    # -- helpers ---------------------------------------------------------------------

    def _update_job(self, job_id: UUID, **values: Any) -> None:
        with self._write_lock, self._engine.begin() as connection:
            self._require_job(connection, job_id)
            connection.execute(
                sa.update(t.jobs).where(t.jobs.c.job_id == str(job_id)).values(**values)
            )

    @staticmethod
    def _require_job(connection: sa.Connection, job_id: UUID) -> None:
        found = connection.execute(
            sa.select(t.jobs.c.job_id).where(t.jobs.c.job_id == str(job_id))
        ).first()
        if found is None:
            raise JobNotFoundError(f"no job {job_id}")

    def _append(
        self, connection: sa.Connection, job_id: UUID, **fields: Any
    ) -> JobEvent:
        self._require_job(connection, job_id)
        seq = (
            connection.execute(
                sa.select(t.jobs.c.last_seq).where(t.jobs.c.job_id == str(job_id))
            ).scalar_one()
            + 1
        )
        event = JobEvent(seq=seq, at=datetime.now(UTC), **fields)
        connection.execute(
            sa.insert(t.job_events).values(
                job_id=str(job_id), seq=seq, event_json=event.model_dump_json()
            )
        )
        connection.execute(
            sa.update(t.jobs).where(t.jobs.c.job_id == str(job_id)).values(last_seq=seq)
        )
        return event


class SettingsRows:
    """Mixed into the repository, which owns ``_engine`` and ``_write_lock``."""

    _engine: sa.Engine
    _write_lock: threading.Lock

    def get_settings_overlay(self, dataset_key: str) -> dict[str, Any] | None:
        with self._engine.connect() as connection:
            row = connection.execute(
                sa.select(t.dataset_settings.c.overlay_json).where(
                    t.dataset_settings.c.dataset_key == dataset_key
                )
            ).first()
        return None if row is None else json.loads(row.overlay_json)

    def set_settings_overlay(self, dataset_key: str, overlay: dict[str, Any]) -> None:
        values = {
            "overlay_json": json.dumps(overlay),
            "updated_at": _text(datetime.now(UTC)),
        }
        with self._write_lock, self._engine.begin() as connection:
            updated = connection.execute(
                sa.update(t.dataset_settings)
                .where(t.dataset_settings.c.dataset_key == dataset_key)
                .values(**values)
            )
            if not updated.rowcount:
                connection.execute(
                    sa.insert(t.dataset_settings).values(
                        dataset_key=dataset_key, **values
                    )
                )

    def delete_settings_overlay(self, dataset_key: str) -> bool:
        with self._write_lock, self._engine.begin() as connection:
            deleted = connection.execute(
                sa.delete(t.dataset_settings).where(
                    t.dataset_settings.c.dataset_key == dataset_key
                )
            )
        return bool(deleted.rowcount)
