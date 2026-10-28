"""Jobs are rows: their state, their numbered events and what a restart does."""

import sqlite3
from uuid import uuid4

import pytest

from dsbf.model import CheckStatus
from dsbf.store import (
    JobNotFoundError,
    JobRequest,
    JobState,
    SqliteRunRepository,
)

pytestmark = pytest.mark.unit

REQUEST = JobRequest(source="/data/a.csv", name="a")


def test_a_new_job_is_queued_and_keeps_what_was_asked(repo):
    job = repo.create_job(
        JobRequest(source="/data/a.csv", settings={"dataset": "a", "depth": "quick"})
    )

    assert job.state is JobState.QUEUED
    assert job.started_at is None
    assert job.request.settings.dataset == "a"
    assert repo.get_job(job.job_id) == job


def test_events_are_numbered_in_order_and_read_after_a_cursor(repo):
    job = repo.create_job(REQUEST)
    repo.start_job(job.job_id)
    repo.append_job_event(
        job.job_id, "check_started", check_id="x", position=1, total=2
    )
    repo.append_job_event(
        job.job_id, "check_finished", check_id="x", status=CheckStatus.OK
    )
    run_id = uuid4()
    done = repo.finish_job(job.job_id, JobState.SUCCEEDED, run_id=run_id)

    events = repo.job_events(job.job_id)
    assert [e.seq for e in events] == [1, 2, 3]
    assert events[-1].kind == "job_finished"
    assert events[-1].run_id == run_id
    assert events[-1].state is JobState.SUCCEEDED
    assert done.last_seq == 3
    assert done.finished_at is not None
    assert [e.seq for e in repo.job_events(job.job_id, after=1, limit=1)] == [2]


def test_jobs_are_listed_newest_first_and_filtered_by_state(repo):
    first = repo.create_job(REQUEST)
    second = repo.create_job(REQUEST)
    repo.finish_job(first.job_id, JobState.FAILED, error="no such file")

    everything = repo.list_jobs()
    failed = repo.list_jobs(state=JobState.FAILED)

    assert [j.job_id for j in everything.items] == [second.job_id, first.job_id]
    assert everything.total == 2
    assert [j.job_id for j in failed.items] == [first.job_id]
    assert failed.items[0].error == "no such file"


def test_an_unknown_job_is_not_found(repo):
    with pytest.raises(JobNotFoundError):
        repo.get_job(uuid4())
    with pytest.raises(JobNotFoundError):
        repo.job_events(uuid4())


def test_a_restart_ends_unfinished_jobs_as_interrupted(tmp_path):
    first = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    queued = first.create_job(REQUEST)
    running = first.create_job(REQUEST)
    first.start_job(running.job_id)
    done = first.create_job(REQUEST)
    first.finish_job(done.job_id, JobState.SUCCEEDED)
    first.close()

    second = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    try:
        assert second.interrupt_unfinished() == 2
        assert second.get_job(queued.job_id).state is JobState.INTERRUPTED
        assert second.get_job(running.job_id).state is JobState.INTERRUPTED
        assert second.get_job(done.job_id).state is JobState.SUCCEEDED
        assert second.job_events(running.job_id)[-1].state is JobState.INTERRUPTED
        assert second.interrupt_unfinished() == 0
    finally:
        second.close()


def test_a_job_outlives_the_deletion_of_its_run(repo, demo_run):
    repo.save(demo_run)
    job = repo.create_job(REQUEST)
    repo.finish_job(job.job_id, JobState.SUCCEEDED, run_id=demo_run.run_id)

    repo.delete(demo_run.run_id)

    assert repo.get_job(job.job_id).run_id == demo_run.run_id


def test_a_snapshot_cap_given_to_save_wins_over_the_repositorys(repo, demo_run):
    repo.save(demo_run, snapshot_rows=0)

    assert repo.snapshot(demo_run.run_id) is None


def test_the_job_tables_are_removed_by_the_downgrade(tmp_path):
    from alembic import command  # noqa: PLC0415
    from alembic.config import Config  # noqa: PLC0415

    from dsbf.store import sqlite as store_sqlite  # noqa: PLC0415

    repo = SqliteRunRepository(tmp_path / "dsbf.db", tmp_path / "artifacts")
    config = Config()
    config.set_main_option("script_location", str(store_sqlite._MIGRATIONS))  # noqa: SLF001
    with repo._engine.begin() as connection:  # noqa: SLF001
        config.attributes["connection"] = connection
        command.downgrade(config, "0002")
    repo.close()

    names = {
        row[0]
        for row in sqlite3.connect(tmp_path / "dsbf.db").execute(
            "select name from sqlite_master"
        )
    }
    assert not {"jobs", "job_events"} & names
    assert "artifacts" in names
