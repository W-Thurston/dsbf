"""Garbage collection keeps the runs the rules keep and leaves no orphaned files."""

import uuid
from datetime import timedelta

import pytest

from dsbf.model import Run

pytestmark = pytest.mark.integration


def aged(run: Run, days: float, key: str | None = "demo") -> Run:
    moment = run.started_at - timedelta(days=days)
    return run.model_copy(
        update={
            "run_id": uuid.uuid4(),
            "started_at": moment,
            "finished_at": moment,
            "dataset": run.dataset.model_copy(update={"key": key}),
        }
    )


@pytest.fixture
def stored(repo, demo_run):
    runs = [aged(demo_run, days) for days in (0, 10, 20, 30)]
    for run in runs:
        repo.save(run)
    return runs


def remaining(repo):
    return {item.run_id for item in repo.list_runs().items}


def test_keep_retains_the_newest_runs_of_each_dataset(repo, demo_run, stored):
    other = aged(demo_run, 50, key="other")
    repo.save(other)
    result = repo.gc(keep=2)
    assert set(result.runs) == {stored[2].run_id, stored[3].run_id}
    assert remaining(repo) == {stored[0].run_id, stored[1].run_id, other.run_id}


def test_a_run_younger_than_the_age_is_kept_beside_the_newest(repo, stored):
    result = repo.gc(keep=1, older_than=timedelta(days=15))
    assert set(result.runs) == {stored[2].run_id, stored[3].run_id}


def test_files_go_with_their_runs_and_orphans_are_removed(repo, stored, tmp_path):
    orphan = tmp_path / "artifacts" / str(uuid.uuid4())
    orphan.mkdir(parents=True)
    (orphan / "x.bin").write_bytes(b"x")
    result = repo.gc(keep=1)
    assert orphan.name in result.orphans
    assert not orphan.exists()
    on_disk = {p.name for p in (tmp_path / "artifacts").iterdir()}
    assert on_disk <= {str(stored[0].run_id)}


def test_a_dry_run_changes_nothing(repo, stored, tmp_path):
    result = repo.gc(keep=1, dry_run=True)
    assert len(result.runs) == 3
    assert len(remaining(repo)) == 4


def test_without_a_rule_only_orphans_go(repo, stored):
    assert repo.gc().runs == ()
    assert len(remaining(repo)) == 4


def test_a_negative_keep_is_refused(repo):
    with pytest.raises(ValueError, match="keep"):
        repo.gc(keep=-1)
