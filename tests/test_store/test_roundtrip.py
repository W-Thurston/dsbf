"""A stored run comes back equal, with its artifacts and its findings."""

from pathlib import Path

import pytest

import dsbf
from dsbf.model import Run
from dsbf.store import RunExistsError, RunNotFoundError
from tests.fixtures.adversarial import build_adversarial_frame
from tests.golden.datasets import KITCHEN5K

pytestmark = pytest.mark.integration

TITANIC = Path(__file__).parents[1] / "golden" / "data" / "titanic.csv"


def assert_stored_equal(repo, run: Run) -> Run:
    repo.save(run)
    back = repo.get(run.run_id)
    assert back == run
    assert [f.id for f in back.findings] == [f.id for f in run.findings]
    for artifact in run.artifacts:
        left, right = run.artifact(artifact.ref), back.artifact(artifact.ref)
        if hasattr(left, "equals"):
            assert left.equals(right)
        else:
            assert left == right
    return back


def kitchen_run(directory: Path) -> Run:
    return dsbf.profile(
        KITCHEN5K.write_csv(directory), depth="deep", time_index="normal_date"
    )


def test_a_small_run_round_trips(repo, demo_run):
    assert_stored_equal(repo, demo_run)


def test_titanic_round_trips(repo):
    assert_stored_equal(repo, dsbf.profile(TITANIC))


def test_the_adversarial_frame_round_trips(repo):
    run = dsbf.profile(build_adversarial_frame(), depth="deep")

    assert_stored_equal(repo, run)


@pytest.mark.slow
def test_kitchen5k_round_trips_with_its_artifacts(repo, tmp_path):
    run = kitchen_run(tmp_path)
    assert run.artifacts

    assert_stored_equal(repo, run)


def test_saving_a_run_twice_is_refused_and_changes_nothing(repo, demo_run):
    repo.save(demo_run)

    with pytest.raises(RunExistsError):
        repo.save(demo_run)

    assert repo.get(demo_run.run_id) == demo_run
    assert repo.list_runs().total == 1


def test_an_unknown_run_is_not_found(repo, demo_run):
    with pytest.raises(RunNotFoundError):
        repo.get(demo_run.run_id)
    with pytest.raises(RunNotFoundError):
        repo.find_findings(demo_run.run_id)
    with pytest.raises(RunNotFoundError):
        repo.delete(demo_run.run_id)


def test_one_result_and_one_finding_are_read_without_the_run(repo, demo_run):
    repo.save(demo_run)
    finding = demo_run.findings[0]

    assert repo.get_finding(demo_run.run_id, finding.id) == finding
    result = demo_run.results[finding.check_id]
    assert repo.get_result(demo_run.run_id, finding.check_id) == result


@pytest.mark.slow
def test_delete_removes_the_run_and_its_files(repo, tmp_path):
    run = kitchen_run(tmp_path)
    repo.save(run)
    directory = tmp_path / "artifacts" / str(run.run_id)
    assert run.artifacts and directory.is_dir()

    repo.delete(run.run_id)

    assert not directory.exists()
    assert repo.list_runs().total == 0
    with pytest.raises(RunNotFoundError):
        repo.get(run.run_id)
