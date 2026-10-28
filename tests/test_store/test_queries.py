"""Listing, finding queries and history: results, order and pages."""

from uuid import uuid4

import pytest

from dsbf.model import Confidence, Severity
from dsbf.store import RunNotFoundError

pytestmark = pytest.mark.integration


def test_runs_list_newest_first_and_page(repo, demo_run, second_demo_run):
    repo.save(second_demo_run)
    repo.save(demo_run)
    newest, oldest = sorted(
        (demo_run, second_demo_run), key=lambda r: r.started_at, reverse=True
    )

    page = repo.list_runs()

    assert [s.run_id for s in page.items] == [newest.run_id, oldest.run_id]
    assert page.total == 2
    assert page.items[0].dataset_key == "demo"
    assert page.items[0].n_findings == len(newest.findings)
    assert page.items[0].max_severity == max(f.severity for f in newest.findings)
    second = repo.list_runs(limit=1, offset=1)
    assert [s.run_id for s in second.items] == [oldest.run_id]
    assert second.total == 2


def test_runs_filter_by_dataset(repo, demo_run):
    repo.save(demo_run)

    assert repo.list_runs(dataset_key="demo").total == 1
    assert repo.list_runs(dataset_key="other").total == 0


def test_a_run_without_a_key_is_stored_ungrouped(repo, demo_run):
    ungrouped = demo_run.model_copy(
        update={"dataset": demo_run.dataset.model_copy(update={"key": ""})}
    )
    repo.save(ungrouped)

    (summary,) = repo.list_runs().items
    assert summary.dataset_key is None
    assert repo.get(ungrouped.run_id).dataset.key == ""


def test_findings_come_in_severity_check_id_order(repo, demo_run):
    repo.save(demo_run)

    page = repo.find_findings(demo_run.run_id, limit=1000)

    assert [f.id for f in page.items] == [f.id for f in demo_run.findings]
    assert page.total == len(demo_run.findings)


def test_findings_page(repo, demo_run):
    repo.save(demo_run)
    everything = [f.id for f in demo_run.findings]
    assert len(everything) > 3

    first = repo.find_findings(demo_run.run_id, limit=3)
    rest = repo.find_findings(demo_run.run_id, limit=3, offset=3)

    assert [f.id for f in first.items] == everything[:3]
    assert [f.id for f in rest.items] == everything[3:6]
    assert first.total == rest.total == len(everything)


def test_findings_filter(repo, demo_run):
    repo.save(demo_run)
    sample = next(f for f in demo_run.findings if f.subject.columns)
    column = sample.subject.columns[0]

    by_column = repo.find_findings(demo_run.run_id, column=column, limit=1000)
    by_code = repo.find_findings(demo_run.run_id, code=sample.code, limit=1000)
    by_dimension = repo.find_findings(
        demo_run.run_id, dimension=sample.dimension, limit=1000
    )
    by_severity = repo.find_findings(
        demo_run.run_id, min_severity=Severity.WARN, limit=1000
    )

    assert {f.id for f in by_column.items} == {
        f.id for f in demo_run.findings if column in f.subject.columns
    }
    assert {f.id for f in by_code.items} == {
        f.id for f in demo_run.findings if f.code == sample.code
    }
    assert {f.id for f in by_dimension.items} == {
        f.id for f in demo_run.findings if f.dimension == sample.dimension
    }
    assert {f.id for f in by_severity.items} == {
        f.id for f in demo_run.findings if f.severity >= Severity.WARN
    }


def test_findings_filter_by_check_and_confidence(repo, demo_run):
    repo.save(demo_run)
    sample = demo_run.findings[0]

    by_check = repo.find_findings(demo_run.run_id, check=sample.check_id, limit=1000)
    by_confidence = repo.find_findings(
        demo_run.run_id, min_confidence=Confidence.HIGH, limit=1000
    )

    assert {f.id for f in by_check.items} == {
        f.id for f in demo_run.findings if f.check_id == sample.check_id
    }
    assert {f.id for f in by_confidence.items} == {
        f.id for f in demo_run.findings if f.confidence >= Confidence.HIGH
    }


@pytest.mark.parametrize("sort", ["check", "code"])
def test_findings_sort_by_check_or_code(repo, demo_run, sort):
    repo.save(demo_run)

    page = repo.find_findings(demo_run.run_id, sort=sort, limit=1000)

    keys = [
        (f.check_id,) if sort == "check" else (f.code, f.check_id) for f in page.items
    ]
    assert keys == sorted(keys)
    assert page.total == len(demo_run.findings)


def test_a_filter_with_no_match_is_an_empty_page(repo, demo_run):
    repo.save(demo_run)

    page = repo.find_findings(demo_run.run_id, code="no_such_code")

    assert page.items == () and page.total == 0


def test_history_follows_one_finding_across_runs_newest_first(
    repo, demo_run, second_demo_run
):
    repo.save(demo_run)
    repo.save(second_demo_run)
    finding = demo_run.findings[0]
    assert finding.id in {f.id for f in second_demo_run.findings}

    page = repo.finding_history("demo", finding_id=finding.id)

    assert page.total == 2
    started = [o.started_at for o in page.items]
    assert started == sorted(started, reverse=True)
    assert {o.run_id for o in page.items} == {demo_run.run_id, second_demo_run.run_id}
    assert all(o.finding.id == finding.id for o in page.items)


def test_history_by_code_and_scoped_to_the_dataset(repo, demo_run):
    repo.save(demo_run)
    code = demo_run.findings[0].code

    page = repo.finding_history("demo", code=code)

    assert page.total == sum(1 for f in demo_run.findings if f.code == code)
    assert repo.finding_history("other", code=code).total == 0
    with pytest.raises(ValueError, match="id or a code"):
        repo.finding_history("demo")


def test_reading_a_missing_result_or_finding_names_it(repo, demo_run):
    repo.save(demo_run)

    with pytest.raises(RunNotFoundError, match="nope"):
        repo.get_result(demo_run.run_id, "nope")
    with pytest.raises(RunNotFoundError, match="nope"):
        repo.get_finding(demo_run.run_id, "nope")
    with pytest.raises(RunNotFoundError):
        repo.get_finding(uuid4(), "nope")


def test_datasets_header_counts_and_column_profile(repo, demo_run):
    repo.save(demo_run)
    page = repo.list_datasets()
    assert page.total == 1
    entry = page.items[0]
    assert (entry.key, entry.n_runs, entry.last_run_id) == (
        demo_run.dataset.key,
        1,
        demo_run.run_id,
    )

    header = repo.get_header(demo_run.run_id)
    assert header.health == demo_run.health
    assert header.dataset == demo_run.dataset

    counts = repo.finding_counts(demo_run.run_id)
    assert sum(c.count for c in counts) == len(demo_run.findings)

    column = demo_run.profile.columns[0]
    assert repo.get_column_profile(demo_run.run_id, column.name) == column
    with pytest.raises(RunNotFoundError, match="no column"):
        repo.get_column_profile(demo_run.run_id, "absent")
