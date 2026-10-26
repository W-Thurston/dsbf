"""A check that asks for a worker process runs in one, and can be killed there."""

import datetime as dt
import glob
import os
import tempfile
import time

import polars as pl
import psutil
import pytest

from dsbf.engine import (
    CheckOverride,
    Limits,
    Registry,
    Requires,
    Selection,
    SemanticTypes,
    check,
    execute,
    make_plan,
)
from dsbf.engine.process import WorkerHost
from dsbf.engine.scheduler import Scheduler
from dsbf.model import CheckStatus, Dimension
from dsbf.profiling import profile_dataset
from tests.test_engine import process_toys as toys
from tests.test_engine.test_parallel import comparable
from tests.test_engine.toys import (
    declare,
    make_dataset,
    null_share,
    numeric_spread,
    run_plan,
)

pytestmark = pytest.mark.integration

KILL = Limits(check_timeout_seconds=1.5, cancel_grace_seconds=0.05)


@pytest.fixture(scope="module", autouse=True)
def warm_worker_server():
    """Start the forkserver once, so a test's timeout doesn't include its start-up."""
    run_plan([toys.py_rows], workers=1)


def run_with(checks, params=None, dataset=None, **options):
    dataset = dataset or make_dataset()
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(checks), Selection(entry_points=False), params=params, profile=profile
    )
    return execute(plan, dataset, profile, dsbf_version="0.0.0", **options)


def shared_frames():
    return glob.glob(os.path.join(tempfile.gettempdir(), "dsbf-frame-*"))


class TestRunsInAProcess:
    def test_its_warnings_come_back_as_one_note(self, recwarn, capfd):
        run = run_with([toys.py_warns])

        result = run.results["py_warns"]
        assert result.status is CheckStatus.OK
        (note,) = result.notes
        assert note.code == "python_warning"
        assert note.message == "RuntimeWarning: constant input in a worker"
        assert note.count == 2
        assert len(recwarn) == 0
        assert "constant input" not in capfd.readouterr().err

    def test_the_check_reads_the_data_in_another_process(self):
        dataset = make_dataset()
        expected = sum(float(v) for v in dataset.values("amount")) + sum(
            float(v) for v in dataset.values("single")
        )

        run = run_with([toys.py_total, toys.py_pid], dataset=dataset)

        result = run.results["py_total"]
        assert result.status is CheckStatus.OK
        (finding,) = result.findings
        assert finding.evidence.metrics["total"] == pytest.approx(expected)
        (where,) = run.results["py_pid"].findings
        assert where.evidence.metrics["pid"] != os.getpid()

    def test_the_worker_sees_the_dataset_as_sampled(self):
        dataset = make_dataset(rows=50, strategy="head")

        run = run_with([toys.py_rows], dataset=dataset)

        (finding,) = run.results["py_rows"].findings
        assert finding.evidence.metrics["rows"] == 50
        assert finding.evidence.sampled is not None
        assert finding.evidence.sampled.original_rows == 200

    def test_a_check_after_it_gets_the_result_it_ran_after(self):
        run = run_with([toys.py_total, toys.py_downstream])

        (total,) = run.results["py_total"].findings
        (seen,) = run.results["py_downstream"].findings
        assert seen.evidence.metrics["total"] == total.evidence.metrics["total"] != -1.0

    def test_the_shared_data_is_removed_when_the_run_ends(self):
        before = set(shared_frames())

        run_with([toys.py_rows])

        assert set(shared_frames()) == before

    @pytest.mark.parametrize("workers", [1, 4])
    def test_the_run_is_the_same_whatever_the_worker_count(self, workers):
        mixed = [null_share, numeric_spread, toys.py_total, toys.py_rows]

        reference = run_with(mixed, workers=2)
        run = run_with(mixed, workers=workers)

        assert comparable(run) == comparable(reference)

    def test_a_worker_check_and_a_thread_check_find_the_same(self):
        @check(
            id="thread_total",
            version="1.0.0",
            kind="analysis",
            applies_to=SemanticTypes("numeric"),
            dimensions=[Dimension.DISTRIBUTION],
        )
        def thread_total(ctx, params):
            toys.py_total.function(ctx, params)

        run = run_with([thread_total, toys.py_total])

        (on_thread,) = run.results["thread_total"].findings
        (in_worker,) = run.results["py_total"].findings
        assert (
            in_worker.evidence.metrics["total"] == on_thread.evidence.metrics["total"]
        )

    def test_several_workers_run_at_once(self):
        started = time.perf_counter()
        run = run_with([toys.py_busy, toys.py_rows, toys.py_total], workers=3)

        assert {r.status for r in run.results.values()} == {CheckStatus.OK}
        assert time.perf_counter() - started < 5


class TestSharedData:
    def test_a_worker_opens_the_same_table_the_caller_has(self):
        frame = pl.DataFrame(
            {
                "cat": pl.Series(["a", "b", None, "a"], dtype=pl.Categorical),
                "enum": pl.Series(["x", "y", "x", None], dtype=pl.Enum(["x", "y"])),
                "lst": [[1, 2], [], None, [3]],
                "ts": pl.Series(
                    [dt.datetime(2024, 1, 1, tzinfo=dt.UTC)] * 4,
                    dtype=pl.Datetime("us", "UTC"),
                ),
                "dec": pl.Series([1, 2, None, 4], dtype=pl.Decimal(10, 2)),
                "small": pl.Series([1, 2, 3, None], dtype=pl.Int8),
                "nested": [{"a": 1}, {"a": None}, None, {"a": 3}],
                "text": ["x", None, "y", "z"],
                "nothing": pl.Series([None] * 4, dtype=pl.Null),
            }
        )
        dataset = make_dataset(frame)

        with WorkerHost(dataset) as host:
            opened = host.share_frame().open()

        assert opened.frame.schema == frame.schema
        assert opened.frame.equals(frame, null_equal=True)
        assert opened.name == dataset.name and opened.source == dataset.source

    def test_a_column_with_an_empty_name_keeps_it(self):
        frame = pl.DataFrame({"": ["a", "b"], "x": [1, 2], " ": ["c", "d"]})

        with WorkerHost(make_dataset(frame)) as host:
            opened = host.share_frame().open()

        assert opened.frame.columns == ["", "x", " "]
        assert opened.frame.equals(frame)


class TestTimeoutKillsTheWorker:
    def test_a_sleeping_worker_is_killed_and_the_run_completes(self, tmp_path):
        pid_file = tmp_path / "pid"

        started = time.perf_counter()
        run = run_with(
            [toys.py_sleeper, toys.py_rows],
            params={"py_sleeper": {"path": str(pid_file)}},
            limits=Limits(check_timeout_seconds=0.8, cancel_grace_seconds=0.05),
        )

        assert time.perf_counter() - started < 5
        result = run.results["py_sleeper"]
        assert result.status is CheckStatus.TIMEOUT
        (note,) = result.notes
        assert note.code == "check_timeout"
        assert note.detail and "killed" in note.detail
        assert run.results["py_rows"].status is CheckStatus.OK
        assert not psutil.pid_exists(int(pid_file.read_text()))

    def test_the_budget_kills_a_worker_too(self, tmp_path):
        pid_file = tmp_path / "pid"

        run = run_with(
            [toys.py_sleeper],
            params={"py_sleeper": {"path": str(pid_file)}},
            limits=Limits(run_budget_seconds=0.8, cancel_grace_seconds=0.05),
        )

        result = run.results["py_sleeper"]
        assert result.status is CheckStatus.TIMEOUT
        assert result.notes[0].code == "run_budget_exhausted"
        assert "killed" in (result.notes[0].detail or "")
        assert not psutil.pid_exists(int(pid_file.read_text()))

    def test_an_interrupt_kills_the_workers_and_removes_the_shared_data(
        self, tmp_path, monkeypatch
    ):
        pid_file = tmp_path / "pid"
        before = set(shared_frames())

        def interrupted_wait(self):
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise KeyboardInterrupt

        monkeypatch.setattr(Scheduler, "_wait", interrupted_wait)

        with pytest.raises(KeyboardInterrupt):
            run_with([toys.py_sleeper], params={"py_sleeper": {"path": str(pid_file)}})

        deadline = time.monotonic() + 5
        pid = int(pid_file.read_text())
        while psutil.pid_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not psutil.pid_exists(pid)
        assert set(shared_frames()) == before


class TestFailuresInAWorker:
    def test_an_exception_comes_back_as_an_error_with_the_workers_traceback(self):
        run = run_with([toys.py_raises, toys.py_rows], limits=KILL)

        result = run.results["py_raises"]
        assert result.status is CheckStatus.ERROR
        assert result.findings == ()
        (note,) = result.notes
        assert note.code == "check_failed"
        assert note.message == "Raised ValueError: boom in worker"
        assert note.exception_type == "ValueError"
        assert note.detail and "py_raises" in note.detail
        assert run.results["py_rows"].status is CheckStatus.OK

    def test_a_worker_that_exits_without_a_result_is_an_error(self):
        run = run_with([toys.py_exits, toys.py_rows], limits=KILL)

        result = run.results["py_exits"]
        assert result.status is CheckStatus.ERROR
        (note,) = result.notes
        assert note.code == "worker_died"
        assert "exited with code 7" in note.message
        assert note.exception_type is None
        assert run.results["py_rows"].status is CheckStatus.OK

    def test_system_exit_in_a_worker_is_an_error_not_a_stop_of_the_run(self):
        run = run_with([toys.py_exits_cleanly, toys.py_rows], limits=KILL)

        result = run.results["py_exits_cleanly"]
        assert result.status is CheckStatus.ERROR
        assert result.notes[0].code == "worker_died"
        assert "code 0" in result.notes[0].message

    def test_a_failed_worker_blocks_its_hard_dependents_as_a_failed_thread_does(self):
        run = run_with(
            [toys.py_raises, declare("later", requires=Requires(checks=["py_raises"]))]
        )

        assert run.results["later"].status is CheckStatus.SKIPPED
        assert run.results["later"].notes[0].blocked_by == ("py_raises",)


class TestCheckThatCannotBeSent:
    def test_a_check_not_importable_by_name_runs_on_a_thread_with_a_note(self):
        local = declare("local_check", runs_in="process")

        run = run_with([local])

        result = run.results["local_check"]
        assert result.status is CheckStatus.OK
        (note,) = result.notes
        assert note.code == "ran_in_thread"
        assert "top level of an importable module" in (note.hint or "")
        assert "local_check" in note.message

    def test_a_dataset_a_file_cannot_hold_runs_on_a_thread_too(self, monkeypatch):
        import polars as pl

        def refuse(self, *args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(pl.DataFrame, "write_ipc", refuse)

        run = run_with([toys.py_rows, toys.py_busy])

        assert {r.status for r in run.results.values()} == {CheckStatus.OK}
        for result in run.results.values():
            assert result.notes[0].code == "ran_in_thread"
            assert "disk full" in result.notes[0].message


class TestAccounting:
    def test_a_worker_reports_its_own_cpu_and_memory(self):
        run = run_with([toys.py_busy])

        timing = run.results["py_busy"].timing
        assert timing.shared_with == 0
        assert timing.cpu_seconds is not None and 0.3 <= timing.cpu_seconds < 1.5
        assert timing.peak_memory_bytes and timing.peak_memory_bytes > 20 * 1024 * 1024
        assert timing.wall_seconds >= 0.3

    def test_a_worker_is_not_counted_as_sharing_with_threads(self):
        run = run_with(
            [
                toys.py_busy,
                declare("on_thread", function=lambda ctx, params: time.sleep(0.2)),
            ],
            workers=2,
        )

        assert run.results["py_busy"].timing.shared_with == 0
        assert run.results["on_thread"].timing.shared_with == 0

    def test_a_killed_worker_has_no_cpu_figure(self):
        run = run_with(
            [toys.py_sleeper],
            limits=Limits(check_timeout_seconds=0.6, cancel_grace_seconds=0.05),
        )

        timing = run.results["py_sleeper"].timing
        assert timing.cpu_seconds is None
        assert timing.wall_seconds >= 0.6


class TestRunsInOverride:
    def test_a_check_that_asks_for_a_process_can_be_put_on_a_thread(self):
        run = run_with(
            [toys.py_pid], check_overrides={"py_pid": CheckOverride(runs_in="thread")}
        )

        (where,) = run.results["py_pid"].findings
        assert where.evidence.metrics["pid"] == os.getpid()

    def test_a_thread_check_can_be_put_in_a_process(self):
        run = run_with(
            [toys.py_total, null_share],
            check_overrides={"null_share": CheckOverride(runs_in="process")},
        )

        assert run.results["null_share"].status is CheckStatus.OK
        assert "ran_in_thread" not in {
            note.code for note in run.results["null_share"].notes
        }
