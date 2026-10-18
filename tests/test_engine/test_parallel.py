"""The checks of a level run together, and the run does not depend on how many."""

import os
import pickle
import threading
import time

import pytest

from dsbf.engine import (
    CheckDefinitionError,
    CheckFinished,
    CheckStarted,
    Limits,
    NoteRecorded,
)
from dsbf.engine.scheduler import MAX_DEFAULT_WORKERS, Scheduler, default_workers
from dsbf.model import CheckStatus
from tests.test_engine.toys import (
    declare,
    null_share,
    numeric_spread,
    quality_summary,
    run_plan,
)

pytestmark = pytest.mark.unit

TOYS = [null_share, numeric_spread, quality_summary]


def statuses(run):
    return {check_id: result.status for check_id, result in run.results.items()}


def sleeper(seconds):
    def sleep(ctx, params):
        time.sleep(seconds)

    return sleep


def meet(parties):
    """A check function that returns only if ``parties`` checks call it at once."""
    barrier = threading.Barrier(parties, timeout=2)

    def wait(ctx, params):
        barrier.wait()

    return wait


def comparable(run):
    """The run without what depends on the clock, which differs between runs."""
    data = run.model_dump(mode="json")
    for key in ("run_id", "started_at", "finished_at"):
        del data[key]
    for result in data["results"].values():
        del result["timing"]
    return data


class TestConcurrency:
    def test_independent_checks_in_a_level_run_at_the_same_time(self):
        together = meet(3)

        run = run_plan([declare(name, function=together) for name in "abc"], workers=3)

        assert set(statuses(run).values()) == {CheckStatus.OK}

    def test_one_worker_runs_them_one_at_a_time(self):
        together = meet(2)

        run = run_plan([declare(name, function=together) for name in "ab"], workers=1)

        assert set(statuses(run).values()) == {CheckStatus.ERROR}

    def test_no_more_run_at_once_than_there_are_workers(self):
        lock = threading.Lock()
        running = peak = 0

        def counted(ctx, params):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with lock:
                running -= 1

        checks = [declare(f"c{i}", function=counted) for i in range(6)]

        run = run_plan(checks, workers=2)

        assert set(statuses(run).values()) == {CheckStatus.OK}
        assert peak == 2

    def test_a_level_waits_for_the_one_before_it(self):
        finished = []

        def first(ctx, params):
            time.sleep(0.15)
            finished.append("first")

        def second(ctx, params):
            finished.append("second")

        run = run_plan(
            [
                declare("first", function=first),
                declare("second", function=second, after=["first"]),
                declare("alongside", function=sleeper(0.01)),
            ],
            workers=4,
        )

        assert finished == ["first", "second"]
        assert set(statuses(run).values()) == {CheckStatus.OK}

    def test_a_slow_check_does_not_hold_up_the_others(self):
        checks = [declare(f"s{i}", function=sleeper(0.3)) for i in range(4)]

        started = time.perf_counter()
        run_plan(checks, workers=4)
        parallel = time.perf_counter() - started
        started = time.perf_counter()
        run_plan(checks, workers=1)
        serial = time.perf_counter() - started

        assert serial >= 1.2
        assert parallel < 0.9


class TestDeterminism:
    @pytest.mark.parametrize("workers", [2, 8])
    def test_the_run_is_the_same_whatever_the_worker_count(self, workers):
        checks = [*TOYS, declare("extra", function=sleeper(0.02))]

        one = run_plan(checks, workers=1)
        many = run_plan(checks, workers=workers)

        assert comparable(many) == comparable(one)
        assert list(many.results) == list(one.results)

    def test_results_are_in_plan_order_even_when_a_later_check_ends_first(self):
        run = run_plan(
            [
                declare("a_slow", function=sleeper(0.2)),
                declare("b_quick"),
            ],
            workers=2,
        )

        assert list(run.results) == ["a_slow", "b_quick"]


class TestEvents:
    def test_every_check_has_one_started_and_one_finished_in_order(self):
        events = []
        checks = [declare(f"c{i}", function=sleeper(0.02)) for i in range(5)]

        run_plan(checks, hooks=[events.append], workers=5)

        for i in range(5):
            mine = [e.kind for e in events if e.check_id == f"c{i}"]
            assert mine == ["check_started", "check_finished"]
        assert len(events) == 10

    def test_notes_come_between_started_and_finished(self):
        def noisy(ctx, params):
            ctx.note("hello", "a note")

        events = []

        run_plan(
            [declare("noisy", function=noisy), declare("quiet")],
            hooks=[events.append],
            workers=2,
        )

        mine = [e.kind for e in events if e.check_id == "noisy"]
        assert mine == ["check_started", "note", "check_finished"]
        assert [e for e in events if isinstance(e, NoteRecorded)][
            0
        ].note.code == "hello"

    def test_hooks_are_called_only_from_the_calling_thread(self):
        threads = set()

        def hook(event):
            threads.add(threading.current_thread())

        run_plan(
            [declare(f"c{i}", function=sleeper(0.02)) for i in range(4)],
            hooks=[hook],
            workers=4,
        )

        assert threads == {threading.current_thread()}

    def test_positions_are_plan_positions_whatever_order_the_checks_end_in(self):
        events = []

        run_plan(
            [declare("a_slow", function=sleeper(0.2)), declare("b_quick")],
            hooks=[events.append],
            workers=2,
        )

        finished = [e for e in events if isinstance(e, CheckFinished)]
        assert [(e.check_id, e.position, e.total) for e in finished] == [
            ("b_quick", 2, 2),
            ("a_slow", 1, 2),
        ]
        started = [e for e in events if isinstance(e, CheckStarted)]
        assert [(e.check_id, e.position) for e in started] == [
            ("a_slow", 1),
            ("b_quick", 2),
        ]


class TestTimeoutsWhileParallel:
    def test_a_check_that_waits_for_a_free_worker_has_not_started_its_clock(self):
        limits = Limits(check_timeout_seconds=0.45, cancel_grace_seconds=0.02)
        checks = [declare(f"c{i}", function=sleeper(0.2)) for i in range(4)]

        run = run_plan(checks, limits=limits, workers=1)

        assert set(statuses(run).values()) == {CheckStatus.OK}

    def test_a_slow_check_times_out_while_its_neighbours_finish(self):
        limits = Limits(check_timeout_seconds=0.2, cancel_grace_seconds=0.02)

        started = time.perf_counter()
        run = run_plan(
            [
                declare("slow", function=sleeper(3)),
                declare("fine", function=sleeper(0.02)),
                declare("also_fine"),
            ],
            limits=limits,
            workers=3,
        )

        assert time.perf_counter() - started < 1.5
        assert statuses(run) == {
            "also_fine": CheckStatus.OK,
            "fine": CheckStatus.OK,
            "slow": CheckStatus.TIMEOUT,
        }

    def test_an_abandoned_thread_frees_its_worker(self):
        limits = Limits(check_timeout_seconds=0.1, cancel_grace_seconds=0.02)

        run = run_plan(
            [
                declare("a_stuck", function=sleeper(3)),
                declare("b_after"),
                declare("c_after"),
            ],
            limits=limits,
            workers=1,
        )

        assert statuses(run) == {
            "a_stuck": CheckStatus.TIMEOUT,
            "b_after": CheckStatus.OK,
            "c_after": CheckStatus.OK,
        }

    def test_the_budget_times_out_what_runs_and_skips_what_has_not_started(self):
        limits = Limits(run_budget_seconds=0.2, cancel_grace_seconds=0.02)
        checks = [declare(f"c{i}", function=sleeper(3)) for i in range(4)]

        run = run_plan(checks, limits=limits, workers=2)

        assert [r.status for r in run.results.values()] == [
            CheckStatus.TIMEOUT,
            CheckStatus.TIMEOUT,
            CheckStatus.SKIPPED,
            CheckStatus.SKIPPED,
        ]
        assert {r.notes[0].code for r in run.results.values()} == {
            "run_budget_exhausted"
        }

    def test_an_interrupt_cancels_every_running_check(self, monkeypatch):
        stopped = threading.Event()
        started = threading.Barrier(3, timeout=2)

        def polite(ctx, params):
            started.wait()
            try:
                while True:
                    ctx.checkpoint()
                    time.sleep(0.01)
            except BaseException:
                stopped.set()
                raise

        def interrupted_wait(self):
            started.wait()
            raise KeyboardInterrupt

        monkeypatch.setattr(Scheduler, "_wait", interrupted_wait)

        with pytest.raises(KeyboardInterrupt):
            run_plan(
                [declare("a", function=polite), declare("b", function=polite)],
                workers=2,
            )

        assert stopped.wait(1)


class TestTiming:
    def test_overlapping_checks_say_how_many_shared_the_figures(self):
        together = meet(2)

        run = run_plan([declare(name, function=together) for name in "ab"], workers=2)

        assert [r.timing.shared_with for r in run.results.values()] == [1, 1]

    def test_a_check_that_ran_alone_shares_with_nobody(self):
        run = run_plan(
            [
                declare("a", function=sleeper(0.02)),
                declare("b", function=sleeper(0.02)),
            ],
            workers=1,
        )

        assert [r.timing.shared_with for r in run.results.values()] == [0, 0]


class TestWorkers:
    def test_fewer_than_one_worker_is_refused_before_anything_runs(self):
        with pytest.raises(ValueError, match="at least 1"):
            run_plan([null_share], workers=0)

    def test_the_default_is_one_per_core_up_to_a_cap(self, monkeypatch):
        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: set(range(64)), raising=False
        )
        assert default_workers() == MAX_DEFAULT_WORKERS

        monkeypatch.setattr(
            os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False
        )
        assert default_workers() == 3

        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(), raising=False)
        assert default_workers() == 1

    def test_the_default_without_affinity_uses_the_cpu_count(self, monkeypatch):
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: None)

        assert default_workers() == 1

    def test_the_scheduler_refuses_a_non_positive_count(self):
        with pytest.raises(ValueError, match="at least 1"):
            Scheduler(workers=0, dataset=None, profile=None, grace=0, meter=None)  # type: ignore[arg-type]


class TestSpec:
    def test_a_check_is_pickled_by_reference(self):
        assert pickle.loads(pickle.dumps(null_share)) is null_share

    def test_a_check_that_is_not_what_its_module_names_cannot_be_pickled(self):
        with pytest.raises(pickle.PicklingError, match="top level of an importable"):
            pickle.dumps(declare("local"))

    def test_runs_in_is_thread_unless_declared(self):
        assert null_share.spec.runs_in == "thread"

    def test_an_unknown_place_to_run_is_refused(self):
        with pytest.raises(CheckDefinitionError, match="runs_in 'cloud'"):
            declare("somewhere", runs_in="cloud")
