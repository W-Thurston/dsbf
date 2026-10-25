"""A check that outlasts its limit ends as ``timeout`` and the run still completes."""

import dataclasses
import threading
import time

import pytest

from dsbf.engine import CheckCancelled, Limits
from dsbf.engine import extras as extras_module
from dsbf.engine.scheduler import Scheduler
from dsbf.model import CheckStatus
from tests.test_engine.toys import (
    declare,
    null_share,
    numeric_spread,
    quality_summary,
    run_plan,
)

pytestmark = pytest.mark.unit

FAST = Limits(check_timeout_seconds=0.15, cancel_grace_seconds=0.05)


def statuses(run):
    return {check_id: result.status for check_id, result in run.results.items()}


def sleeper(seconds=2.0):
    def sleep(ctx, params):
        time.sleep(seconds)

    return sleep


class TestCheckTimeout:
    def test_a_sleeping_check_times_out_without_stalling_the_run(self):
        # `numeric_spread` runs in a worker process, whose start-up and data transfer
        # count against its limit, so the limit is well above `FAST`; the sleeper
        # outlasts the whole run, which must not wait for it.
        limits = Limits(check_timeout_seconds=1.0, cancel_grace_seconds=0.05)
        slow = declare("slow", function=sleeper(6.0))
        started = time.perf_counter()

        run = run_plan([null_share, slow, numeric_spread], limits=limits)

        assert time.perf_counter() - started < 4.0
        assert statuses(run) == {
            "null_share": CheckStatus.OK,
            "numeric_spread": CheckStatus.OK,
            "slow": CheckStatus.TIMEOUT,
        }
        assert run.health.counts.timeout == 1

    def test_the_note_names_the_limit_and_that_the_check_could_not_be_stopped(self):
        run = run_plan([declare("slow", function=sleeper())], limits=FAST)

        result = run.results["slow"]
        (note,) = result.notes
        assert note.code == "check_timeout"
        assert note.message == "Did not finish within 0.15s."
        assert note.hint and "exclude" in note.hint
        assert note.detail and "cannot be interrupted" in note.detail
        assert result.findings == () and result.output is None
        assert 0.15 <= result.timing.wall_seconds < 1.0
        assert "Did not finish within 0.15s." in run.render_health()

    def test_the_hard_dependents_of_a_timed_out_check_are_skipped(self):
        slow = dataclasses.replace(null_share, function=sleeper())

        run = run_plan([slow, numeric_spread, quality_summary], limits=FAST)

        assert statuses(run) == {
            "null_share": CheckStatus.TIMEOUT,
            "numeric_spread": CheckStatus.OK,
            "quality_summary": CheckStatus.SKIPPED,
        }
        note = run.results["quality_summary"].notes[0]
        assert note.blocked_by == ("null_share",)
        assert "Requires 'null_share' (timeout)" in note.message

    def test_a_check_inside_its_limit_is_untouched(self):
        run = run_plan(
            [null_share, numeric_spread, quality_summary],
            limits=Limits(check_timeout_seconds=30),
        )

        assert run.health.counts.ok == 3

    def test_without_a_limit_nothing_times_out(self):
        run = run_plan([declare("slowish", function=sleeper(0.3))])

        assert statuses(run) == {"slowish": CheckStatus.OK}


class TestCooperativeCancellation:
    def test_a_check_that_uses_its_context_stops_at_the_next_call(self):
        stopped = threading.Event()

        def polite(ctx, params):
            try:
                while True:
                    ctx.checkpoint()
                    time.sleep(0.01)
            except CheckCancelled:
                stopped.set()
                raise

        run = run_plan([declare("polite", function=polite)], limits=FAST)

        result = run.results["polite"]
        assert result.status is CheckStatus.TIMEOUT
        assert result.notes[0].detail == (
            "The check noticed the cancellation and stopped."
        )
        assert stopped.wait(1)

    @pytest.mark.parametrize(
        "call",
        [
            lambda ctx: ctx.note("n", "a note"),
            lambda ctx: ctx.result("anything"),
            lambda ctx: ctx.output(),
        ],
        ids=["note", "result", "output"],
    )
    def test_each_context_method_stops_a_cancelled_check(self, call):
        stopped = threading.Event()

        def uses_context(ctx, params):
            try:
                while True:
                    try:
                        call(ctx)
                    except Exception:
                        pass  # result() refuses an undeclared name until the cancel
                    time.sleep(0.01)
            except CheckCancelled:
                stopped.set()
                raise

        run = run_plan([declare("uses_context", function=uses_context)], limits=FAST)

        assert run.results["uses_context"].status is CheckStatus.TIMEOUT
        assert stopped.wait(1)

    def test_a_check_that_swallows_exceptions_is_still_cancelled(self):
        stopped = threading.Event()

        def stubborn(ctx, params):
            while not stopped.is_set():
                try:
                    ctx.checkpoint()
                    time.sleep(0.01)
                except Exception:  # noqa: S110 - cancellation must not be caught here
                    pass

        run = run_plan([declare("stubborn", function=stubborn)], limits=FAST)

        assert run.results["stubborn"].status is CheckStatus.TIMEOUT
        assert "stopped" in (run.results["stubborn"].notes[0].detail or "")
        stopped.set()

    def test_the_context_without_a_cancel_event_never_cancels(self):
        from dsbf.engine import CheckContext
        from dsbf.profiling import profile_dataset
        from tests.test_engine.toys import make_dataset

        dataset = make_dataset()
        context = CheckContext(null_share.spec, dataset, profile_dataset(dataset))

        context.checkpoint()


class TestRunBudget:
    def test_checks_after_the_budget_is_gone_are_skipped(self):
        slow = declare("a_slow", function=sleeper(0.4))
        later = declare("b_later")

        run = run_plan(
            [slow, later],
            limits=Limits(run_budget_seconds=0.1, cancel_grace_seconds=0.02),
            workers=1,
        )

        assert statuses(run) == {
            "a_slow": CheckStatus.TIMEOUT,
            "b_later": CheckStatus.SKIPPED,
        }
        timeout, skipped = (run.results[i].notes[0] for i in ("a_slow", "b_later"))
        assert timeout.code == skipped.code == "run_budget_exhausted"
        assert "while a_slow was running" in timeout.message
        assert "before this check started" in skipped.message
        assert skipped.blocked_by == ()
        assert skipped.hint and "budget" in skipped.hint

    def test_the_smaller_of_the_check_timeout_and_the_budget_applies(self):
        slow = declare("slow", function=sleeper())

        by_check = run_plan(
            [slow],
            limits=Limits(
                check_timeout_seconds=0.1,
                run_budget_seconds=30,
                cancel_grace_seconds=0.02,
            ),
        )
        by_budget = run_plan(
            [slow],
            limits=Limits(
                check_timeout_seconds=30,
                run_budget_seconds=0.1,
                cancel_grace_seconds=0.02,
            ),
        )

        assert by_check.results["slow"].notes[0].code == "check_timeout"
        assert by_budget.results["slow"].notes[0].code == "run_budget_exhausted"

    def test_a_budget_the_run_stays_within_changes_nothing(self):
        run = run_plan(
            [null_share, numeric_spread, quality_summary],
            limits=Limits(run_budget_seconds=60),
        )

        assert run.health.counts.ok == 3

    def test_a_check_that_cannot_run_keeps_its_own_status_over_budget(
        self, monkeypatch
    ):
        real = extras_module.find_spec
        monkeypatch.setattr(
            extras_module,
            "find_spec",
            lambda m: None if m == "statsmodels" else real(m),
        )
        slow = declare("a_slow", function=sleeper(0.4))
        missing = declare("b_missing", extras=["timeseries"])

        run = run_plan(
            [slow, missing],
            limits=Limits(run_budget_seconds=0.1, cancel_grace_seconds=0.02),
        )

        assert statuses(run) == {
            "a_slow": CheckStatus.TIMEOUT,
            "b_missing": CheckStatus.UNAVAILABLE,
        }


class TestOtherBaseExceptions:
    def test_system_exit_in_a_check_reaches_the_caller(self):
        def leaves(ctx, params):
            raise SystemExit(3)

        with pytest.raises(SystemExit):
            run_plan([declare("leaves", function=leaves)], limits=FAST)

    def test_a_keyboard_interrupt_while_waiting_cancels_the_check(self, monkeypatch):
        stopped = threading.Event()

        def polite(ctx, params):
            try:
                while True:
                    ctx.checkpoint()
                    time.sleep(0.01)
            except CheckCancelled:
                stopped.set()
                raise

        def interrupted_wait(self):
            time.sleep(0.05)
            raise KeyboardInterrupt

        monkeypatch.setattr(Scheduler, "_wait", interrupted_wait)

        with pytest.raises(KeyboardInterrupt):
            run_plan([declare("polite", function=polite)], limits=FAST)

        assert stopped.wait(1)
