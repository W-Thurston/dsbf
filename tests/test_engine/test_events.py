"""The executor reports every check to its hooks, in order, from the calling thread."""

import dataclasses
import logging
import threading

import pytest

from dsbf.engine import (
    CheckFinished,
    CheckStarted,
    Limits,
    NoteRecorded,
)
from dsbf.engine import extras as extras_module
from dsbf.model import CheckStatus
from tests.test_engine.toys import (
    declare,
    null_share,
    numeric_spread,
    quality_summary,
    run_plan,
)

pytestmark = pytest.mark.unit

TOYS = (null_share, numeric_spread, quality_summary)


def explode(ctx, params):
    raise RuntimeError("boom")


def summary(events):
    return [(event.kind, event.check_id) for event in events]


def test_each_check_is_started_then_finished_in_plan_order():
    events = []

    run = run_plan(TOYS, hooks=[events.append])

    assert summary(events) == [
        ("check_started", "null_share"),
        ("check_finished", "null_share"),
        ("check_started", "numeric_spread"),
        ("check_finished", "numeric_spread"),
        ("check_started", "quality_summary"),
        ("check_finished", "quality_summary"),
    ]
    finished = [e for e in events if isinstance(e, CheckFinished)]
    assert [e.result for e in finished] == [run.results[e.check_id] for e in finished]


def test_position_and_total_count_the_plan():
    events = []

    run_plan(TOYS, hooks=[events.append])

    assert [(e.position, e.total) for e in events if isinstance(e, CheckStarted)] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]


def test_each_note_is_reported_between_start_and_finish():
    def noisy(ctx, params):
        ctx.note("first", "one")
        ctx.note("second", "two")

    events = []

    run_plan([declare("noisy", function=noisy)], hooks=[events.append])

    assert summary(events) == [
        ("check_started", "noisy"),
        ("note", "noisy"),
        ("note", "noisy"),
        ("check_finished", "noisy"),
    ]
    assert [e.note.code for e in events if isinstance(e, NoteRecorded)] == [
        "first",
        "second",
    ]


def test_a_check_that_is_not_called_has_no_start_but_still_finishes(monkeypatch):
    real = extras_module.find_spec
    monkeypatch.setattr(
        extras_module, "find_spec", lambda m: None if m == "statsmodels" else real(m)
    )
    events = []

    run_plan([declare("acf", extras=["timeseries"])], hooks=[events.append])

    assert summary(events) == [("note", "acf"), ("check_finished", "acf")]
    assert events[-1].result.status is CheckStatus.UNAVAILABLE


def test_an_error_and_its_skipped_dependent_both_finish_in_order():
    events = []

    run_plan(
        [
            dataclasses.replace(null_share, function=explode),
            numeric_spread,
            quality_summary,
        ],
        hooks=[events.append],
    )

    finished = [
        (e.check_id, e.result.status) for e in events if e.kind == "check_finished"
    ]
    assert finished == [
        ("null_share", CheckStatus.ERROR),
        ("numeric_spread", CheckStatus.OK),
        ("quality_summary", CheckStatus.SKIPPED),
    ]
    assert [e.check_id for e in events if e.kind == "check_started"] == [
        "null_share",
        "numeric_spread",
    ]


def test_a_timeout_is_reported_with_its_note():
    events = []

    run_plan(
        [declare("slow", function=lambda ctx, params: threading.Event().wait(2))],
        limits=Limits(check_timeout_seconds=0.1, cancel_grace_seconds=0.02),
        hooks=[events.append],
    )

    assert summary(events) == [
        ("check_started", "slow"),
        ("note", "slow"),
        ("check_finished", "slow"),
    ]
    assert events[1].note.code == "check_timeout"


def test_every_hook_hears_every_event():
    first, second = [], []

    run_plan(TOYS, hooks=[first.append, second.append])

    assert first == second and len(first) == 6


def test_hooks_run_on_the_calling_thread():
    threads = set()

    run_plan(TOYS, hooks=[lambda event: threads.add(threading.current_thread())])

    assert threads == {threading.current_thread()}


def test_a_hook_that_raises_is_logged_once_and_dropped(caplog):
    calls = []

    def broken(event):
        calls.append(event.kind)
        raise ValueError("progress bar broke")

    healthy = []

    with caplog.at_level(logging.WARNING, logger="dsbf.engine.events"):
        run = run_plan(TOYS, hooks=[broken, healthy.append])

    assert run.health.counts.ok == 3
    assert calls == ["check_started"]
    assert len(healthy) == 6
    (record,) = caplog.records
    assert "receives no further events" in record.getMessage()
    assert record.exc_info and record.exc_info[0] is ValueError


def test_without_hooks_nothing_is_emitted():
    assert run_plan(TOYS).health.counts.ok == 3
