"""A run can be stopped from another thread, between checks."""

import threading

import pytest

from dsbf.engine import CheckFinished, CheckStarted, RunCancelled
from tests.test_engine.test_executor import TOYS
from tests.test_engine.toys import run_plan

pytestmark = pytest.mark.unit


def test_a_set_event_stops_the_run_before_the_next_check_starts():
    cancel = threading.Event()
    seen = []

    def hook(event):
        seen.append(event)
        if isinstance(event, CheckFinished):
            cancel.set()

    with pytest.raises(RunCancelled):
        run_plan(TOYS, workers=1, hooks=[hook], cancel=cancel)

    assert sum(isinstance(e, CheckStarted) for e in seen) == 1
    assert sum(isinstance(e, CheckFinished) for e in seen) == 1


def test_an_event_set_before_the_run_starts_no_check():
    cancel = threading.Event()
    cancel.set()
    seen = []

    with pytest.raises(RunCancelled):
        run_plan(TOYS, workers=1, hooks=[seen.append], cancel=cancel)

    assert seen == []


def test_an_unset_event_changes_nothing():
    run = run_plan(TOYS, workers=1, cancel=threading.Event())
    assert set(run.results) == {toy.id for toy in TOYS}
