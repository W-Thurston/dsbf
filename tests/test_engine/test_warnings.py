"""A warning a check raises becomes a note on its result, not output on stderr."""

import threading
import time
import warnings

import pytest

from dsbf.engine import DiskCache, Limits
from dsbf.engine.capture import (
    MAX_WARNING_KINDS,
    WarningLog,
    capture_warnings,
    collecting,
)
from dsbf.model import CheckStatus, Severity
from tests.test_engine.toys import declare, run_plan

pytestmark = pytest.mark.unit


def warns(*messages, category=RuntimeWarning):
    """A check function that raises each of ``messages`` as a warning, in order."""

    def check(ctx, params):
        for message in messages:
            warnings.warn(message, category, stacklevel=1)

    return check


def warning_notes(result):
    return [note for note in result.notes if note.code == "python_warning"]


@pytest.fixture
def default_filters():
    """The interpreter's own filters, whatever the test runner installed."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        for category in (DeprecationWarning, PendingDeprecationWarning):
            warnings.simplefilter("ignore", category)
        warnings.simplefilter("default")
        yield


class TestNotes:
    @pytest.mark.parametrize("workers", [1, 4])
    def test_a_repeated_warning_is_one_note_with_its_count(
        self, workers, default_filters, capsys
    ):
        checks = [
            declare("twice", function=warns("constant input", "constant input")),
            declare("other", function=warns("another thing")),
            declare("quiet"),
        ]

        run = run_plan(checks, workers=workers)

        (note,) = warning_notes(run.results["twice"])
        assert note.message == "RuntimeWarning: constant input"
        assert note.count == 2
        assert note.severity is Severity.NOTICE
        assert "test_warnings.py" in (note.detail or "")
        assert [n.message for n in warning_notes(run.results["other"])] == [
            "RuntimeWarning: another thing"
        ]
        assert warning_notes(run.results["quiet"]) == []
        assert run.results["twice"].status is CheckStatus.OK
        assert capsys.readouterr().err == ""

    def test_nothing_reaches_the_callers_handler(self, recwarn):
        run = run_plan([declare("c", function=warns("x", "x"))], workers=2)

        assert len(recwarn) == 0
        assert warning_notes(run.results["c"])[0].count == 2

    def test_a_second_run_records_the_same_note(self, default_filters, capsys):
        checks = [declare("twice", function=warns("again", "again"))]

        first = run_plan(checks, workers=1)
        second = run_plan(checks, workers=1)

        assert warning_notes(first.results["twice"]) == warning_notes(
            second.results["twice"]
        )
        assert capsys.readouterr().err == ""

    def test_the_callers_filters_and_handler_are_restored(self, default_filters):
        filters, handler = list(warnings.filters), warnings.showwarning

        run_plan([declare("twice", function=warns("x", "x"))], workers=2)

        assert warnings.filters == filters
        assert warnings.showwarning is handler

    def test_an_ignore_filter_still_holds(self, default_filters):
        warnings.filterwarnings("ignore", message="hidden")

        run = run_plan([declare("c", function=warns("hidden", "shown"))], workers=1)

        assert [n.message for n in warning_notes(run.results["c"])] == [
            "RuntimeWarning: shown"
        ]

    def test_distinct_messages_past_the_cap_are_counted_together(self, default_filters):
        extra = 5
        messages = [f"value {i}" for i in range(MAX_WARNING_KINDS + extra)]

        run = run_plan([declare("c", function=warns(*messages, "value 0"))])

        notes = warning_notes(run.results["c"])
        assert len(notes) == MAX_WARNING_KINDS + 1
        assert notes[0].count == 2
        assert notes[-1].count == extra
        assert notes[-1].message == "5 more warnings with other messages, not listed."

    def test_an_error_keeps_the_warnings_before_it(self, default_filters):
        def fails(ctx, params):
            warnings.warn("first this", UserWarning, stacklevel=1)
            raise ValueError("then this")

        run = run_plan([declare("c", function=fails)])

        result = run.results["c"]
        assert result.status is CheckStatus.ERROR
        assert result.notes[0].code == "check_failed"
        assert [n.message for n in warning_notes(result)] == ["UserWarning: first this"]

    def test_a_timeout_keeps_the_warnings_so_far(self, default_filters):
        def slow(ctx, params):
            warnings.warn("before the deadline", UserWarning, stacklevel=1)
            time.sleep(1)

        limits = Limits(check_timeout_seconds=0.2, cancel_grace_seconds=0.05)
        run = run_plan([declare("c", function=slow)], limits=limits)

        result = run.results["c"]
        assert result.status is CheckStatus.TIMEOUT
        assert [n.message for n in warning_notes(result)] == [
            "UserWarning: before the deadline"
        ]

    def test_a_cached_result_replays_its_note(self, default_filters, tmp_path):
        calls = []

        def counted(ctx, params):
            calls.append(1)
            warnings.warn("cached", UserWarning, stacklevel=1)

        checks = [declare("c", function=counted)]
        cache = DiskCache(tmp_path)

        first = run_plan(checks, cache=cache)
        second = run_plan(checks, cache=cache)

        assert len(calls) == 1
        assert second.results["c"].timing.cached
        assert warning_notes(second.results["c"]) == warning_notes(first.results["c"])


class TestRouting:
    def test_a_warning_from_another_thread_reaches_the_original_handler(self):
        seen = []
        log = WarningLog()

        def raise_one():
            warnings.warn("from elsewhere", UserWarning, stacklevel=1)

        def record(message, category, filename, lineno, file=None, line=None):
            seen.append(str(message))

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.showwarning = record
            with capture_warnings(), collecting(log):
                for _ in range(2):
                    other = threading.Thread(target=raise_one)
                    other.start()
                    other.join()

        assert seen == ["from elsewhere", "from elsewhere"]
        assert log.notes() == ()

    def test_a_filter_added_during_a_run_is_kept(self):
        with warnings.catch_warnings():
            before = list(warnings.filters)
            with capture_warnings():
                # As a library does when a check first imports it.
                warnings.filterwarnings("ignore", message="added meanwhile")
            (added, *rest) = warnings.filters
            assert added[0] == "ignore" and "added meanwhile" in str(added[1])
            assert rest == before

    def test_concurrent_runs_share_one_installation(self):
        with warnings.catch_warnings():
            handler = warnings.showwarning
            outer = capture_warnings()
            outer.__enter__()
            with capture_warnings():
                pass
            assert warnings.showwarning is not handler  # the outer run still routes
            outer.__exit__(None, None, None)
            assert warnings.showwarning is handler
