# dsbf/engine/capture.py
"""Turn the Python warnings a check raises into notes on its result.

A warning is not a log record: left alone it reaches the caller's stderr, and under
the default filters only the first time it is raised at a place in the code, for the
life of the process. While a run lasts the executor routes every warning raised on a
check's thread to that check's ``WarningLog``, and the check's result carries one
``python_warning`` note for each category and message, with a count.
"""

import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from dsbf.model import Note, Severity

# The distinct warnings one check keeps; the rest are counted in one more note, so a
# check that warns once per value cannot fill the run with notes.
MAX_WARNING_KINDS = 20

_HINT = (
    "The check's code or a library it calls raised it; the result may be less "
    "reliable for this data."
)
# Filter actions that show a warning only the first time; the run shows every one.
_ONCE_ACTIONS = frozenset({"default", "module", "once"})

_lock = threading.Lock()
_depth = 0
_saved: warnings.catch_warnings | None = None
_installed: list[Any] = []  # the run's filters, as installed
_original: Callable[..., None] | None = None
_local = threading.local()


class WarningLog:
    """The warnings one check raised, by category and message, in the order seen.

    A thread run's log is read by the scheduler while the check may still be
    raising, so adding and reading are locked.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[type[Warning], str], int] = {}
        self._first_at: dict[tuple[type[Warning], str], str] = {}
        self._more = 0

    def add(
        self, category: type[Warning], text: str, filename: str, lineno: int
    ) -> None:
        key = (category, text)
        with self._lock:
            if key in self._counts:
                self._counts[key] += 1
            elif len(self._counts) < MAX_WARNING_KINDS:
                self._counts[key] = 1
                self._first_at[key] = f"{filename}:{lineno}"
            else:
                self._more += 1

    def notes(self) -> tuple[Note, ...]:
        """One ``python_warning`` note per kind, and one for the kinds past the cap."""
        with self._lock:
            kinds = [
                (key, count, self._first_at[key]) for key, count in self._counts.items()
            ]
            more = self._more
        notes = [
            Note(
                code="python_warning",
                message=f"{category.__name__}: {text}" if text else category.__name__,
                severity=Severity.NOTICE,
                hint=_HINT,
                detail=f"First raised at {where}.",
                count=count,
            )
            for (category, text), count, where in kinds
        ]
        if more:
            notes.append(
                Note(
                    code="python_warning",
                    message=(
                        f"{more} more warning{'s' if more > 1 else ''} with other "
                        "messages, not listed."
                    ),
                    severity=Severity.NOTICE,
                    hint=_HINT,
                    detail=f"Only the first {MAX_WARNING_KINDS} kinds are listed.",
                    count=more,
                )
            )
        return tuple(notes)


@contextmanager
def capture_warnings() -> Iterator[None]:
    """Route warnings to the logs of the threads that raise them, while it lasts.

    The process's warning filters are changed for as long as any run is in this
    block: a filter that shows a warning once (``default``, ``module``, ``once``)
    shows it every time, so a log counts each repeat and a second run sees what the
    first did; ``ignore`` and ``error`` filters hold. A warning from a thread with
    no log goes to the ``warnings.showwarning`` that was in place, under the same
    filters, so the caller's other threads see each repeat while a run lasts.
    Concurrent and nested runs share one installation; the last to leave restores
    the filters and ``warnings.showwarning`` as they were, keeping any filter added
    meanwhile, such as the one a library adds when a check first imports it.
    """
    global _depth, _saved, _original, _installed
    with _lock:
        if _depth == 0:
            saved = warnings.catch_warnings()
            saved.__enter__()
            if warnings.showwarning is not _show:
                _original = warnings.showwarning
            # ``catch_warnings`` restores the list it saved, so a new one is safe.
            warnings.filters = [
                (
                    "always" if action in _ONCE_ACTIONS else action,
                    message,
                    category,
                    module,
                    lineno,
                )
                for action, message, category, module, lineno in warnings.filters
            ]
            # Appending is what clears the registries of warnings already shown.
            warnings.simplefilter("always", append=True)
            warnings.showwarning = _show
            _saved = saved
            _installed = list(warnings.filters)
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _saved is not None:
                added = [f for f in warnings.filters if f not in _installed]
                _saved.__exit__(None, None, None)
                _saved = None
                if added:
                    warnings.filters = [*added, *warnings.filters]


@contextmanager
def collecting(log: WarningLog) -> Iterator[WarningLog]:
    """Add the warnings this thread raises to ``log``, inside ``capture_warnings``."""
    previous = getattr(_local, "log", None)
    _local.log = log
    try:
        yield log
    finally:
        _local.log = previous


def _show(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    log: WarningLog | None = getattr(_local, "log", None)
    if log is not None:
        log.add(category, str(message), filename, lineno)
    elif _original is not None:
        _original(message, category, filename, lineno, file, line)
