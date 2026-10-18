# dsbf/engine/events.py
"""What the executor reports while a run is under way, and who hears it."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Literal

from dsbf.model import CheckResult, Note

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckStarted:
    """A check is about to be called; checks that are never called don't emit this."""

    kind: ClassVar[Literal["check_started"]] = "check_started"
    check_id: str
    position: int  # 1-based, in plan order
    total: int


@dataclass(frozen=True)
class NoteRecorded:
    """A note on a check's result, whether the check or the executor wrote it."""

    kind: ClassVar[Literal["note"]] = "note"
    check_id: str
    note: Note
    position: int
    total: int


@dataclass(frozen=True)
class CheckFinished:
    """A check has its result; every planned check emits exactly one."""

    kind: ClassVar[Literal["check_finished"]] = "check_finished"
    check_id: str
    result: CheckResult
    position: int
    total: int


Event = CheckStarted | NoteRecorded | CheckFinished
Hook = Callable[[Event], None]


class Dispatcher:
    """Deliver events to hooks, on the calling thread, in the order they happen.

    A hook that raises is logged and receives nothing further in this run: a broken
    progress bar must not stop an analysis or flood the log.
    """

    def __init__(self, hooks: Iterable[Hook]) -> None:
        self._hooks = list(hooks)

    def emit(self, event: Event) -> None:
        for hook in list(self._hooks):
            try:
                hook(event)
            except Exception:
                self._hooks.remove(hook)
                logger.warning(
                    "Event hook %r raised on %s for %s; it receives no further "
                    "events in this run.",
                    hook,
                    event.kind,
                    event.check_id,
                    exc_info=True,
                )
