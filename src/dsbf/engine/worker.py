# dsbf/engine/worker.py
"""Run one check on its own thread, so the executor can stop waiting for it."""

import queue
import threading
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from dsbf.data import Dataset
from dsbf.engine.capture import WarningLog, collecting
from dsbf.engine.context import CheckContext
from dsbf.engine.errors import CheckContractError
from dsbf.engine.plan import PlanEntry
from dsbf.engine.reference import Reference
from dsbf.engine.runner import run_check
from dsbf.model import CheckResult, DatasetProfile, Note


@dataclass(frozen=True)
class Failure:
    """An exception from a check, in a form a worker process can send back."""

    type_name: str
    message: str
    detail: str  # the formatted traceback
    contract: bool  # the check broke the context's rules: a bug in the check

    @property
    def summary(self) -> str:
        """The traceback's last line, the exception and its message."""
        lines = self.detail.strip().splitlines()
        return lines[-1] if lines else ""

    @classmethod
    def of(cls, error: Exception) -> "Failure":
        return cls(
            type_name=type(error).__name__,
            message=str(error) or type(error).__name__,
            detail="".join(traceback.format_exception(error)),
            contract=isinstance(error, CheckContractError),
        )


@dataclass(frozen=True)
class Attempt:
    """How a check's turn ended: a result, a failure, or running out of time.

    ``notes`` are the ``python_warning`` notes of what the check raised, however its
    turn ended.
    """

    result: CheckResult | None = None
    failure: Failure | None = None
    timed_out: bool = False
    stopped: bool = False  # a thread after a timeout: it left within the grace
    killed: bool = False  # a worker process after a timeout: it was terminated
    cpu_seconds: float | None = None  # a worker process's own, when it finished
    notes: tuple[Note, ...] = ()


class Running(Protocol):
    """A check that has been started and has not ended its turn."""

    expires: float | None  # the monotonic time the scheduler must look again

    def step(self, now: float) -> Attempt | None:
        """The attempt if the check's turn is over at ``now``, else None.

        Raises:
            BaseException: anything that is not an ``Exception``
                (``KeyboardInterrupt``, ``SystemExit``) that the check raised.
        """

    def cancel(self) -> None:
        """Stop the check if it can be stopped; it is not waited for."""


class ThreadRun:
    """A check on a daemon thread, which Python cannot stop.

    After ``deadline`` the context is cancelled, so a check that still uses it stops
    at its next call; ``grace`` seconds later the thread is abandoned, and
    ``Attempt.stopped`` says whether it had left by then. The thread puts itself on
    ``finished`` when it ends, to wake the scheduler. The warnings it raises are
    logged for its result (see ``dsbf.engine.capture``).
    """

    def __init__(
        self,
        entry: PlanEntry,
        dataset: Dataset,
        profile: DatasetProfile,
        upstream: Mapping[str, CheckResult],
        reference: Reference | None,
        *,
        deadline: float | None,
        grace: float,
        finished: "queue.Queue[object]",
    ) -> None:
        self.expires = deadline
        self._deadline = deadline
        self._grace = grace
        self._cancel = threading.Event()
        self._timed_out = False
        self._outcome: list[CheckResult | BaseException] = []
        self._finished = finished
        self._warnings = WarningLog()
        self._thread = threading.Thread(
            target=self._work,
            args=(entry, dataset, profile, upstream, reference),
            name=f"dsbf-check-{entry.id}",
            daemon=True,
        )
        self._thread.start()

    def _work(
        self,
        entry: PlanEntry,
        dataset: Dataset,
        profile: DatasetProfile,
        upstream: Mapping[str, CheckResult],
        reference: Reference | None,
    ) -> None:
        try:
            with collecting(self._warnings):
                context = CheckContext(
                    entry.check.spec,
                    dataset,
                    profile,
                    entry.params,
                    upstream=upstream,
                    reference=reference,
                    cancel=self._cancel,
                )
                self._outcome.append(run_check(entry.check, context))
        except BaseException as error:
            self._outcome.append(error)
        finally:
            self._finished.put(self)

    def step(self, now: float) -> Attempt | None:
        notes = self._warnings.notes
        if self._outcome:
            (done,) = self._outcome
            if self._timed_out:
                return Attempt(timed_out=True, stopped=True, notes=notes())
            if isinstance(done, CheckResult):
                return Attempt(result=done, notes=notes())
            if isinstance(done, Exception):
                return Attempt(failure=Failure.of(done), notes=notes())
            raise done
        if not self._timed_out and self._deadline is not None and now >= self._deadline:
            self._timed_out = True
            self._cancel.set()
            self.expires = now + self._grace
        if self._timed_out and self.expires is not None and now >= self.expires:
            return Attempt(timed_out=True, stopped=False, notes=notes())
        return None

    def cancel(self) -> None:
        self._cancel.set()
