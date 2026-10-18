# dsbf/engine/worker.py
"""Run one check on its own thread, so the executor can stop waiting for it."""

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from dsbf.data import Dataset
from dsbf.engine.context import CheckContext
from dsbf.engine.plan import PlanEntry
from dsbf.engine.runner import run_check
from dsbf.model import CheckResult, DatasetProfile


@dataclass(frozen=True)
class Attempt:
    """How a check's turn ended: a result, an exception, or running out of time."""

    result: CheckResult | None = None
    error: Exception | None = None
    timed_out: bool = False
    stopped: bool = False  # after a timeout: the check left its thread in the grace


def attempt_check(
    entry: PlanEntry,
    dataset: Dataset,
    profile: DatasetProfile,
    upstream: Mapping[str, CheckResult],
    *,
    limit: float | None,
    grace: float,
) -> Attempt:
    """Build the context and run the check, waiting at most ``limit`` seconds.

    The check runs on a daemon thread, which Python cannot stop. On a timeout the
    context is cancelled, so a check that still uses it stops at its next call;
    after ``grace`` seconds the thread is abandoned, and ``stopped`` says whether it
    had left by then. An exception is returned in ``error``; anything that is not an
    ``Exception`` (``KeyboardInterrupt``, ``SystemExit``) is raised here, on the
    calling thread.
    """
    cancel = threading.Event()
    outcome: list[CheckResult | BaseException] = []

    def work() -> None:
        try:
            context = CheckContext(
                entry.check.spec,
                dataset,
                profile,
                entry.params,
                upstream={
                    name: upstream[name]
                    for name in (*entry.requires, *entry.after)
                    if name in upstream
                },
                cancel=cancel,
            )
            outcome.append(run_check(entry.check, context))
        except BaseException as error:
            outcome.append(error)

    worker = threading.Thread(target=work, name=f"dsbf-check-{entry.id}", daemon=True)
    worker.start()
    try:
        worker.join(limit)
        if worker.is_alive():
            cancel.set()
            worker.join(grace)
            return Attempt(timed_out=True, stopped=not worker.is_alive())
    except BaseException:
        cancel.set()
        raise
    (done,) = outcome
    if isinstance(done, CheckResult):
        return Attempt(result=done)
    if isinstance(done, Exception):
        return Attempt(error=done)
    raise done
