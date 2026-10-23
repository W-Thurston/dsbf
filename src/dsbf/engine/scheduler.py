# dsbf/engine/scheduler.py
"""Run the checks of one plan level at the same time, on threads and in workers."""

import os
import queue
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType

import psutil

from dsbf.data import Dataset
from dsbf.engine.meter import Meter, Window
from dsbf.engine.plan import PlanEntry
from dsbf.engine.process import NotShippable, ProcessRun, WorkerHost
from dsbf.engine.reference import Reference
from dsbf.engine.worker import Attempt, Running, ThreadRun
from dsbf.model import CheckResult, DatasetProfile, Note, Severity

MAX_DEFAULT_WORKERS = 8


def default_workers() -> int:
    """One worker per core this process may use, at most ``MAX_DEFAULT_WORKERS``."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cores = os.cpu_count() or 1
    return max(1, min(cores, MAX_DEFAULT_WORKERS))


@dataclass(frozen=True)
class Launch:
    """A check that will be started, and the time it may take."""

    entry: PlanEntry
    limit: float | None
    by_budget: bool  # ``limit`` is what was left of the run budget


@dataclass
class _Active:
    launch: Launch
    run: Running
    window: Window
    fallback: str | None  # why a process check is on a thread, if it is


class Scheduler:
    """Starts checks up to ``workers`` at a time and waits for them on one thread.

    The calling thread does all the waiting and hands every result back, so the
    caller emits events and keeps its state without locks. A check's deadline runs
    from the moment it starts, not from when it was queued. Use as a context
    manager: leaving it early stops what is still running and removes the worker
    processes' shared data.
    """

    def __init__(
        self,
        *,
        workers: int,
        dataset: Dataset,
        profile: DatasetProfile,
        grace: float,
        meter: Meter,
        reference: Reference | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError(f"workers must be at least 1, not {workers!r}")
        self._workers = workers
        self._dataset = dataset
        self._profile = profile
        self._reference = reference
        self._grace = grace
        self._meter = meter
        self._host = WorkerHost(dataset)
        self._finished: queue.Queue[object] = queue.Queue()
        self._active: list[_Active] = []

    def __enter__(self) -> "Scheduler":
        self._host.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop_all()
        self._host.__exit__(exc_type, exc, traceback)

    def run_level(
        self,
        entries: Sequence[PlanEntry],
        upstream: Mapping[str, CheckResult],
        *,
        prepare: Callable[[PlanEntry], CheckResult | Launch],
        started: Callable[[PlanEntry], None],
        finish: Callable[[Launch, Attempt, Window], CheckResult],
    ) -> Iterator[tuple[PlanEntry, CheckResult]]:
        """Run ``entries``, which depend on nothing in each other, and yield results.

        ``prepare`` is called for each entry when a worker is free for it, in
        order, and either gives the entry's result at once (it will not run) or says
        how to run it; ``started`` is called after the check starts; ``finish`` turns
        how a check's turn ended into its result. Results come in the order the
        checks end, which is the order of ``entries`` when ``workers`` is 1.
        """
        pending = deque(entries)
        try:
            while pending or self._active:
                while pending and len(self._active) < self._workers:
                    entry = pending.popleft()
                    prepared = prepare(entry)
                    if isinstance(prepared, CheckResult):
                        yield entry, prepared
                        continue
                    self._active.append(self._launch(prepared, upstream))
                    started(entry)
                if not self._active:
                    continue
                self._wait()
                now = time.monotonic()
                for active in list(self._active):
                    attempt = active.run.step(now)
                    if attempt is None:
                        continue
                    self._active.remove(active)
                    yield active.launch.entry, self._conclude(active, attempt, finish)
        except BaseException:
            self._stop_all()
            raise

    def _launch(self, launch: Launch, upstream: Mapping[str, CheckResult]) -> _Active:
        entry = launch.entry
        deadline = None if launch.limit is None else time.monotonic() + launch.limit
        visible = {
            name: upstream[name]
            for name in entry.check.spec.upstream
            if name in upstream
        }
        fallback = None
        if entry.check.spec.runs_in == "process":
            try:
                run, window = self._start_worker(entry, visible, deadline)
                return _Active(launch, run, window, None)
            except NotShippable as error:
                fallback = error.reason
        window = self._meter.open_window()
        run = ThreadRun(
            entry,
            self._dataset,
            self._profile,
            visible,
            self._reference,
            deadline=deadline,
            grace=self._grace,
            finished=self._finished,
        )
        return _Active(launch, run, window, fallback)

    def _start_worker(
        self,
        entry: PlanEntry,
        upstream: Mapping[str, CheckResult],
        deadline: float | None,
    ) -> tuple[ProcessRun, Window]:
        """Start ``entry`` in a worker process, with a window on that process.

        Raises:
            NotShippable: the check, its inputs or the data cannot go to a worker.
        """
        payload = self._host.package(entry, self._profile, upstream, self._reference)
        window = self._meter.open_window(worker=True)
        try:
            run = self._host.start(payload, deadline=deadline, finished=self._finished)
        except NotShippable:
            self._meter.close_window(window)
            raise
        window.attach(psutil.Process(run.pid))
        return run, window

    def _wait(self) -> None:
        """Block until a check may have ended or the nearest deadline passes."""
        expiries = [
            active.run.expires
            for active in self._active
            if active.run.expires is not None
        ]
        timeout = max(0.0, min(expiries) - time.monotonic()) if expiries else None
        try:
            self._finished.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            try:
                self._finished.get_nowait()
            except queue.Empty:
                return

    def _conclude(
        self,
        active: _Active,
        attempt: Attempt,
        finish: Callable[[Launch, Attempt, Window], CheckResult],
    ) -> CheckResult:
        window = active.window
        self._meter.close_window(window)
        if not window.shares_process:
            window.cpu_seconds = attempt.cpu_seconds
        result = finish(active.launch, attempt, window)
        if active.fallback is not None:
            note = Note(
                code="ran_in_thread",
                message=(
                    f"{result.check_id} asks for a worker process but ran on a "
                    f"thread: {active.fallback}"
                ),
                severity=Severity.NOTICE,
                hint=(
                    "Declare the check with @check at the top level of an importable "
                    "module so a worker can import it. Until then it cannot be "
                    "killed on a timeout."
                ),
            )
            result = result.model_copy(update={"notes": (*result.notes, note)})
        return result

    def _stop_all(self) -> None:
        """Cancel every running check; a thread is not waited for."""
        active, self._active = self._active, []
        for item in active:
            item.run.cancel()
            self._meter.close_window(item.window)
