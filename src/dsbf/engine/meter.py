# dsbf/engine/meter.py
"""Approximate resource use of each check: wall time, CPU and peak memory."""

import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from types import TracebackType
from typing import Self

import psutil

from dsbf.model import Timing

DEFAULT_SAMPLE_SECONDS = 0.01


class Window:
    """The measurements of one check, open from ``Meter.window()`` until it exits.

    A window of a check on a thread measures this process: its CPU and memory are
    shared with every window that overlapped it, and ``shared_with`` counts them.
    A window of a check in a worker process is ``attach``-ed to that process, and
    then measures it alone.
    """

    def __init__(
        self, process: psutil.Process, sampled: bool, own_process: bool = False
    ) -> None:
        self.started_at = datetime.now(UTC)
        self._clock = time.perf_counter()
        self._cpu = time.process_time()
        self._process = process
        self._sampled = sampled
        self._own_process = own_process
        self._attached = False
        self._overlapped: set[Window] = set()
        self.peak_bytes: int | None = None
        if sampled and not own_process:
            self.peak_bytes = self.resident()
        self.wall_seconds = 0.0
        self.cpu_seconds: float | None = None

    def resident(self) -> int:
        return self._process.memory_info().rss

    def attach(self, worker: psutil.Process) -> None:
        """Measure the memory of ``worker`` from now on, in place of this process's."""
        self._process = worker
        self._attached = True
        if self._sampled:
            self.sample_now(0)

    @property
    def shared_with(self) -> int:
        return len(self._overlapped)

    def overlap(self, other: "Window") -> None:
        self._overlapped.add(other)
        other._overlapped.add(self)

    @property
    def shares_process(self) -> bool:
        return not self._own_process

    def sample(self, resident: int) -> None:
        if self.peak_bytes is None or resident > self.peak_bytes:
            self.peak_bytes = resident

    def sample_now(self, resident: int) -> None:
        """Record ``resident`` (this process's), or the worker's own, if attached."""
        if self._own_process:
            if not self._attached:
                return
            try:
                resident = self.resident()
            except psutil.Error:
                return  # the worker has gone; its last sample stands
        self.sample(resident)

    def close(self) -> None:
        self.wall_seconds = time.perf_counter() - self._clock
        if self.shares_process:
            self.cpu_seconds = time.process_time() - self._cpu
        if self._sampled:
            if self.shares_process:
                self.sample(self.resident())
            else:
                self.sample_now(0)

    @property
    def timing(self) -> Timing:
        return Timing(
            wall_seconds=self.wall_seconds,
            cpu_seconds=self.cpu_seconds,
            peak_memory_bytes=self.peak_bytes,
            shared_with=self.shared_with,
            started_at=self.started_at,
        )


class Meter:
    """Measure checks while a run executes; use as a context manager.

    A sampler thread reads the process's resident memory every ``sample_seconds``
    and raises the peak of every open window, so windows may overlap. A peak is the
    whole process's, data included, and a spike between samples is missed: the
    figure can fall short of the true peak. Pass ``None`` to skip memory.
    """

    def __init__(self, sample_seconds: float | None = DEFAULT_SAMPLE_SECONDS) -> None:
        if sample_seconds is not None and not sample_seconds > 0:
            raise ValueError(
                f"memory_sample_seconds must be positive, not {sample_seconds!r}"
            )
        self._interval = sample_seconds
        self._process = psutil.Process()
        self._open: set[Window] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None

    def __enter__(self) -> Self:
        if self._interval is not None:
            self._sampler = threading.Thread(
                target=self._sample, name="dsbf-meter", daemon=True
            )
            self._sampler.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join()

    def open_window(self, worker: bool = False) -> Window:
        """Start measuring a check; pass the result to ``close_window`` when it ends.

        Pass ``worker=True`` for a check that runs in a process of its own: its
        window measures nothing until ``attach`` names that process, and it never
        counts as overlapping the checks on threads.
        """
        window = Window(
            self._process, sampled=self._interval is not None, own_process=worker
        )
        with self._lock:
            if window.shares_process:
                for other in self._open:
                    if other.shares_process:
                        window.overlap(other)
            self._open.add(window)
        return window

    def close_window(self, window: Window) -> None:
        with self._lock:
            self._open.discard(window)
        window.close()

    @contextmanager
    def window(self, worker: bool = False):
        """Measure what happens inside the ``with`` block; read ``.timing`` after."""
        window = self.open_window(worker)
        try:
            yield window
        finally:
            self.close_window(window)

    def _sample(self) -> None:
        assert self._interval is not None
        while not self._stop.wait(self._interval):
            resident = self._process.memory_info().rss
            with self._lock:
                for window in self._open:
                    window.sample_now(resident)
