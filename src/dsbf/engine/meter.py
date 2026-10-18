# dsbf/engine/meter.py
"""Approximate resource use of each check: wall time, process CPU and peak memory."""

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
    """The measurements of one check, open from ``Meter.window()`` until it exits."""

    def __init__(self, process: psutil.Process, sampled: bool) -> None:
        self.started_at = datetime.now(UTC)
        self._clock = time.perf_counter()
        self._cpu = time.process_time()
        self._process = process
        self._sampled = sampled
        self.peak_bytes: int | None = self.resident() if sampled else None
        self.wall_seconds = 0.0
        self.cpu_seconds = 0.0

    def resident(self) -> int:
        return self._process.memory_info().rss

    def sample(self, resident: int) -> None:
        if self.peak_bytes is None or resident > self.peak_bytes:
            self.peak_bytes = resident

    def close(self) -> None:
        self.wall_seconds = time.perf_counter() - self._clock
        self.cpu_seconds = time.process_time() - self._cpu
        if self._sampled:
            self.sample(self.resident())

    @property
    def timing(self) -> Timing:
        return Timing(
            wall_seconds=self.wall_seconds,
            cpu_seconds=self.cpu_seconds,
            peak_memory_bytes=self.peak_bytes,
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

    @contextmanager
    def window(self):
        """Measure what happens inside the ``with`` block; read ``.timing`` after."""
        window = Window(self._process, sampled=self._interval is not None)
        with self._lock:
            self._open.add(window)
        try:
            yield window
        finally:
            with self._lock:
                self._open.discard(window)
            window.close()

    def _sample(self) -> None:
        assert self._interval is not None
        while not self._stop.wait(self._interval):
            resident = self._process.memory_info().rss
            with self._lock:
                for window in self._open:
                    window.sample(resident)
