# dsbf/engine/process.py
"""Run a check in a worker process, which the executor can terminate."""

import multiprocessing
import pickle
import queue
import signal
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from multiprocessing.connection import Connection
from multiprocessing.context import ForkServerContext, SpawnContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType
from typing import Any

import polars as pl
import pyarrow as pa

from dsbf.data import Dataset
from dsbf.engine.check import Check
from dsbf.engine.context import CheckContext
from dsbf.engine.plan import PlanEntry
from dsbf.engine.runner import run_check
from dsbf.engine.worker import Attempt, Failure
from dsbf.model import CheckParams, CheckResult, DatasetProfile
from dsbf.model.evidence import SamplingInfo
from dsbf.model.manifest import SourceRef

_TERMINATE_WAIT_SECONDS = 1.0


class NotShippable(Exception):
    """A check cannot go to a worker process; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@cache
def _context() -> ForkServerContext | SpawnContext:
    """A context that starts clean workers: a fork would copy Polars' thread pool.

    ``forkserver`` where there is one, preloaded with the engine so a worker starts
    in milliseconds; ``spawn`` elsewhere.
    """
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")
        assert isinstance(context, ForkServerContext)
        context.set_forkserver_preload(["dsbf.engine"])
        return context
    context = multiprocessing.get_context("spawn")
    assert isinstance(context, SpawnContext)
    return context


def _map_frame(path: str) -> pl.DataFrame:
    """Open the shared file without copying it: the workers share its pages.

    ``pl.read_ipc`` reads the file into memory (400 MB for 1M x 50 floats, in every
    worker); Arrow's memory map hands Polars the file's own pages instead.
    """
    table = pa.ipc.open_file(pa.memory_map(path)).read_all()
    frame = pl.from_arrow(table)
    assert isinstance(frame, pl.DataFrame)
    return frame


@dataclass(frozen=True)
class SharedFrame:
    """A dataset as a worker opens it: the frame is a file it memory-maps."""

    path: str
    name: str
    source: SourceRef
    coerced: dict[str, str]
    sampling: SamplingInfo | None
    source_fingerprint: str | None
    value_cache_bytes: int

    def open(self) -> Dataset:
        return Dataset(
            frame=_map_frame(self.path),
            name=self.name,
            source=self.source,
            coerced=self.coerced,
            sampling=self.sampling,
            source_fingerprint=self.source_fingerprint,
            value_cache_bytes=self.value_cache_bytes,
        )


@dataclass(frozen=True)
class _Job:
    check: Check
    params: CheckParams
    profile: DatasetProfile
    upstream: dict[str, CheckResult]
    frame: SharedFrame


def _worker_main(connection: Connection, payload: bytes) -> None:
    """Entry point of a worker: run the job and send back one message.

    The message is ``("result", CheckResult, cpu)`` or ``("failure", Failure, cpu)``,
    with the CPU seconds the worker spent from the moment it had the job. A worker
    that exits any other way sends nothing, and the executor reports that.
    """
    # The parent decides what a Ctrl-C does to a run; a worker ignores it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cpu = time.process_time()
    message: tuple[str, Any, float]
    try:
        job: _Job = pickle.loads(payload)
        context = CheckContext(
            job.check.spec,
            job.frame.open(),
            job.profile,
            job.params,
            upstream=job.upstream,
        )
        message = ("result", run_check(job.check, context), 0.0)
    except Exception as error:
        message = ("failure", Failure.of(error), 0.0)
    message = (message[0], message[1], time.process_time() - cpu)
    try:
        connection.send(message)
    except Exception as error:  # a result that cannot be pickled
        connection.send(("failure", Failure.of(error), message[2]))
    finally:
        connection.close()


class ProcessRun:
    """A check in a worker process, ended by its message, its death or a kill."""

    def __init__(
        self,
        process: BaseProcess,
        connection: Connection,
        *,
        deadline: float | None,
        finished: "queue.Queue[object]",
    ) -> None:
        self.expires = deadline
        self._process = process
        self._connection = connection
        self._finished = finished
        self._closed = False
        self._arrived = threading.Event()
        self._message: tuple[str, Any, float] | None = None
        self._collector = threading.Thread(
            target=self._collect, name=f"dsbf-worker-{process.pid}", daemon=True
        )
        self._collector.start()

    @property
    def pid(self) -> int:
        assert self._process.pid is not None
        return self._process.pid

    def _collect(self) -> None:
        try:
            self._message = self._connection.recv()
        except (EOFError, OSError):
            self._message = None  # the worker ended without a message
        finally:
            self._arrived.set()
            self._finished.put(self)

    def step(self, now: float) -> Attempt | None:
        if self._arrived.is_set():
            self._reap()
            message = self._message
            died = self._died() if message is None else None
            self._close()
            if message is None:
                return Attempt(failure=died)
            kind, payload, cpu = message
            if kind == "result":
                return Attempt(result=payload, cpu_seconds=cpu)
            return Attempt(failure=payload, cpu_seconds=cpu)
        if self.expires is not None and now >= self.expires:
            self._kill()
            return Attempt(timed_out=True, killed=True)
        return None

    def cancel(self) -> None:
        self._kill()

    def _died(self) -> Failure:
        code = self._process.exitcode
        how = (
            f"was killed by signal {-code}"
            if code and code < 0
            else f"exited with code {code}"
        )
        return Failure(
            type_name="WorkerDied",
            message=f"The worker process {how} without returning a result.",
            detail="",
            contract=False,
        )

    def _reap(self) -> None:
        """Wait for a worker that has sent its message to exit; kill it if it won't."""
        self._process.join(_TERMINATE_WAIT_SECONDS)
        if self._process.is_alive():
            self._kill()

    def _kill(self) -> None:
        if self._closed:
            return
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(_TERMINATE_WAIT_SECONDS)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        self._collector.join(_TERMINATE_WAIT_SECONDS)
        self._close()

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            self._connection.close()
            self._process.close()


class WorkerHost:
    """Starts worker processes for one run and cleans up after them.

    The dataset's frame is written once, the first time a check needs a worker, to
    a file in a temporary directory that every worker memory-maps instead of
    copying. Use as a context manager so the file goes when the run ends.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._frame: SharedFrame | None = None
        self._unshareable: str | None = None

    def __enter__(self) -> "WorkerHost":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def package(
        self,
        entry: PlanEntry,
        profile: DatasetProfile,
        upstream: Mapping[str, CheckResult],
    ) -> bytes:
        """Everything a worker needs to run ``entry``, ready to send.

        Raises:
            NotShippable: the check, its parameters or the data cannot be sent.
        """
        job = _Job(
            check=entry.check,
            params=entry.params,
            profile=profile,
            upstream=dict(upstream),
            frame=self.share_frame(),
        )
        try:
            return pickle.dumps(job)
        except Exception as error:
            raise NotShippable(str(error)) from error

    def start(
        self,
        payload: bytes,
        *,
        deadline: float | None,
        finished: "queue.Queue[object]",
    ) -> ProcessRun:
        """Start a worker on ``payload``.

        Raises:
            NotShippable: the operating system would not start a process.
        """
        context = _context()
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker_main, args=(sender, payload), daemon=True
        )
        try:
            process.start()
        except Exception as error:
            receiver.close()
            sender.close()
            raise NotShippable(f"could not start a worker process: {error}") from error
        sender.close()  # so the receiver sees the end of the worker's side
        return ProcessRun(process, receiver, deadline=deadline, finished=finished)

    def share_frame(self) -> SharedFrame:
        """The dataset as a worker opens it, written to the shared file on first use.

        Raises:
            NotShippable: the frame cannot be written as Arrow.
        """
        if self._unshareable is not None:
            raise NotShippable(self._unshareable)
        if self._frame is None:
            dataset = self._dataset
            directory = tempfile.TemporaryDirectory(prefix="dsbf-frame-")
            self._directory = directory
            path = Path(directory.name) / "frame.arrow"
            try:
                dataset.frame.write_ipc(path, compression="uncompressed")
            except Exception as error:
                self._unshareable = (
                    f"the data cannot be shared with a worker process: {error}"
                )
                raise NotShippable(self._unshareable) from error
            self._frame = SharedFrame(
                path=str(path),
                name=dataset.name,
                source=dataset.source,
                coerced=dict(dataset.coerced),
                sampling=dataset.sampling,
                source_fingerprint=dataset.source_fingerprint,
                value_cache_bytes=dataset.value_cache_bytes,
            )
        return self._frame
