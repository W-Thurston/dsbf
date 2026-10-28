# dsbf/store/artifacts.py
"""Where artifact payloads and data snapshots are kept: files behind fsspec."""

import hashlib
from typing import Protocol

import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq

from dsbf.store.repository import StoreError


class ArtifactNotFoundError(StoreError):
    """The store has no file for this artifact."""


class ArtifactStore(Protocol):
    """Files addressed by a run id and a run-relative path."""

    def put(self, run_id: str, path: str, data: bytes) -> tuple[int, str]:
        """Write ``data``; return the bytes written and their SHA-256."""

    def read(self, run_id: str, path: str) -> bytes:
        """The whole file.

        Raises:
            ArtifactNotFoundError: no such file.
        """

    def read_rows(self, run_id: str, path: str, start: int, stop: int) -> pa.Table:
        """Rows ``start`` up to ``stop`` of a Parquet file, reading only the row
        groups that hold them.

        Raises:
            ArtifactNotFoundError: no such file.
        """

    def exists(self, run_id: str, path: str) -> bool:
        """Whether the file is stored."""

    def delete_run(self, run_id: str) -> None:
        """Remove every file of a run."""


class FsspecArtifactStore:
    """Files under a root URL: a local directory now, an object store later."""

    def __init__(self, root: str) -> None:
        self._fs, self._root = fsspec.core.url_to_fs(root)

    def _path(self, run_id: str, path: str = "") -> str:
        return f"{self._root.rstrip('/')}/{run_id}/{path}".rstrip("/")

    def put(self, run_id: str, path: str, data: bytes) -> tuple[int, str]:
        target = self._path(run_id, path)
        self._fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
        self._fs.pipe_file(target, data)
        return len(data), hashlib.sha256(data).hexdigest()

    def read(self, run_id: str, path: str) -> bytes:
        try:
            return self._fs.cat_file(self._path(run_id, path))
        except FileNotFoundError as error:
            raise _missing(run_id, path) from error

    def read_rows(self, run_id: str, path: str, start: int, stop: int) -> pa.Table:
        try:
            handle = self._fs.open(self._path(run_id, path), "rb")
        except FileNotFoundError as error:
            raise _missing(run_id, path) from error
        with handle:
            file = pq.ParquetFile(handle)
            groups, offset = [], 0
            for index in range(file.num_row_groups):
                rows = file.metadata.row_group(index).num_rows
                if offset < stop and offset + rows > start:
                    first = max(start - offset, 0)
                    last = min(stop - offset, rows)
                    groups.append(file.read_row_group(index).slice(first, last - first))
                offset += rows
            if not groups:
                return file.schema_arrow.empty_table()
            return pa.concat_tables(groups)

    def exists(self, run_id: str, path: str) -> bool:
        return bool(self._fs.exists(self._path(run_id, path)))

    def delete_run(self, run_id: str) -> None:
        directory = self._path(run_id)
        if self._fs.exists(directory):
            self._fs.rm(directory, recursive=True)


def _missing(run_id: str, path: str) -> ArtifactNotFoundError:
    return ArtifactNotFoundError(f"run {run_id} has no stored file {path!r}")
