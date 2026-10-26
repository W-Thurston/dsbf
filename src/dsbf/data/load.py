# dsbf/data/load.py
"""The single entry point that turns a source into a Dataset."""

import os
from dataclasses import replace
from pathlib import Path

import polars as pl

from dsbf.data.dataset import Dataset
from dsbf.data.errors import SourceReadError, UnsupportedSourceError
from dsbf.data.files import read_file
from dsbf.data.frames import adapt_frame
from dsbf.model.manifest import SourceRef

_IN_MEMORY_NAME = "dataframe"


def load(
    source: object, *, name: str | None = None, format: str | None = None
) -> Dataset:
    """Load ``source`` as a Polars-backed :class:`Dataset`.

    ``source`` is a pandas, Polars (eager or lazy) or Arrow table, or the path of a
    CSV, TSV, Parquet, Excel (first sheet), JSON (an array of records) or NDJSON
    file. The format comes from the extension unless ``format`` names it. ``name``
    defaults to the file stem, or ``dataframe`` for a table in memory.

    Raises:
        UnsupportedSourceError: the object or format is not one DSBF loads.
        SourceReadError: the data is missing, empty, corrupt or malformed.
        SourceDependencyError: reading it needs an extra that is not installed.
    """
    if isinstance(source, Dataset):
        return source if name is None else replace(source, name=name)
    if isinstance(source, str | os.PathLike):
        return _load_file(source, name, format)
    label = type(source).__name__
    if format is not None:
        raise UnsupportedSourceError(
            "format applies to files, not to a table in memory",
            source=label,
            hint="Drop format=, or pass a path.",
        )
    adapted = adapt_frame(source, label)
    if adapted is None:
        raise UnsupportedSourceError(
            f"Cannot load a {label} object",
            source=label,
            hint=(
                "Pass a pandas, Polars or Arrow table, or the path of a CSV, "
                "Parquet, Excel, JSON or NDJSON file."
            ),
        )
    frame, coerced = adapted
    _require_columns(frame, label)
    return Dataset(
        frame=frame,
        name=name or _IN_MEMORY_NAME,
        source=SourceRef(kind="dataframe"),
        coerced=coerced,
    )


def _load_file(
    source: str | os.PathLike, name: str | None, format: str | None
) -> Dataset:
    text = os.fspath(source)
    if "://" in text:
        raise UnsupportedSourceError(
            f"Remote source '{text}' is not supported",
            source=text,
            hint="Download the file and pass its local path.",
        )
    path = Path(text).expanduser()
    frame, used = read_file(path, format)
    _require_columns(frame, str(path))
    return Dataset(
        frame=frame,
        name=name or path.stem or _IN_MEMORY_NAME,
        source=SourceRef(kind="file", location=str(path.resolve()), format=used),
    )


def _require_columns(frame: pl.DataFrame, label: str) -> None:
    if frame.width == 0:
        raise SourceReadError(
            "The source has no columns", source=label, hint="Check that it holds data."
        )
    unsafe = [name for name in frame.columns if _is_pattern(name)]
    if unsafe:
        raise SourceReadError(
            "Column names Polars reads as patterns cannot be profiled: "
            + ", ".join(repr(name) for name in unsafe),
            source=label,
            hint="Rename them, for example 'star' for '*'.",
        )


def _is_pattern(name: str) -> bool:
    # `pl.col` reads `*` as every column and `^...$` as a regular expression, and
    # every Series method selects its own name that way, so a Series called this
    # fails or selects nothing wherever it goes.
    return name == "*" or (name.startswith("^") and name.endswith("$"))
