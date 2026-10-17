# dsbf/data/files.py
"""Readers for files that fit in memory: CSV, TSV, Parquet, Excel, JSON and NDJSON."""

from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import polars as pl

from dsbf.data.errors import (
    SourceDependencyError,
    SourceError,
    SourceReadError,
    UnsupportedSourceError,
)

FORMAT_BY_EXTENSION = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".json": "json",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
}

# The strings pandas reads as missing, so a CSV profiles the same as it did through
# `pd.read_csv`. Polars reads only the empty string as null by default.
NULL_TOKENS = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]


def _read_csv(path: Path, separator: str = ",") -> pl.DataFrame:
    # A full scan to infer types: a text value far down must not fail an integer column.
    return pl.read_csv(
        path, separator=separator, null_values=NULL_TOKENS, infer_schema_length=None
    )


def _read_excel(path: Path) -> pl.DataFrame:
    if find_spec("fastexcel") is None:
        raise SourceDependencyError(
            "Reading Excel files needs the fastexcel package",
            source=str(path),
            hint="Install it with `pip install 'dsbf[excel]'`.",
        )
    return pl.read_excel(path)


_READERS: dict[str, Callable[[Path], pl.DataFrame]] = {
    "csv": _read_csv,
    "tsv": lambda path: _read_csv(path, separator="\t"),
    "parquet": pl.read_parquet,
    "excel": _read_excel,
    "json": pl.read_json,
    "ndjson": pl.read_ndjson,
}

_READ_HINTS = {
    "csv": (
        "Check that every row has the same number of fields and the file is UTF-8; "
        "for tab-separated data use a .tsv extension or format='tsv'."
    ),
    "tsv": "Check that every row has the same number of fields and the file is UTF-8.",
    "parquet": "The file may be truncated or not Parquet; check its extension.",
    "excel": "Check that the file is a valid workbook and is not password-protected.",
    "json": (
        "A .json file must hold an array of records; for one record per line "
        "use a .ndjson extension or format='ndjson'."
    ),
    "ndjson": "Each line must be one complete JSON object.",
}


def read_file(path: Path, format: str | None) -> tuple[pl.DataFrame, str]:
    """Read ``path`` and return the frame and the format that was used."""
    label = str(path)
    if not path.exists():
        raise SourceReadError(
            f"The file '{label}' does not exist",
            source=label,
            hint=f"Check the path; a relative path is read from {Path.cwd()}.",
        )
    if path.is_dir():
        raise UnsupportedSourceError(
            f"'{label}' is a directory",
            source=label,
            hint="Pass one file; directories and globs are not supported yet.",
        )
    if format is None:
        format = FORMAT_BY_EXTENSION.get(path.suffix.lower())
        if format is None:
            raise UnsupportedSourceError(
                f"Cannot tell the format of '{path.name}' from its extension",
                source=label,
                hint=(
                    f"Use one of {', '.join(sorted(FORMAT_BY_EXTENSION))}, "
                    "or pass format= (one of "
                    f"{', '.join(sorted(_READERS))})."
                ),
            )
    elif format not in _READERS:
        raise UnsupportedSourceError(
            f"Unknown file format '{format}'",
            source=label,
            hint=f"Use one of {', '.join(sorted(_READERS))}.",
        )
    try:
        frame = _READERS[format](path)
    except SourceError:
        raise
    except Exception as exc:
        raise SourceReadError(
            f"Could not read '{path.name}' as {format}: {_first_line(exc)}",
            source=label,
            hint=_READ_HINTS[format],
        ) from exc
    return frame, format


def _first_line(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__
