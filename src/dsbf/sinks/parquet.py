# dsbf/sinks/parquet.py
"""The Parquet sinks: one file each for findings, column profiles and results."""

from pathlib import Path

import pyarrow.parquet as pq

from dsbf.model import Run
from dsbf.sinks.registry import Sink, register_alias, register_sink
from dsbf.sinks.tables import TABLES, Table


def _writer(table: Table):
    def write(run: Run, path: Path) -> None:
        pq.write_table(table.build(run), path, compression="zstd")

    return write


for _table in TABLES.values():
    register_sink(Sink(f"{_table.name}_parquet", _table.filename, _writer(_table)))

PARQUET_SINKS = tuple(f"{name}_parquet" for name in TABLES)
register_alias("parquet", PARQUET_SINKS)
