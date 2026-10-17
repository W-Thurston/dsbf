# dsbf/data/dataset.py
"""The loaded data every later stage works from."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import polars as pl

from dsbf.model.manifest import SourceRef


@dataclass(frozen=True, slots=True)
class Dataset:
    """A Polars-backed table and where it came from.

    ``frame`` is not copied from a pandas or Arrow input where the memory can be
    shared. With pandas 2.x, editing the pandas frame in place afterwards changes
    ``frame`` too; pass ``df.copy()`` to ``load`` if that can happen. pandas 3's
    copy-on-write makes it impossible.
    """

    frame: pl.DataFrame
    name: str
    source: SourceRef
    coerced: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}),
        metadata={"doc": "Columns that had to change type to load, with the reason."},
    )

    @property
    def n_rows(self) -> int:
        return self.frame.height

    @property
    def n_columns(self) -> int:
        return self.frame.width

    @property
    def columns(self) -> list[str]:
        return self.frame.columns

    @property
    def schema(self) -> pl.Schema:
        return self.frame.schema
