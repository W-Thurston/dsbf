# dsbf/model/subject.py
"""What a finding is about: the dataset, one column, a pair, a group or a row set."""

from typing import Literal, Self

from pydantic import model_validator

from dsbf.model.base import Contract

SubjectKind = Literal["dataset", "column", "column_pair", "column_group", "row_set"]


class Subject(Contract):
    """The thing a finding or result refers to.

    Columns are kept as an ordered tuple of opaque strings, never joined into a
    single key, so names containing ``|``, brackets, unicode or nothing at all
    stay unambiguous.
    """

    kind: SubjectKind
    columns: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_columns(self) -> Self:
        count = len(self.columns)
        if self.kind == "dataset" and count != 0:
            raise ValueError("a dataset subject has no columns")
        if self.kind == "column" and count != 1:
            raise ValueError("a column subject has exactly one column")
        if self.kind == "column_pair" and count != 2:
            raise ValueError("a column_pair subject has exactly two columns")
        if self.kind == "column_group" and count < 2:
            raise ValueError("a column_group subject has at least two columns")
        if len(set(self.columns)) != count:
            raise ValueError("the columns of a subject are distinct")
        return self

    @classmethod
    def dataset(cls) -> Self:
        return cls(kind="dataset")

    @classmethod
    def column(cls, name: str) -> Self:
        return cls(kind="column", columns=(name,))

    @classmethod
    def column_pair(cls, first: str, second: str) -> Self:
        return cls(kind="column_pair", columns=(first, second))

    @classmethod
    def column_group(cls, *names: str) -> Self:
        return cls(kind="column_group", columns=names)

    @classmethod
    def row_set(cls, *names: str) -> Self:
        """Rows picked out by the given columns (for example duplicates)."""
        return cls(kind="row_set", columns=names)
