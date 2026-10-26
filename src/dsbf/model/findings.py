# dsbf/model/findings.py
"""A run's findings as a view that can be filtered and handed to a dataframe."""

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Self

from dsbf.model.finding import Finding
from dsbf.model.policy import Matcher

if TYPE_CHECKING:
    import polars as pl

__all__ = ["Findings"]


class Findings(tuple[Finding, ...]):
    """Findings in order, as a tuple that also answers questions about itself."""

    __slots__ = ()

    def filter(self, **conditions: Any) -> Self:
        """The findings that meet every condition, in the same order.

        The keys and operators are those of a policy rule's ``when``: ``dimension``,
        ``code``, ``check``, ``column`` (a pair matches on either column), ``tag``,
        ``severity=">=warn"``, ``confidence`` and ``metric={"null_share": ">0.3"}``.
        With no conditions every finding is kept. Raises ``ValueError`` for a key or
        a value a rule would reject.
        """
        if not conditions:
            return self
        matcher = Matcher(**conditions)
        return type(self)(f for f in self if matcher.matches(f))

    def to_polars(self) -> "pl.DataFrame":
        """One row per finding: what it is, where, how serious, and what it says.

        Evidence metrics are left out because their values have mixed types; the
        columns are the same, with no rows, when there are no findings.
        """
        import polars as pl

        return pl.DataFrame(
            {
                "id": [f.id for f in self],
                "check_id": [f.check_id for f in self],
                "code": [f.code for f in self],
                "dimension": [f.dimension.value for f in self],
                "severity": [f.severity.value for f in self],
                "confidence": [f.confidence.value for f in self],
                "subject_kind": [f.subject.kind for f in self],
                "columns": [list(f.subject.columns) for f in self],
                "title": [f.title for f in self],
                "message": [f.message for f in self],
                "tags": [list(f.tags) for f in self],
            },
            schema={
                "id": pl.String,
                "check_id": pl.String,
                "code": pl.String,
                "dimension": pl.String,
                "severity": pl.String,
                "confidence": pl.String,
                "subject_kind": pl.String,
                "columns": pl.List(pl.String),
                "title": pl.String,
                "message": pl.String,
                "tags": pl.List(pl.String),
            },
        )

    def to_pandas(self) -> Any:
        """The ``to_polars`` table as a pandas frame; needs the ``pandas`` extra.

        Typed ``Any`` so the pandas extra is never imported, even for type checking.
        """
        if find_spec("pandas") is None:
            raise ImportError(
                "Findings.to_pandas needs the pandas package. "
                "Install it with `pip install 'dsbf[pandas]'`."
            )
        return self.to_polars().to_pandas()
