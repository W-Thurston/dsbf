# dsbf/checks/dtype_optimizations.py
"""Columns stored in a wider type than their values need."""

from typing import Literal

from pydantic import Field

from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    CategoricalProfile,
    CheckParams,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    NumericProfile,
    Severity,
    Subject,
)

Reason = Literal["fits_bool", "fits_int", "dictionary"]

# Bytes per value of the signed integer types a column can be narrowed to, and of the
# types it can be narrowed from; unsigned and floating point columns are left alone.
_INTEGER_BYTES = {"Int64": 8, "Int32": 4, "Int16": 2, "Int8": 1}
_RANGES = (("Int8", 2**7), ("Int16", 2**15), ("Int32", 2**31))
_BOOLEAN_WORDS = {"true", "false", "yes", "no"}
_OFFSET_BYTES = 4  # the offset a string column keeps per value


class DtypeOptimizationsParams(CheckParams):
    min_saving_bytes: int = Field(
        1024,
        ge=0,
        description="The estimated saving below which a column is not reported, so "
        "a small column does not add noise.",
    )


class DtypeSuggestion(Contract):
    current_dtype: str
    suggested_dtype: str
    reason: Reason
    estimated_saving_bytes: int = Field(ge=0)


class DtypeOptimizationsOutput(CheckOutputModel):
    suggestions: dict[str, DtypeSuggestion] = Field(
        description="Each column with a narrower type that holds its values, by name."
    )
    total_saving_bytes: int = Field(ge=0)


def _listed_values(column: CategoricalProfile) -> list[object] | None:
    """Every value of the column, if the profile lists them all."""
    if column.counts.distinct > len(column.top_values):
        return None
    return [entry.value for entry in column.top_values]


def _integer_range(
    column: NumericProfile | CategoricalProfile,
) -> tuple[int, int] | None:
    """The smallest and largest value of an integer column, if the profile has them.

    A numeric column has them as statistics. A categorical column of integer codes has
    none, so they come from its values when it lists every one.
    """
    if isinstance(column, NumericProfile):
        stats = column.stats
        if stats is None or stats.min is None or stats.max is None:
            return None
        return int(stats.min), int(stats.max)
    values = _listed_values(column)
    if not values or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in values
    ):
        return None
    return min(values), max(values)  # type: ignore[type-var]


def _string_bytes(column: CategoricalProfile) -> float | None:
    """What a value of a string column takes: its characters and its offset."""
    if column.lengths is None:
        return None
    return column.lengths.mean + _OFFSET_BYTES


def _suggest(
    column: NumericProfile | CategoricalProfile,
) -> tuple[str, Reason, int] | None:
    """The type to narrow ``column`` to, why, and the estimated bytes saved.

    None if no narrower type is shown to hold the values.
    """
    dtype, rows = column.physical_dtype, column.counts.rows
    if dtype in _INTEGER_BYTES:
        bounds = _integer_range(column)
        if bounds is None:
            return None
        if isinstance(column, CategoricalProfile):
            values = _listed_values(column) or []
            if values and set(values) <= {0, 1}:
                return "Boolean", "fits_bool", (_INTEGER_BYTES[dtype] - 1) * rows
        low, high = bounds
        for name, limit in _RANGES:
            if (
                _INTEGER_BYTES[name] < _INTEGER_BYTES[dtype]
                and -limit <= low
                and high < limit
            ):
                return (
                    name,
                    "fits_int",
                    (_INTEGER_BYTES[dtype] - _INTEGER_BYTES[name]) * rows,
                )
        return None
    if dtype == "String" and isinstance(column, CategoricalProfile):
        value_bytes = _string_bytes(column)
        if value_bytes is None:
            return None
        values = _listed_values(column)
        if (
            values
            and len(values) <= 2
            and {str(v).lower() for v in values} <= _BOOLEAN_WORDS
        ):
            return "Boolean", "fits_bool", int(rows * (value_bytes - 1))
        distinct = column.counts.distinct
        code = 1 if distinct < 2**8 else 2 if distinct < 2**16 else 4
        dictionary = rows * code + distinct * value_bytes
        return "Categorical", "dictionary", int(rows * value_bytes - dictionary)
    return None


@check(
    id="dtype_optimizations",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric", "categorical"),
    dimensions=[Dimension.USABILITY],
    params=DtypeOptimizationsParams,
    output=DtypeOptimizationsOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    codes=["smaller_dtype"],
    packs=["core"],
    tags=["memory", "dtype"],
)
def dtype_optimizations(
    ctx: CheckContext, params: DtypeOptimizationsParams
) -> CheckOutput:
    """Suggest a narrower type for integer, two-valued and low-cardinality columns.

    The profile holds no memory per column, so a saving is estimated from the type's
    width times the rows, and for strings from the mean length plus a 4-byte offset
    per value, against a dictionary of the distinct values and one small code per
    row. A column is narrowed only when the profile shows its values fit: an integer
    column's range is its minimum and maximum, or, for a column of codes the profile
    types categorical, its values when it lists them all. Floating point columns are
    never suggested: nothing in the profile shows float32 holds their precision.
    """
    suggestions: dict[str, DtypeSuggestion] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, (NumericProfile, CategoricalProfile)):
            continue
        found = _suggest(column)
        if found is None:
            continue
        target, reason, saving = found
        if saving < params.min_saving_bytes:
            continue
        suggestions[column.name] = DtypeSuggestion(
            current_dtype=column.physical_dtype,
            suggested_dtype=target,
            reason=reason,
            estimated_saving_bytes=saving,
        )
        ctx.finding(
            code="smaller_dtype",
            subject=Subject.column(column.name),
            severity=Severity.INFO,
            confidence=Confidence.MODERATE,
            title=f"{column.physical_dtype} → {target} "
            f"(~{saving / 1024:,.1f} KiB saved)",
            message=(
                f"{column.name!r} is stored as {column.physical_dtype}, and its "
                f"values {_because(reason)}. Storing it as {target} would save about "
                f"{saving / 1024:,.1f} KiB of {column.counts.rows:,} rows "
                "(an estimate from the type's width). A narrower type overflows "
                "silently on later data outside the range seen here, so convert it "
                "only if the range is known to be stable."
            ),
            evidence=Evidence(
                metrics={
                    "current_dtype": column.physical_dtype,
                    "suggested_dtype": target,
                    "estimated_saving_bytes": saving,
                    "n_unique": column.counts.distinct,
                },
                n_used=column.counts.non_null,
                n_total=column.counts.rows,
                method="dtype_width_estimate",
            ),
        )
    total = sum(s.estimated_saving_bytes for s in suggestions.values())
    return ctx.output(
        DtypeOptimizationsOutput(suggestions=suggestions, total_saving_bytes=total)
    )


def _because(reason: Reason) -> str:
    return {
        "fits_bool": "are two values that map to true and false",
        "fits_int": "fit a narrower integer type",
        "dictionary": "are few and repeat, so a dictionary stores each once",
    }[reason]
