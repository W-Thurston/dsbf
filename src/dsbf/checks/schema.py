# dsbf/checks/schema.py
"""The data against a schema the caller declares."""

from collections import Counter
from collections.abc import Callable

import numpy as np
from pydantic import Field, model_validator

from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    CheckParams,
    CheckStatus,
    ColumnProfile,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    NumericProfile,
    SemanticType,
    Severity,
    Subject,
)

_EXAMPLES = 5

# Records one finding of a column: its code, title, message and evidence metrics.
Report = Callable[..., None]


class ColumnExpectation(Contract):
    """What one column must look like. Every field but `required` is optional."""

    required: bool = Field(True, description="The column must exist.")
    type: SemanticType | None = Field(
        None, description="The semantic type the profile gives the column."
    )
    dtype: str | None = Field(
        None,
        description="The storage type as the profile names it, such as `Int64` or "
        "`String`; compared without regard to case.",
    )
    nullable: bool | None = Field(
        None, description="False when the column must have no nulls."
    )
    min: float | None = Field(None, description="No value may be below this.")
    max: float | None = Field(None, description="No value may be above this.")
    allowed: tuple[bool | int | float | str, ...] | None = Field(
        None, description="The only values the column may hold."
    )

    @model_validator(mode="after")
    def _ordered(self) -> "ColumnExpectation":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min is at most max")
        return self


class SchemaParams(CheckParams):
    columns: dict[str, ColumnExpectation] = Field(
        default_factory=dict,
        description="The expectation for each column, by name. With none, the check "
        "is not applicable.",
    )


class SchemaOutput(CheckOutputModel):
    checked: int = Field(ge=0, description="Columns with an expectation.")
    missing_columns: tuple[str, ...] = Field(
        description="Required columns the data does not have."
    )
    violations: dict[str, tuple[str, ...]] = Field(
        description="For each column that breaks its expectation, the finding codes."
    )
    valid: bool = Field(description="The data meets every expectation.")


@check(
    id="schema",
    version="1.0.0",
    kind="analysis",
    # Every column, so a column that is missing is known to be missing.
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    params=SchemaParams,
    output=SchemaOutput,
    requires=Requires(profile=["counts"], params=["columns"]),
    # The profile keeps ten top values and a minimum and maximum, so allowed values
    # and the count of values out of range are read from the column; one pass each.
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["schema", "contract", "validation"],
)
def schema(ctx: CheckContext, params: SchemaParams) -> CheckOutput:
    """Compare the columns with the expected types, nulls, ranges and values."""
    if not params.columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_columns",
            "No schema is declared, so there is nothing to validate the data against.",
            severity=Severity.INFO,
            hint="Set the `columns` parameter to a mapping of column name to "
            "expectation.",
        )
    present = {column.name: column for column in ctx.profile.columns}
    missing = tuple(
        name
        for name, expected in params.columns.items()
        if expected.required and name not in present
    )
    if missing:
        ctx.finding(
            code="missing_column",
            subject=Subject.dataset(),
            severity=Severity.ERROR,
            confidence=Confidence.HIGH,
            title=f"Missing required column{'s' if len(missing) > 1 else ''}: "
            f"{', '.join(missing)}",
            message=(
                f"The data has no column named {', '.join(map(repr, missing))}, "
                "which the schema requires."
            ),
            evidence=Evidence(
                metrics={"missing": missing, "n_missing": len(missing)},
                n_used=len(params.columns) - len(missing),
                n_total=len(params.columns),
                method="required_columns",
            ),
        )
    violations: dict[str, list[str]] = {}
    for name, expected in params.columns.items():
        column = present.get(name)
        if column is None:
            continue
        for code in _check_column(ctx, column, expected):
            violations.setdefault(name, []).append(code)
    return ctx.output(
        SchemaOutput(
            checked=len(params.columns),
            missing_columns=missing,
            violations={name: tuple(codes) for name, codes in violations.items()},
            valid=not missing and not violations,
        )
    )


def _check_column(
    ctx: CheckContext, column: ColumnProfile, expected: ColumnExpectation
) -> list[str]:
    """Record a finding for each way ``column`` breaks ``expected``, and list them."""
    counts = column.counts
    found: list[str] = []

    def report(code: str, title: str, message: str, **metrics: object) -> None:
        found.append(code)
        ctx.finding(
            code=code,
            subject=Subject.column(column.name),
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=title,
            message=message,
            evidence=Evidence(
                metrics=metrics,  # type: ignore[arg-type]
                n_used=counts.non_null,
                n_total=counts.rows,
                method="schema",
            ),
        )

    name = column.name
    if expected.type is not None and column.semantic_type != expected.type:
        report(
            "type_mismatch",
            f"Expected {expected.type}, found {column.semantic_type}",
            f"{name!r} should be {expected.type} but the profile types it "
            f"{column.semantic_type}.",
            expected=expected.type,
            actual=column.semantic_type,
        )
    if expected.dtype is not None and (
        column.physical_dtype.lower() != expected.dtype.lower()
    ):
        report(
            "dtype_mismatch",
            f"Expected {expected.dtype}, found {column.physical_dtype}",
            f"{name!r} should be stored as {expected.dtype} but is "
            f"{column.physical_dtype}.",
            expected=expected.dtype,
            actual=column.physical_dtype,
        )
    if expected.nullable is False and counts.nulls:
        report(
            "unexpected_nulls",
            f"{counts.nulls:,} null values in a column that allows none",
            f"{name!r} must have no nulls but {counts.nulls:,} of its "
            f"{counts.rows:,} values are null.",
            null_count=counts.nulls,
        )
    if expected.min is not None or expected.max is not None:
        _check_range(ctx, column, expected, report)
    if expected.allowed is not None:
        _check_allowed(ctx, column, expected.allowed, report)
    return found


def _check_range(
    ctx: CheckContext,
    column: ColumnProfile,
    expected: ColumnExpectation,
    report: Report,
) -> None:
    stats = column.stats if isinstance(column, NumericProfile) else None
    if not isinstance(column, NumericProfile):
        ctx.note(
            "range_on_non_numeric",
            f"{column.name!r} is {column.semantic_type}, so its min and max were "
            "not checked.",
            severity=Severity.INFO,
        )
        return
    if stats is None or stats.min is None or stats.max is None:
        return
    below = expected.min is not None and stats.min < expected.min
    above = expected.max is not None and stats.max > expected.max
    if not (below or above):
        return
    values = ctx.values(column.name)
    if below:
        count = int(np.count_nonzero(values < expected.min))
        report(
            "below_min",
            f"{count:,} values below {expected.min:g}",
            f"{count:,} values of {column.name!r} are below the minimum "
            f"{expected.min:g}; the lowest is {stats.min:g}.",
            bound=expected.min,
            observed=stats.min,
            count=count,
        )
    if above:
        count = int(np.count_nonzero(values > expected.max))
        report(
            "above_max",
            f"{count:,} values above {expected.max:g}",
            f"{count:,} values of {column.name!r} are above the maximum "
            f"{expected.max:g}; the highest is {stats.max:g}.",
            bound=expected.max,
            observed=stats.max,
            count=count,
        )


def _check_allowed(
    ctx: CheckContext,
    column: ColumnProfile,
    allowed: tuple[bool | int | float | str, ...],
    report: Report,
) -> None:
    allowed_set = set(allowed)
    seen = Counter(ctx.values(column.name).tolist())
    outside = Counter({v: n for v, n in seen.items() if v not in allowed_set})
    if not outside:
        return
    rows = sum(outside.values())
    examples = tuple(str(value) for value, _ in outside.most_common(_EXAMPLES))
    report(
        "unexpected_values",
        f"{len(outside):,} values outside the allowed set",
        f"{column.name!r} holds {len(outside):,} values not in the allowed set "
        f"({rows:,} rows), such as {', '.join(map(repr, examples))}.",
        distinct_values=len(outside),
        rows=rows,
        examples=examples,
    )
