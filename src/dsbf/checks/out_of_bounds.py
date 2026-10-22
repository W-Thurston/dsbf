# dsbf/checks/out_of_bounds.py
"""Numeric values outside the range the caller declares for their column."""

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
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class Bounds(Contract):
    """The range a column's values belong in; either end may be open."""

    min: float | None = Field(None, description="No value may be below this.")
    max: float | None = Field(None, description="No value may be above this.")

    @model_validator(mode="after")
    def _ordered(self) -> "Bounds":
        if self.min is None and self.max is None:
            raise ValueError("set min, max or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min is at most max")
        return self


class OutOfBoundsParams(CheckParams):
    bounds: dict[str, Bounds] = Field(
        default_factory=dict,
        description="The range of each column, by name. With none, the check is not "
        "applicable: a column's name does not say what range it should hold.",
    )


class ColumnBounds(Contract):
    below: int = Field(ge=0, description="Values under `min`.")
    above: int = Field(ge=0, description="Values over `max`.")
    lowest: float | None = Field(description="The lowest value under `min`.")
    highest: float | None = Field(description="The highest value over `max`.")


class OutOfBoundsOutput(CheckOutputModel):
    columns: dict[str, ColumnBounds] = Field(
        description="Every numeric column that has bounds, by name."
    )
    unchecked: dict[str, str] = Field(
        description="Columns with bounds that were not checked, with the reason."
    )


@check(
    id="out_of_bounds",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.VALIDITY],
    params=OutOfBoundsParams,
    output=OutOfBoundsOutput,
    requires=Requires(profile=["counts"]),
    # One comparison of each bounded column's values with its bounds.
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["bounds", "validation"],
)
def out_of_bounds(ctx: CheckContext, params: OutOfBoundsParams) -> CheckOutput:
    """Count the values of each bounded numeric column outside its range."""
    if not params.bounds:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_bounds",
            "No bounds are declared, so there is nothing to compare the values with.",
            severity=Severity.INFO,
            hint="Set the `bounds` parameter to a mapping of column name to "
            "{min, max}.",
        )
    in_scope = set(ctx.columns)
    columns: dict[str, ColumnBounds] = {}
    unchecked: dict[str, str] = {}
    for name, bounds in params.bounds.items():
        if name not in in_scope:
            reason = "not a numeric column of the data"
            unchecked[name] = reason
            ctx.note(
                "bounds_not_checked",
                f"{name!r} has bounds but is {reason}, so they were not checked.",
            )
            continue
        values = ctx.values(name)
        below_mask = values < bounds.min if bounds.min is not None else None
        above_mask = values > bounds.max if bounds.max is not None else None
        below = int(np.count_nonzero(below_mask)) if below_mask is not None else 0
        above = int(np.count_nonzero(above_mask)) if above_mask is not None else 0
        lowest = _extreme(values[below_mask], np.min) if below else None
        highest = _extreme(values[above_mask], np.max) if above else None
        columns[name] = ColumnBounds(
            below=below, above=above, lowest=lowest, highest=highest
        )
        if not below + above:
            continue
        ctx.finding(
            code="out_of_bounds",
            subject=Subject.column(name),
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"Out-of-bounds values ({below + above:,} rows)",
            message=_message(name, bounds, below, above, lowest, highest),
            evidence=Evidence(
                metrics={
                    "violation_count": below + above,
                    "below_count": below,
                    "above_count": above,
                    "lowest_below": lowest,
                    "highest_above": highest,
                    "min": bounds.min,
                    "max": bounds.max,
                },
                n_used=int(values.size),
                n_total=ctx.profile.n_rows,
                method="declared_bounds",
            ),
        )
    return ctx.output(OutOfBoundsOutput(columns=columns, unchecked=unchecked))


def _extreme(violations: np.ndarray, reduce) -> float | None:
    """The farthest violation, leaving out infinities (None if only they are left)."""
    finite = violations[np.isfinite(violations)]
    return float(reduce(finite)) if finite.size else None


def _message(
    name: str,
    bounds: Bounds,
    below: int,
    above: int,
    lowest: float | None,
    highest: float | None,
) -> str:
    parts = []
    if below:
        reached = f" (lowest {lowest:g})" if lowest is not None else ""
        parts.append(f"{below:,} below the minimum {bounds.min:g}{reached}")
    if above:
        reached = f" (highest {highest:g})" if highest is not None else ""
        parts.append(f"{above:,} above the maximum {bounds.max:g}{reached}")
    return (
        f"{name!r} has {' and '.join(parts)}. Values outside the declared range can "
        "be data entry errors, a unit mismatch such as a temperature in the wrong "
        "scale, or real cases the range does not allow for; find out which before "
        "treating them as valid."
    )
