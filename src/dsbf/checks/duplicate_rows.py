# dsbf/checks/duplicate_rows.py
"""Rows that repeat an earlier row."""

from pydantic import Field

from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    check,
)
from dsbf.model import (
    CheckParams,
    Confidence,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class DuplicateRowsParams(CheckParams):
    warn_share: float = Field(
        0.01,
        ge=0,
        le=1,
        description="Share of rows above which the finding is a warning.",
    )


class DuplicateRowsOutput(CheckOutputModel):
    duplicate_rows: int = Field(ge=0, description="Rows equal to an earlier row.")
    rows: int = Field(ge=0)
    duplicate_share: float = Field(ge=0, le=1)


@check(
    id="duplicate_rows",
    version="1.0.0",
    kind="rule",
    applies_to=None,
    dimensions=[Dimension.VALIDITY],
    params=DuplicateRowsParams,
    output=DuplicateRowsOutput,
    requires=Requires(profile=["duplicate_rows"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["duplicates", "rows"],
)
def duplicate_rows(ctx: CheckContext, params: DuplicateRowsParams) -> CheckOutput:
    """Report the rows that repeat an earlier row, in every column."""
    rows = ctx.profile.n_rows
    # `requires` keeps the check from running when the profile lacks the count.
    count = ctx.profile.duplicate_rows or 0
    share = count / rows if rows else 0.0
    if count:
        ctx.finding(
            code="duplicate_rows",
            subject=Subject.dataset(),
            severity=Severity.WARN if share > params.warn_share else Severity.INFO,
            confidence=Confidence.HIGH,
            title=f"Duplicate rows ({count:,}, {share:.1%})",
            message=(
                f"{count:,} of {rows:,} rows ({share:.1%}) repeat an earlier row in "
                "every column. They can come from loading overlapping batches, a "
                "join that multiplied rows, or a sample drawn with replacement. "
                "Decide whether they are repeated observations or an artefact "
                "before counting them twice."
            ),
            evidence=Evidence(
                metrics={"duplicate_rows": count, "duplicate_share": share},
                n_used=rows,
                n_total=rows,
                method="repeated_rows",
            ),
        )
    return ctx.output(
        DuplicateRowsOutput(duplicate_rows=count, rows=rows, duplicate_share=share)
    )
