# dsbf/checks/duplicate_columns.py
"""Columns that hold the same values as another column."""

from collections import defaultdict

from pydantic import Field

from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    NoParams,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    Confidence,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class DuplicateColumnsOutput(CheckOutputModel):
    groups: list[list[str]] = Field(
        description="Each set of columns equal to one another, in dataset order."
    )
    pairs: int = Field(ge=0, description="Pairs of equal columns in all the groups.")


@check(
    id="duplicate_columns",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.REDUNDANCY],
    params=NoParams,
    output=DuplicateColumnsOutput,
    requires=Requires(profile=["counts", "content_hash"]),
    cost=Cost.CHEAP,
    codes=["duplicate_columns"],
    packs=["core"],
    tags=["duplicates", "redundancy"],
)
def duplicate_columns(ctx: CheckContext, params: NoParams) -> CheckOutput:
    """Group the columns that have the same values in the same rows.

    Two columns are equal when their 64-bit content digests and storage types are. The
    digest follows row order, so the same values in another order do not match. A
    chance match between two columns is about one in 2^64.
    """
    members: dict[tuple[str | None, str], list[str]] = defaultdict(list)
    for column in ctx.profile.columns:
        members[(column.content_hash, column.physical_dtype)].append(column.name)
    groups = [names for names in members.values() if len(names) > 1]
    groups.sort(key=lambda names: ctx.profile.column(names[0]).position)
    pairs = 0
    for names in groups:
        size = len(names)
        pairs += size * (size - 1) // 2
        rows = ctx.profile.n_rows
        ctx.finding(
            code="duplicate_columns",
            subject=Subject.column_group(*names),
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"{size} columns hold the same values",
            message=(
                f"{', '.join(repr(name) for name in names)} have the same values "
                "in every row, by their content digests. Keep one and drop the "
                "rest, or find out why they were copied."
            ),
            evidence=Evidence(
                metrics={"n_columns": size, "n_pairs": size * (size - 1) // 2},
                n_used=rows,
                n_total=rows,
                method="content_hash",
            ),
        )
    return ctx.output(DuplicateColumnsOutput(groups=groups, pairs=pairs))
