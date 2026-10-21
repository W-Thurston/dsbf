# dsbf/checks/dominant_value.py
"""Categorical and boolean columns in which one value is most of the rows."""

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
    BooleanProfile,
    CategoricalProfile,
    CheckParams,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class DominantValueParams(CheckParams):
    notice_share: float = Field(
        0.70, gt=0, le=1, description="Share from which a value is reported."
    )
    warn_share: float = Field(
        0.90, gt=0, le=1, description="Share from which the finding is a warning."
    )
    flag_share: float = Field(
        0.95, gt=0, le=1, description="Share from which a column is flagged."
    )

    @model_validator(mode="after")
    def _ordered(self) -> "DominantValueParams":
        if not self.notice_share <= self.warn_share <= self.flag_share:
            raise ValueError("notice_share <= warn_share <= flag_share")
        return self


class ColumnDominance(Contract):
    top_value: str = Field(description="The most frequent value, as text.")
    top_count: int = Field(ge=1)
    non_null: int = Field(ge=1)
    top_share: float = Field(gt=0, le=1)
    distinct: int = Field(ge=1)
    flagged: bool = Field(description="The share is at least `flag_share`.")


class DominantValueOutput(CheckOutputModel):
    columns: dict[str, ColumnDominance] = Field(
        description="Each categorical or boolean column whose most frequent value "
        "is known, by name."
    )


def _top(column: CategoricalProfile | BooleanProfile) -> tuple[str, int] | None:
    """The most frequent value and its count, or None if the profile lacks it.

    A categorical column keeps no top value when its values are too long to list,
    so it is skipped, not read as having no dominant value.
    """
    if isinstance(column, BooleanProfile):
        if column.true_count == column.false_count == 0:
            return None
        if column.true_count >= column.false_count:
            return "True", column.true_count
        return "False", column.false_count
    if not column.top_values:
        return None
    first = column.top_values[0]
    return str(first.value), first.count


@check(
    id="dominant_value",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("categorical", "boolean"),
    dimensions=[Dimension.DISTRIBUTION],
    params=DominantValueParams,
    output=DominantValueOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["dominant-value", "imbalance"],
)
def dominant_value(ctx: CheckContext, params: DominantValueParams) -> CheckOutput:
    """Flag columns in which the most frequent value is most of the non-null rows."""
    columns: dict[str, ColumnDominance] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, (CategoricalProfile, BooleanProfile)):
            continue
        top = _top(column)
        counts = column.counts
        if top is None or counts.non_null == 0:
            continue
        value, count = top
        share = count / counts.non_null
        columns[column.name] = ColumnDominance(
            top_value=value,
            top_count=count,
            non_null=counts.non_null,
            top_share=share,
            distinct=counts.distinct,
            flagged=share >= params.flag_share,
        )
        # A column of one value is the constant check's.
        if share < params.notice_share or counts.distinct <= 1:
            continue
        severity = Severity.WARN if share >= params.warn_share else Severity.INFO
        ctx.finding(
            code="dominant_value",
            subject=Subject.column(column.name),
            severity=severity,
            confidence=Confidence.HIGH,
            title=f"Dominant value {value!r} ({share:.1%})",
            message=(
                f"{value!r} is {count:,} of the {counts.non_null:,} non-null values "
                f"of {column.name!r} ({share:.1%}), among {counts.distinct:,} "
                "distinct values. Little is left to tell rows apart, and a model "
                "that predicts this column does well by always giving the common "
                "value. Check whether it is a default value, and stratify any "
                "split on it."
            ),
            evidence=Evidence(
                metrics={
                    "top_count": count,
                    "top_share": share,
                    "n_unique": counts.distinct,
                },
                n_used=counts.non_null,
                n_total=counts.rows,
                method="top_value_share",
            ),
        )
    return ctx.output(DominantValueOutput(columns=columns))
