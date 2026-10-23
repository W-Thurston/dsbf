# dsbf/checks/categorical_encoding.py
"""How to turn a categorical column into numbers for a model."""

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
    Action,
    CategoricalProfile,
    CheckParams,
    CheckStatus,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Guidance,
    Severity,
    Subject,
)

Strategy = Literal["binary", "one_hot", "frequency"]

# The `method` of the `encode` actions this check emits; the export that turns
# guidance into a pipeline reads them. `target` is an extra action beside a column's
# strategy, never a strategy of its own.
ENCODE_METHODS: tuple[str, ...] = ("binary", "one_hot", "frequency", "target")


class CategoricalEncodingParams(CheckParams):
    max_one_hot: int = Field(
        10,
        ge=2,
        description="Most distinct values for which one-hot columns are suggested.",
    )
    min_rows_per_level: int = Field(
        20,
        ge=1,
        description="Rows per distinct value below which one-hot columns would be "
        "too sparse to estimate, so the frequency of each value is suggested.",
    )
    target_folds: int = Field(
        5,
        ge=2,
        description="Folds of the out-of-fold target encoding suggested for a "
        "column that predicts the target.",
    )
    target_smoothing: float = Field(
        20,
        gt=0,
        description="Rows of prior the target encoding adds to each value, which "
        "pulls the mean of a rare value toward the overall mean.",
    )


class ColumnEncoding(Contract):
    cardinality: int = Field(ge=2, description="Distinct non-null values.")
    rows_per_level: float = Field(gt=0, description="Non-null rows per distinct value.")
    strategy: Strategy
    rarest_level_count: int | None = Field(
        description="Rows of the least frequent value; known only when the profile "
        "lists every value."
    )
    target_encoding: bool = Field(
        description="An out-of-fold target encoding is suggested too: the column "
        "has a frequency encoding, `mutual_information` finds it informative about "
        "the target, and `target_leakage` does not flag it."
    )


class CategoricalEncodingOutput(CheckOutputModel):
    columns: dict[str, ColumnEncoding] = Field(
        description="Each categorical column with at least two values, by name."
    )


def _strategy(
    distinct: int, rows_per_level: float, params: CategoricalEncodingParams
) -> tuple[Strategy, str]:
    """The encoding for a column and why, in words."""
    if distinct == 2:
        return "binary", "it has two values, which one 0/1 column holds"
    if distinct > params.max_one_hot:
        return (
            "frequency",
            f"it has {distinct} values, more than {params.max_one_hot} one-hot "
            "columns can carry without a wide, sparse matrix",
        )
    if rows_per_level < params.min_rows_per_level:
        return (
            "frequency",
            f"each value has only {rows_per_level:.0f} rows on average, fewer than "
            f"the {params.min_rows_per_level} a one-hot column needs to be estimated",
        )
    return "one_hot", f"it has {distinct} values, few enough for one column each"


_DETAIL = {
    "binary": "Map the two values to 0 and 1.",
    "one_hot": "Make one 0/1 column per value and drop one, so the columns are not "
    "collinear.",
    "frequency": "Replace each value by its share of the rows, fitted on the "
    "training split. Frequency is not a ranking: a model can mistake it for one.",
}


@check(
    id="categorical_encoding",
    version="1.1.0",
    kind="rule",
    applies_to=SemanticTypes("categorical"),
    dimensions=[Dimension.USABILITY],
    params=CategoricalEncodingParams,
    output=CategoricalEncodingOutput,
    requires=Requires(profile=["counts"]),
    after=["mutual_information", "target_leakage"],
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["encoding", "ml"],
)
def categorical_encoding(
    ctx: CheckContext, params: CategoricalEncodingParams
) -> CheckOutput:
    """Suggest an encoding for each categorical column from its cardinality."""
    columns: dict[str, ColumnEncoding] = {}
    predictive, leaking = _target_evidence(ctx)
    for column in ctx.profile.columns:
        if not isinstance(column, CategoricalProfile):
            continue
        counts = column.counts
        if counts.distinct < 2:  # `constant_columns` reports these
            continue
        rows_per_level = counts.non_null / counts.distinct
        strategy, reason = _strategy(counts.distinct, rows_per_level, params)
        listed = len(column.top_values) >= counts.distinct
        rarest = min((v.count for v in column.top_values), default=None)
        with_target = (
            strategy == "frequency"
            and column.name in predictive
            and column.name not in leaking
        )
        columns[column.name] = ColumnEncoding(
            cardinality=counts.distinct,
            rows_per_level=rows_per_level,
            strategy=strategy,
            rarest_level_count=rarest if listed else None,
            target_encoding=with_target,
        )
        ctx.finding(
            code="encoding_strategy",
            subject=Subject.column(column.name),
            severity=Severity.INFO,
            confidence=Confidence.HIGH if counts.distinct_exact else Confidence.LOW,
            title=f"Encode as {strategy.replace('_', '-')} ({counts.distinct} values)",
            message=(
                f"{column.name!r} has {counts.distinct:,} distinct values in "
                f"{counts.non_null:,} rows ({rows_per_level:,.0f} rows per value); "
                f"{reason}. A tree-based model can read integer codes directly; "
                "linear, distance-based and neural models need the encoding."
                + (
                    ""
                    if counts.distinct_exact
                    else " The count of values is an estimate."
                )
            ),
            evidence=Evidence(
                metrics={
                    "cardinality": counts.distinct,
                    "rows_per_level": rows_per_level,
                    "rarest_level_count": rarest if listed else None,
                    "strategy": strategy,
                    "target_encoding": with_target,
                },
                n_used=counts.non_null,
                n_total=counts.rows,
                method="cardinality_rows_per_level",
            ),
            guidance=Guidance(
                phase="ml",
                actions=(
                    Action(
                        action="encode",
                        columns=(column.name,),
                        method=strategy,
                        condition=reason,
                        detail=_DETAIL[strategy],
                        params={"n_levels": counts.distinct},
                    ),
                    *(
                        (_target_action(column.name, predictive[column.name], params),)
                        if with_target
                        else ()
                    ),
                ),
            ),
        )
    return ctx.output(CategoricalEncodingOutput(columns=columns))


def _target_evidence(ctx: CheckContext) -> tuple[dict[str, float], set[str]]:
    """Columns informative about the target, and columns flagged as leaking it.

    Target encoding needs a target that is binary or numeric (a column per class
    would be needed otherwise), so a target of more classes yields no column.
    """
    informative: dict[str, float] = {}
    information = ctx.result("mutual_information")
    if (
        information is not None
        and information.status is CheckStatus.OK
        and information.output is not None
        and information.output["target_classes"] in (None, 2)
    ):
        informative = {
            f["feature"]: f["normalised"]
            for f in information.output["features"]
            if f["notable"]
        }
    leaking: set[str] = set()
    leakage = ctx.result("target_leakage")
    if (
        leakage is not None
        and leakage.status is CheckStatus.OK
        and leakage.output is not None
    ):
        leaking = {f["feature"] for f in leakage.output["features"] if f["flagged"]}
    return informative, leaking


def _target_action(
    name: str, normalised: float, params: CategoricalEncodingParams
) -> Action:
    return Action(
        action="encode",
        columns=(name,),
        method="target",
        condition=(
            f"{name!r} is informative about the target (mutual information "
            f"{normalised:.1%} of its uncertainty)"
        ),
        detail=(
            "Replace each value by the mean of the target for it, smoothed toward "
            "the overall mean, and computed out of fold: each row is encoded from "
            "the folds that exclude it, fitted on the training split only. An "
            "encoding fitted on a row's own target leaks it."
        ),
        params={"folds": params.target_folds, "smoothing": params.target_smoothing},
    )
