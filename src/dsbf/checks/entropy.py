# dsbf/checks/entropy.py
"""How evenly the values of a categorical, boolean or text column are spread."""

import math

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
    BooleanProfile,
    CategoricalProfile,
    CheckParams,
    Contract,
    Cost,
    Dimension,
    TextProfile,
)


class EntropyParams(CheckParams):
    min_rows: int = Field(
        30,
        ge=1,
        description="Non-null values below which the estimates are called unstable: "
        "rare values may be missing, so the entropy is underestimated.",
    )


class ColumnEntropy(Contract):
    entropy_bits: float = Field(ge=0, description="Shannon entropy, in bits.")
    max_bits: float = Field(
        ge=0, description="The most it could be: log2 of the distinct values."
    )
    evenness: float = Field(
        ge=0,
        le=1,
        description="Entropy over its maximum: 0 when one value dominates, 1 when "
        "all are equally likely (and 1 for a column of one value).",
    )
    distinct: int = Field(ge=1)
    non_null: int = Field(ge=1)


class EntropyOutput(CheckOutputModel):
    columns: dict[str, ColumnEntropy] = Field(
        description="Each categorical, boolean or text column with values, by name."
    )


@check(
    id="entropy",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("categorical", "text", "boolean"),
    dimensions=[Dimension.DISTRIBUTION],
    params=EntropyParams,
    output=EntropyOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["information", "distribution"],
)
def entropy(ctx: CheckContext, params: EntropyParams) -> CheckOutput:
    """Report the Shannon entropy of each categorical, boolean and text column.

    This raises no finding: a skewed column is `dominant_value`'s to report, and a
    column of one value is `constant_columns`'s. The values feed the distribution view.
    """
    columns: dict[str, ColumnEntropy] = {}
    for column in ctx.profile.columns:
        if not isinstance(column, (BooleanProfile, CategoricalProfile, TextProfile)):
            continue
        bits, counts = column.entropy_bits, column.counts
        if bits is None or counts.non_null == 0:
            continue
        top = math.log2(counts.distinct)
        columns[column.name] = ColumnEntropy(
            entropy_bits=bits,
            max_bits=top,
            evenness=min(bits / top, 1.0) if top > 0 else 1.0,
            distinct=counts.distinct,
            non_null=counts.non_null,
        )
    if columns and ctx.profile.n_rows < params.min_rows:
        ctx.note(
            "small_sample",
            f"The dataset has {ctx.profile.n_rows} rows, under {params.min_rows}: "
            "entropy estimates are unstable, and rare values may be missing, "
            "which makes the entropy too low.",
            hint="Read the entropy values cautiously, or confirm them with a resample.",
        )
    return ctx.output(EntropyOutput(columns=columns))
