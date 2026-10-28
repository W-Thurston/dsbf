# dsbf/checks/mixed_types.py
"""String columns that hold more than one kind of value."""

import polars as pl
from pydantic import Field

from dsbf.checks.strings import (
    STRING_TYPES,
    distinct_values,
    examples,
    holds_strings,
    value_kind,
)
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
from dsbf.profiling.types import DATE_FORMATS

# Words are words: a column of true, false and maybe is a label column, not a mix.
_PLAIN_KINDS = frozenset({"boolean", "text"})


class MixedTypesParams(CheckParams):
    min_share: float = Field(
        0.05,
        gt=0,
        le=0.5,
        description="Share of the values a kind needs to count as present.",
    )
    ignore_null_tokens: bool = Field(
        True,
        description="Leave out values that stand for a missing value (`NA`, `null`, "
        "`none`, `-`, an empty or blank string). Off, they are a kind of their own.",
    )
    min_values: int = Field(10, ge=1, description="Values a column needs to be read.")
    max_examples: int = Field(3, ge=1, le=10, description="Examples listed per kind.")


class KindCount(Contract):
    rows: int = Field(ge=1)
    share: float = Field(gt=0, le=1)
    examples: tuple[str, ...]


class MixedColumn(Contract):
    n_used: int = Field(ge=1, description="Values counted, null tokens excluded.")
    kinds: dict[str, KindCount] = Field(description="The kinds present, by name.")
    source: str = Field(
        description="`content` if the values mix kinds, `load` if the column held "
        "mixed Python types when it was loaded, `both` if both."
    )


class MixedTypesOutput(CheckOutputModel):
    columns: dict[str, MixedColumn] = Field(
        description="Each column that mixes kinds, by name."
    )
    columns_read: int = Field(ge=0)
    skipped_columns: dict[str, str] = Field(
        description="Columns left out, with the reason."
    )


@check(
    id="mixed_types",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes(*STRING_TYPES),
    dimensions=[Dimension.VALIDITY],
    params=MixedTypesParams,
    output=MixedTypesOutput,
    requires=Requires(profile=["counts"]),
    # Regular expressions, and the date formats for values that could be dates.
    cost=Cost.CHEAP,
    codes=["mixed_value_types"],
    packs=["core"],
    tags=["type", "format"],
)
def mixed_types(ctx: CheckContext, params: MixedTypesParams) -> CheckOutput:
    """Report columns that mix numbers, dates, booleans and words."""
    columns: dict[str, MixedColumn] = {}
    skipped: dict[str, str] = {}
    coerced = ctx.coerced
    read = 0
    for name in ctx.columns:
        ctx.checkpoint()
        profile = ctx.profile.column(name)
        if not holds_strings(profile):
            skipped[name] = f"stored as {profile.physical_dtype}, not as strings"
            continue
        if profile.counts.non_null < params.min_values:
            skipped[name] = f"fewer than {params.min_values} values"
            continue
        read += 1
        found = _read(ctx, name, params, loaded_mixed=name in coerced)
        if found is not None:
            columns[name] = found
            _report(ctx, name, found, coerced.get(name))
    if not read:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_strings",
            f"No column in scope holds at least {params.min_values} non-null strings.",
        )
    return ctx.output(
        MixedTypesOutput(columns=columns, columns_read=read, skipped_columns=skipped)
    )


def _read(
    ctx: CheckContext, name: str, params: MixedTypesParams, *, loaded_mixed: bool
) -> MixedColumn | None:
    kinded = value_kind(distinct_values(ctx.values(name)), DATE_FORMATS)
    if params.ignore_null_tokens:
        kinded = kinded.filter(pl.col("kind") != "null_token")
    total = int(kinded["count"].sum())
    if not total:
        return None
    rows = (
        kinded.group_by("kind")
        .agg(rows=pl.col("count").sum())
        .sort("rows", descending=True)
    )
    present = {
        row["kind"]: row["rows"]
        for row in rows.iter_rows(named=True)
        if row["rows"] / total >= params.min_share
    }
    by_content = len(present) >= 2 and not set(present) <= _PLAIN_KINDS
    if not (by_content or loaded_mixed):
        return None
    shown = (
        {row["kind"]: row["rows"] for row in rows.iter_rows(named=True)}
        if not by_content
        else present
    )
    return MixedColumn(
        n_used=total,
        kinds={
            kind: KindCount(
                rows=count,
                share=count / total,
                examples=tuple(
                    examples(kinded, pl.col("kind") == kind, params.max_examples)
                ),
            )
            for kind, count in shown.items()
        },
        source="both"
        if by_content and loaded_mixed
        else "content"
        if by_content
        else "load",
    )


def _report(
    ctx: CheckContext, name: str, found: MixedColumn, loaded: str | None
) -> None:
    ranked = sorted(found.kinds.items(), key=lambda item: -item[1].rows)
    summary = ", ".join(
        f"{kind} {count.share:.0%} (e.g. {_quoted(count.examples)})"
        for kind, count in ranked
    )
    from_load = (
        " It was loaded from a column of mixed Python types and converted to text."
        if loaded
        else ""
    )
    ctx.finding(
        code="mixed_value_types",
        subject=Subject.column(name),
        severity=Severity.WARN,
        confidence=Confidence.HIGH
        if found.source != "content"
        else Confidence.MODERATE,
        title=f"Mixes {len(found.kinds)} kinds of value",
        message=(
            f"{name!r} holds more than one kind of value: {summary}.{from_load} "
            "Sums, sorts and casts on a column like this fail or give wrong "
            "results; look at the minority kind to see whether it is an entry "
            "error, a placeholder or a different field."
        ),
        evidence=Evidence(
            metrics={
                "kinds": len(found.kinds),
                "source": found.source,
                **{f"share_{kind}": c.share for kind, c in found.kinds.items()},
            },
            n_used=found.n_used,
            n_total=ctx.profile.n_rows,
            method="value_kinds" if found.source != "load" else "mixed_on_load",
        ),
    )


def _quoted(values: tuple[str, ...], limit: int = 2) -> str:
    return ", ".join(repr(value) for value in values[:limit])
