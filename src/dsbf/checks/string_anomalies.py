# dsbf/checks/string_anomalies.py
"""Strings that differ from each other in ways no reader can see."""

import numpy as np
import polars as pl
from pydantic import Field

from dsbf.checks.strings import (
    INVISIBLE,
    STRING_TYPES,
    distinct_values,
    examples,
    holds_strings,
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

# Free text is long-tailed and has no casing to agree on, so it gets neither the case
# nor the length test; whitespace and invisible characters are mistakes in any string.
_LABEL_TYPES = ("categorical", "identifier")
_EDGE_SPACE = r"^\s|\s$"


class StringAnomaliesParams(CheckParams):
    min_values: int = Field(
        10, ge=1, description="Non-null values a column needs to be read."
    )
    fence: float = Field(
        3.0,
        gt=0,
        description="Multiples of the interquartile range beyond the quartiles of the "
        "value lengths from which a length is an outlier.",
    )
    max_examples: int = Field(
        5, ge=1, le=20, description="Examples listed per anomaly."
    )


class CaseGroup(Contract):
    folded: str = Field(description="The lower-cased form the variants share.")
    variants: tuple[str, ...] = Field(description="The spellings, most frequent first.")
    rows: int = Field(ge=2)


class MixedCase(Contract):
    groups: int = Field(ge=1, description="Sets of values that differ only by case.")
    values: int = Field(ge=2, description="Distinct values in those sets.")
    rows: int = Field(ge=2, description="Rows holding a value of such a set.")
    minority_rows: int = Field(
        ge=1,
        description="Rows that do not hold the most frequent spelling of their set.",
    )
    examples: tuple[CaseGroup, ...] = Field(description="The sets with most rows.")


class Counted(Contract):
    rows: int = Field(ge=1)
    values: int = Field(ge=1, description="Distinct values affected.")
    examples: tuple[str, ...] = Field(
        description="The most frequent affected values; invisible characters are "
        "shown as escapes."
    )


class LengthOutliers(Contract):
    rows: int = Field(ge=1)
    values: int = Field(ge=1)
    lower_fence: float = Field(ge=0)
    upper_fence: float = Field(ge=0)
    examples: tuple[str, ...]


class ColumnAnomalies(Contract):
    n_used: int = Field(ge=1, description="Non-null values read.")
    mixed_case: MixedCase | None = None
    edge_whitespace: Counted | None = None
    invisible_characters: Counted | None = None
    length_outliers: LengthOutliers | None = None


class StringAnomaliesOutput(CheckOutputModel):
    columns: dict[str, ColumnAnomalies] = Field(
        description="Each column with at least one anomaly, by name."
    )
    columns_read: int = Field(ge=0)
    skipped_columns: dict[str, str] = Field(
        description="Columns left out, with the reason."
    )


@check(
    id="string_anomalies",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes(*STRING_TYPES),
    dimensions=[Dimension.VALIDITY],
    params=StringAnomaliesParams,
    output=StringAnomaliesOutput,
    requires=Requires(profile=["counts"]),
    # Polars string operations over each column's distinct values.
    cost=Cost.CHEAP,
    codes=["edge_whitespace", "invisible_characters", "length_outliers", "mixed_case"],
    packs=["core"],
    tags=["string", "whitespace", "case", "unicode"],
)
def string_anomalies(ctx: CheckContext, params: StringAnomaliesParams) -> CheckOutput:
    """Find values that differ only by case, padding, hidden characters or length."""
    columns: dict[str, ColumnAnomalies] = {}
    skipped: dict[str, str] = {}
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
        distinct = distinct_values(ctx.values(name))
        n_used = int(distinct["count"].sum())
        label = profile.semantic_type in _LABEL_TYPES
        found = ColumnAnomalies(
            n_used=n_used,
            mixed_case=_mixed_case(distinct, params) if label else None,
            edge_whitespace=_counted(
                distinct,
                pl.col("value").str.contains(_EDGE_SPACE),
                params,
                escape=False,
            ),
            invisible_characters=_counted(
                distinct, pl.col("value").str.contains(INVISIBLE), params, escape=True
            ),
            length_outliers=_length_outliers(distinct, params) if label else None,
        )
        if any(
            (
                found.mixed_case,
                found.edge_whitespace,
                found.invisible_characters,
                found.length_outliers,
            )
        ):
            columns[name] = found
            _report(ctx, name, found)
    if not read:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_strings",
            f"No column in scope holds at least {params.min_values} non-null strings.",
        )
    return ctx.output(
        StringAnomaliesOutput(
            columns=columns, columns_read=read, skipped_columns=skipped
        )
    )


def _mixed_case(
    distinct: pl.DataFrame, params: StringAnomaliesParams
) -> MixedCase | None:
    folded = distinct.with_columns(folded=pl.col("value").str.to_lowercase())
    groups = (
        folded.group_by("folded")
        .agg(
            pl.col("value").sort_by(["count", "value"], descending=[True, False]),
            rows=pl.col("count").sum(),
            top=pl.col("count").max(),
            n=pl.len(),
        )
        .filter(pl.col("n") > 1)
        .sort(["rows", "folded"], descending=[True, False])
    )
    if groups.is_empty():
        return None
    return MixedCase(
        groups=groups.height,
        values=int(groups["n"].sum()),
        rows=int(groups["rows"].sum()),
        minority_rows=int((groups["rows"] - groups["top"]).sum()),
        examples=tuple(
            CaseGroup(
                folded=row["folded"], variants=tuple(row["value"][:5]), rows=row["rows"]
            )
            for row in groups.head(params.max_examples).iter_rows(named=True)
        ),
    )


def _counted(
    distinct: pl.DataFrame,
    matches: pl.Expr,
    params: StringAnomaliesParams,
    *,
    escape: bool,
) -> Counted | None:
    affected = distinct.filter(matches)
    if affected.is_empty():
        return None
    shown = examples(distinct, matches, params.max_examples)
    if escape:
        shown = [value.encode("unicode_escape").decode("ascii") for value in shown]
    return Counted(
        rows=int(affected["count"].sum()),
        values=affected.height,
        examples=tuple(shown),
    )


def _length_outliers(
    distinct: pl.DataFrame, params: StringAnomaliesParams
) -> LengthOutliers | None:
    lengths = distinct.with_columns(length=pl.col("value").str.len_chars())
    q1, q3 = _weighted_quartiles(
        lengths["length"].to_numpy(), lengths["count"].to_numpy()
    )
    # When most values share a length the quartiles meet, and a value of any other
    # length would be an outlier; one character is the least spread a length has.
    spread = max(q3 - q1, 1.0)
    upper = q3 + params.fence * spread
    lower = max(q1 - params.fence * spread, 0.0)
    far = (pl.col("length") > upper) | (pl.col("length") < lower)
    affected = lengths.filter(far)
    if affected.is_empty():
        return None
    return LengthOutliers(
        rows=int(affected["count"].sum()),
        values=affected.height,
        lower_fence=lower,
        upper_fence=upper,
        examples=tuple(examples(lengths, far, params.max_examples)),
    )


def _weighted_quartiles(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """The first and third quartile of ``values`` repeated ``weights`` times.

    Linear interpolation, as ``numpy.quantile`` does on the repeated values.
    """
    order = np.argsort(values, kind="stable")
    sorted_values = values[order].astype(np.float64)
    ends = np.cumsum(weights[order])
    total = int(ends[-1])

    def at(position: int) -> float:
        return float(sorted_values[np.searchsorted(ends, position, side="right")])

    def quantile(q: float) -> float:
        position = q * (total - 1)
        low = int(np.floor(position))
        high = min(low + 1, total - 1)
        weight = position - low
        return at(low) * (1 - weight) + at(high) * weight

    return quantile(0.25), quantile(0.75)


def _report(ctx: CheckContext, name: str, found: ColumnAnomalies) -> None:
    subject = Subject.column(name)
    total = ctx.profile.n_rows
    n = found.n_used

    def evidence(method: str, **metrics: object) -> Evidence:
        return Evidence(metrics=metrics, n_used=n, n_total=total, method=method)  # type: ignore[arg-type]

    if found.mixed_case:
        case = found.mixed_case
        shown = "; ".join(
            f"{' / '.join(repr(v) for v in group.variants)}"
            for group in case.examples[:3]
        )
        ctx.finding(
            code="mixed_case",
            subject=subject,
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"{case.groups:,} sets of values differ only by case",
            message=(
                f"{name!r} holds {case.values:,} values that fall into {case.groups:,} "
                f"sets differing only by capitalisation ({shown}). They count as "
                f"separate categories, so counts and groupings are split; "
                f"{case.minority_rows:,} rows ({case.minority_rows / n:.1%}) do not "
                "hold the most frequent spelling of their set."
            ),
            evidence=evidence(
                "case_folded_groups",
                groups=case.groups,
                values=case.values,
                affected_rows=case.rows,
                minority_rows=case.minority_rows,
                affected_share=case.rows / n,
            ),
        )
    if found.edge_whitespace:
        pad = found.edge_whitespace
        ctx.finding(
            code="edge_whitespace",
            subject=subject,
            severity=Severity.INFO,
            confidence=Confidence.HIGH,
            title=f"{pad.rows:,} values have leading or trailing whitespace",
            message=(
                f"{pad.rows:,} of {n:,} values of {name!r} ({pad.rows / n:.1%}) start "
                f"or end with whitespace, e.g. {_quoted(pad.examples)}. "
                "A padded value is a different string from its trimmed form, so "
                "joins and groupings split on it."
            ),
            evidence=evidence(
                "edge_whitespace",
                affected_rows=pad.rows,
                values=pad.values,
                affected_share=pad.rows / n,
            ),
        )
    if found.invisible_characters:
        hidden = found.invisible_characters
        ctx.finding(
            code="invisible_characters",
            subject=subject,
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"{hidden.rows:,} values hold invisible characters",
            message=(
                f"{hidden.rows:,} of {n:,} values of {name!r} ({hidden.rows / n:.1%}) "
                "hold a zero-width, non-breaking, bidirectional or control character, "
                f"shown here escaped: {', '.join(hidden.examples[:3])}. Two values "
                "that look the same can compare as different."
            ),
            evidence=evidence(
                "invisible_characters",
                affected_rows=hidden.rows,
                values=hidden.values,
                affected_share=hidden.rows / n,
            ),
        )
    if found.length_outliers:
        long = found.length_outliers
        ctx.finding(
            code="length_outliers",
            subject=subject,
            severity=Severity.INFO,
            confidence=Confidence.MODERATE,
            title=f"{long.rows:,} values have an unusual length",
            message=(
                f"{long.rows:,} of {n:,} values of {name!r} ({long.rows / n:.1%}) are "
                f"shorter than {long.lower_fence:.0f} or longer than "
                f"{long.upper_fence:.0f} characters, far outside the range of the "
                f"rest, e.g. {_quoted(long.examples, 2)}. Long "
                "values in a label column are often concatenated fields, error "
                "messages or markup."
            ),
            evidence=evidence(
                "length_iqr_fence",
                affected_rows=long.rows,
                values=long.values,
                affected_share=long.rows / n,
                lower_fence=long.lower_fence,
                upper_fence=long.upper_fence,
            ),
        )


def _quoted(values: tuple[str, ...], limit: int = 3) -> str:
    return ", ".join(repr(value) for value in values[:limit])
