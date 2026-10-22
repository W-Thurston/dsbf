# dsbf/checks/associations.py
"""How strongly each numeric column moves with the others, and with categories."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.stats import kendalltau, rankdata

from dsbf.checks.pairs import (
    Band,
    finite,
    fisher_interval,
    level_codes,
    strength_band,
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
    Interval,
    Severity,
    Subject,
)

Method = Literal["auto", "pearson", "spearman", "kendall"]
PairMetric = Literal["pearson_r", "spearman_r", "kendalls_tau", "eta"]
LABELS: dict[str, str] = {
    "pearson_r": "Pearson r",
    "spearman_r": "Spearman ρ",
    "kendalls_tau": "Kendall's τ",
    "eta": "η",
}
_METHOD_METRIC: dict[str, PairMetric] = {
    "pearson": "pearson_r",
    "spearman": "spearman_r",
    "kendall": "kendalls_tau",
}


class AssociationsParams(CheckParams):
    method: Method = Field(
        "auto",
        description="The statistic for a numeric pair. `auto` uses Kendall's τ-b "
        "below `small_sample` complete rows and Pearson's r from there on.",
    )
    small_sample: int = Field(
        30, ge=5, description="Complete rows below which `auto` uses Kendall's τ."
    )
    min_n: int = Field(
        8,
        ge=4,
        description="Complete rows a pair needs. Routing to a statistic happens "
        "after this gate, never before it, so a small frame gets Kendall's τ.",
    )
    max_levels: int = Field(
        50,
        ge=2,
        description="Most levels a categorical column may have; η over more levels "
        "is inflated by the groups' own noise.",
    )
    interval_level: float = Field(
        0.95, gt=0, lt=1, description="Coverage of the Fisher z intervals."
    )
    redundancy_threshold: float = Field(
        0.99,
        gt=0,
        le=1,
        description="|Pearson r| from which two numeric columns are redundant.",
    )
    max_pairs: int = Field(
        10_000,
        ge=1,
        description="Most pairs tested; with more, the check is skipped rather "
        "than test an arbitrary part of them.",
    )
    max_rows: int = Field(
        200_000,
        ge=1000,
        description="Most rows read; a larger frame is sampled with `seed`.",
    )
    seed: int = Field(42, description="Seed of the row sample.")
    max_findings: int = Field(
        50, ge=1, description="Most findings; the redundant and strongest pairs first."
    )


class PairAssociation(Contract):
    first: str
    second: str
    metric: PairMetric
    value: float = Field(
        ge=-1,
        le=1,
        description="Signed for r, ρ and τ; η is between 0 and 1 and has no sign.",
    )
    strength: Band = Field(description="The band of |value| on the scale of `metric`.")
    interval: Interval | None = Field(
        description="Fisher z interval; None for η and for a pair too small for one."
    )
    n_used: int = Field(ge=0, description="Rows where both columns have a value.")
    pearson_r: float | None = Field(
        ge=-1, le=1, description="For a numeric pair, whatever `metric` is."
    )


class AssociationsOutput(CheckOutputModel):
    pairs: list[PairAssociation] = Field(description="Every pair that was tested.")
    pairs_tested: int = Field(ge=0)
    pairs_flagged: int = Field(
        ge=0, description="Pairs that are redundant or in the strong band."
    )
    untested: dict[str, int] = Field(
        description="Pairs not tested, by reason: too_few_rows, too_few_groups, "
        "no_variation or overflow."
    )
    skipped_columns: dict[str, str] = Field(
        description="Categorical columns left out, with the reason."
    )
    rows_used: int = Field(ge=0, description="Rows read, after any sampling.")
    rows_total: int = Field(ge=0)


@dataclass
class _Numeric:
    """One numeric column over the rows read."""

    values: np.ndarray  # float64, with NaN where the value is missing or infinite
    ok: np.ndarray
    complete: bool
    _unit: np.ndarray | None = field(default=None, repr=False)
    _rank_unit: np.ndarray | None = field(default=None, repr=False)

    def unit(self, rank: bool) -> np.ndarray:
        """The centred, unit-length column (or its ranks), for a complete column.

        Raises ``_Undefined`` when the column has no spread or overflows.
        """
        if rank:
            if self._rank_unit is None:
                self._rank_unit = _unit(rankdata(self.values))
            return self._rank_unit
        if self._unit is None:
            self._unit = _unit(self.values)
        return self._unit


class _Undefined(Exception):
    """A pair whose statistic does not exist; the reason is the message."""


def _unit(values: np.ndarray) -> np.ndarray:
    """``values`` centred and scaled to length 1; correlation is a dot product.

    The column is divided by its largest magnitude first, so values near 1e308 do not
    overflow.
    """
    with np.errstate(all="ignore"):
        biggest = float(np.max(np.abs(values)))
        if not np.isfinite(biggest):
            raise _Undefined("overflow")
        if biggest == 0:
            raise _Undefined("no_variation")
        scaled = values / biggest
        centred = scaled - scaled.mean()
        length = float(np.sqrt(np.sum(centred * centred)))
    if not np.isfinite(length):
        raise _Undefined("overflow")
    if length == 0:
        raise _Undefined("no_variation")
    return centred / length


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


@check(
    id="associations",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric", "categorical", "boolean"),
    dimensions=[Dimension.RELATIONSHIP, Dimension.REDUNDANCY],
    params=AssociationsParams,
    output=AssociationsOutput,
    requires=Requires(profile=["counts"]),
    # One pass over the rows per pair; ranks are computed once per complete column.
    cost=Cost.MODERATE,
    packs=["core"],
    tags=["correlation", "association", "redundancy"],
)
def associations(ctx: CheckContext, params: AssociationsParams) -> CheckOutput:
    """Measure each numeric column against every other numeric and categorical one."""
    numeric_names: list[str] = []
    categorical_names: list[str] = []
    skipped_columns: dict[str, str] = {}
    for name in ctx.columns:
        column = ctx.profile.column(name)
        if column.semantic_type == "numeric":
            numeric_names.append(name)
            continue
        distinct = column.counts.distinct
        if distinct < 2:
            skipped_columns[name] = "fewer than two levels"
        elif distinct > params.max_levels:
            skipped_columns[name] = f"more than {params.max_levels} levels"
        else:
            categorical_names.append(name)
    n_numeric, n_categorical = len(numeric_names), len(categorical_names)
    n_pairs = n_numeric * (n_numeric - 1) // 2 + n_numeric * n_categorical
    if not n_pairs:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_pairs",
            "Needs a numeric column and another numeric column or a categorical "
            "one with 2 to "
            f"{params.max_levels} levels; there are {n_numeric} numeric and "
            f"{n_categorical} categorical columns.",
            severity=Severity.INFO,
        )
    if n_pairs > params.max_pairs:
        return ctx.stop(
            CheckStatus.SKIPPED,
            "too_many_pairs",
            f"{n_pairs:,} pairs would be tested, over the limit of "
            f"{params.max_pairs:,}.",
            hint="Raise `max_pairs`, or run the check on fewer columns.",
        )

    total_rows = ctx.profile.n_rows
    if total_rows > params.max_rows:
        rng = np.random.default_rng(params.seed)
        rows = np.sort(rng.choice(total_rows, params.max_rows, replace=False))
        ctx.note(
            "rows_sampled",
            f"Read {rows.size:,} of {total_rows:,} rows, sampled with seed "
            f"{params.seed}.",
        )
    else:
        rows = None
    rows_used = total_rows if rows is None else int(rows.size)

    numeric = {name: _read_numeric(ctx, name, rows) for name in numeric_names}
    codes: dict[str, tuple[np.ndarray, int]] = {}
    for name in categorical_names:
        column_codes = ctx.values(name, dropna=False)
        found, levels = level_codes(
            column_codes if rows is None else column_codes[rows]
        )
        codes[name] = (found, int(levels.size))

    pairs: list[PairAssociation] = []
    untested: dict[str, int] = {}

    def test(measure, first: str, second: str, left, right) -> None:
        ctx.checkpoint()
        try:
            pairs.append(measure(first, second, left, right, params))
        except _Undefined as why:
            untested[str(why)] = untested.get(str(why), 0) + 1

    for position, first in enumerate(numeric_names):
        for second in numeric_names[position + 1 :]:
            test(_numeric_pair, first, second, numeric[first], numeric[second])
        for second in categorical_names:
            test(_eta_pair, first, second, numeric[first], codes[second])

    flagged = [p for p in pairs if _is_flagged(p, params)]
    shown = sorted(flagged, key=_priority(params))[: params.max_findings]
    if len(flagged) > len(shown):
        ctx.note(
            "findings_capped",
            f"{len(flagged):,} pairs are redundant or strongly associated; the "
            f"findings cover the first {len(shown):,}, all {len(flagged):,} are in "
            "the output.",
        )
    for pair in shown:
        _report(ctx, pair, len(pairs), len(flagged), params)
    return ctx.output(
        AssociationsOutput(
            pairs=pairs,
            pairs_tested=len(pairs),
            pairs_flagged=len(flagged),
            untested=untested,
            skipped_columns=skipped_columns,
            rows_used=rows_used,
            rows_total=total_rows,
        )
    )


def _read_numeric(ctx: CheckContext, name: str, rows: np.ndarray | None) -> _Numeric:
    raw = ctx.values(name, dropna=False)
    values = np.asarray(raw if rows is None else raw[rows], dtype=np.float64)
    values = np.where(finite(values), values, np.nan)
    ok = ~np.isnan(values)
    return _Numeric(values=values, ok=ok, complete=bool(ok.all()))


def _numeric_pair(
    first: str,
    second: str,
    left: _Numeric,
    right: _Numeric,
    params: AssociationsParams,
) -> PairAssociation:
    both = left.ok & right.ok
    n_used = int(np.count_nonzero(both))
    if n_used < params.min_n:
        raise _Undefined("too_few_rows")
    clean = left.complete and right.complete
    pearson = _correlation(
        *(
            (left.unit(False), right.unit(False))
            if clean
            else (_unit(left.values[both]), _unit(right.values[both]))
        )
    )
    metric: PairMetric
    if params.method == "auto":
        metric = "kendalls_tau" if n_used < params.small_sample else "pearson_r"
    else:
        metric = _METHOD_METRIC[params.method]
    if metric == "pearson_r":
        value = pearson
    elif metric == "spearman_r":
        value = _correlation(
            *(
                (left.unit(True), right.unit(True))
                if clean
                else (
                    _unit(rankdata(left.values[both])),
                    _unit(rankdata(right.values[both])),
                )
            )
        )
    else:
        value = _kendall(left.values[both], right.values[both])
    interval = fisher_interval(value, n_used, metric, params.interval_level)
    return PairAssociation(
        first=first,
        second=second,
        metric=metric,
        value=value,
        strength=strength_band(value, metric),
        interval=(
            Interval(
                low=interval[0],
                high=interval[1],
                level=params.interval_level,
                method="fisher_z",
            )
            if interval
            else None
        ),
        n_used=n_used,
        pearson_r=pearson,
    )


def _kendall(first: np.ndarray, second: np.ndarray) -> float:
    """Kendall's τ-b, which corrects for ties."""
    tau = kendalltau(first, second).statistic
    if not np.isfinite(tau):
        raise _Undefined("no_variation")
    return float(np.clip(tau, -1.0, 1.0))


def _eta_pair(
    first: str,
    second: str,
    numeric: _Numeric,
    categorical: tuple[np.ndarray, int],
    params: AssociationsParams,
) -> PairAssociation:
    groups, n_levels = categorical
    keep = numeric.ok & (groups >= 0)
    x, g = numeric.values[keep], groups[keep]
    counts = np.bincount(g, minlength=n_levels)
    # A group of one is fitted exactly, which inflates η; leave such groups out.
    big = counts[g] >= 2
    x, g = x[big], g[big]
    if np.count_nonzero(counts >= 2) < 2:
        raise _Undefined("too_few_groups")
    n_used = int(x.size)
    if n_used < params.min_n:
        raise _Undefined("too_few_rows")
    with np.errstate(all="ignore"):
        biggest = float(np.max(np.abs(x)))
        if not np.isfinite(biggest):
            raise _Undefined("overflow")
        if biggest == 0:
            raise _Undefined("no_variation")
        x = x / biggest
        x = x - x.mean()
        counts = np.bincount(g, minlength=n_levels)
        sums = np.bincount(g, weights=x, minlength=n_levels)
        present = counts > 0
        between = float(np.sum(sums[present] ** 2 / counts[present]))
        total = float(np.sum(x * x))
    if not np.isfinite(total):
        raise _Undefined("overflow")
    if total == 0:
        raise _Undefined("no_variation")
    value = float(np.sqrt(np.clip(between / total, 0.0, 1.0)))
    return PairAssociation(
        first=first,
        second=second,
        metric="eta",
        value=value,
        strength=strength_band(value, "eta"),
        interval=None,
        n_used=n_used,
        pearson_r=None,
    )


def _is_redundant(pair: PairAssociation, params: AssociationsParams) -> bool:
    return (
        pair.pearson_r is not None
        and abs(pair.pearson_r) >= params.redundancy_threshold
    )


def _is_flagged(pair: PairAssociation, params: AssociationsParams) -> bool:
    return _is_redundant(pair, params) or pair.strength == "strong"


def _priority(params: AssociationsParams):
    """Redundant pairs first, then by strength; ties keep the order tested."""

    def key(pair: PairAssociation) -> tuple[int, float]:
        return (0 if _is_redundant(pair, params) else 1, -abs(pair.value))

    return key


def _confidence(n_used: int) -> Confidence:
    if n_used >= 100:
        return Confidence.HIGH
    return Confidence.MODERATE if n_used >= 30 else Confidence.LOW


def _report(
    ctx: CheckContext,
    pair: PairAssociation,
    tested: int,
    flagged: int,
    params: AssociationsParams,
) -> None:
    redundant = _is_redundant(pair, params)
    label = LABELS[pair.metric]
    first, second = pair.first, pair.second
    metrics: dict[str, object] = {
        "metric": pair.metric,
        "value": pair.value,
        "abs_value": abs(pair.value),
        "strength": pair.strength,
        "pairs_tested": tested,
        "pairs_flagged": flagged,
    }
    if pair.pearson_r is not None:
        metrics["pearson_r"] = pair.pearson_r
        metrics["abs_pearson_r"] = abs(pair.pearson_r)
    if pair.interval is not None:
        metrics["interval_low"] = pair.interval.low
        metrics["interval_high"] = pair.interval.high
    reach = (
        f" (95% interval {pair.interval.low:.2f} to {pair.interval.high:.2f})"
        if pair.interval is not None
        else ""
    )
    scan = (
        f"{tested:,} pairs were tested and {flagged:,} are flagged, so some of "
        "the strong ones can arise by chance in a wide frame."
        if tested > 1
        else ""
    )
    if redundant:
        assert pair.pearson_r is not None
        code, dimension, severity = (
            "redundant_columns",
            Dimension.REDUNDANCY,
            Severity.WARN,
        )
        title = f"Redundant columns (r = {pair.pearson_r:.3f})"
        message = (
            f"{first!r} and {second!r} move together almost exactly (Pearson r = "
            f"{pair.pearson_r:.4f} over {pair.n_used:,} rows{reach}). One is likely "
            "derived from the other, such as a unit conversion, a ratio or a "
            "running total, or both measure the same thing. Keeping both adds no "
            "information and makes linear coefficients unstable. This says nothing "
            "about a target: whether either column leaks one is a separate question."
        )
    else:
        code, dimension, severity = (
            "strong_association",
            Dimension.RELATIONSHIP,
            Severity.INFO,
        )
        title = f"Strong association ({label} = {pair.value:.2f})"
        message = (
            f"{first!r} and {second!r} are strongly associated ({label} = "
            f"{pair.value:.3f} over {pair.n_used:,} rows{reach}). {scan} An "
            "association shows the columns vary together, not that one causes the "
            "other; check whether one is derived from the other before using both "
            "as features."
        ).replace("  ", " ")
    ctx.finding(
        code=code,
        subject=Subject.column_pair(first, second),
        dimension=dimension,
        severity=severity,
        confidence=_confidence(pair.n_used),
        title=title,
        message=message,
        evidence=Evidence(
            metrics=metrics,  # type: ignore[arg-type]
            n_used=pair.n_used,
            n_total=ctx.profile.n_rows,
            method=pair.metric,
            interval=pair.interval,
        ),
    )
