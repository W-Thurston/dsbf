# dsbf/checks/contingency.py
"""Pairs of categorical columns that are not independent, and how strongly."""

import numpy as np
from pydantic import Field

from dsbf.checks.pairs import (
    Band,
    chi2_p_value,
    contingency_table,
    cramers_v,
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
    AssumptionCheck,
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

_EXPECTED_FLOOR = 5.0
_MAX_SPARSE_SHARE = 0.2


class ContingencyParams(CheckParams):
    alpha: float = Field(
        0.05, gt=0, lt=1, description="Significance level of the chi-squared test."
    )
    min_v: float = Field(
        0.1,
        gt=0,
        lt=1,
        description="Bias-corrected V from which a significant pair is reported "
        "and gets a bootstrap interval.",
    )
    min_n: int = Field(10, ge=5, description="Complete rows a pair needs.")
    max_levels: int = Field(
        50, ge=2, description="Most levels a column may have to be compared."
    )
    top_n: int = Field(
        10,
        ge=2,
        description="Levels per column printed in a pair's table; the statistics "
        "always use all the levels.",
    )
    interval_level: float = Field(
        0.95, gt=0, lt=1, description="Coverage of the bootstrap interval."
    )
    n_boot: int = Field(500, ge=100, description="Bootstrap resamples of a table.")
    seed: int = Field(42, description="Seed of the bootstrap.")
    max_pairs: int = Field(
        5_000,
        ge=1,
        description="Most pairs tested; with more, the check is skipped rather "
        "than test an arbitrary part of them.",
    )
    max_findings: int = Field(
        50, ge=1, description="Most findings; the strongest pairs first."
    )


class PairTable(Contract):
    first: str
    second: str
    n_used: int = Field(ge=1, description="Rows where both columns have a value.")
    levels: tuple[int, int] = Field(
        description="Levels of `first` and of `second` among those rows."
    )
    chi2: float = Field(ge=0)
    dof: int = Field(ge=1)
    p_value: float = Field(ge=0, le=1)
    cramers_v: float = Field(
        ge=0, le=1, description="Plain V, which grows with the number of levels."
    )
    cramers_v_corrected: float = Field(
        ge=0, le=1, description="V with Bergsma's (2013) correction for small samples."
    )
    strength: Band = Field(description="The band of the corrected V.")
    interval: Interval | None = Field(
        description="Percentile bootstrap interval of the corrected V; only for a "
        "pair that reaches `min_v`."
    )
    sparse_share: float = Field(
        ge=0, le=1, description="Share of cells with an expected count under 5."
    )
    table: dict[str, dict[str, int]] = Field(
        description="Counts for the most frequent levels, `first` down the side."
    )
    truncated: bool = Field(description="The table leaves out some levels.")


class ContingencyOutput(CheckOutputModel):
    pairs: list[PairTable] = Field(description="Every pair that was tested.")
    pairs_tested: int = Field(ge=0)
    pairs_flagged: int = Field(
        ge=0, description="Pairs that are significant and reach `min_v`."
    )
    untested: dict[str, int] = Field(
        description="Pairs not tested, by reason: too_few_rows or too_few_levels."
    )
    skipped_columns: dict[str, str] = Field(
        description="Columns left out, with the reason."
    )


@check(
    id="contingency",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("categorical", "boolean"),
    dimensions=[Dimension.RELATIONSHIP],
    params=ContingencyParams,
    output=ContingencyOutput,
    requires=Requires(profile=["counts"]),
    # One table per pair; a bootstrap only for the pairs worth reading.
    cost=Cost.MODERATE,
    codes=["categorical_association"],
    packs=["core"],
    tags=["categorical", "association", "chi_squared"],
)
def contingency(ctx: CheckContext, params: ContingencyParams) -> CheckOutput:
    """Test each pair of categorical columns for independence and size the link."""
    names: list[str] = []
    skipped_columns: dict[str, str] = {}
    for name in ctx.columns:
        distinct = ctx.profile.column(name).counts.distinct
        if distinct < 2:
            skipped_columns[name] = "fewer than two levels"
        elif distinct > params.max_levels:
            skipped_columns[name] = f"more than {params.max_levels} levels"
        else:
            names.append(name)
    n_pairs = len(names) * (len(names) - 1) // 2
    if not n_pairs:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_pairs",
            f"Needs two categorical columns with 2 to {params.max_levels} levels; "
            f"there are {len(names)}.",
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

    coded = {name: level_codes(ctx.values(name, dropna=False)) for name in names}
    rng = np.random.default_rng(params.seed)
    pairs: list[PairTable] = []
    untested: dict[str, int] = {}
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            ctx.checkpoint()
            table, rows, cols = contingency_table(coded[first][0], coded[second][0])
            if int(table.sum()) < params.min_n:
                untested["too_few_rows"] = untested.get("too_few_rows", 0) + 1
            elif min(table.shape) < 2:
                untested["too_few_levels"] = untested.get("too_few_levels", 0) + 1
            else:
                labels = (coded[first][1][rows], coded[second][1][cols])
                pairs.append(_test(first, second, table, labels, rng, params))

    flagged = [p for p in pairs if _is_flagged(p, params)]
    shown = sorted(flagged, key=lambda p: -p.cramers_v_corrected)[: params.max_findings]
    if len(flagged) > len(shown):
        ctx.note(
            "findings_capped",
            f"{len(flagged):,} pairs are associated; the findings cover the "
            f"strongest {len(shown):,}, all {len(flagged):,} are in the output.",
        )
    for pair in shown:
        _report(ctx, pair, len(pairs), len(flagged), params)
    return ctx.output(
        ContingencyOutput(
            pairs=pairs,
            pairs_tested=len(pairs),
            pairs_flagged=len(flagged),
            untested=untested,
            skipped_columns=skipped_columns,
        )
    )


def _test(
    first: str,
    second: str,
    table: np.ndarray,
    labels: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    params: ContingencyParams,
) -> PairTable:
    n = int(table.sum())
    chi2, plain, corrected = (float(x) for x in cramers_v(table))
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    value = max(0.0, min(1.0, corrected))
    interval = None
    if value >= params.min_v:
        interval = _bootstrap(table, rng, params)
    shown, truncated = _printed(table, labels, params.top_n)
    return PairTable(
        first=first,
        second=second,
        n_used=n,
        levels=(table.shape[0], table.shape[1]),
        chi2=chi2,
        dof=(table.shape[0] - 1) * (table.shape[1] - 1),
        p_value=chi2_p_value(chi2, table),
        cramers_v=plain,
        cramers_v_corrected=value,
        strength=strength_band(value, "cramers_v"),
        interval=interval,
        sparse_share=float(np.mean(expected < _EXPECTED_FLOOR)),
        table=shown,
        truncated=truncated,
    )


def _bootstrap(
    table: np.ndarray, rng: np.random.Generator, params: ContingencyParams
) -> Interval | None:
    """Percentile interval of the corrected V from resampled tables.

    Resampling the rows of a table is a multinomial draw over its cells.
    """
    n = int(table.sum())
    draws = rng.multinomial(n, table.ravel() / n, size=params.n_boot)
    _, _, resampled = cramers_v(draws.reshape(params.n_boot, *table.shape))
    tail = (1 - params.interval_level) / 2
    low, high = np.quantile(resampled, [tail, 1 - tail])
    return Interval(
        low=float(low),
        high=float(high),
        level=params.interval_level,
        method="percentile_bootstrap",
    )


def _printed(
    table: np.ndarray, labels: tuple[np.ndarray, np.ndarray], top_n: int
) -> tuple[dict[str, dict[str, int]], bool]:
    """The counts of the ``top_n`` most frequent levels of each column."""
    rows = _most_frequent(table.sum(axis=1), top_n)
    cols = _most_frequent(table.sum(axis=0), top_n)
    shown = {
        str(labels[0][r]): {str(labels[1][c]): int(table[r, c]) for c in cols}
        for r in rows
    }
    return shown, len(rows) < table.shape[0] or len(cols) < table.shape[1]


def _most_frequent(counts: np.ndarray, top_n: int) -> np.ndarray:
    """Indexes of the largest counts (the earlier level wins a tie), in level order."""
    order = np.argsort(-counts, kind="stable")[:top_n]
    return np.sort(order)


def _is_flagged(pair: PairTable, params: ContingencyParams) -> bool:
    return pair.p_value < params.alpha and pair.cramers_v_corrected >= params.min_v


def _report(
    ctx: CheckContext,
    pair: PairTable,
    tested: int,
    flagged: int,
    params: ContingencyParams,
) -> None:
    sparse = pair.sparse_share > _MAX_SPARSE_SHARE
    reach = (
        f" (95% bootstrap interval {pair.interval.low:.2f} to {pair.interval.high:.2f})"
        if pair.interval is not None
        else ""
    )
    caution = (
        " More than a fifth of the cells expect fewer than 5 rows, so the "
        "chi-squared p-value is unreliable; trust the effect size and its interval."
        if sparse
        else ""
    )
    scan = (
        f" {tested:,} pairs were tested and {flagged:,} are flagged, so some "
        "can arise by chance in a wide frame."
        if tested > 1
        else ""
    )
    metrics: dict[str, object] = {
        "cramers_v_corrected": pair.cramers_v_corrected,
        "cramers_v": pair.cramers_v,
        "chi2": pair.chi2,
        "dof": pair.dof,
        "p_value": pair.p_value,
        "strength": pair.strength,
        "sparse_share": pair.sparse_share,
        "pairs_tested": tested,
        "pairs_flagged": flagged,
    }
    if pair.interval is not None:
        metrics["interval_low"] = pair.interval.low
        metrics["interval_high"] = pair.interval.high
    ctx.finding(
        code="categorical_association",
        subject=Subject.column_pair(pair.first, pair.second),
        severity=Severity.WARN if pair.strength == "strong" else Severity.INFO,
        confidence=(
            Confidence.LOW
            if sparse
            else Confidence.HIGH
            if pair.n_used >= 100
            else Confidence.MODERATE
        ),
        title=(
            f"Associated categories ({pair.strength}, "
            f"V = {pair.cramers_v_corrected:.2f})"
        ),
        message=(
            f"{pair.first!r} and {pair.second!r} are not independent: bias-corrected "
            f"Cramér's V = {pair.cramers_v_corrected:.3f}{reach}, "
            f"χ² = {pair.chi2:,.1f} (p = {pair.p_value:.4g}) over {pair.n_used:,} "
            f"rows.{caution}{scan} The table shows which combinations occur more "
            "or less often than independence predicts; a strong link means the "
            "second column adds little beyond the first as a feature."
        ),
        evidence=Evidence(
            metrics=metrics,  # type: ignore[arg-type]
            n_used=pair.n_used,
            n_total=ctx.profile.n_rows,
            method="cramers_v_bias_corrected",
            assumptions=(
                AssumptionCheck(
                    name="expected_counts_at_least_5",
                    satisfied=not sparse,
                    method="chi2_expected",
                    detail=(
                        f"{pair.sparse_share:.0%} of cells expect fewer than 5 rows."
                    ),
                ),
            ),
            interval=pair.interval,
        ),
    )
