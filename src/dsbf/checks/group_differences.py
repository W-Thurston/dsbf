# dsbf/checks/group_differences.py
"""Numeric columns whose values differ between the groups a categorical column makes."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.stats import chi2, mannwhitneyu, rankdata
from scipy.stats import f as f_distribution

from dsbf.checks.pairs import (
    Correction,
    Undefined,
    adjust_p_values,
    finite,
    level_codes,
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

Method = Literal["anova", "kruskal"]
# Values drawn per bootstrap chunk, so memory stays at a few megabytes.
_CHUNK_ELEMENTS = 1_000_000
# Largest ratio of group standard deviations at which classic ANOVA is still safe.
_SD_RATIO_LIMIT = 2.0
# Within-group share of the variation under which the groups have no spread inside.
_SPREAD_FLOOR = 1e-14


class GroupDifferencesParams(CheckParams):
    alpha: float = Field(
        0.05, gt=0, lt=1, description="Level of the adjusted p-value for a difference."
    )
    correction: Correction = Field(
        "fdr_bh",
        description="Adjustment of the p-values for the number of pairs tested, "
        "separately for each method. `fdr_bh` controls the false discovery rate.",
    )
    min_effect: float = Field(
        0.01,
        ge=0,
        lt=1,
        description="Effect size (ω² or ε²) from which a significant pair is "
        "reported; a huge sample makes a negligible difference significant.",
    )
    min_group_n: int = Field(
        5, ge=2, description="Rows a group needs; a smaller group is left out."
    )
    max_levels: int = Field(
        20, ge=2, description="Most levels a categorical column may have."
    )
    max_contrast_levels: int = Field(
        10,
        ge=2,
        description="Most groups for which a flagged pair gets every pairwise "
        "Mann–Whitney contrast.",
    )
    interval_level: float = Field(
        0.95, gt=0, lt=1, description="Coverage of the bootstrap intervals."
    )
    n_boot: int = Field(200, ge=100, description="Bootstrap resamples of a pair.")
    seed: int = Field(42, description="Seed of the bootstrap and of the row sample.")
    max_pairs: int = Field(
        5_000,
        ge=1,
        description="Most pairs tested; with more, the check is skipped rather "
        "than test an arbitrary part of them.",
    )
    max_rows: int = Field(
        200_000,
        ge=1000,
        description="Most rows read; a larger frame is sampled with `seed`.",
    )
    max_findings: int = Field(
        50, ge=1, description="Most findings; the largest effects first."
    )


class Contrast(Contract):
    first_level: str
    second_level: str
    n_first: int = Field(ge=1)
    n_second: int = Field(ge=1)
    p_value: float = Field(ge=0, le=1, description="Mann–Whitney U, two-sided.")
    p_adjusted: float = Field(
        ge=0, le=1, description="Holm adjustment over this pair's contrasts."
    )
    rank_biserial: float = Field(
        ge=-1,
        le=1,
        description="Probability that a row of the first level is above one of the "
        "second, minus the reverse; positive when the first level is higher.",
    )
    interval: Interval | None = Field(
        description="Bootstrap interval of `rank_biserial`; only for a significant "
        "contrast."
    )


class PairComparison(Contract):
    numeric: str
    categorical: str
    n_used: int = Field(ge=0, description="Rows in the groups compared.")
    n_groups: int = Field(ge=2)
    groups_dropped: int = Field(
        ge=0, description="Levels left out for having too few rows."
    )
    smallest_group: int = Field(ge=1)
    largest_group: int = Field(ge=1)
    f_statistic: float | None = Field(
        description="None when the groups have no spread within them."
    )
    anova_p: float = Field(ge=0, le=1)
    anova_p_adjusted: float = Field(ge=0, le=1)
    omega_squared: float = Field(
        ge=0,
        le=1,
        description="Share of variance the groups explain, corrected for "
        "bias; the effect size of the ANOVA.",
    )
    eta_squared: float = Field(ge=0, le=1, description="The uncorrected share.")
    h_statistic: float = Field(ge=0, description="Kruskal–Wallis H, tie-corrected.")
    kruskal_p: float = Field(ge=0, le=1)
    kruskal_p_adjusted: float = Field(ge=0, le=1)
    epsilon_squared: float = Field(
        ge=0, le=1, description="H / (n - 1): the effect size of the rank test."
    )
    sd_ratio: float | None = Field(
        description="Largest over smallest group standard deviation; ANOVA is "
        "doubtful above 2. None when a group has no spread."
    )
    flagged: bool
    flagged_by: tuple[Method, ...]
    omega_interval: Interval | None = Field(
        description="Percentile bootstrap interval of ω²; only for a flagged pair."
    )
    epsilon_interval: Interval | None = Field(
        description="Percentile bootstrap interval of ε², on the ranks; only for a "
        "flagged pair."
    )
    top_group: str = Field(description="The level with the highest mean rank.")
    bottom_group: str = Field(description="The level with the lowest mean rank.")
    contrasts: list[Contrast] = Field(
        description="Every pair of groups, for a flagged pair."
    )
    contrasts_skipped: bool = Field(
        description="A flagged pair with more groups than `max_contrast_levels`."
    )


class GroupDifferencesOutput(CheckOutputModel):
    pairs: list[PairComparison] = Field(description="Every pair that was tested.")
    pairs_tested: int = Field(ge=0)
    pairs_flagged: int = Field(
        ge=0, description="Pairs significant after adjustment with a reportable effect."
    )
    tests_run: int = Field(
        ge=0, description="Hypothesis tests run: two per pair, plus the contrasts."
    )
    correction: str
    alpha: float
    untested: dict[str, int] = Field(
        description="Pairs not tested, by reason: too_few_groups, no_variation or "
        "overflow."
    )
    skipped_columns: dict[str, str] = Field(
        description="Categorical columns left out, with the reason."
    )
    rows_used: int = Field(ge=0, description="Rows read, after any sampling.")
    rows_total: int = Field(ge=0)


@dataclass
class _Numeric:
    """One numeric column over the rows read."""

    scaled: np.ndarray  # divided by its largest magnitude, NaN where missing
    ok: np.ndarray
    n_ok: int
    _ranks: np.ndarray | None = field(default=None, repr=False)

    def ranks(self) -> np.ndarray:
        """Mid-ranks of the values present, in row order."""
        if self._ranks is None:
            self._ranks = rankdata(self.scaled[self.ok])
        return self._ranks


@check(
    id="group_differences",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric", "categorical", "boolean"),
    dimensions=[Dimension.RELATIONSHIP],
    params=GroupDifferencesParams,
    output=GroupDifferencesOutput,
    requires=Requires(profile=["counts"]),
    # Group sums per pair; bootstraps and contrasts only for the pairs worth reading.
    cost=Cost.MODERATE,
    packs=["core"],
    tags=["anova", "kruskal_wallis", "mann_whitney", "group_difference"],
)
def group_differences(ctx: CheckContext, params: GroupDifferencesParams) -> CheckOutput:
    """Test whether each numeric column differs between the groups of each category."""
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
    n_pairs = len(numeric_names) * len(categorical_names)
    if not n_pairs:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_pairs",
            "Needs a numeric column and a categorical one with 2 to "
            f"{params.max_levels} levels; there are {len(numeric_names)} numeric "
            f"and {len(categorical_names)} categorical columns.",
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
    rng = np.random.default_rng(params.seed)
    rows = None
    if total_rows > params.max_rows:
        rows = np.sort(rng.choice(total_rows, params.max_rows, replace=False))
        ctx.note(
            "rows_sampled",
            f"Read {rows.size:,} of {total_rows:,} rows, sampled with seed "
            f"{params.seed}.",
        )
    rows_used = total_rows if rows is None else int(rows.size)

    numeric = {name: _read_numeric(ctx, name, rows) for name in numeric_names}
    coded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in categorical_names:
        raw = ctx.values(name, dropna=False)
        coded[name] = level_codes(raw if rows is None else raw[rows])

    pairs: list[PairComparison] = []
    untested: dict[str, int] = {}
    for first in numeric_names:
        for second in categorical_names:
            ctx.checkpoint()
            try:
                pairs.append(
                    _compare(first, second, numeric[first], coded[second], params)
                )
            except Undefined as why:
                untested[str(why)] = untested.get(str(why), 0) + 1

    pairs = _adjusted(pairs, params)
    flagged = [p for p in pairs if p.flagged]
    shown = sorted(flagged, key=lambda p: -_effect(p))[: params.max_findings]
    if len(flagged) > len(shown):
        ctx.note(
            "findings_capped",
            f"{len(flagged):,} pairs differ between groups; the findings cover the "
            f"largest {len(shown):,} effects, all {len(flagged):,} are in the output.",
        )
    # A flagged pair beyond the findings cap keeps its p-values but no interval.
    detailed = {
        (p.numeric, p.categorical): _detail(
            p, numeric[p.numeric], coded[p.categorical], rng, params
        )
        for p in shown
    }
    pairs = [detailed.get((p.numeric, p.categorical), p) for p in pairs]
    contrasts_run = sum(len(p.contrasts) for p in detailed.values())
    for pair in detailed.values():
        _report(ctx, pair, len(pairs), len(flagged), params)
    return ctx.output(
        GroupDifferencesOutput(
            pairs=pairs,
            pairs_tested=len(pairs),
            pairs_flagged=len(flagged),
            tests_run=2 * len(pairs) + contrasts_run,
            correction=params.correction,
            alpha=params.alpha,
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
    with np.errstate(all="ignore"):
        biggest = float(np.max(np.abs(values[ok]))) if ok.any() else 0.0
        scaled = values / biggest if biggest > 0 else values
    return _Numeric(scaled=scaled, ok=ok, n_ok=int(ok.sum()))


@dataclass
class _Groups:
    """The rows of one pair, in groups of at least ``min_group_n``."""

    x: np.ndarray  # scaled values
    g: np.ndarray  # group index, 0..k-1
    levels: np.ndarray  # label of each group
    counts: np.ndarray
    dropped: int
    ranks: np.ndarray


def _groups(
    numeric: _Numeric,
    coded: tuple[np.ndarray, np.ndarray],
    params: GroupDifferencesParams,
) -> _Groups:
    codes, labels = coded
    keep = numeric.ok & (codes >= 0)
    x, g = numeric.scaled[keep], codes[keep]
    counts = np.bincount(g, minlength=labels.size)
    big = counts >= params.min_group_n
    n_big = int(big.sum())
    if n_big < 2:
        raise Undefined("too_few_groups")
    whole = bool(big.all()) and int(keep.sum()) == numeric.n_ok
    if not whole:
        rows = big[g]
        x, g = x[rows], g[rows]
    remap = np.cumsum(big) - 1
    g = remap[g]
    if x.max() == x.min():
        raise Undefined("no_variation")
    ranks = numeric.ranks() if whole else rankdata(x)
    return _Groups(
        x=x,
        g=g,
        levels=labels[big],
        counts=counts[big],
        dropped=int((counts > 0).sum()) - n_big,
        ranks=ranks,
    )


def _compare(
    first: str,
    second: str,
    numeric: _Numeric,
    coded: tuple[np.ndarray, np.ndarray],
    params: GroupDifferencesParams,
) -> PairComparison:
    data = _groups(numeric, coded, params)
    k, n = data.counts.size, int(data.counts.sum())
    means = np.bincount(data.g, weights=data.x, minlength=k) / data.counts
    deviations = data.x - means[data.g]
    ssw = float(deviations @ deviations)
    ssb = float(np.sum(data.counts * (means - data.x.mean()) ** 2))
    omega, eta = _anova_effect(
        data.counts, means[None, :], np.array([[ssw]]), np.array([ssb])
    )
    if ssw > _SPREAD_FLOOR * (ssb + ssw):
        f_stat = (ssb / (k - 1)) / (ssw / (n - k))
        anova_p = float(f_distribution.sf(f_stat, k - 1, n - k))
        f_value: float | None = float(f_stat) if np.isfinite(f_stat) else None
    else:
        f_value, anova_p = None, 0.0

    rank_means = np.bincount(data.g, weights=data.ranks, minlength=k) / data.counts
    centred = data.ranks - data.ranks.mean()
    sst_ranks = float(centred @ centred)
    if sst_ranks == 0:
        raise Undefined("no_variation")
    ssb_ranks = float(np.sum(data.counts * (rank_means - data.ranks.mean()) ** 2))
    epsilon = float(np.clip(ssb_ranks / sst_ranks, 0.0, 1.0))
    h_stat = epsilon * (n - 1)
    kruskal_p = float(chi2.sf(h_stat, k - 1))

    spreads = np.sqrt(
        np.bincount(data.g, weights=deviations**2, minlength=k) / (data.counts - 1)
    )
    sd_ratio = float(spreads.max() / spreads.min()) if spreads.min() > 0 else None
    return PairComparison(
        numeric=first,
        categorical=second,
        n_used=n,
        n_groups=k,
        groups_dropped=data.dropped,
        smallest_group=int(data.counts.min()),
        largest_group=int(data.counts.max()),
        f_statistic=f_value,
        anova_p=anova_p,
        anova_p_adjusted=anova_p,
        omega_squared=float(omega[0]),
        eta_squared=float(eta[0]),
        h_statistic=h_stat,
        kruskal_p=kruskal_p,
        kruskal_p_adjusted=kruskal_p,
        epsilon_squared=epsilon,
        sd_ratio=sd_ratio,
        flagged=False,
        flagged_by=(),
        omega_interval=None,
        epsilon_interval=None,
        top_group=str(data.levels[int(np.argmax(rank_means))]),
        bottom_group=str(data.levels[int(np.argmin(rank_means))]),
        contrasts=[],
        contrasts_skipped=False,
    )


def _anova_effect(
    counts: np.ndarray, means: np.ndarray, ssw: np.ndarray, ssb: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """ω² and η² for each row of group ``means`` (and within-group sums of squares).

    ``ssw`` has one column per group in a resample and one column in total for the
    data; ``ssb`` is given only when the between-group sum of squares is known.
    """
    k, n = counts.size, float(counts.sum())
    if ssb is None:
        grand = (means * counts).sum(axis=1) / n
        ssb = (counts * (means - grand[:, None]) ** 2).sum(axis=1)
    sse = ssw.sum(axis=1)
    total = ssb + sse
    msw = sse / (n - k)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(total > 0, (ssb - (k - 1) * msw) / (total + msw), 0.0)
        eta = np.where(total > 0, ssb / total, 0.0)
    return np.clip(omega, 0.0, 1.0), np.clip(eta, 0.0, 1.0)


def _adjusted(
    pairs: list[PairComparison], params: GroupDifferencesParams
) -> list[PairComparison]:
    """Adjust each method's p-values over all the pairs, and flag the pairs."""
    if not pairs:
        return pairs
    anova = adjust_p_values(np.array([p.anova_p for p in pairs]), params.correction)
    kruskal = adjust_p_values(np.array([p.kruskal_p for p in pairs]), params.correction)
    done = []
    for pair, a, k in zip(pairs, anova, kruskal, strict=True):
        by: list[Method] = []
        if a < params.alpha and pair.omega_squared >= params.min_effect:
            by.append("anova")
        if k < params.alpha and pair.epsilon_squared >= params.min_effect:
            by.append("kruskal")
        done.append(
            pair.model_copy(
                update={
                    "anova_p_adjusted": float(a),
                    "kruskal_p_adjusted": float(k),
                    "flagged": bool(by),
                    "flagged_by": tuple(by),
                }
            )
        )
    return done


def _effect(pair: PairComparison) -> float:
    return max(pair.omega_squared, pair.epsilon_squared)


def _resample(
    groups: list[np.ndarray], params: GroupDifferencesParams, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Means and within-group sums of squares of stratified bootstrap resamples."""
    k = len(groups)
    chunk = max(1, _CHUNK_ELEMENTS // sum(g.size for g in groups))
    means = np.empty((params.n_boot, k))
    ssw = np.empty((params.n_boot, k))
    for start in range(0, params.n_boot, chunk):
        stop = min(start + chunk, params.n_boot)
        for i, values in enumerate(groups):
            sample = values[rng.integers(0, values.size, (stop - start, values.size))]
            mean = sample.mean(axis=1)
            deviations = sample - mean[:, None]
            means[start:stop, i] = mean
            ssw[start:stop, i] = np.einsum("ij,ij->i", deviations, deviations)
    return means, ssw


def _interval(draws: np.ndarray, low: float, high: float, params) -> Interval:
    tail = (1 - params.interval_level) / 2
    lower, upper = np.quantile(draws, [tail, 1 - tail])
    return Interval(
        low=float(np.clip(lower, low, high)),
        high=float(np.clip(upper, low, high)),
        level=params.interval_level,
        method="percentile_bootstrap",
    )


def _detail(
    pair: PairComparison,
    numeric: _Numeric,
    coded: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    params: GroupDifferencesParams,
) -> PairComparison:
    """The intervals and contrasts of a flagged pair."""
    data = _groups(numeric, coded, params)
    k = data.counts.size
    by_group = np.argsort(data.g, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(data.counts)])
    sorted_x, sorted_ranks = data.x[by_group], data.ranks[by_group]
    update: dict[str, object] = {}
    if "anova" in pair.flagged_by:
        values = [sorted_x[bounds[i] : bounds[i + 1]] for i in range(k)]
        means, ssw = _resample(values, params, rng)
        omega, _ = _anova_effect(data.counts, means, ssw, None)
        update["omega_interval"] = _interval(omega, 0.0, 1.0, params)
    if "kruskal" in pair.flagged_by:
        values = [sorted_ranks[bounds[i] : bounds[i + 1]] for i in range(k)]
        means, ssw = _resample(values, params, rng)
        grand = (means * data.counts).sum(axis=1) / data.counts.sum()
        ssb = (data.counts * (means - grand[:, None]) ** 2).sum(axis=1)
        total = ssb + ssw.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            epsilon = np.where(total > 0, ssb / total, 0.0)
        update["epsilon_interval"] = _interval(epsilon, 0.0, 1.0, params)
    if k > params.max_contrast_levels:
        update["contrasts_skipped"] = True
    else:
        update["contrasts"] = _contrasts(data, sorted_x, bounds, rng, params)
    return pair.model_copy(update=update)


def _contrasts(
    data: _Groups,
    sorted_x: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    params: GroupDifferencesParams,
) -> list[Contrast]:
    k = data.counts.size
    raw: list[tuple[int, int, float, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            a = sorted_x[bounds[i] : bounds[i + 1]]
            b = sorted_x[bounds[j] : bounds[j + 1]]
            pooled = rankdata(np.concatenate([a, b]))
            ranks_a, ranks_b = pooled[: a.size], pooled[a.size :]
            r = 2 * (ranks_a.mean() - ranks_b.mean()) / pooled.size
            p = float(mannwhitneyu(a, b).pvalue)
            raw.append((i, j, p if np.isfinite(p) else 1.0, float(np.clip(r, -1, 1))))
    adjusted = adjust_p_values(np.array([p for _, _, p, _ in raw]), "holm")
    contrasts = []
    for (i, j, p, r), p_adjusted in zip(raw, adjusted, strict=True):
        interval = None
        if p_adjusted < params.alpha:
            a = sorted_x[bounds[i] : bounds[i + 1]]
            b = sorted_x[bounds[j] : bounds[j + 1]]
            pooled = rankdata(np.concatenate([a, b]))
            means, _ = _resample([pooled[: a.size], pooled[a.size :]], params, rng)
            draws = 2 * (means[:, 0] - means[:, 1]) / pooled.size
            interval = _interval(draws, -1.0, 1.0, params)
        contrasts.append(
            Contrast(
                first_level=str(data.levels[i]),
                second_level=str(data.levels[j]),
                n_first=int(data.counts[i]),
                n_second=int(data.counts[j]),
                p_value=p,
                p_adjusted=float(p_adjusted),
                rank_biserial=r,
                interval=interval,
            )
        )
    return contrasts


def _confidence(pair: PairComparison) -> Confidence:
    if pair.smallest_group < 10 or pair.n_used < 30:
        return Confidence.LOW
    return Confidence.HIGH if pair.smallest_group >= 30 else Confidence.MODERATE


def _size(effect: float) -> str:
    return "large" if effect >= 0.14 else "medium" if effect >= 0.06 else "small"


def _report(
    ctx: CheckContext,
    pair: PairComparison,
    tested: int,
    flagged: int,
    params: GroupDifferencesParams,
) -> None:
    anova = "anova" in pair.flagged_by
    kruskal = "kruskal" in pair.flagged_by
    headline = "anova" if anova else "kruskal"
    effect = pair.omega_squared if anova else pair.epsilon_squared
    interval = pair.omega_interval if anova else pair.epsilon_interval
    symbol = "ω²" if anova else "ε²"
    reach = (
        f" ({params.interval_level:.0%} interval {interval.low:.2f} to "
        f"{interval.high:.2f})"
        if interval is not None
        else ""
    )
    first, second = pair.numeric, pair.categorical
    tests = " and ".join(
        {"anova": "ANOVA", "kruskal": "the Kruskal–Wallis test"}[m]
        for m in pair.flagged_by
    )
    disagreement = (
        " The rank test finds no difference, so the means differ without the "
        "typical value moving, which is often a few extreme values."
        if anova and not kruskal
        else " The ANOVA of the means finds no difference, so the groups differ in "
        "where their values sit but not in their means."
        if kruskal and not anova
        else ""
    )
    variance = (
        " The groups' spreads differ by more than a factor of "
        f"{_SD_RATIO_LIMIT:g}, which ANOVA assumes they do not; trust the rank "
        "test."
        if pair.sd_ratio is not None and pair.sd_ratio > _SD_RATIO_LIMIT and anova
        else ""
    )
    scan = (
        f" {tested:,} pairs were tested and {flagged:,} are flagged; the p-values "
        f"are adjusted ({params.correction}) for that."
        if tested > 1
        else ""
    )
    metrics: dict[str, object] = {
        "method": headline,
        "effect": effect,
        "omega_squared": pair.omega_squared,
        "eta_squared": pair.eta_squared,
        "epsilon_squared": pair.epsilon_squared,
        "anova_p_adjusted": pair.anova_p_adjusted,
        "kruskal_p_adjusted": pair.kruskal_p_adjusted,
        "h_statistic": pair.h_statistic,
        "n_groups": pair.n_groups,
        "smallest_group": pair.smallest_group,
        "top_group": pair.top_group,
        "bottom_group": pair.bottom_group,
        "correction": params.correction,
        "pairs_tested": tested,
        "pairs_flagged": flagged,
        "contrasts": len(pair.contrasts),
    }
    if pair.f_statistic is not None:
        metrics["f_statistic"] = pair.f_statistic
    if pair.sd_ratio is not None:
        metrics["sd_ratio"] = pair.sd_ratio
    if interval is not None:
        metrics["interval_low"] = interval.low
        metrics["interval_high"] = interval.high
    ctx.finding(
        code="group_difference",
        subject=Subject.column_pair(first, second),
        severity=Severity.INFO,
        confidence=_confidence(pair),
        title=f"{first!r} differs by {second!r} ({symbol} = {effect:.2f})",
        message=(
            f"{first!r} differs between the {pair.n_groups} groups of {second!r}: "
            f"{tests} find{'s' if len(pair.flagged_by) == 1 else ''} it "
            f"(adjusted p = {min(pair.anova_p_adjusted, pair.kruskal_p_adjusted):.2g}, "
            f"{symbol} = {effect:.3f}{reach}, a {_size(effect)} effect, over "
            f"{pair.n_used:,} rows). Values run highest in {pair.top_group!r} and "
            f"lowest in {pair.bottom_group!r}.{disagreement}{variance}{scan} A "
            "difference between groups is not a cause; check whether the grouping "
            "column leaks the numeric one or the reverse before using both."
        ),
        evidence=Evidence(
            metrics=metrics,  # type: ignore[arg-type]
            n_used=pair.n_used,
            n_total=ctx.profile.n_rows,
            method="+".join(pair.flagged_by),
            assumptions=(
                AssumptionCheck(
                    name="similar_group_spreads",
                    satisfied=(
                        None
                        if pair.sd_ratio is None
                        else pair.sd_ratio <= _SD_RATIO_LIMIT
                    ),
                    method="sd_ratio",
                    detail=(
                        "a group has no spread"
                        if pair.sd_ratio is None
                        else f"the largest group SD is {pair.sd_ratio:.1f} times "
                        "the smallest"
                    ),
                ),
                AssumptionCheck(
                    name="groups_of_at_least_5_rows",
                    satisfied=pair.smallest_group >= params.min_group_n,
                    method="group_sizes",
                    detail=(
                        f"{pair.smallest_group:,} to {pair.largest_group:,} rows per "
                        "group"
                    ),
                ),
            ),
            interval=interval,
        ),
    )
