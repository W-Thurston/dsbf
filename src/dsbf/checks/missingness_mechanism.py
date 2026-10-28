# dsbf/checks/missingness_mechanism.py
"""Whether the rows where a column is missing differ from the rows where it is not."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.stats import norm, rankdata

from dsbf.checks.pairs import (
    Correction,
    adjust_p_values,
    chi2_p_value,
    contingency_table,
    cramers_v,
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

Assessment = Literal["depends_on_observed", "no_dependence_found", "untestable"]
TestName = Literal["mann_whitney", "chi_squared"]
EffectName = Literal["rank_biserial", "cramers_v"]

# Predictors that are read as numbers (a datetime as its epoch) or as levels.
_NUMERIC_TYPES = ("numeric", "datetime")
_LEVEL_TYPES = ("categorical", "boolean")
# The Jaccard index from which two columns are said to go missing together.
_TOGETHER = 0.5
_STRONG_EFFECT = 0.3


class MissingnessMechanismParams(CheckParams):
    alpha: float = Field(
        0.05, gt=0, lt=1, description="Level of the adjusted p-value for a dependence."
    )
    correction: Correction = Field(
        "fdr_bh",
        description="Adjustment of the p-values for the number of tests run over "
        "all the columns with nulls and all their predictors. `fdr_bh` controls the "
        "false discovery rate.",
    )
    min_effect: float = Field(
        0.1,
        ge=0,
        lt=1,
        description="Effect size (|rank-biserial r| or Cramér's V) from which a "
        "significant test is reported; a huge sample makes a weak dependence "
        "significant.",
    )
    min_null_share: float = Field(
        0.01,
        ge=0,
        lt=1,
        description="Share of rows that must be null for a column to be analysed.",
    )
    min_group_n: int = Field(
        8,
        ge=2,
        description="Rows each of the missing and observed groups needs for a test.",
    )
    max_levels: int = Field(
        20, ge=2, description="Most levels a categorical predictor may have."
    )
    max_subjects: int = Field(
        30,
        ge=1,
        description="Most columns with nulls analysed, those with most nulls first.",
    )
    max_rows: int = Field(
        200_000,
        ge=1000,
        description="Most rows read; a larger frame is sampled with `seed`.",
    )
    seed: int = Field(42, description="Seed of the row sample.")
    max_listed: int = Field(
        10, ge=1, description="Most dependences listed for one column."
    )
    min_report_share: float = Field(
        0.05,
        ge=0,
        lt=1,
        description="Null share from which a column with no dependence found is "
        "still reported as a finding.",
    )
    warn_null_share: float = Field(
        0.2,
        gt=0,
        lt=1,
        description="Null share above which a strong dependence is a warning.",
    )


class Predictor(Contract):
    column: str
    test: TestName
    n_missing: int = Field(
        ge=1, description="Rows of the test where the column is null."
    )
    n_observed: int = Field(ge=1, description="Rows of the test where it is not.")
    statistic: float = Field(description="z for Mann–Whitney, χ² for chi-squared.")
    p_value: float = Field(ge=0, le=1)
    p_adjusted: float = Field(ge=0, le=1)
    effect_name: EffectName
    effect: float = Field(
        ge=-1,
        le=1,
        description="Rank-biserial r, positive when the predictor is higher where "
        "the column is null; or Cramér's V (bias-corrected).",
    )
    flagged: bool


class MissingColumn(Contract):
    n_used: int = Field(ge=1, description="Rows read.")
    n_missing: int = Field(ge=1)
    null_share: float = Field(gt=0, le=1)
    assessment: Assessment
    reason: str | None = Field(description="Why a column is `untestable`.")
    tests_run: int = Field(ge=0)
    tests_flagged: int = Field(ge=0)
    dependences: list[Predictor] = Field(
        description="The predictors with a flagged test, the largest effect first."
    )
    strongest: Predictor | None = Field(
        description="The largest effect among all the tests, flagged or not."
    )
    missing_with: dict[str, float] = Field(
        description="Columns that are null in the same rows (Jaccard of at least "
        "0.5), from the profile."
    )


class MissingnessMechanismOutput(CheckOutputModel):
    columns: dict[str, MissingColumn] = Field(
        description="Each analysed column with nulls, by name."
    )
    tests_run: int = Field(ge=0)
    tests_flagged: int = Field(ge=0)
    correction: Correction
    rows_used: int = Field(ge=0)
    skipped_columns: dict[str, str] = Field(
        description="Columns with nulls that were not analysed, with the reason."
    )
    unusable_predictors: dict[str, str] = Field(
        description="Columns left out as predictors, with the reason."
    )


@check(
    id="missingness_mechanism",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.RELATIONSHIP],
    params=MissingnessMechanismParams,
    output=MissingnessMechanismOutput,
    requires=Requires(profile=["counts"]),
    # One rank per numeric predictor and a matrix product with the null indicators.
    cost=Cost.MODERATE,
    codes=["missingness_depends_on_observed", "missingness_unexplained"],
    packs=["core"],
    tags=["missingness", "mcar", "mar", "mann_whitney", "chi_squared"],
)
def missingness_mechanism(
    ctx: CheckContext, params: MissingnessMechanismParams
) -> CheckOutput:
    """Test what the rows with a null in a column have in common.

    The null of each column is a 0/1 indicator, compared with every other column on
    the rows where that column has a value: numbers by Mann–Whitney, levels by
    chi-squared. Every p-value is adjusted together. A dependence says the nulls are
    not a random sample of the rows (not MCAR); it cannot tell missing at random
    from missing not at random, which depends on the unobserved values.
    """
    subjects, skipped = _subjects(ctx, params)
    if not subjects:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_missing_columns",
            "No column has enough nulls to analyse: at least "
            f"{params.min_null_share:.0%} of the rows and "
            f"{2 * params.min_group_n} rows.",
            severity=Severity.INFO,
        )
    rows = _rows(ctx, params)
    indicators = np.column_stack(
        [_is_null(ctx.values(name, dropna=False))[rows] for name in subjects]
    ).astype(np.float64)
    n_used = indicators.shape[0]
    tests, unusable = _run_tests(ctx, params, subjects, indicators, rows)

    adjusted = adjust_p_values(np.array([t.p_value for t in tests]), params.correction)
    predictors = [
        t.finish(float(p), params) for t, p in zip(tests, adjusted, strict=True)
    ]
    together = _together(ctx)
    columns: dict[str, MissingColumn] = {}
    for position, name in enumerate(subjects):
        own = [
            pred
            for t, pred in zip(tests, predictors, strict=True)
            if t.subject == position
        ]
        n_missing = int(indicators[:, position].sum())
        columns[name] = _column(own, n_used, n_missing, together.get(name, {}), params)
    for name, found in columns.items():
        _report(ctx, name, found, params, len(tests))
    flagged = sum(c.tests_flagged for c in columns.values())
    return ctx.output(
        MissingnessMechanismOutput(
            columns=columns,
            tests_run=len(tests),
            tests_flagged=flagged,
            correction=params.correction,
            rows_used=n_used,
            skipped_columns=skipped,
            unusable_predictors=unusable,
        )
    )


@dataclass
class _Test:
    """A test before its p-value is adjusted."""

    subject: int
    column: str
    test: TestName
    n_missing: int
    n_observed: int
    statistic: float
    p_value: float
    effect_name: EffectName
    effect: float

    def finish(
        self, p_adjusted: float, params: MissingnessMechanismParams
    ) -> Predictor:
        return Predictor(
            column=self.column,
            test=self.test,
            n_missing=self.n_missing,
            n_observed=self.n_observed,
            statistic=self.statistic,
            p_value=self.p_value,
            p_adjusted=p_adjusted,
            effect_name=self.effect_name,
            effect=self.effect,
            flagged=p_adjusted < params.alpha and abs(self.effect) >= params.min_effect,
        )


def _subjects(
    ctx: CheckContext, params: MissingnessMechanismParams
) -> tuple[list[str], dict[str, str]]:
    """The columns to analyse (most nulls first), and why the others are not."""
    candidates: list[tuple[int, str]] = []
    skipped: dict[str, str] = {}
    for column in ctx.profile.columns:
        counts = column.counts
        if not counts.nulls:
            continue
        floor = max(params.min_group_n, math.ceil(params.min_null_share * counts.rows))
        if counts.nulls < floor:
            skipped[column.name] = (
                f"{counts.nulls:,} nulls, fewer than the {floor:,} asked for"
            )
        elif counts.non_null < params.min_group_n:
            skipped[column.name] = f"{counts.non_null:,} values, too few to compare"
        else:
            candidates.append((counts.nulls, column.name))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, name in candidates[params.max_subjects :]:
        skipped[name] = f"beyond the {params.max_subjects} columns with most nulls"
    return [name for _, name in candidates[: params.max_subjects]], skipped


def _rows(ctx: CheckContext, params: MissingnessMechanismParams) -> np.ndarray | slice:
    """The rows read: all of them, or a seeded sample in row order."""
    n = ctx.profile.n_rows
    if n <= params.max_rows:
        return slice(None)
    chosen = np.random.default_rng(params.seed).choice(
        n, params.max_rows, replace=False
    )
    return np.sort(chosen)


def _is_null(values: np.ndarray) -> np.ndarray:
    """True where a column read with ``dropna=False`` holds a null."""
    kind = values.dtype.kind
    if kind == "f":
        return np.isnan(values)
    if kind in "mM":
        return np.isnat(values)
    if kind == "O":
        return np.fromiter(
            (v is None or v != v for v in values), dtype=bool, count=values.size
        )
    return np.zeros(values.size, dtype=bool)


def _run_tests(
    ctx: CheckContext,
    params: MissingnessMechanismParams,
    subjects: list[str],
    indicators: np.ndarray,
    rows: np.ndarray | slice,
) -> tuple[list[_Test], dict[str, str]]:
    """Test every subject's indicator against every other column."""
    tests: list[_Test] = []
    unusable: dict[str, str] = {}
    for name in ctx.columns:
        ctx.checkpoint()
        column = ctx.profile.column(name)
        kind = column.semantic_type
        if kind in _NUMERIC_TYPES:
            values = ctx.values(name, dropna=False)[rows]
            if values.dtype.kind not in "iufmM":
                unusable[name] = f"{kind} stored as {column.physical_dtype}, not read"
                continue
            tests.extend(_numeric_tests(name, values, subjects, indicators, params))
        elif kind in _LEVEL_TYPES:
            if column.counts.distinct < 2:
                unusable[name] = "fewer than two levels"
            elif column.counts.distinct > params.max_levels:
                unusable[name] = f"more than {params.max_levels} levels"
            else:
                codes, _ = level_codes(ctx.values(name, dropna=False)[rows])
                tests.extend(_level_tests(name, codes, subjects, indicators, params))
        else:
            unusable[name] = f"a {kind} column is not compared"
    return tests, unusable


def _numeric_tests(
    name: str,
    values: np.ndarray,
    subjects: list[str],
    indicators: np.ndarray,
    params: MissingnessMechanismParams,
) -> list[_Test]:
    """Mann–Whitney of ``name`` between the rows null and not null in each subject.

    The test runs on the rows where ``name`` has a finite value, which are the same
    for every subject, so the ranks are computed once and each subject's rank sum is
    a product with its indicator column.
    """
    if values.dtype.kind in "mM":
        usable = ~np.isnat(values)
        numbers = values.astype("datetime64[us]").astype(np.int64).astype(np.float64)
    else:
        numbers = values.astype(np.float64, copy=False)
        usable = np.isfinite(numbers)
    n = int(usable.sum())
    if n < 2 * params.min_group_n:
        return []
    ranks = np.zeros(numbers.size)
    ranks[usable] = rankdata(numbers[usable])
    _, ties = np.unique(numbers[usable], return_counts=True)
    tie_term = float(np.sum(ties.astype(np.float64) ** 3 - ties))
    if tie_term >= n**3 - n:
        return []
    weight = usable.astype(np.float64)
    n1 = weight @ indicators
    rank_sum = ranks @ indicators
    n2 = n - n1
    found: list[_Test] = []
    for subject, own in enumerate(subjects):
        if (
            own == name
            or n1[subject] < params.min_group_n
            or n2[subject] < params.min_group_n
        ):
            continue
        found.append(
            _mann_whitney(
                subject,
                name,
                float(rank_sum[subject]),
                int(n1[subject]),
                int(n2[subject]),
                tie_term,
            )
        )
    return found


def _mann_whitney(
    subject: int, name: str, rank_sum: float, n1: int, n2: int, tie_term: float
) -> _Test:
    """Mann–Whitney U from the rank sum of group 1, with the normal approximation."""
    total = n1 + n2
    u1 = rank_sum - n1 * (n1 + 1) / 2
    mean = n1 * n2 / 2
    variance = n1 * n2 / 12 * ((total + 1) - tie_term / (total * (total - 1)))
    deviation = abs(u1 - mean)
    z = max(deviation - 0.5, 0.0) / np.sqrt(variance)
    return _Test(
        subject=subject,
        column=name,
        test="mann_whitney",
        n_missing=n1,
        n_observed=n2,
        statistic=float(np.sign(u1 - mean) * z),
        p_value=float(min(1.0, 2 * norm.sf(z))),
        effect_name="rank_biserial",
        effect=float(np.clip(2 * u1 / (n1 * n2) - 1, -1, 1)),
    )


def _level_tests(
    name: str,
    codes: np.ndarray,
    subjects: list[str],
    indicators: np.ndarray,
    params: MissingnessMechanismParams,
) -> list[_Test]:
    """Chi-squared of the levels of ``name`` against each subject's indicator."""
    found: list[_Test] = []
    present = codes >= 0
    for subject, own in enumerate(subjects):
        if own == name:
            continue
        flags = indicators[present, subject].astype(np.int64)
        table, _, _ = contingency_table(flags, codes[present])
        # The indicator is a code too: 0 observed, 1 missing, both present.
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        n_observed, n_missing = int(table[0].sum()), int(table[1].sum())
        if min(n_observed, n_missing) < params.min_group_n:
            continue
        chi2, _, corrected = (float(x) for x in cramers_v(table))
        found.append(
            _Test(
                subject=subject,
                column=name,
                test="chi_squared",
                n_missing=n_missing,
                n_observed=n_observed,
                statistic=chi2,
                p_value=chi2_p_value(chi2, table),
                effect_name="cramers_v",
                effect=float(min(1.0, max(0.0, corrected))),
            )
        )
    return found


def _together(ctx: CheckContext) -> dict[str, dict[str, float]]:
    """For each column, the columns that are null in the same rows, from the profile."""
    found: dict[str, dict[str, float]] = {}
    for pair in ctx.profile.null_pairs:
        if pair.jaccard >= _TOGETHER:
            found.setdefault(pair.a, {})[pair.b] = pair.jaccard
            found.setdefault(pair.b, {})[pair.a] = pair.jaccard
    return found


def _column(
    own: list[Predictor],
    n_used: int,
    n_missing: int,
    together: dict[str, float],
    params: MissingnessMechanismParams,
) -> MissingColumn:
    flagged = sorted(
        (p for p in own if p.flagged), key=lambda p: (-abs(p.effect), p.column)
    )
    strongest = max(own, key=lambda p: (abs(p.effect), p.column), default=None)
    if not own:
        assessment: Assessment = "untestable"
        reason = "no other column has enough rows in both groups to compare"
    else:
        assessment = "depends_on_observed" if flagged else "no_dependence_found"
        reason = None
    return MissingColumn(
        n_used=n_used,
        n_missing=n_missing,
        null_share=n_missing / n_used,
        assessment=assessment,
        reason=reason,
        tests_run=len(own),
        tests_flagged=len(flagged),
        dependences=flagged[: params.max_listed],
        strongest=strongest,
        missing_with=dict(sorted(together.items())),
    )


def _quoted_predictors(found: MissingColumn, limit: int = 3) -> str:
    parts = []
    for p in found.dependences[:limit]:
        if p.test == "mann_whitney":
            where = "higher" if p.effect > 0 else "lower"
            parts.append(
                f"{p.column!r} ({where} where it is null, r = {p.effect:+.2f})"
            )
        else:
            parts.append(f"{p.column!r} (V = {p.effect:.2f})")
    return ", ".join(parts)


def _report(
    ctx: CheckContext,
    name: str,
    found: MissingColumn,
    params: MissingnessMechanismParams,
    total_tests: int,
) -> None:
    if found.assessment == "untestable":
        return
    subject = Subject.column(name)
    biggest = max((abs(p.effect) for p in found.dependences), default=0.0)
    metrics: dict[str, object] = {
        "null_share": found.null_share,
        "n_missing": found.n_missing,
        "tests_run": found.tests_run,
        "tests_flagged": found.tests_flagged,
        "max_effect": biggest,
        "correction": params.correction,
    }
    if found.strongest is not None:
        metrics["strongest_effect"] = abs(found.strongest.effect)
        metrics["strongest_p_adjusted"] = found.strongest.p_adjusted
    together = (
        f" It is null in the same rows as {_names(found.missing_with)}."
        if found.missing_with
        else ""
    )
    evidence = Evidence(
        metrics=metrics,  # type: ignore[arg-type]
        n_used=found.n_used,
        n_total=ctx.profile.n_rows,
        method="indicator_tests",
    )
    cause = (
        "A dependence does not separate data missing at random from data missing "
        "because of the values themselves; only knowing how the column was "
        "collected does."
    )
    if found.assessment == "depends_on_observed":
        strong = biggest >= _STRONG_EFFECT
        worst = min(p.p_adjusted for p in found.dependences)
        solid = (
            strong
            and worst < 0.001
            and min(p.n_missing for p in found.dependences) >= 30
        )
        ctx.finding(
            code="missingness_depends_on_observed",
            subject=subject,
            severity=(
                Severity.WARN
                if strong and found.null_share >= params.warn_null_share
                else Severity.INFO
            ),
            confidence=Confidence.HIGH if solid else Confidence.MODERATE,
            title=f"Nulls follow other columns ({found.tests_flagged} of "
            f"{found.tests_run} tests)",
            message=(
                f"{found.n_missing:,} of {found.n_used:,} values of {name!r} "
                f"({found.null_share:.1%}) are null, and the rows with a null differ "
                f"from the rest in {_quoted_predictors(found)}. {found.tests_flagged} "
                f"of {found.tests_run} tests flagged after adjusting for "
                f"the {total_tests:,} tests run on all columns (adjusted p < "
                f"{params.alpha}, effect of at least {params.min_effect}). Dropping "
                "the rows with a null would bias "
                f"what the other columns show.{together} {cause}"
            ),
            evidence=evidence,
        )
    elif found.null_share >= params.min_report_share:
        ctx.finding(
            code="missingness_unexplained",
            subject=subject,
            severity=Severity.INFO,
            confidence=Confidence.LOW,
            title="Nulls do not follow the other columns",
            message=(
                f"{found.n_missing:,} of {found.n_used:,} values of {name!r} "
                f"({found.null_share:.1%}) are null, and none of {found.tests_run} "
                "tests found the rows with a null to differ from the rest. That is "
                "consistent with nulls that fall at random, but does not show it: "
                f"the tests miss small effects in small groups.{together} {cause}"
            ),
            evidence=evidence,
        )


def _names(columns: dict[str, float]) -> str:
    return ", ".join(repr(name) for name in columns)
