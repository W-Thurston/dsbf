# dsbf/checks/collinearity.py
"""Numeric columns that a linear combination of the others nearly reproduces."""

from typing import Literal

import numpy as np
from pydantic import Field

from dsbf.checks.pairs import Undefined, unit_column
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

Tier = Literal["low", "moderate", "high", "severe"]

# The method stamped on every finding: the VIF of a column is 1 / (1 - R²) of its
# regression on the other columns *with an intercept*, which is the diagonal of the
# inverse correlation matrix. Without the intercept the VIF of a column with a large
# mean is inflated by the means alone (statsmodels before 0.15 computed that one).
METHOD = "vif_centred"
_VIF_CAP = 1e6
_EIGENVALUE_FLOOR = 1e-12
_SEVERE = 20.0


class CollinearityParams(CheckParams):
    vif_threshold: float = Field(
        10.0, gt=1, description="VIF from which a column is a warning."
    )
    moderate_threshold: float = Field(
        5.0, gt=1, description="VIF from which a column is reported at all."
    )
    max_missing_share: float = Field(
        0.4,
        ge=0,
        lt=1,
        description="Share of missing values above which a column is left out "
        "before any row is dropped, so a sparse column does not cost every row.",
    )
    min_rows: int = Field(
        10,
        ge=3,
        description="Complete rows needed, at least the number of columns plus one.",
    )
    max_columns: int = Field(
        200,
        ge=2,
        description="Most columns; with more, the check is skipped rather than "
        "read an arbitrary part of them.",
    )
    max_rows: int = Field(
        100_000,
        ge=1000,
        description="Most rows read; a larger frame is sampled with `seed`.",
    )
    seed: int = Field(42, description="Seed of the row sample.")
    max_findings: int = Field(
        50, ge=1, description="Most findings; the highest VIFs first."
    )


class Partner(Contract):
    column: str
    r: float = Field(ge=-1, le=1, description="Pearson r with the column.")


class ColumnVif(Contract):
    column: str
    vif: float = Field(
        ge=1, le=_VIF_CAP, description="1 for a column the others do not predict."
    )
    capped: bool = Field(
        description=f"The column is an exact linear combination of others and its "
        f"VIF is reported as {_VIF_CAP:.0e}."
    )
    tier: Tier = Field(
        description="low under 5, moderate, high from 10, severe from 20."
    )
    partners: tuple[Partner, ...] = Field(
        description="The columns it correlates with most; empty for a low VIF."
    )


class CollinearityOutput(CheckOutputModel):
    definition: str = Field(description="How the VIF is defined: `vif_centred`.")
    columns: list[ColumnVif] = Field(description="Every column the VIF is defined for.")
    n_used: int = Field(
        ge=0, description="Rows with a value in every column used, after sampling."
    )
    n_columns: int = Field(ge=0, description="Columns the VIFs were computed from.")
    rows_total: int = Field(ge=0)
    rows_read: int = Field(ge=0, description="Rows read, after any sampling.")
    columns_flagged: int = Field(ge=0, description="Columns at or above the threshold.")
    skipped_columns: dict[str, str] = Field(
        description="Columns left out before rows were dropped, with the reason."
    )
    dropped_for_rows: list[str] = Field(
        description="Columns dropped, sparsest first, so that enough rows were left "
        "with a value in every column."
    )
    condition_number: float = Field(
        ge=1, description="Largest over smallest eigenvalue of the correlation matrix."
    )


@check(
    id="collinearity",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.REDUNDANCY],
    params=CollinearityParams,
    output=CollinearityOutput,
    requires=Requires(profile=["counts"]),
    # One correlation matrix and its eigendecomposition.
    cost=Cost.MODERATE,
    codes=["multicollinearity"],
    packs=["core"],
    tags=["multicollinearity", "vif", "redundancy"],
)
def collinearity(ctx: CheckContext, params: CollinearityParams) -> CheckOutput:
    """Measure how well each numeric column is predicted by all the others."""
    names = list(ctx.columns)
    if len(names) < 2:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "too_few_columns",
            f"Needs two numeric columns; there are {len(names)}.",
            severity=Severity.INFO,
        )
    if len(names) > params.max_columns:
        return ctx.stop(
            CheckStatus.SKIPPED,
            "too_many_columns",
            f"{len(names):,} columns would need a {len(names):,} × {len(names):,} "
            f"matrix, over the limit of {params.max_columns:,}.",
            hint="Raise `max_columns`, or run the check on fewer columns.",
        )

    total_rows = ctx.profile.n_rows
    rows = None
    if total_rows > params.max_rows:
        rng = np.random.default_rng(params.seed)
        rows = np.sort(rng.choice(total_rows, params.max_rows, replace=False))
        ctx.note(
            "rows_sampled",
            f"Read {rows.size:,} of {total_rows:,} rows, sampled with seed "
            f"{params.seed}.",
        )
    rows_read = total_rows if rows is None else int(rows.size)

    columns: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    for name in names:
        values = _read(ctx, name, rows)
        present = ~np.isnan(values)
        share = 1.0 - float(present.mean()) if values.size else 1.0
        if share > params.max_missing_share:
            skipped[name] = f"{share:.0%} of values are missing"
        elif not present.any() or np.nanmin(values) == np.nanmax(values):
            skipped[name] = "no variation"
        else:
            columns[name] = values

    dropped: list[str] = []
    while True:
        ctx.checkpoint()
        kept = list(columns)
        if len(kept) < 2:
            return ctx.stop(
                CheckStatus.NOT_APPLICABLE,
                "too_few_columns",
                f"Needs two numeric columns with values; {len(kept)} remain after "
                f"leaving out {len(skipped) + len(dropped)}.",
                severity=Severity.INFO,
            )
        matrix = np.column_stack([columns[name] for name in kept])
        complete = ~np.isnan(matrix).any(axis=1)
        needed = max(len(kept) + 1, params.min_rows)
        if int(complete.sum()) >= needed:
            units, undefined = _unit_columns(matrix[complete], kept)
            if not undefined:
                break
            # A column constant over the complete rows: leave it out and retry.
            for name, why in undefined.items():
                skipped[name] = (
                    "no variation among the complete rows"
                    if why == "no_variation"
                    else "values too large to compute with"
                )
                del columns[name]
            continue
        sparsest = max(kept, key=lambda n: float(np.isnan(columns[n]).sum()))
        if float(np.isnan(columns[sparsest]).sum()) == 0:
            return ctx.stop(
                CheckStatus.NOT_APPLICABLE,
                "too_few_rows",
                f"{int(complete.sum()):,} rows are complete for {len(kept)} columns, "
                f"fewer than the {needed:,} needed.",
                severity=Severity.INFO,
                hint="Collect more rows, or run the check on fewer columns.",
            )
        dropped.append(sparsest)
        del columns[sparsest]

    n_used = int(complete.sum())
    corr = units.T @ units
    np.fill_diagonal(corr, 1.0)
    eigenvalues, vectors = np.linalg.eigh(corr)
    floored = np.maximum(eigenvalues, _EIGENVALUE_FLOOR)
    raw_vif = (vectors**2 / floored).sum(axis=1)
    condition = float(min(floored.max() / floored.min(), 1e12))

    results: list[ColumnVif] = []
    for i, name in enumerate(kept):
        vif = float(raw_vif[i])
        capped = vif >= _VIF_CAP
        vif = float(np.clip(vif, 1.0, _VIF_CAP))
        tier = _tier(vif, params)
        results.append(
            ColumnVif(
                column=name,
                vif=vif,
                capped=capped,
                tier=tier,
                partners=(
                    _partners(corr, kept, i) if vif >= params.moderate_threshold else ()
                ),
            )
        )

    flagged = sorted(
        (c for c in results if c.vif >= params.moderate_threshold),
        key=lambda c: -c.vif,
    )
    shown = flagged[: params.max_findings]
    if len(flagged) > len(shown):
        ctx.note(
            "findings_capped",
            f"{len(flagged):,} columns have a VIF of {params.moderate_threshold:g} "
            f"or more; the findings cover the highest {len(shown):,}, all are in "
            "the output.",
        )
    for column in shown:
        _report(ctx, column, n_used, len(kept), params)
    return ctx.output(
        CollinearityOutput(
            definition=METHOD,
            columns=results,
            n_used=n_used,
            n_columns=len(kept),
            rows_total=total_rows,
            rows_read=rows_read,
            columns_flagged=sum(c.vif >= params.vif_threshold for c in results),
            skipped_columns=skipped,
            dropped_for_rows=dropped,
            condition_number=condition,
        )
    )


def _read(ctx: CheckContext, name: str, rows: np.ndarray | None) -> np.ndarray:
    raw = ctx.values(name, dropna=False)
    values = np.asarray(raw if rows is None else raw[rows], dtype=np.float64)
    return np.where(np.isfinite(values), values, np.nan)


def _unit_columns(
    matrix: np.ndarray, names: list[str]
) -> tuple[np.ndarray, dict[str, str]]:
    """The columns of ``matrix`` centred and scaled, and the ones that cannot be."""
    units: list[np.ndarray] = []
    undefined: dict[str, str] = {}
    for i, name in enumerate(names):
        try:
            units.append(unit_column(matrix[:, i]))
        except Undefined as why:
            undefined[name] = str(why)
    return (np.column_stack(units) if not undefined else matrix), undefined


def _tier(vif: float, params: CollinearityParams) -> Tier:
    if vif >= _SEVERE:
        return "severe"
    if vif >= params.vif_threshold:
        return "high"
    return "moderate" if vif >= params.moderate_threshold else "low"


def _partners(corr: np.ndarray, names: list[str], index: int) -> tuple[Partner, ...]:
    row = np.clip(corr[index], -1.0, 1.0)
    order = [j for j in np.argsort(-np.abs(row), kind="stable") if j != index][:2]
    return tuple(Partner(column=names[j], r=float(row[j])) for j in order)


def _confidence(n_used: int) -> Confidence:
    if n_used >= 100:
        return Confidence.HIGH
    return Confidence.MODERATE if n_used >= 30 else Confidence.LOW


def _report(
    ctx: CheckContext,
    column: ColumnVif,
    n_used: int,
    n_columns: int,
    params: CollinearityParams,
) -> None:
    name, vif = column.column, column.vif
    shown = f"{vif:,.1f}" if not column.capped else f"at least {_VIF_CAP:,.0f}"
    near = " and ".join(f"{p.column!r} (r = {p.r:.2f})" for p in column.partners)
    severity = Severity.WARN if vif >= params.vif_threshold else Severity.INFO
    metrics: dict[str, object] = {
        "vif": vif,
        "capped": column.capped,
        "tier": column.tier,
        "vif_threshold": params.vif_threshold,
        "n_columns": n_columns,
    }
    for position, partner in enumerate(column.partners, start=1):
        metrics[f"partner_{position}"] = partner.column
        metrics[f"partner_{position}_r"] = partner.r
    ctx.finding(
        code="multicollinearity",
        subject=Subject.column(name),
        severity=severity,
        confidence=_confidence(n_used),
        title=f"{column.tier.capitalize()} multicollinearity (VIF = {vif:,.1f})"
        if not column.capped
        else "Exact linear combination of other columns",
        message=(
            f"{name!r} has a variance inflation factor of {shown}: the other "
            f"{n_columns - 1} numeric columns predict {1 - 1 / vif:.1%} of its "
            f"variance (computed over {n_used:,} rows, with an intercept). "
            f"It correlates most with {near}. Linear models cannot tell these "
            "columns apart, so their coefficients are unstable; tree-based models "
            "are not harmed, but split importance between the columns."
        ),
        evidence=Evidence(
            metrics=metrics,  # type: ignore[arg-type]
            n_used=n_used,
            n_total=ctx.profile.n_rows,
            method=METHOD,
        ),
    )
