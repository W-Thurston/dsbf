# dsbf/checks/reference_drift.py
"""Columns whose type, missingness or distribution differ from a reference dataset."""

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import chi2_contingency

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
    CheckStatus,
    ColumnProfile,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    NumericProfile,
    Severity,
    Subject,
)

Compared = Literal["numeric", "categorical", "none"]
Drift = Literal["none", "moderate", "high"]
# Added to each bin's count on both sides, so an empty bin does not make PSI infinite.
_SMOOTHING = 0.5
_SEVERITIES = (Severity.INFO, Severity.NOTICE, Severity.WARN, Severity.ERROR)
_TOP_SHIFTS = 3


class ReferenceDriftParams(CheckParams):
    psi_warn: float = Field(
        0.1,
        gt=0,
        description="Population stability index from which a numeric column has "
        "moderate drift. 0.1 and 0.25 are the conventional limits.",
    )
    psi_high: float = Field(
        0.25, gt=0, description="Index from which the drift is high."
    )
    tvd_warn: float = Field(
        0.2,
        gt=0,
        le=1,
        description="Total variation distance between the shares of a categorical "
        "column's values from which it has moderate drift.",
    )
    tvd_high: float = Field(
        0.4, gt=0, le=1, description="Distance from which the drift is high."
    )
    bins: int = Field(
        10,
        ge=2,
        le=100,
        description="Most bins a numeric distribution is compared on. The "
        "reference's histogram is merged into bins that each hold at least this "
        "share of it; PSI grows with the number of bins, and its usual limits "
        "assume about ten.",
    )
    significance: float = Field(
        0.01,
        gt=0,
        lt=1,
        description="Level of the chi-squared test that the two samples are of one "
        "distribution (on the PSI for a number, on the counts for a category). A "
        "distance below its limits or a test above this level is no drift: a small "
        "sample has a large distance by chance, and a huge one a significant "
        "difference too small to matter.",
    )
    missing_shift: float = Field(
        0.1,
        gt=0,
        le=1,
        description="Change in a column's share of nulls from which it is reported; "
        "twice the change is a warning.",
    )
    target: str | None = Field(
        None,
        description="The column to predict. `dsbf.profile(target=...)` sets it; "
        "drift in the target is reported one level higher.",
    )
    target_scale: float = Field(
        0.5,
        gt=0,
        le=1,
        description="The target is held to this share of the PSI and distance "
        "limits: a model learns the target itself, and a 15-point move in the "
        "share of positives is a total variation distance of 0.15.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> "ReferenceDriftParams":
        if self.psi_warn > self.psi_high:
            raise ValueError("psi_warn is at most psi_high")
        if self.tvd_warn > self.tvd_high:
            raise ValueError("tvd_warn is at most tvd_high")
        return self


class ColumnDrift(Contract):
    compared: Compared = Field(
        description="`numeric` or `categorical` by the distance used, `none` when the "
        "column's distribution is not compared."
    )
    reason: str | None = Field(description="Why a column is not compared, else None.")
    drift: Drift
    psi: float | None = Field(description="Population stability index; numeric only.")
    js_divergence: float | None = Field(
        description="Jensen–Shannon divergence in bits, between 0 and 1; numeric only."
    )
    tvd: float | None = Field(description="Total variation distance; categorical only.")
    null_share: float
    reference_null_share: float | None = Field(
        description="None when the reference lacks the column."
    )


class ReferenceDriftOutput(CheckOutputModel):
    reference: str
    reference_kind: str
    reference_rows: int = Field(ge=0, description="Rows of the reference's source.")
    reference_rows_profiled: int = Field(ge=0)
    rows: int = Field(ge=0, description="Rows profiled in this dataset.")
    columns_added: tuple[str, ...] = Field(
        description="Columns of this dataset the reference lacks."
    )
    columns_dropped: tuple[str, ...] = Field(
        description="Columns of the reference this dataset lacks."
    )
    columns: dict[str, ColumnDrift]
    drifted: tuple[str, ...] = Field(description="Columns with moderate or high drift.")


def coarsen(
    edges: Sequence[float], counts: Sequence[int], min_count: int
) -> tuple[list[float], list[int]]:
    """Merge neighbouring bins until each holds ``min_count`` or the rest of them."""
    kept = [edges[0]]
    merged: list[int] = []
    held = 0
    for index, count in enumerate(counts):
        held += count
        if held >= min_count:
            merged.append(held)
            kept.append(edges[index + 1])
            held = 0
    if held:
        if merged:
            merged[-1] += held
            kept[-1] = edges[-1]
        else:
            merged, kept = [held], [edges[0], edges[-1]]
    return kept, merged


def bin_counts(edges: Sequence[float], values: np.ndarray) -> np.ndarray:
    """Count ``values`` below the first edge, in each bin, then above the last.

    The bins are left-closed and the last is closed on both sides, as in a
    histogram; a point mass is the edges ``[v, v]``.
    """
    bins = len(edges) - 1
    place = np.searchsorted(np.asarray(edges), values, side="right")
    place[values == edges[-1]] = bins
    return np.bincount(place, minlength=bins + 2)


def population_stability(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI of two count vectors, each bin smoothed by half a count."""
    bins = len(reference)
    p = (reference + _SMOOTHING) / (reference.sum() + _SMOOTHING * bins)
    q = (current + _SMOOTHING) / (current.sum() + _SMOOTHING * bins)
    return float(np.sum((q - p) * np.log(q / p)))


def jensen_shannon(reference: np.ndarray, current: np.ndarray) -> float:
    """Jensen–Shannon divergence, in bits, of two count vectors."""
    p = reference / reference.sum()
    q = current / current.sum()
    mix = (p + q) / 2

    def divergence(a: np.ndarray) -> float:
        used = a > 0
        return float(np.sum(a[used] * np.log2(a[used] / mix[used])))

    return min(1.0, max(0.0, (divergence(p) + divergence(q)) / 2))


def total_variation(
    reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray
) -> float:
    """Half the summed absolute difference of the shares of two count vectors."""
    p = np.asarray(reference, dtype=float)
    q = np.asarray(current, dtype=float)
    return float(np.abs(p / p.sum() - q / q.sum()).sum() / 2)


def _homogeneity_p(
    reference: Sequence[float] | np.ndarray, current: Sequence[float] | np.ndarray
) -> float:
    """The p-value of the chi-squared test that two count vectors are of one law."""
    table = np.array([reference, current], dtype=float)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(chi2_contingency(table)[1])


def _tvd_noise(reference: Sequence[float], current: Sequence[float]) -> float:
    """The total variation distance two samples of one distribution are apart.

    Each share differs by about its standard error, so half the sum of the expected
    absolute differences. It is large for a small sample of many values, which a
    fixed limit would call drift.
    """
    ref = np.asarray(reference, dtype=float)
    now = np.asarray(current, dtype=float)
    pooled = (ref + now) / (ref.sum() + now.sum())
    spread = np.sqrt(pooled * (1 - pooled) * (1 / ref.sum() + 1 / now.sum()))
    return float(0.5 * np.sqrt(2 / np.pi) * spread.sum())


def _raised(severity: Severity) -> Severity:
    return _SEVERITIES[min(_SEVERITIES.index(severity) + 1, len(_SEVERITIES) - 1)]


def _confidence(*sizes: int) -> Confidence:
    smallest = min(sizes)
    if smallest >= 200:
        return Confidence.HIGH
    return Confidence.MODERATE if smallest >= 30 else Confidence.LOW


def _share(count: int, rows: int) -> float:
    return count / rows if rows else 0.0


def _reference_bins(
    column: NumericProfile, bins: int
) -> tuple[list[float], list[int]] | None:
    """The reference's coarsened bins, or None when it has no values to bin."""
    if column.histogram is not None:
        total = column.histogram.total
        floor = max(1, math.ceil(total / bins))
        return coarsen(column.histogram.edges, column.histogram.counts, floor)
    stats = column.stats
    if stats is not None and stats.min is not None and stats.min == stats.max:
        # A constant column keeps no histogram; its one value is a point mass.
        finite = column.counts.non_null - column.non_finite
        if finite > 0:
            return [stats.min, stats.min], [finite]
    return None


class _Level:
    """The drift level of a distance and the severity it is reported at."""

    def __init__(self, value: float, warn: float, high: float) -> None:
        self.drift: Drift = (
            "high" if value >= high else "moderate" if value >= warn else "none"
        )
        self.severity = {
            "none": Severity.INFO,
            "moderate": Severity.NOTICE,
            "high": Severity.WARN,
        }[self.drift]


@check(
    id="reference_drift",
    version="1.1.0",
    kind="analysis",
    # Every column, so a column that has changed type or gone missing is known.
    applies_to=SemanticTypes(),
    dimensions=[Dimension.DISTRIBUTION, Dimension.VALIDITY, Dimension.COMPLETENESS],
    params=ReferenceDriftParams,
    output=ReferenceDriftOutput,
    requires=Requires(
        profile=["counts"],
        reference=True,
    ),
    # One pass over each numeric and categorical column's values.
    cost=Cost.MODERATE,
    packs=["core"],
    tags=["drift", "reference", "schema", "comparison"],
)
def reference_drift(ctx: CheckContext, params: ReferenceDriftParams) -> CheckOutput:
    """Compare each column's type, nulls and distribution with the reference's."""
    reference = ctx.reference
    ref_columns = {c.name: c for c in reference.profile.columns}
    current = {c.name: c for c in ctx.profile.columns}
    shared = [name for name in current if name in ref_columns]
    if not shared:
        return ctx.stop(
            CheckStatus.SKIPPED,
            "no_shared_columns",
            f"No column of the data is in the reference {reference.name!r}, "
            "so there is nothing to compare.",
            hint="Check that both tables have the same column names.",
        )
    added = tuple(name for name in current if name not in ref_columns)
    dropped = tuple(name for name in ref_columns if name not in current)
    rows = ctx.profile.n_rows
    ref_rows = reference.profile.n_rows
    _note_samplings(ctx)
    _report_schema(ctx, reference.name, added, dropped, len(current), len(ref_columns))

    target = params.target
    if target is not None and target not in current:
        ctx.note(
            "target_not_found",
            f"The target {target!r} is not a column of the data, so its drift is "
            "reported like any other column's.",
            severity=Severity.INFO,
        )
        target = None
    if target is not None and target not in ref_columns:
        ctx.finding(
            code="target_missing_in_reference",
            subject=Subject.column(target),
            dimension=Dimension.VALIDITY,
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"The target {target!r} is not in the reference",
            message=(
                f"The reference {reference.name!r} has no column {target!r}, so the "
                "drift of the target cannot be measured."
            ),
            evidence=Evidence(
                metrics={"target_missing": True, "reference": reference.name},
                n_used=0,
                n_total=rows,
                method="schema_comparison",
            ),
        )

    results: dict[str, ColumnDrift] = {}
    for name in shared:
        ctx.checkpoint()
        results[name] = _compare_column(
            ctx, params, current[name], ref_columns[name], is_target=name == target
        )
    return ctx.output(
        ReferenceDriftOutput(
            reference=reference.name,
            reference_kind=reference.kind,
            reference_rows=reference.n_rows,
            reference_rows_profiled=ref_rows,
            rows=rows,
            columns_added=added,
            columns_dropped=dropped,
            columns=results,
            drifted=tuple(n for n, d in results.items() if d.drift != "none"),
        )
    )


def _note_samplings(ctx: CheckContext) -> None:
    reference = ctx.reference
    ours, theirs = ctx.sampling, reference.sampling
    if ours is None and theirs is None:
        return
    here = (
        f"{ours.sampled_rows:,} of {ours.original_rows:,} rows"
        if ours
        else f"all {ctx.profile.n_rows:,} rows"
    )
    there = (
        f"{theirs.sampled_rows:,} of {theirs.original_rows:,} rows"
        if theirs
        else f"all {reference.profile.n_rows:,} rows"
    )
    ctx.note(
        "sampled_comparison",
        f"The data is {here} and the reference {reference.name!r} is {there}; "
        "small differences in a distribution can come from the samples.",
        severity=Severity.INFO,
    )


def _report_schema(
    ctx: CheckContext,
    reference: str,
    added: tuple[str, ...],
    dropped: tuple[str, ...],
    n_columns: int,
    n_reference: int,
) -> None:
    for code, names, noun, where in (
        ("columns_added", added, "new", "are not in the reference"),
        ("columns_dropped", dropped, "missing", "are in the reference but not here"),
    ):
        if not names:
            continue
        listed = ", ".join(map(repr, names[:10]))
        more = f" and {len(names) - 10} more" if len(names) > 10 else ""
        ctx.finding(
            code=code,
            subject=Subject.dataset(),
            dimension=Dimension.VALIDITY,
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"{len(names):,} {noun} column{'s' if len(names) > 1 else ''} "
            f"against the reference: {listed}{more}",
            message=(
                f"{len(names):,} column{'s' if len(names) > 1 else ''} {where} "
                f"{reference!r}: {listed}{more}. Code or a model built on the "
                "reference expects its columns."
            ),
            evidence=Evidence(
                metrics={"columns": names, "n_columns": len(names)},
                n_used=n_columns,
                n_total=max(n_columns, n_reference),
                method="schema_comparison",
            ),
        )


def _compare_column(
    ctx: CheckContext,
    params: ReferenceDriftParams,
    column: ColumnProfile,
    ref: ColumnProfile,
    *,
    is_target: bool,
) -> ColumnDrift:
    name = column.name
    counts, ref_counts = column.counts, ref.counts
    null_share = _share(counts.nulls, counts.rows)
    ref_null_share = _share(ref_counts.nulls, ref_counts.rows)
    shift = abs(null_share - ref_null_share)
    if shift >= params.missing_shift:
        ctx.finding(
            code="missingness_shift",
            subject=Subject.column(name),
            dimension=Dimension.COMPLETENESS,
            severity=Severity.WARN
            if shift >= 2 * params.missing_shift
            else Severity.NOTICE,
            confidence=_confidence(counts.rows, ref_counts.rows),
            title=f"{name!r}: nulls went from {ref_null_share:.1%} to {null_share:.1%}",
            message=(
                f"{name!r} is {null_share:.1%} null here and was "
                f"{ref_null_share:.1%} null in the reference: a change of "
                f"{shift:.1%}. A source that stopped filling a field, or a join "
                "that lost matches, shows up this way."
            ),
            evidence=Evidence(
                metrics={
                    "null_share": null_share,
                    "reference_null_share": ref_null_share,
                    "change": null_share - ref_null_share,
                    "threshold": params.missing_shift,
                },
                n_used=counts.rows,
                n_total=counts.rows,
                method="null_share_difference",
            ),
        )
    base = {
        "null_share": null_share,
        "reference_null_share": ref_null_share,
    }
    if _family(column.physical_dtype) != _family(ref.physical_dtype):
        ctx.finding(
            code="type_changed",
            subject=Subject.column(name),
            dimension=Dimension.VALIDITY,
            severity=Severity.WARN,
            confidence=Confidence.HIGH,
            title=f"{name!r} was {ref.physical_dtype}, now {column.physical_dtype}",
            message=(
                f"{name!r} is stored as {column.physical_dtype} here and was "
                f"{ref.physical_dtype} in the reference. Its distribution is not "
                "compared."
            ),
            evidence=Evidence(
                metrics={
                    "type_changed": True,
                    "physical_dtype": column.physical_dtype,
                    "reference_physical_dtype": ref.physical_dtype,
                    "semantic_type": column.semantic_type,
                    "reference_semantic_type": ref.semantic_type,
                },
                n_used=counts.non_null,
                n_total=counts.rows,
                method="schema_comparison",
            ),
        )
        return _undetermined("the type changed", base)
    # The reference says how to compare: a constant or few-valued number is
    # `categorical` to the profile in one dataset and `numeric` in the other.
    if ref_counts.non_null == 0:
        return _undetermined("the reference column has no values", base)
    if isinstance(ref, NumericProfile):
        return _compare_numeric(ctx, params, column, ref, base, is_target)
    if isinstance(ref, CategoricalProfile | BooleanProfile):
        return _compare_categorical(ctx, params, column, ref, base, is_target)
    return _undetermined(
        f"a {ref.semantic_type} column's distribution is not compared", base
    )


def _family(dtype: str) -> str:
    """The kind of storage a physical dtype is: whole and fractional numbers are one."""
    name = dtype.lower()
    if name.startswith(("int", "uint", "float", "decimal")):
        return "number"
    if name.startswith(("string", "utf8", "categorical", "enum", "object")):
        return "text"
    if name.startswith(("date", "time", "duration")):
        return "temporal"
    return name


def _undetermined(reason: str, base: dict[str, float]) -> ColumnDrift:
    return ColumnDrift(
        compared="none",
        reason=reason,
        drift="none",
        psi=None,
        js_divergence=None,
        tvd=None,
        **base,
    )


def _compare_numeric(
    ctx: CheckContext,
    params: ReferenceDriftParams,
    column: ColumnProfile,
    ref: NumericProfile,
    base: dict[str, float],
    is_target: bool,
) -> ColumnDrift:
    name = column.name
    bins = _reference_bins(ref, params.bins)
    values = np.asarray(ctx.values(name), dtype=float)
    values = values[np.isfinite(values)]
    if bins is None or not len(values):
        return _undetermined("one side has no finite values to compare", base)
    edges, ref_counts = bins
    n_ref = sum(ref_counts)
    reference = np.array([0, *ref_counts, 0])
    now = bin_counts(edges, values)
    psi = population_stability(reference, now)
    js = jensen_shannon(reference, now)
    # What PSI reaches between two samples of one distribution. It is subtracted, so
    # that a small sample, whose PSI is high by chance alone, is not called drifting.
    noise = (len(reference) - 1) * (1 / n_ref + 1 / len(values))
    scale = params.target_scale if is_target else 1.0
    p_value = _homogeneity_p(reference, now)
    level = _Level(
        max(0.0, psi - noise) if p_value < params.significance else 0.0,
        params.psi_warn * scale,
        params.psi_high * scale,
    )
    if level.drift != "none":
        outside = float((now[0] + now[-1]) / len(values))
        _report_shift(
            ctx,
            column,
            is_target=is_target,
            level=level,
            n_used=len(values),
            n_reference=n_ref,
            method="psi",
            metrics={
                "psi": psi,
                "psi_excess": max(0.0, psi - noise),
                "js_divergence": js,
                "psi_noise": noise,
                "p_value": p_value,
                "n_bins": len(ref_counts),
                "share_outside_reference_range": outside,
                "mean": getattr(getattr(column, "stats", None), "mean", None),
                "reference_mean": ref.stats.mean if ref.stats else None,
                "warn": params.psi_warn * scale,
                "high": params.psi_high * scale,
            },
            what=f"PSI {psi:.2f} (JS divergence {js:.2f})",
        )
    return ColumnDrift(
        compared="numeric",
        reason=None,
        drift=level.drift,
        psi=psi,
        js_divergence=js,
        tvd=None,
        **base,
    )


def _levels(column: ColumnProfile) -> tuple[dict[Hashable, int], bool]:
    """A reference column's listed values and whether they are all of them."""
    if isinstance(column, BooleanProfile):
        listed: dict[Hashable, int] = {
            True: column.true_count,
            False: column.false_count,
        }
        return {k: v for k, v in listed.items() if v}, True
    assert isinstance(column, CategoricalProfile)
    listed = {v.value: v.count for v in column.top_values}
    return listed, sum(listed.values()) == column.counts.non_null


def _compare_categorical(
    ctx: CheckContext,
    params: ReferenceDriftParams,
    column: ColumnProfile,
    ref: CategoricalProfile | BooleanProfile,
    base: dict[str, float],
    is_target: bool,
) -> ColumnDrift:
    listed, complete = _levels(ref)
    n_ref = ref.counts.non_null
    n_now = column.counts.non_null
    if not listed or n_now == 0:
        return _undetermined("one side has no values to compare", base)
    seen = _counts(ctx, column)
    keys = list(listed)
    ref_vector = [listed[key] for key in keys]
    now_vector = [seen.get(key, 0) for key in keys]
    # The rest: values the reference did not list. Its own are none if it listed all.
    ref_vector.append(n_ref - sum(ref_vector))
    now_vector.append(n_now - sum(now_vector))
    tvd = total_variation(ref_vector, now_vector)
    noise = _tvd_noise(ref_vector, now_vector)
    scale = params.target_scale if is_target else 1.0
    p_value = _homogeneity_p(ref_vector, now_vector)
    level = _Level(
        max(0.0, tvd - noise) if p_value < params.significance else 0.0,
        params.tvd_warn * scale,
        params.tvd_high * scale,
    )
    if level.drift != "none":
        p = np.asarray(ref_vector, dtype=float) / n_ref
        q = np.asarray(now_vector, dtype=float) / n_now
        order = np.argsort(-np.abs(q - p))[:_TOP_SHIFTS]
        names = [*(str(key) for key in keys), "(other values)"]
        _report_shift(
            ctx,
            column,
            is_target=is_target,
            level=level,
            n_used=n_now,
            n_reference=n_ref,
            method="tvd",
            metrics={
                "tvd": tvd,
                "tvd_noise": noise,
                "p_value": p_value,
                "listed_values_complete": complete,
                "largest_shifts": {names[i]: float(q[i] - p[i]) for i in order},
                "new_value_share": float(q[-1]) if complete else None,
                "warn": params.tvd_warn * scale,
                "high": params.tvd_high * scale,
            },
            what=f"total variation distance {tvd:.2f}",
        )
    return ColumnDrift(
        compared="categorical",
        reason=None,
        drift=level.drift,
        psi=None,
        js_divergence=None,
        tvd=tvd,
        **base,
    )


def _counts(ctx: CheckContext, column: ColumnProfile) -> Counter[Hashable]:
    if isinstance(column, BooleanProfile):
        return Counter({True: column.true_count, False: column.false_count})
    return Counter(ctx.values(column.name).tolist())


def _report_shift(
    ctx: CheckContext,
    column: ColumnProfile,
    *,
    is_target: bool,
    level: _Level,
    n_used: int,
    n_reference: int,
    method: str,
    metrics: dict,
    what: str,
) -> None:
    name = column.name
    reference = ctx.reference
    severity = _raised(level.severity) if is_target else level.severity
    role = "target" if is_target else "column"
    ctx.finding(
        code="target_drift" if is_target else "distribution_shift",
        subject=Subject.column(name),
        dimension=Dimension.DISTRIBUTION,
        severity=severity,
        confidence=_confidence(n_used, n_reference),
        title=f"{level.drift.capitalize()} drift in the {role} {name!r}: {what}",
        message=(
            f"The distribution of the {role} {name!r} differs from the reference "
            f"{reference.name!r}: {what}, {level.drift} drift. "
            + (
                "A model trained on the reference learned a different target. "
                if is_target
                else "A model or rule fitted on the reference may no longer hold. "
            )
            + "Check what changed in how the data is collected before retraining."
        ),
        evidence=Evidence(
            metrics={
                **metrics,
                "reference": reference.name,
                "reference_kind": reference.kind,
                "reference_rows": n_reference,
                "level": level.drift,
            },
            n_used=n_used,
            n_total=column.counts.rows,
            method=method,
        ),
    )
