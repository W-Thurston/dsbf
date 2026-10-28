# dsbf/checks/target_leakage.py
"""Features that carry the target, or that are known only after it is."""

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from dsbf.checks.pairs import Undefined
from dsbf.checks.target import (
    AssociationMetric,
    Kind,
    Target,
    Variable,
    associate,
    class_variable,
    null_mask,
    numeric_variable,
    read,
    resolve_target,
    sample_rows,
)
from dsbf.checks.timeseries import read_instants
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
    CheckParams,
    Confidence,
    Contract,
    Cost,
    DatetimeProfile,
    Dimension,
    Evidence,
    Guidance,
    Severity,
    Subject,
)

Source = Literal["value", "part", "null_indicator", "timestamp"]
Metric = AssociationMetric | Literal["later_share"]

_CODES: dict[Source, str] = {
    "value": "target_leakage",
    "part": "derived_from_target",
    "null_indicator": "missingness_tracks_target",
    "timestamp": "timestamp_after_event",
}
_METHODS: dict[Metric, str] = {
    "pearson_r": "pearson_spearman_max",
    "spearman_r": "pearson_spearman_max",
    "eta": "correlation_ratio",
    "cramers_v": "cramers_v_bias_corrected",
    "later_share": "later_than_time_index",
}

# An identifier is cut where a run of other characters ends and where letters meet
# digits, and its first parts are tested; `outcome_1_7` is `outcome`, `1`, `7`.
_PART_BOUNDARY = re.compile(
    r"[^A-Za-z0-9]+|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])"
)
_PARTS = 4
_PART_ROWS = 50_000
_CONFIDENT_PAIRS = 100


class TargetLeakageParams(CheckParams):
    target: str | None = Field(
        None,
        description="The column to predict. `dsbf.profile(target=...)` sets it; "
        "without one the check is not applicable.",
    )
    time_index: str | None = Field(
        None,
        description="The datetime column that dates each row's event. "
        "`dsbf.profile(time_index=...)` sets it; without one, timestamps are not "
        "compared with it.",
    )
    warn_at: float = Field(
        0.93,
        gt=0,
        le=1,
        description="Association with the target from which a feature is reported "
        "as a possible leak. A feature that gets 3 labels in 100 wrong is at 0.94, "
        "one that gets 5 in 100 wrong at 0.90, and an honest feature seldom gets "
        "within 10.",
    )
    error_at: float = Field(
        0.99,
        gt=0,
        le=1,
        description="Association from which the finding is an error: the feature "
        "is, to within noise, a copy or a recode of the target.",
    )
    after_share: float = Field(
        0.95,
        gt=0,
        le=1,
        description="Share of the rows in which a datetime column is later than the "
        "time index from which it is reported as known only after the event.",
    )
    min_n: int = Field(
        30, ge=5, description="Complete rows a feature needs to be measured."
    )
    max_levels: int = Field(
        50,
        ge=2,
        description="Most levels a class column, or a part of an identifier, may "
        "have; V over more is inflated by the levels' own noise.",
    )
    max_rows: int = Field(
        200_000, ge=1000, description="Most rows read; more are sampled with `seed`."
    )
    seed: int = Field(42, description="Seed of the row sample.")
    max_findings: int = Field(50, ge=1, description="Most findings, strongest first.")


class FeatureAssociation(Contract):
    feature: str
    source: Source = Field(
        description="What was tested: the column's `value`, a `part` of its text, "
        "its `null_indicator`, or a `timestamp` compared with the time index."
    )
    part: int | None = Field(
        description="For a `part`, its position in the text, from 1."
    )
    metric: Metric
    value: float = Field(ge=0, le=1)
    n_used: int = Field(ge=0)
    flagged: bool


class TargetLeakageOutput(CheckOutputModel):
    target: str
    target_kind: Kind
    features: list[FeatureAssociation] = Field(
        description="Every test that was made, strongest first."
    )
    features_tested: int = Field(ge=0)
    features_flagged: int = Field(ge=0)
    untested: dict[str, int] = Field(
        description="Tests not made, by reason: too_few_rows, too_few_groups, "
        "no_variation or overflow."
    )
    skipped_columns: dict[str, str] = Field(
        description="Columns whose value was not measured, with the reason."
    )
    rows_used: int = Field(ge=0, description="Rows read, after any sampling.")
    rows_total: int = Field(ge=0)


@dataclass(frozen=True)
class _Test:
    feature: str
    source: Source
    part: int | None
    metric: Metric
    value: float
    n_used: int


def _text_parts(raw: np.ndarray) -> list[np.ndarray]:
    """The first parts of each text value, as one object array per position."""
    columns = [np.full(raw.size, None, dtype=object) for _ in range(_PARTS)]
    for row, value in enumerate(raw):
        if value is None or value != value:
            continue
        for position, piece in enumerate(
            [p for p in _PART_BOUNDARY.split(str(value)) if p][:_PARTS]
        ):
            columns[position][row] = piece
    return columns


def _instants(
    ctx: CheckContext, name: str, column: DatetimeProfile, rows: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    return read_instants(read(ctx, name, rows), column)


def _time_index(ctx: CheckContext, params: TargetLeakageParams) -> str | None:
    """The time index if it is a datetime column of the data, else None with a note."""
    name = params.time_index
    if name is None:
        ctx.note(
            "no_time_index",
            "No time index is set, so datetime columns are not compared with the "
            "time of the event.",
            hint="Pass time_index=... to dsbf.profile(), or set the `time_index` "
            "parameter.",
        )
        return None
    if name not in ctx.columns or not isinstance(
        ctx.profile.column(name), DatetimeProfile
    ):
        ctx.note(
            "time_index_not_a_datetime",
            f"The time index {name!r} is not a column the profile typed as "
            "datetime, so datetime columns are not compared with it.",
            severity=Severity.WARN,
        )
        return None
    return name


def _null_indicator(raw: np.ndarray, params: TargetLeakageParams) -> Variable | None:
    """Whether each row is null, or None when either group is too small to test."""
    nulls = null_mask(raw)
    n_null = int(np.count_nonzero(nulls))
    if min(n_null, nulls.size - n_null) < params.min_n:
        return None
    return Variable("class", nulls.astype(np.int64), np.empty(0), 2)


def _strongest_part(
    raw: np.ndarray,
    target: Target,
    params: TargetLeakageParams,
    ctx: CheckContext,
) -> _Test | None:
    """The part of a text column that carries the target best, if any can be tested."""
    positions = None
    if raw.size > _PART_ROWS:
        positions = np.sort(
            np.random.default_rng(params.seed).choice(
                raw.size, _PART_ROWS, replace=False
            )
        )
        raw = raw[positions]
        target = target.take(positions)
    best: _Test | None = None
    for number, part in enumerate(_text_parts(raw), start=1):
        ctx.checkpoint()
        feature = class_variable(part)
        if not 2 <= feature.n_levels <= params.max_levels:
            continue
        try:
            found = associate(feature, target, min_n=params.min_n)
        except Undefined:
            continue
        if best is None or found.value > best.value:
            best = _Test("", "part", number, found.metric, found.value, found.n_used)
    return best


def _later_share(
    ctx: CheckContext,
    name: str,
    column: DatetimeProfile,
    rows: np.ndarray | None,
    index: tuple[np.ndarray, np.ndarray],
    params: TargetLeakageParams,
) -> _Test | None:
    """The share of rows in which the column is after the index, if enough compare."""
    micros, present = _instants(ctx, name, column, rows)
    both = present & index[1]
    n_used = int(np.count_nonzero(both))
    if n_used < params.min_n:
        return None
    share = float(np.mean(micros[both] > index[0][both]))
    return _Test(name, "timestamp", None, "later_share", share, n_used)


@check(
    id="target_leakage",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.LEAKAGE],
    params=TargetLeakageParams,
    output=TargetLeakageOutput,
    requires=Requires(profile=["counts"], params=["target"]),
    cost=Cost.MODERATE,
    codes=[
        "derived_from_target",
        "missingness_tracks_target",
        "target_leakage",
        "timestamp_after_event",
    ],
    packs=["core"],
    tags=["target", "leakage", "ml-readiness"],
)
def target_leakage(ctx: CheckContext, params: TargetLeakageParams) -> CheckOutput:
    """Flag features that predict the target almost perfectly, or follow it in time.

    Four things are looked for, each a reason to ask whether the value exists when
    the model predicts: a feature that is nearly a copy of the target, by the
    association that suits the pair of types; an identifier or other text with a part
    that carries the target; a feature that is present for some outcomes only; and a
    datetime column that is later than the time index.
    """
    rows = sample_rows(ctx, params.max_rows, params.seed)
    resolved = resolve_target(ctx, params.target, rows, params.max_levels)
    if isinstance(resolved, CheckOutput):
        return resolved
    target = resolved

    total_rows = ctx.profile.n_rows
    rows_used = total_rows if rows is None else int(rows.size)
    tests: list[_Test] = []
    untested: dict[str, int] = {}
    skipped: dict[str, str] = {}

    def measure(
        name: str, source: Source, feature: Variable, part: int | None = None
    ) -> None:
        ctx.checkpoint()
        try:
            found = associate(feature, target, min_n=params.min_n)
        except Undefined as why:
            untested[str(why)] = untested.get(str(why), 0) + 1
            return
        tests.append(_Test(name, source, part, found.metric, found.value, found.n_used))

    def measure_parts(name: str, raw: np.ndarray) -> None:
        best = _strongest_part(raw, target, params, ctx)
        if best is not None:
            tests.append(
                _Test(name, "part", best.part, best.metric, best.value, best.n_used)
            )

    index_name = _time_index(ctx, params)
    index = None
    if index_name is not None:
        index = _instants(ctx, index_name, _datetime(ctx, index_name), rows)

    for name in ctx.columns:
        if name == target.name:
            continue
        column = ctx.profile.column(name)
        kind = column.semantic_type
        raw = read(ctx, name, rows)
        if column.counts.nulls:
            indicator = _null_indicator(raw, params)
            if indicator is not None:
                measure(name, "null_indicator", indicator)
        if kind == "numeric":
            measure(name, "value", numeric_variable(raw))
        elif kind in ("categorical", "boolean"):
            distinct = column.counts.distinct
            if distinct < 2:
                skipped[name] = "fewer than two levels"
            elif distinct > params.max_levels:
                skipped[name] = f"more than {params.max_levels} levels"
                measure_parts(name, raw)
            else:
                measure(name, "value", class_variable(raw))
        elif kind == "identifier":
            skipped[name] = "identifier"
            measure_parts(name, raw)
        elif kind == "datetime":
            skipped[name] = "datetime"
            if index is not None and name != index_name:
                assert isinstance(column, DatetimeProfile)
                later = _later_share(ctx, name, column, rows, index, params)
                if later is not None:
                    tests.append(later)
        else:
            skipped[name] = kind

    tests.sort(key=lambda t: -t.value)
    flagged = [t for t in tests if _is_flagged(t, params)]
    shown = flagged[: params.max_findings]
    if len(flagged) > len(shown):
        ctx.note(
            "findings_capped",
            f"{len(flagged):,} tests are flagged; the findings cover the strongest "
            f"{len(shown):,}, all of them are in the output.",
        )
    for test in shown:
        _report(ctx, test, target, params, total_rows, index_name)
    return ctx.output(
        TargetLeakageOutput(
            target=target.name,
            target_kind=target.kind,
            features=[
                FeatureAssociation(
                    feature=t.feature,
                    source=t.source,
                    part=t.part,
                    metric=t.metric,
                    value=t.value,
                    n_used=t.n_used,
                    flagged=_is_flagged(t, params),
                )
                for t in tests
            ],
            features_tested=len(tests),
            features_flagged=len(flagged),
            untested=untested,
            skipped_columns=skipped,
            rows_used=rows_used,
            rows_total=total_rows,
        )
    )


def _datetime(ctx: CheckContext, name: str) -> DatetimeProfile:
    column = ctx.profile.column(name)
    assert isinstance(column, DatetimeProfile)
    return column


def _is_flagged(test: _Test, params: TargetLeakageParams) -> bool:
    threshold = params.after_share if test.source == "timestamp" else params.warn_at
    return test.value >= threshold


def _report(
    ctx: CheckContext,
    test: _Test,
    target: Target,
    params: TargetLeakageParams,
    total_rows: int,
    index: str | None,
) -> None:
    name, value = test.feature, test.value
    severity = Severity.WARN
    confidence = (
        Confidence.HIGH if test.n_used >= _CONFIDENT_PAIRS else Confidence.MODERATE
    )
    if test.source == "timestamp":
        confidence = Confidence.MODERATE
        title = "Timestamp after the event"
        message = (
            f"{name!r} is later than the time index {index!r} in {value:.1%} of "
            f"{test.n_used:,} rows. If it records something that happened after the "
            f"event, it holds information the model will not have when it predicts "
            f"{target.name!r}; a scheduled date (a due or renewal date) is later by "
            "design and is no leak. Check what the column records."
        )
        detail = (
            f"Leave {name!r} out of the features unless it is known at the time of "
            "the event."
        )
    else:
        if value >= params.error_at:
            severity = Severity.ERROR
        if test.source == "part":
            title = f"Part {test.part} of {name!r} carries the target"
            what = (
                f"part {test.part} of its text (cut at punctuation and where letters "
                "meet digits) "
            )
            message = (
                f"{what}is associated with the target {target.name!r} at "
                f"{value:.3f} (bias-corrected Cramér's V, {test.n_used:,} rows), so "
                f"{name!r} may be built from the outcome. An identifier that spells "
                "out the outcome lets a model read the answer."
            )
            detail = (
                f"Drop {name!r}, or rebuild it without the part that spells the "
                "outcome."
            )
        elif test.source == "null_indicator":
            title = f"Missing values of {name!r} track the target"
            message = (
                f"Whether {name!r} is missing is associated with the target "
                f"{target.name!r} at {value:.3f} (bias-corrected Cramér's V, "
                f"{test.n_used:,} rows), so the column may be filled in only for some "
                "outcomes, for example after the outcome is known."
            )
            detail = (
                f"Find out when {name!r} is filled in; if it is after the event, "
                "leave it out of the features."
            )
        else:
            title = f"{name!r} is almost the target"
            message = (
                f"{name!r} is associated with the target {target.name!r} at "
                f"{value:.3f} ({_LABEL[test.metric]}, {test.n_used:,} rows). A feature "
                "this close to the target is often a copy, a recode, or a quantity "
                "recorded after the outcome, which the model will not have when it "
                "predicts; it can also be a true, very strong cause."
            )
            detail = (
                f"Check how {name!r} is produced, and drop it if it follows the target."
            )
    ctx.finding(
        code=_CODES[test.source],
        subject=Subject.column(name),
        severity=severity if test.source != "timestamp" else Severity.WARN,
        confidence=confidence,
        title=title,
        message=message,
        evidence=Evidence(
            metrics={
                "association": value,
                "metric": test.metric,
                "target": target.name,
                "threshold": (
                    params.after_share if test.source == "timestamp" else params.warn_at
                ),
                "part": test.part,
            },
            n_used=test.n_used,
            n_total=total_rows,
            method=_METHODS[test.metric],
        ),
        guidance=Guidance(
            phase="ml",
            actions=(
                Action(
                    action="drop",
                    columns=(name,),
                    condition=f"unless {_KNOWN}",
                    detail=detail,
                ),
            ),
        ),
    )


_KNOWN = "its value is known when the model has to predict"
_LABEL: dict[Metric, str] = {
    "pearson_r": "|Pearson r|",
    "spearman_r": "|Spearman ρ|",
    "eta": "η",
    "cramers_v": "bias-corrected Cramér's V",
    "later_share": "share of rows",
}
