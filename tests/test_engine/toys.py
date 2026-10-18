"""Three small checks, one of each kind, and the data they run on."""

import datetime as dt

import numpy as np
import polars as pl
from pydantic import Field

from dsbf.data import Dataset, load
from dsbf.engine import (
    CheckContext,
    CheckOutputModel,
    Requires,
    SemanticTypes,
    check,
    run_check,
)
from dsbf.model import (
    CheckParams,
    CheckResult,
    Confidence,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)
from dsbf.profile import profile_dataset

N_ROWS = 200


def build_frame() -> pl.DataFrame:
    """One column of each semantic type, with nulls in two of them."""
    rng = np.random.default_rng(0)
    amount = rng.normal(100, 15, N_ROWS)
    return pl.DataFrame(
        {
            "amount": [None if i % 4 == 0 else float(v) for i, v in enumerate(amount)],
            "single": [1.5] + [None] * (N_ROWS - 1),
            "colour": rng.choice(["red", "green", "blue"], N_ROWS),
            "flag": rng.integers(0, 2, N_ROWS).astype(bool),
            "when": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(N_ROWS)],
            "comment": [
                f"the quick brown fox jumps over lazy dog {i % 60}"
                for i in range(N_ROWS)
            ],
            "row_id": [f"ID{i:05d}" for i in range(N_ROWS)],
            "empty": pl.Series([None] * N_ROWS, dtype=pl.String),
        }
    )


EXPECTED_TYPES = {
    "amount": "numeric",
    "single": "numeric",
    "colour": "categorical",
    "flag": "boolean",
    "when": "datetime",
    "comment": "text",
    "row_id": "identifier",
    "empty": "unknown",
}


class SpreadParams(CheckParams):
    min_std: float = Field(0.5, ge=0, description="Spread below this is reported.")


class SpreadOutput(CheckOutputModel):
    std: dict[str, float | None]
    columns_read: int


@check(
    id="null_share",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.COMPLETENESS],
    requires=Requires(profile=["counts"]),
    packs=["core"],
)
def null_share(ctx: CheckContext, params: CheckParams):
    """Report each column that has nulls."""
    for profile in ctx.profile.columns:
        counts = profile.counts
        if counts.nulls == 0:
            continue
        ctx.finding(
            code="has_nulls",
            subject=Subject.column(profile.name),
            severity=Severity.NOTICE,
            confidence=Confidence.HIGH,
            title="Column has nulls",
            message=f"{profile.name!r} is {counts.nulls / counts.rows:.0%} null.",
            evidence=Evidence(
                metrics={"nulls": counts.nulls},
                n_used=counts.rows,
                n_total=counts.rows,
                method="null_count",
            ),
        )
    return None


@check(
    id="numeric_spread",
    version="1.2.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.DISTRIBUTION, Dimension.VALIDITY],
    params=SpreadParams,
    output=SpreadOutput,
    cost=Cost.MODERATE,
    packs=["core"],
)
def numeric_spread(ctx: CheckContext, params: SpreadParams):
    """Standard deviation of each numeric column."""
    stds: dict[str, float | None] = {}
    for column in ctx.columns:
        values = ctx.values(column)
        std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        stds[column] = std
        if std is not None and std < params.min_std:
            ctx.finding(
                code="low_spread",
                subject=Subject.column(column),
                dimension=Dimension.DISTRIBUTION,
                severity=Severity.WARN,
                confidence=Confidence.MODERATE,
                title="Almost constant",
                message=f"{column!r} has a standard deviation of {std:.3g}.",
                evidence=Evidence(
                    metrics={"std": std},
                    n_used=len(values),
                    n_total=ctx.profile.n_rows,
                    method="sample_std",
                ),
            )
    return ctx.output(SpreadOutput(std=stds, columns_read=len(ctx.columns)))


class SummaryOutput(CheckOutputModel):
    findings_seen: int


@check(
    id="quality_summary",
    version="0.1.0",
    kind="synthesis",
    applies_to=None,
    dimensions=[Dimension.USABILITY],
    requires=Requires(checks=["null_share"]),
    after=["numeric_spread"],
    output=SummaryOutput,
    cost=Cost.CHEAP,
)
def quality_summary(ctx: CheckContext, params: CheckParams):
    """Roll up the findings of the checks before it."""
    seen = 0
    for check_id in ("null_share", "numeric_spread"):
        upstream = ctx.result(check_id)
        seen += len(upstream.findings) if upstream else 0
    ctx.finding(
        code="findings_rolled_up",
        subject=Subject.dataset(),
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        title="Findings rolled up",
        message=f"{seen} findings from the checks before.",
        evidence=Evidence(
            metrics={"findings": seen},
            n_used=ctx.profile.n_rows,
            n_total=ctx.profile.n_rows,
            method="count",
        ),
    )
    return ctx.output(SummaryOutput(findings_seen=seen))


def make_dataset(frame: pl.DataFrame | None = None, **sample) -> Dataset:
    dataset = load(build_frame() if frame is None else frame)
    return dataset.sample(**sample) if sample else dataset


def run_toys(dataset: Dataset | None = None) -> dict[str, CheckResult]:
    """Run the three toy checks in dependency order, as an executor would."""
    dataset = dataset or make_dataset()
    profile = profile_dataset(dataset)
    results: dict[str, CheckResult] = {}
    for toy in (null_share, numeric_spread, quality_summary):
        context = CheckContext(
            toy.spec,
            dataset,
            profile,
            toy.resolve_params(),
            upstream=results,
        )
        results[toy.id] = run_check(toy, context)
    return results
