# dsbf/checks/quality_dimensions.py
"""Report the five quality dimensions of a dataset from the findings of other checks."""

from pydantic import Field

from dsbf.checks.synthesis import (
    QUALITY_DIMENSIONS,
    QUALITY_INPUTS,
    FindingRef,
    InputReport,
    Level,
    columns_of,
    coverage,
    finding_ref,
    inputs_of,
    is_assessed,
    max_severity,
    read_inputs,
    traffic_light,
    worst_first,
)
from dsbf.engine import CheckContext, CheckOutput, CheckOutputModel, NoParams, check
from dsbf.model import Contract, Cost, Severity, Stage


class QualityDimension(Contract):
    assessed: bool = Field(
        description="False when no input check was read or had nothing to read, such "
        "as Leakage without a target: the dimension has no level, and green would "
        "claim a clean bill."
    )
    level: Level | None = Field(
        description="None when the dimension was not assessed. Green is no finding; "
        "an error is red; a warning is amber, or red above half the columns; a "
        "finding below a warning is blue up to 30% of the columns and amber above."
    )
    coverage: float = Field(
        ge=0,
        le=1,
        description="The share of input checks that were read, or had no column of "
        "their kind to read. A check that was not planned, failed, or needs a "
        "setting the caller did not give lowers it.",
    )
    inputs: tuple[InputReport, ...]
    affected_columns: tuple[str, ...]
    affected_count: int = Field(ge=0)
    pct_affected: float = Field(ge=0, le=1)
    max_severity: Severity | None
    findings: tuple[FindingRef, ...] = Field(
        description="The findings counted, the most severe first."
    )


class QualityDimensionsOutput(CheckOutputModel):
    total_columns: int = Field(ge=0)
    dimensions: dict[str, QualityDimension] = Field(
        description="Completeness, validity, usability, redundancy and leakage."
    )
    clean_columns: tuple[str, ...] = Field(
        description="The columns no dimension flags. A column no check could read is "
        "here too: read it with the coverage of the dimensions."
    )


@check(
    id="quality_dimensions",
    version="1.1.0",
    kind="synthesis",
    applies_to=None,
    dimensions=QUALITY_DIMENSIONS,
    output=QualityDimensionsOutput,
    after=inputs_of(QUALITY_INPUTS.values()),
    cost=Cost.CHEAP,
    stage=Stage.REPORT,
    packs=["core"],
    tags=["quality", "summary"],
)
def quality_dimensions(ctx: CheckContext, params: NoParams) -> CheckOutput:
    """Report completeness, validity, usability, redundancy and leakage."""
    total = len(ctx.columns)
    dimensions: dict[str, QualityDimension] = {}
    flagged: set[str] = set()
    for dimension in QUALITY_DIMENSIONS:
        inputs, found = read_inputs(ctx, QUALITY_INPUTS[dimension], dimension)
        columns = columns_of(found)
        flagged.update(columns)
        assessed = is_assessed(inputs)
        pct = len(columns) / total if total else 0.0
        worst = max_severity(found)
        dimensions[dimension.value] = QualityDimension(
            assessed=assessed,
            level=traffic_light(pct, bool(found), worst) if assessed else None,
            coverage=coverage(inputs),
            inputs=tuple(inputs),
            affected_columns=tuple(columns),
            affected_count=len(columns),
            pct_affected=pct,
            max_severity=worst,
            findings=tuple(finding_ref(f) for f in worst_first(found)),
        )
    return ctx.output(
        QualityDimensionsOutput(
            total_columns=total,
            dimensions=dimensions,
            clean_columns=tuple(c for c in ctx.columns if c not in flagged),
        )
    )
