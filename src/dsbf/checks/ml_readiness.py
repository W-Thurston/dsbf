# dsbf/checks/ml_readiness.py
"""Report what a dataset needs before modelling, from the findings of other checks."""

from typing import Literal

from pydantic import Field

from dsbf.checks.synthesis import (
    READINESS_INPUTS,
    READINESS_LABELS,
    FindingRef,
    InputReport,
    columns_of,
    coverage,
    finding_ref,
    inputs_of,
    is_assessed,
    read_inputs,
    worst_first,
)
from dsbf.engine import CheckContext, CheckOutput, CheckOutputModel, NoParams, check
from dsbf.model import Contract, Cost, Dimension, Severity, Stage

Gate = Literal["ready", "needs_work", "not_ready", "not_assessed"]
ReadinessLevel = Literal["green", "amber", "red"]


class ReadinessDimension(Contract):
    label: str
    assessed: bool = Field(
        description="False when no input check was read or had nothing to read, "
        "such as Leakage risk without a target."
    )
    level: ReadinessLevel | None = Field(
        description="Red for an error, amber for a warning, else green; None when "
        "the dimension was not assessed. Unlike a quality dimension it does not "
        "depend on how many columns are flagged: one leaking column is enough."
    )
    coverage: float = Field(ge=0, le=1)
    inputs: tuple[InputReport, ...]
    action_columns: tuple[str, ...] = Field(
        description="The columns with a warning or an error in this dimension."
    )
    advisory_columns: tuple[str, ...] = Field(
        description="The columns whose findings here are all below a warning."
    )
    findings: tuple[FindingRef, ...] = Field(
        description="The most severe first, with the `ml` actions each carries."
    )


class MlReadinessOutput(CheckOutputModel):
    gate: Gate = Field(
        description="`not_ready` if an assessed dimension is red, `needs_work` if one "
        "is amber, else `ready`; `not_assessed` when no dimension was."
    )
    complete: bool = Field(
        description="False when a dimension was not assessed, so `ready` holds only "
        "for the dimensions that were."
    )
    unassessed: tuple[str, ...]
    total_columns: int = Field(ge=0)
    dimensions: dict[str, ReadinessDimension]
    action_columns: tuple[str, ...] = Field(
        description="Columns with a warning or an error in any dimension."
    )
    advisory_columns: tuple[str, ...] = Field(
        description="Columns with only findings below a warning, in every dimension."
    )
    clean_columns: tuple[str, ...] = Field(
        description="Columns no dimension has a finding for."
    )


def _level(findings: list) -> ReadinessLevel:
    worst = max((f.severity for f in findings), default=None)
    if worst is not None and worst >= Severity.ERROR:
        return "red"
    if worst is not None and worst >= Severity.WARN:
        return "amber"
    return "green"


@check(
    id="ml_readiness",
    version="1.0.0",
    kind="synthesis",
    applies_to=None,
    dimensions=[Dimension.USABILITY],
    output=MlReadinessOutput,
    after=inputs_of(READINESS_INPUTS.values()),
    cost=Cost.CHEAP,
    stage=Stage.REPORT,
    packs=["core"],
    tags=["ml", "summary"],
)
def ml_readiness(ctx: CheckContext, params: NoParams) -> CheckOutput:
    """Report the work a dataset needs before modelling."""
    dimensions: dict[str, ReadinessDimension] = {}
    action: set[str] = set()
    advisory: set[str] = set()
    for name, check_ids in READINESS_INPUTS.items():
        inputs, found = read_inputs(ctx, check_ids)
        loud = {
            c for f in found if f.severity >= Severity.WARN for c in f.subject.columns
        }
        quiet = set(columns_of(found)) - loud
        action |= loud
        advisory |= quiet
        assessed = is_assessed(inputs)
        dimensions[name] = ReadinessDimension(
            label=READINESS_LABELS[name],
            assessed=assessed,
            level=_level(found) if assessed else None,
            coverage=coverage(inputs),
            inputs=tuple(inputs),
            action_columns=tuple(sorted(loud)),
            advisory_columns=tuple(sorted(quiet)),
            findings=tuple(
                finding_ref(f, with_actions=True) for f in worst_first(found)
            ),
        )
    advisory -= action
    levels = [d.level for d in dimensions.values() if d.level is not None]
    gate: Gate = (
        "not_assessed"
        if not levels
        else "not_ready"
        if "red" in levels
        else "needs_work"
        if "amber" in levels
        else "ready"
    )
    unassessed = tuple(name for name, d in dimensions.items() if not d.assessed)
    return ctx.output(
        MlReadinessOutput(
            gate=gate,
            complete=not unassessed,
            unassessed=unassessed,
            total_columns=len(ctx.columns),
            dimensions=dimensions,
            action_columns=tuple(sorted(action)),
            advisory_columns=tuple(sorted(advisory)),
            clean_columns=tuple(
                c for c in ctx.columns if c not in action and c not in advisory
            ),
        )
    )
