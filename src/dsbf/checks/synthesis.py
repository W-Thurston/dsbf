# dsbf/checks/synthesis.py
"""What the synthesis checks share: their inputs, input states and levels.

``quality_dimensions`` and ``ml_readiness`` report on other checks' findings. Each lists
the checks it reads by hand, in a dimension's tuple here, because the checks are
declared in their own modules and a check cannot see another's spec. A unit test holds
the lists to the specs, so a new check that declares a quality dimension is not
forgotten.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field

from dsbf.engine import CheckContext
from dsbf.model import (
    Action,
    CheckResult,
    CheckStatus,
    Contract,
    Dimension,
    Finding,
    Severity,
)

Level = Literal["green", "blue", "amber", "red"]
InputState = Literal[
    "assessed", "nothing_to_check", "needs_setting", "not_run", "failed"
]

# The five quality dimensions, in the order a report lists them.
QUALITY_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension.COMPLETENESS,
    Dimension.VALIDITY,
    Dimension.USABILITY,
    Dimension.REDUNDANCY,
    Dimension.LEAKAGE,
)

# Checks that advise how to prepare a column, or compare the data with another dataset.
# They declare a quality dimension but report no flaw in the data, so a column they
# mention is not thereby flagged.
ADVICE_CHECKS: frozenset[str] = frozenset(
    {
        "categorical_encoding",
        "numeric_binning",
        "transformations",
        "imputation",
        "dtype_optimizations",
        "datetime_features",
        "reference_drift",
    }
)

# Each quality dimension to the checks that declare it and report a flaw. A check that
# declares two reads for each only the findings of that dimension.
QUALITY_INPUTS: Mapping[Dimension, tuple[str, ...]] = {
    Dimension.COMPLETENESS: ("null_rate",),
    Dimension.VALIDITY: (
        "datetime_consistency",
        "duplicate_rows",
        "format_violations",
        "fuzzy_duplicates",
        "mixed_types",
        "out_of_bounds",
        "schema",
        "string_anomalies",
        "text_encoding",
    ),
    Dimension.USABILITY: (
        "constant_columns",
        "encoded_columns",
        "high_cardinality",
        "id_columns",
        "near_zero_variance",
        "sample_size",
    ),
    Dimension.REDUNDANCY: ("associations", "collinearity", "duplicate_columns"),
    Dimension.LEAKAGE: ("target_leakage",),
}

# The ML-readiness dimensions, each with the checks whose findings say what to do
# before modelling.
READINESS_LABELS: Mapping[str, str] = {
    "transformations": "Transformations needed",
    "encoding": "Encoding required",
    "missingness": "Missingness impact",
    "leakage": "Leakage risk",
    "unusable": "Unusable features",
}
READINESS_INPUTS: Mapping[str, tuple[str, ...]] = {
    "transformations": (
        "bimodality",
        "multivariate_outliers",
        "numeric_binning",
        "outliers",
        "skewness",
        "transformations",
    ),
    "encoding": ("categorical_encoding", "high_cardinality"),
    "missingness": ("imputation", "missingness_mechanism", "null_rate"),
    "leakage": ("target_leakage",),
    "unusable": (
        "constant_columns",
        "dominant_value",
        "id_columns",
        "near_zero_variance",
        "zeros",
    ),
}

# Why a check was not applicable: the caller gave it nothing to work with. Any other
# reason is that the data held nothing of the kind it reads.
NEEDS_SETTING_CODES: frozenset[str] = frozenset(
    {"no_target", "no_bounds", "no_schema", "no_reference"}
)
_COVERED: frozenset[InputState] = frozenset({"assessed", "nothing_to_check"})


def inputs_of(groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    """Every check id in ``groups``, once and sorted: a synthesis check's ``after``."""
    return tuple(sorted({check_id for group in groups for check_id in group}))


class InputReport(Contract):
    """One check a dimension reads, and whether it could be read."""

    check_id: str
    state: InputState = Field(
        description="`assessed`: the check completed. `nothing_to_check`: the data "
        "held no column of the kind it reads. `needs_setting`: it needs what the "
        "caller did not give (a target, bounds, a schema or a reference). "
        "`not_run`: it was not in the plan. `failed`: it errored, timed out, was "
        "skipped or lacked an extra."
    )
    findings: int = Field(ge=0, description="How many findings the dimension took.")
    reason: str | None = Field(
        description="The check's own note, for any state but `assessed`."
    )


class FindingRef(Contract):
    """A finding a dimension counted, by reference."""

    id: str
    check_id: str
    code: str
    severity: Severity
    columns: tuple[str, ...]
    title: str
    actions: tuple[Action, ...] = Field(
        default=(), description="The `ml` actions the finding carries."
    )


def input_state(result: CheckResult | None) -> tuple[InputState, str | None]:
    """How a check's result reads as an input, and its note when it did not complete."""
    if result is None:
        return "not_run", "The check was not in this run's plan."
    if result.status is CheckStatus.OK:
        return "assessed", None
    note = result.notes[0]
    if result.status is CheckStatus.NOT_APPLICABLE:
        return (
            "needs_setting" if note.code in NEEDS_SETTING_CODES else "nothing_to_check",
            note.message,
        )
    return "failed", note.message


def coverage(inputs: Sequence[InputReport]) -> float:
    """The share of inputs that were read or had nothing to read."""
    if not inputs:
        return 0.0
    return sum(i.state in _COVERED for i in inputs) / len(inputs)


def is_assessed(inputs: Sequence[InputReport]) -> bool:
    """Whether any input was read or had nothing to read.

    A dimension whose every input is missing, failed or needs a setting the caller did
    not give is not assessed; one whose checks found no column of their kind is: there
    was nothing to flag.
    """
    return any(i.state in _COVERED for i in inputs)


def read_inputs(
    ctx: CheckContext,
    check_ids: Iterable[str],
    dimension: Dimension | None = None,
) -> tuple[list[InputReport], list[Finding]]:
    """The state of each input check and the findings to count from them.

    With ``dimension``, a finding counts only if it speaks to that dimension.
    """
    reports: list[InputReport] = []
    findings: list[Finding] = []
    for check_id in check_ids:
        result = ctx.result(check_id)
        state, reason = input_state(result)
        taken = [
            finding
            for finding in (result.findings if result is not None else ())
            if dimension is None or finding.dimension == dimension
        ]
        findings += taken
        reports.append(
            InputReport(
                check_id=check_id, state=state, findings=len(taken), reason=reason
            )
        )
    return reports, findings


def finding_ref(finding: Finding, *, with_actions: bool = False) -> FindingRef:
    actions = (
        finding.guidance.actions
        if with_actions and finding.guidance and finding.guidance.phase == "ml"
        else ()
    )
    return FindingRef(
        id=finding.id,
        check_id=finding.check_id,
        code=finding.code,
        severity=finding.severity,
        columns=finding.subject.columns,
        title=finding.title,
        actions=actions,
    )


def worst_first(findings: Iterable[Finding]) -> list[Finding]:
    """The findings, the most severe first, then by check and subject."""
    return sorted(
        findings,
        key=lambda f: (
            -list(Severity).index(f.severity),
            f.check_id,
            f.subject.columns,
            f.code,
        ),
    )


def max_severity(findings: Sequence[Finding]) -> Severity | None:
    return max((f.severity for f in findings), default=None)


def columns_of(findings: Iterable[Finding]) -> list[str]:
    """The columns the findings are about, once each, in the profile's order of use."""
    return list(dict.fromkeys(c for f in findings for c in f.subject.columns))


def traffic_light(
    pct_affected: float, any_affected: bool, severity: Severity | None
) -> Level:
    """A dimension's level from the share of columns it flags and its worst finding.

    Green is no finding. An error is red; a warning is amber, and red above 15% of the
    columns. Findings below a warning are blue under 10% of the columns, amber to 30%
    and red above, so a pattern across many columns is not mistaken for a coincidence.
    """
    if not any_affected:
        return "green"
    if severity is not None and severity >= Severity.ERROR:
        return "red"
    if severity is not None and severity >= Severity.WARN:
        return "red" if pct_affected > 0.15 else "amber"
    if pct_affected < 0.10:
        return "blue"
    return "amber" if pct_affected <= 0.30 else "red"
