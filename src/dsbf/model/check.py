# dsbf/model/check.py
"""The outcome of one check: how it ended, why, and what it found."""

from typing import Any, Self

from pydantic import Field, model_validator

from dsbf.model.base import Contract
from dsbf.model.finding import Finding
from dsbf.model.scrub import json_safe
from dsbf.model.timing import Timing
from dsbf.model.vocabulary import CheckStatus, Severity


class Note(Contract):
    """An operational message: why a check was skipped, failed or degraded."""

    code: str = Field(
        min_length=1, description="Machine-readable, for example `dependency_failed`."
    )
    message: str = Field(min_length=1, description="What happened, in a sentence.")
    severity: Severity = Severity.INFO
    hint: str | None = Field(
        default=None, description="What the reader can do about it."
    )
    blocked_by: tuple[str, ...] = Field(
        default=(), description="The checks whose outcome caused a skip."
    )
    exception_type: str | None = None
    detail: str | None = Field(
        default=None, description="Technical detail, for example a traceback summary."
    )
    count: int | None = Field(
        default=None,
        ge=1,
        description="How many times it happened, for a note that can repeat, such "
        "as a `python_warning`.",
    )


class CheckResult(Contract):
    """What running one check produced."""

    check_id: str = Field(min_length=1)
    check_version: str = Field(min_length=1)
    status: CheckStatus
    notes: tuple[Note, ...] = Field(
        default=(), description="Always present when the status is not `ok`."
    )
    findings: tuple[Finding, ...] = ()
    output: dict[str, Any] | None = Field(
        default=None, description="The check's own output; bulky parts live elsewhere."
    )
    non_finite_outputs: tuple[str, ...] = Field(
        default=(),
        description="Keys of `output` holding a NaN, infinite or NaT value, now null.",
    )
    timing: Timing
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="The resolved parameters, for reproducibility.",
    )

    @model_validator(mode="before")
    @classmethod
    def _make_json_safe(cls, data: Any) -> Any:
        """Make `output` and `params` JSON-safe.

        An undefined number in `output` becomes null and its key is flagged; one in
        `params` is an error, since parameters are not measurements.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        output = data.get("output")
        if isinstance(output, dict):
            undefined = set(data.get("non_finite_outputs") or ())
            cleaned: dict[str, Any] = {}
            for key, value in output.items():
                if not isinstance(key, str):
                    raise ValueError(f"output keys must be strings, not {key!r}")
                cleaned[key], is_undefined = json_safe(value)
                if is_undefined:
                    undefined.add(key)
            data["output"] = cleaned
            data["non_finite_outputs"] = tuple(sorted(undefined))
        params = data.get("params")
        if isinstance(params, dict):
            data["params"] = json_safe(params, null_non_finite=False)[0]
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.status != CheckStatus.OK and not self.notes:
            raise ValueError(f"a {self.status.value} result carries a note saying why")
        if self.status != CheckStatus.OK and self.findings:
            raise ValueError(f"a {self.status.value} result has no findings")
        for finding in self.findings:
            if (finding.check_id, finding.check_version) != (
                self.check_id,
                self.check_version,
            ):
                raise ValueError(
                    f"finding {finding.id} belongs to {finding.check_id} "
                    f"{finding.check_version}, not {self.check_id} {self.check_version}"
                )
        return self
