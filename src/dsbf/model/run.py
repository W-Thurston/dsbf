# dsbf/model/run.py
"""The result of executing a plan on a dataset."""

import re
from collections.abc import Iterable
from typing import Self
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator, model_validator

from dsbf.model.base import Contract
from dsbf.model.check import CheckResult
from dsbf.model.finding import Finding
from dsbf.model.health import RunHealth
from dsbf.model.manifest import DatasetManifest, PlanManifest
from dsbf.model.policy import PolicyOutcome
from dsbf.model.profile import DatasetProfile
from dsbf.model.vocabulary import Severity

# Semver. Additive changes are minor and breaking changes are major. The 1.x
# compatibility guarantee begins with the first release.
SCHEMA_VERSION = "0.1.0"

_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_SEVERITY_FIRST = {severity: -rank for rank, severity in enumerate(Severity)}


def order_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Highest severity first; ties by check, then id, so the order is stable."""
    return tuple(
        sorted(findings, key=lambda f: (_SEVERITY_FIRST[f.severity], f.check_id, f.id))
    )


class Run(Contract):
    """Everything one execution produced, in one serialisable object."""

    schema_version: str = SCHEMA_VERSION
    run_id: UUID
    started_at: AwareDatetime
    finished_at: AwareDatetime
    dataset: DatasetManifest
    plan: PlanManifest
    profile: DatasetProfile
    results: dict[str, CheckResult] = Field(
        description="One result per planned check, keyed by check id."
    )
    findings: tuple[Finding, ...] = Field(
        description="Every result's findings, highest severity first."
    )
    health: RunHealth
    policy: PolicyOutcome | None = Field(
        default=None,
        description="What the user's policy decided; absent when none was applied.",
    )

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: str) -> str:
        match = _SEMVER.fullmatch(value)
        if not match:
            raise ValueError(f"{value!r} is not a major.minor.patch version")
        if match.group(1) != _SEMVER.fullmatch(SCHEMA_VERSION).group(1):  # type: ignore[union-attr]
            raise ValueError(
                f"schema version {value} is not readable by schema {SCHEMA_VERSION}"
            )
        return value

    @classmethod
    def build(cls, *, results: Iterable[CheckResult], **fields: object) -> Self:
        """Make a run, deriving its findings and health from the check results."""
        by_id: dict[str, CheckResult] = {}
        for result in results:
            if result.check_id in by_id:
                raise ValueError(f"two results for check {result.check_id}")
            by_id[result.check_id] = result
        return cls(
            results=by_id,
            findings=order_findings(f for r in by_id.values() for f in r.findings),
            health=RunHealth.from_results(by_id),
            **fields,  # type: ignore[arg-type]
        )

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.started_at > self.finished_at:
            raise ValueError("a run starts before it finishes")
        self._check_inputs()
        self._check_results()
        self._check_policy()
        return self

    def _check_inputs(self) -> None:
        if self.profile.n_rows != self.dataset.analysed_rows:
            raise ValueError(
                f"the profile covers {self.profile.n_rows} rows but the dataset "
                f"analysed {self.dataset.analysed_rows}"
            )
        if self.profile.n_columns != self.dataset.n_columns:
            raise ValueError(
                f"the profile has {self.profile.n_columns} columns but the "
                f"dataset has {self.dataset.n_columns}"
            )

    def _check_results(self) -> None:
        for key, result in self.results.items():
            if key != result.check_id:
                raise ValueError(f"result for {result.check_id} is keyed {key!r}")
        if set(self.results) != set(self.plan.check_ids):
            missing = sorted(set(self.plan.check_ids) - set(self.results))
            extra = sorted(set(self.results) - set(self.plan.check_ids))
            raise ValueError(
                f"results and plan disagree (no result for {missing}, "
                f"unplanned results for {extra})"
            )
        for planned in self.plan.checks:
            version = self.results[planned.check_id].check_version
            if version != planned.check_version:
                raise ValueError(
                    f"{planned.check_id} was planned at version "
                    f"{planned.check_version} but ran at {version}"
                )
        expected = order_findings(f for r in self.results.values() for f in r.findings)
        if self.findings != expected:
            raise ValueError(
                "findings are not the results' findings, highest severity first"
            )
        if len({f.id for f in self.findings}) != len(self.findings):
            raise ValueError("finding ids are unique within a run")
        if self.health != RunHealth.from_results(self.results):
            raise ValueError("health does not match the check results")

    def _check_policy(self) -> None:
        if self.policy is None:
            return
        finding_ids = {f.id for f in self.findings}
        for outcome in self.policy.rules:
            name = outcome.rule.name
            unknown = [i for i in outcome.finding_ids if i not in finding_ids]
            if unknown:
                raise ValueError(f"rule {name!r} matched unknown findings {unknown}")
            status = outcome.rule.when.health_status
            for check_id in outcome.check_ids:
                if check_id not in self.results:
                    raise ValueError(f"rule {name!r} matched unknown check {check_id}")
                actual = self.results[check_id].status
                if actual != status:
                    raise ValueError(
                        f"rule {name!r} matched {check_id}, whose status is "
                        f"{actual}, not {status}"
                    )

    def render_health(self, pointer: str | None = None) -> str:
        """The plain-text health summary, headed by what the run looked at."""
        sampling = self.dataset.sampling
        scan = f"sample of {sampling.sampled_rows:,} rows" if sampling else "full scan"
        header = (
            f"DSBF run {str(self.run_id)[:4]}…  {self.dataset.name}  "
            f"({self.dataset.n_rows:,} × {self.dataset.n_columns}, {scan})   "
            f"{len(self.results)} checks"
        )
        return self.health.render(header=header, pointer=pointer)
