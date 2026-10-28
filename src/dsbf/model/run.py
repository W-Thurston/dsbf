# dsbf/model/run.py
"""The result of executing a plan on a dataset."""

import re
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Self
from uuid import UUID

from pydantic import AwareDatetime, Field, PrivateAttr, field_validator, model_validator

from dsbf.model.artifact import ArtifactRef, Payload
from dsbf.model.base import Contract, fields_equal
from dsbf.model.check import CheckResult, Note
from dsbf.model.finding import Finding
from dsbf.model.findings import Findings
from dsbf.model.health import RunHealth
from dsbf.model.manifest import DatasetManifest, PlanManifest
from dsbf.model.notebook import render_run_html
from dsbf.model.policy import PolicyOutcome
from dsbf.model.profile import DatasetProfile
from dsbf.model.vocabulary import Severity

# Semver. Additive changes are minor and breaking changes are major. The 1.x
# compatibility guarantee begins with the first release.
SCHEMA_VERSION = "0.12.0"

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
    health: RunHealth
    notes: tuple[Note, ...] = Field(
        default=(),
        description="What happened to the run as a whole rather than to one check: "
        "a plugin that failed to load, an empty plan, a result cache that failed.",
    )
    policy: PolicyOutcome | None = Field(
        default=None,
        description="What the user's policy decided; absent when none was applied.",
    )

    _artifact_dir: Path | None = PrivateAttr(default=None)

    def __eq__(self, other: object) -> bool:
        return fields_equal(self, other)

    @cached_property
    def findings(self) -> Findings:
        """Every result's findings, highest severity first.

        Derived from ``results`` rather than stored, so a run carries each finding once.
        The view filters (``run.findings.filter(severity=">=warn")``) and converts to
        a dataframe.
        """
        return Findings(
            order_findings(f for r in self.results.values() for f in r.findings)
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
        """Make a run, deriving its health from the check results."""
        by_id: dict[str, CheckResult] = {}
        for result in results:
            if result.check_id in by_id:
                raise ValueError(f"two results for check {result.check_id}")
            by_id[result.check_id] = result
        return cls(
            results=by_id,
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
        self._check_artifacts()
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
        if len({f.id for f in self.findings}) != len(self.findings):
            raise ValueError("finding ids are unique within a run")
        if self.health != RunHealth.from_results(self.results):
            raise ValueError("health does not match the check results")

    def _check_artifacts(self) -> None:
        refs = {a.ref for r in self.results.values() for a in r.artifacts}
        for finding in self.findings:
            ref = finding.evidence.details_ref
            if ref is not None and ref not in refs:
                raise ValueError(
                    f"finding {finding.id} points at artifact {ref!r}, "
                    "which no result of the run has"
                )

    def artifact(self, ref: str) -> Payload:
        """The payload of artifact ``ref`` (``<check_id>/<name>``).

        A ``pyarrow.Table`` for a table, a JSON value otherwise. It is in memory for a
        run that was just made and read from the files beside ``run.json`` for one
        that was read back.

        Raises:
            KeyError: the run has no such artifact, or its file is missing; the
                message lists the refs the run has.
        """
        for result in self.results.values():
            payloads = result._payloads  # noqa: SLF001
            if ref in payloads:
                return payloads[ref]
        known = {a.ref: a for r in self.results.values() for a in r.artifacts}
        if ref not in known:
            raise KeyError(f"no artifact {ref!r}; the run has {sorted(known)}")
        from dsbf.sinks import read_artifact  # noqa: PLC0415

        if self._artifact_dir is None:
            raise KeyError(
                f"artifact {ref!r} is not in memory and the run has no files"
            )
        return read_artifact(self._artifact_dir, known[ref])

    @property
    def artifacts(self) -> tuple[ArtifactRef, ...]:
        """Every artifact of every result."""
        return tuple(a for r in self.results.values() for a in r.artifacts)

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
        return self.health.render(header=header, pointer=pointer, notes=self.notes)

    def write(
        self, directory: str | Path, sinks: Iterable[str] = ("json",)
    ) -> dict[str, Path]:
        """Write the run into ``directory``; return the file each sink wrote.

        ``sinks`` names the formats: ``json`` (``run.json``, the whole run) and
        ``ndjson`` (``findings.ndjson``, one finding per line), and ``parquet``
        (``findings.parquet``, ``profile.parquet`` and ``results.parquet``; each of
        ``findings_parquet``, ``profile_parquet`` and ``results_parquet`` writes one),
        and ``html`` (``report.html``, a self-contained page).

        Raises:
            ValueError: a sink name is unknown; nothing is written.
        """
        from dsbf.sinks import write_run  # noqa: PLC0415 (the model stays free of I/O)

        return write_run(self, directory, sinks)

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a run back from a ``run.json`` or the directory that holds one.

        Raises:
            RunReadError: the file is missing, is not JSON, or does not match the
                run schema (including a schema from another major version).
        """
        from dsbf.sinks import read_run  # noqa: PLC0415

        return read_run(path)  # type: ignore[return-value]

    def _repr_html_(self) -> str:
        """What a notebook shows: the health summary and the top findings."""
        return render_run_html(self)
