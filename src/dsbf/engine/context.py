# dsbf/engine/context.py
"""What a check is given while it runs, and nothing beyond its declared scope."""

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from dsbf.data import Dataset
from dsbf.engine.check import CheckOutputModel, CheckSpec
from dsbf.engine.eligibility import eligibility
from dsbf.engine.errors import CheckCancelled, CheckContractError
from dsbf.engine.reference import Reference
from dsbf.model import (
    CheckParams,
    CheckResult,
    CheckStatus,
    ColumnProfile,
    Confidence,
    DatasetProfile,
    Dimension,
    Evidence,
    Finding,
    Guidance,
    Note,
    NullPair,
    SamplingInfo,
    Severity,
    Subject,
)

_STOP_STATUSES = (
    CheckStatus.NOT_APPLICABLE,
    CheckStatus.UNAVAILABLE,
    CheckStatus.SKIPPED,
    CheckStatus.ERROR,
)


@dataclass(frozen=True)
class CheckOutput:
    """A check's output, validated against the model it declared.

    ``status`` is ``ok`` unless the check ended through ``CheckContext.stop``.
    """

    data: dict[str, Any] | None
    status: CheckStatus = CheckStatus.OK


@dataclass(frozen=True)
class ProfileView:
    """The profile as one check may see it: dataset facts and its own columns."""

    n_rows: int
    approx_memory_bytes: int | None
    duplicate_rows: int | None
    rows_with_any_null: int | None
    columns: tuple[ColumnProfile, ...]
    null_pairs: tuple[NullPair, ...] = field(
        metadata={"doc": "Pairs of columns null together, both within `columns`."}
    )
    _by_name: dict[str, ColumnProfile] = field(repr=False, compare=False)

    def column(self, name: str) -> ColumnProfile:
        """The profile of ``name``, if the check applies to it."""
        try:
            return self._by_name[name]
        except KeyError:
            raise CheckContractError(
                f"No column {name!r} in this check's scope; "
                f"it applies to {len(self.columns)} columns"
            ) from None


class CheckContext:
    """The view of the data a check runs against.

    The caller builds one per check from the check's spec, the dataset, its profile,
    the resolved parameters, the results of the checks it may read and, for a check
    that requires one, the reference dataset. A check
    reaches columns only through ``columns``, ``values`` and ``profile``, all of
    which are limited to the columns the check can read: those ``applies_to``
    selects that also have the profile features it requires, as in the plan.
    ``eligibility`` says why a check has nothing to run on.

    When the executor gives up on a check for taking too long it sets ``cancel``;
    from then on every method here, and ``checkpoint``, raises ``CheckCancelled``.
    """

    def __init__(
        self,
        spec: CheckSpec,
        dataset: Dataset,
        profile: DatasetProfile,
        params: CheckParams | None = None,
        *,
        upstream: Mapping[str, CheckResult] | None = None,
        reference: Reference | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        if [column.name for column in profile.columns] != dataset.columns:
            raise ValueError("the profile and the dataset have different columns")
        if profile.n_rows != dataset.n_rows:
            raise ValueError(
                f"the profile counts {profile.n_rows} rows, "
                f"the dataset {dataset.n_rows}"
            )
        self.spec = spec
        self.params = params if params is not None else spec.params()
        self._dataset = dataset
        self._upstream = dict(upstream or {})
        self._reference = reference if spec.requires.reference else None
        self._cancel = cancel
        self._findings: list[Finding] = []
        self._finding_ids: set[str] = set()
        self._notes: list[Note] = []

        self.eligibility = eligibility(
            spec,
            profile,
            has_reference=self._reference is not None,
            params={name: getattr(self.params, name) for name in spec.requires.params},
        )
        eligible = set(self.eligibility.columns or ())
        scoped = tuple(c for c in profile.columns if c.name in eligible)
        self.columns: tuple[str, ...] = tuple(column.name for column in scoped)
        names = self._scope = frozenset(self.columns)
        self.profile = ProfileView(
            n_rows=profile.n_rows,
            approx_memory_bytes=profile.approx_memory_bytes,
            duplicate_rows=profile.duplicate_rows,
            rows_with_any_null=profile.rows_with_any_null,
            columns=scoped,
            null_pairs=tuple(
                pair
                for pair in (profile.missingness.pairs if profile.missingness else ())
                if pair.a in names and pair.b in names
            ),
            _by_name={column.name: column for column in scoped},
        )

    @property
    def sampling(self) -> SamplingInfo | None:
        """How the dataset was sampled, or None if the check sees all of it."""
        return self._dataset.sampling

    @property
    def reference(self) -> Reference:
        """The reference dataset the run compares with.

        Raises:
            CheckContractError: the check did not declare ``Requires(reference=True)``.
        """
        self.checkpoint()
        if not self.spec.requires.reference or self._reference is None:
            raise CheckContractError(
                f"{self.spec.id} did not declare requires.reference, "
                "so it cannot read the reference"
            )
        return self._reference

    @property
    def coerced(self) -> Mapping[str, str]:
        """The columns in scope that had to change type to load, with the reason."""
        return MappingProxyType(
            {c: why for c, why in self._dataset.coerced.items() if c in self._scope}
        )

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(self._findings)

    @property
    def notes(self) -> tuple[Note, ...]:
        return tuple(self._notes)

    def checkpoint(self) -> None:
        """Stop here if the executor has given up on this check.

        Every method of the context calls this, so a check that keeps using its
        context stops by itself. A long loop that doesn't should call it once per
        iteration; that is the only way to interrupt a check that has timed out.

        Raises:
            CheckCancelled: the check's time is up. Let it propagate.
        """
        if self._cancel is not None and self._cancel.is_set():
            raise CheckCancelled(self.spec.id)

    def values(self, column: str, dropna: bool = True) -> np.ndarray:
        """The column as a read-only NumPy array; see ``Dataset.values``.

        Only an analysis check reads values, and only for columns it applies to.
        """
        self.checkpoint()
        if self.spec.kind != "analysis":
            raise CheckContractError(
                f"{self.spec.id} is a {self.spec.kind} check, which does not read "
                "values; declare it kind='analysis' or use the profile"
            )
        self._require_in_scope(column)
        return self._dataset.values(column, dropna=dropna)

    def result(self, check_id: str) -> CheckResult | None:
        """The result of a check this one declared in ``requires`` or ``after``.

        None when that check didn't run, which can only happen for ``after``.
        """
        self.checkpoint()
        if check_id not in self.spec.upstream:
            raise CheckContractError(
                f"{self.spec.id} did not declare {check_id!r} in requires.checks "
                "or after, so it cannot read its result"
            )
        return self._upstream.get(check_id)

    def finding(
        self,
        *,
        code: str,
        subject: Subject,
        severity: Severity,
        confidence: Confidence,
        title: str,
        message: str,
        evidence: Evidence,
        dimension: Dimension | None = None,
        guidance: Guidance | None = None,
        tags: tuple[str, ...] = (),
    ) -> Finding:
        """Record a finding for this check and return it.

        The id comes from the check, the subject and the code, so a check states
        each (subject, code) once. ``dimension`` may be left out when the check
        declares just one. A finding about a sample carries the sample's provenance.
        """
        self.checkpoint()
        spec = self.spec
        if dimension is None:
            if len(spec.dimensions) != 1:
                raise CheckContractError(
                    f"{spec.id} declares {len(spec.dimensions)} dimensions, "
                    "so each finding names one"
                )
            dimension = spec.dimensions[0]
        elif dimension not in spec.dimensions:
            raise CheckContractError(
                f"{spec.id} did not declare dimension {dimension.value!r}; "
                f"it declares {', '.join(d.value for d in spec.dimensions)}"
            )
        for column in subject.columns:
            self._require_in_scope(column)
        if evidence.sampled is None and self.sampling is not None:
            evidence = evidence.model_copy(update={"sampled": self.sampling})
        finding = Finding.build(
            check_id=spec.id,
            check_version=spec.version,
            subject=subject,
            code=code,
            dimension=dimension,
            severity=severity,
            confidence=confidence,
            title=title,
            message=message,
            evidence=evidence,
            guidance=guidance,
            tags=tags,
        )
        if finding.id in self._finding_ids:
            raise CheckContractError(
                f"{spec.id} already reported {code!r} for this subject"
            )
        self._finding_ids.add(finding.id)
        self._findings.append(finding)
        return finding

    def note(
        self,
        code: str,
        message: str,
        *,
        severity: Severity = Severity.INFO,
        hint: str | None = None,
        exception_type: str | None = None,
        detail: str | None = None,
    ) -> Note:
        """Record an operational message, such as a cap that was applied."""
        self.checkpoint()
        note = Note(
            code=code,
            message=message,
            severity=severity,
            hint=hint,
            exception_type=exception_type,
            detail=detail,
        )
        self._notes.append(note)
        return note

    def stop(
        self,
        status: CheckStatus,
        code: str,
        message: str,
        *,
        severity: Severity = Severity.WARN,
        hint: str | None = None,
        exception_type: str | None = None,
        detail: str | None = None,
    ) -> CheckOutput:
        """End the check without an output, in a status other than ``ok``.

        For a check that finds out, part way, that it cannot give an answer and can
        say why without raising: ``skipped`` when something it needs is absent,
        ``not_applicable`` when the data leaves it nothing to read, ``error`` when it
        failed and caught the failure. The note says why; the findings and notes
        recorded so far are kept only as notes, and the findings are dropped, as for
        an exception. Return the result from the check.

        Raises:
            CheckContractError: ``status`` is ``ok`` (use ``output``), or a status
                the executor alone decides (``unavailable``, ``timeout``).
        """
        self.checkpoint()
        if status not in _STOP_STATUSES:
            allowed = ", ".join(s.value for s in _STOP_STATUSES)
            raise CheckContractError(
                f"{self.spec.id} cannot stop with status {status.value!r}; "
                f"choose from {allowed}"
            )
        self.note(
            code,
            message,
            severity=severity,
            hint=hint,
            exception_type=exception_type,
            detail=detail,
        )
        return CheckOutput(data=None, status=status)

    def output(self, model: CheckOutputModel | None = None) -> CheckOutput:
        """Wrap the check's output, which must be an instance of its declared model."""
        self.checkpoint()
        declared = self.spec.output
        if model is None:
            return CheckOutput(data=None)
        if declared is None:
            raise CheckContractError(
                f"{self.spec.id} declares no output model, so it returns no output"
            )
        if not isinstance(model, declared):
            raise CheckContractError(
                f"{self.spec.id} declares output {declared.__name__}, "
                f"not {type(model).__name__}"
            )
        return CheckOutput(data=model.model_dump(mode="json"))

    def _require_in_scope(self, column: str) -> None:
        if column not in self._scope:
            applies = self.spec.applies_to
            scope = applies.describe() if applies else "no columns"
            raise CheckContractError(
                f"{self.spec.id} applies to {scope}; column {column!r} is outside "
                "that scope"
            )
