# dsbf/model/finding.py
"""One statement about the data, and the advice that goes with it."""

import hashlib
import json
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from dsbf.model.base import Contract, Scalar
from dsbf.model.evidence import Evidence
from dsbf.model.subject import Subject
from dsbf.model.vocabulary import Confidence, Dimension, Severity

GuidancePhase = Literal["eda", "ml"]


class Action(Contract):
    """A step a person or an agent can take, in a form a program can read."""

    action: str = Field(min_length=1, description="The verb, for example `transform`.")
    columns: tuple[str, ...] = Field(
        default=(),
        description="The columns the step applies to; empty for the dataset.",
    )
    method: str | None = Field(default=None, description="For example `log1p`.")
    condition: str | None = Field(
        default=None, description="When the step applies, in words."
    )
    detail: str | None = Field(
        default=None, description="What the step does, in words."
    )
    params: dict[str, Scalar] = Field(
        default_factory=dict,
        description="Anything else the step needs, for example `from_dtype`.",
    )


class Guidance(Contract):
    """What to do about a finding, from one point of view."""

    phase: GuidancePhase = Field(
        description=(
            "`eda` describes the data as it is; `ml` says what to do before modelling."
        )
    )
    actions: tuple[Action, ...] = ()


class Finding(Contract):
    """One statement about the data: what, where, how much, and why to believe it."""

    id: str = Field(
        description="Stable across runs for the same check, subject and code.",
        min_length=1,
    )
    check_id: str = Field(min_length=1)
    check_version: str = Field(min_length=1)
    code: str = Field(
        min_length=1, description="Machine-readable kind, for example `high_null_rate`."
    )
    subject: Subject
    dimension: Dimension
    severity: Severity
    confidence: Confidence
    title: str
    message: str = Field(description="Self-contained, with the observed values.")
    evidence: Evidence
    guidance: Guidance | None = None
    tags: tuple[str, ...] = ()

    @staticmethod
    def make_id(check_id: str, subject: Subject, code: str) -> str:
        """The id for a check, subject and code.

        The inputs are encoded as a JSON array, so no choice of names can make two
        different inputs produce the same text. The columns keep the order given:
        a symmetric check puts them in a canonical order before calling this.
        """
        text = json.dumps(
            [check_id, subject.kind, list(subject.columns), code], ensure_ascii=False
        )
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @classmethod
    def build(
        cls, *, check_id: str, subject: Subject, code: str, **fields: Any
    ) -> Self:
        """Make a finding, deriving its id from the check, subject and code."""
        return cls(
            id=cls.make_id(check_id, subject, code),
            check_id=check_id,
            subject=subject,
            code=code,
            **fields,
        )

    @model_validator(mode="after")
    def _check_id(self) -> Self:
        expected = self.make_id(self.check_id, self.subject, self.code)
        if self.id != expected:
            raise ValueError(
                f"id {self.id!r} does not match its check, subject and code "
                f"(expected {expected!r})"
            )
        return self
