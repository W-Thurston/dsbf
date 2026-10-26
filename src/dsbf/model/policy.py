# dsbf/model/policy.py
"""Declarative policy rules over findings and run health, and what they decided.

A policy is data, never code: each rule pairs a ``when`` matcher with one or more
``then`` actions. The loose spellings a person writes in YAML (``severity: ">=error"``,
``then: fail``, ``then: {notify: slack:#data-quality}``) are normalised into the typed
models below when they are validated, so the models also read back what they dump.
"""

import math
import re
from collections.abc import Callable, Mapping
from enum import StrEnum
from operator import eq, ge, gt, le, lt, ne
from typing import Annotated, Any, Literal, Self

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from dsbf.model.base import Contract, FiniteFloat
from dsbf.model.finding import Finding
from dsbf.model.vocabulary import CheckStatus, Confidence, Dimension, Severity


class Operator(StrEnum):
    """How a comparison relates the observed value to the one in the rule."""

    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"


_COMPARE: dict[Operator, Callable[[Any, Any], bool]] = {
    Operator.EQ: eq,
    Operator.NE: ne,
    Operator.GE: ge,
    Operator.GT: gt,
    Operator.LE: le,
    Operator.LT: lt,
}
_ORDERING = frozenset({Operator.GE, Operator.GT, Operator.LE, Operator.LT})
# Two-character operators first, so ">=" is not read as ">" followed by "=".
_OPERATORS = (">=", "<=", "==", "!=", ">", "<")
_OPERATOR_CHARS = "<>=!"


def _split_operator(text: str) -> tuple[str, str]:
    """Split ``">=error"`` into ``(">=", "error")``; no operator means equality."""
    stripped = text.strip()
    for operator in _OPERATORS:
        if stripped.startswith(operator):
            return operator, stripped[len(operator) :].strip()
    if stripped[:1] and stripped[0] in _OPERATOR_CHARS:
        raise ValueError(
            f"{text!r} starts with an unknown operator; use one of "
            f"{', '.join(_OPERATORS)}"
        )
    return "==", stripped


def _operand(text: str) -> Any:
    """Read a metric operand: a number, ``true``/``false``, or else the text itself."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        return {"true": True, "false": False}.get(text, text)
    if not math.isfinite(number):
        raise ValueError(f"{text!r} is not a finite number")
    return number


def _comparison_input(value: Any, *, numeric: bool = False) -> Any:
    """Normalise what a rule author wrote into ``{"op": ..., "value": ...}``."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        operator, text = _split_operator(value)
        if not text:
            raise ValueError(f"{value!r} has no value after the operator")
        return {"op": operator, "value": _operand(text) if numeric else text}
    return {"op": "==", "value": value}


def _ordinal_input(value: Any) -> Any:
    return _comparison_input(value)


def _metric_input(value: Any) -> Any:
    return _comparison_input(value, numeric=True)


class SeverityComparison(Contract):
    """Compare a finding's severity, for example ``>=error``."""

    op: Operator = Operator.EQ
    value: Severity


class ConfidenceComparison(Contract):
    """Compare a finding's confidence, for example ``>=moderate``."""

    op: Operator = Operator.EQ
    value: Confidence


class MetricComparison(Contract):
    """Compare one number in a finding's evidence, for example ``>0.3``."""

    op: Operator = Operator.EQ
    value: bool | int | FiniteFloat | str

    @model_validator(mode="after")
    def _ordering_needs_a_number(self) -> Self:
        numeric = isinstance(self.value, int | float) and not isinstance(
            self.value, bool
        )
        if self.op in _ORDERING and not numeric:
            raise ValueError(
                f"operator {self.op.value!r} compares numbers, not {self.value!r}"
            )
        return self


# The input types are what the published validation schema accepts: the stored
# `{op, value}` form or the text a person writes.
_Severity = Annotated[
    SeverityComparison,
    BeforeValidator(_ordinal_input, json_schema_input_type=str | SeverityComparison),
]
_Confidence = Annotated[
    ConfidenceComparison,
    BeforeValidator(_ordinal_input, json_schema_input_type=str | ConfidenceComparison),
]
_Metric = Annotated[
    MetricComparison,
    BeforeValidator(
        _metric_input,
        json_schema_input_type=bool | int | FiniteFloat | str | MetricComparison,
    ),
]

_FINDING_KEYS = (
    "dimension",
    "code",
    "column",
    "tag",
    "severity",
    "confidence",
    "metric",
    "new_since",
)


class Matcher(Contract):
    """The ``when`` of a rule: every key given must hold, and at least one is needed.

    Finding keys select findings; ``health.status`` selects checks that ended in that
    status and may be narrowed with ``check`` only. A bare value means equality;
    ``severity``, ``confidence`` and ``metric`` values may carry an operator
    (``">=error"``, ``">0.3"``), which must be quoted in YAML.
    """

    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    dimension: Dimension | None = None
    code: str | None = Field(
        default=None,
        min_length=1,
        description="A finding's machine-readable kind, for example `high_null_rate`.",
    )
    check: str | None = Field(
        default=None, min_length=1, description="The id of the check concerned."
    )
    column: str | None = Field(
        default=None,
        min_length=1,
        description="A column the finding's subject names; a pair matches on either.",
    )
    tag: str | None = Field(
        default=None, min_length=1, description="A tag the finding carries."
    )
    severity: _Severity | None = None
    confidence: _Confidence | None = None
    metric: dict[str, _Metric] | None = Field(
        default=None,
        min_length=1,
        description="Evidence metric name to comparison; every entry must hold.",
    )
    new_since: Literal["previous_run"] | None = Field(
        default=None,
        description="Only findings absent from the previous run; needs one to compare.",
    )
    health_status: CheckStatus | None = Field(
        default=None,
        alias="health.status",
        description="Only checks that ended in this status.",
    )

    @model_validator(mode="after")
    def _check_keys(self) -> Self:
        used = [k for k in _FINDING_KEYS if getattr(self, k) is not None]
        if self.health_status is not None and used:
            raise ValueError(
                f"'health.status' selects checks and can't be combined with "
                f"{', '.join(used)}; only 'check' may accompany it"
            )
        if not used and self.health_status is None and self.check is None:
            raise ValueError(
                "a rule needs at least one condition; an empty 'when' would "
                "match everything"
            )
        return self

    @property
    def selects_checks(self) -> bool:
        """True when the matcher is about check health rather than findings."""
        return self.health_status is not None

    def matches(self, finding: Finding) -> bool:
        """True when every finding condition the matcher gives holds for ``finding``."""
        return (
            (self.dimension is None or finding.dimension == self.dimension)
            and (self.code is None or finding.code == self.code)
            and (self.check is None or finding.check_id == self.check)
            and (self.column is None or self.column in finding.subject.columns)
            and (self.tag is None or self.tag in finding.tags)
            and (
                self.severity is None
                or _COMPARE[self.severity.op](finding.severity, self.severity.value)
            )
            and (
                self.confidence is None
                or _COMPARE[self.confidence.op](
                    finding.confidence, self.confidence.value
                )
            )
            and all(
                _metric_holds(finding.evidence.metrics.get(name), comparison)
                for name, comparison in (self.metric or {}).items()
            )
        )


def _metric_holds(observed: Any, comparison: MetricComparison) -> bool:
    """A missing, null, list or mapping metric never holds; a bool is not a number."""
    if observed is None or isinstance(observed, tuple | list | dict):
        return False
    wanted = comparison.value
    if _is_number(observed) != _is_number(wanted):
        # A number is never equal to text or a flag, but it is different from them.
        return comparison.op is Operator.NE
    if isinstance(observed, bool) != isinstance(wanted, bool):
        return comparison.op is Operator.NE
    return _COMPARE[comparison.op](observed, wanted)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class ActionKind(StrEnum):
    """What a rule does when it matches."""

    ANNOTATE = "annotate"
    WARN = "warn"
    FAIL = "fail"
    NOTIFY = "notify"


_TARGET = re.compile(r"[a-z][a-z0-9_+.-]*:\S.*")


class PolicyAction(Contract):
    """One step a matched rule takes. Only `fail` affects the exit code."""

    kind: ActionKind
    target: str | None = Field(
        default=None,
        description="Where `notify` sends it, as `scheme:destination`.",
    )

    @model_validator(mode="after")
    def _check_target(self) -> Self:
        if self.kind is ActionKind.NOTIFY:
            if self.target is None or not _TARGET.fullmatch(self.target):
                raise ValueError(
                    "notify needs a target written scheme:destination, "
                    "for example `{notify: slack:#data-quality}`; "
                    f"got {self.target!r}"
                )
        elif self.target is not None:
            raise ValueError(f"{self.kind.value} takes no target; only notify does")
        return self


def _one_action(item: Any) -> Any:
    if isinstance(item, str):
        if item == ActionKind.NOTIFY:
            raise ValueError(
                "notify needs a target; write `{notify: slack:#data-quality}`"
            )
        return {"kind": item}
    if isinstance(item, Mapping) and "kind" not in item:
        if len(item) != 1:
            raise ValueError(
                f"an action is one name or one `{{notify: target}}`, got keys "
                f"{', '.join(map(repr, item))}"
            )
        (name, target), *_ = item.items()
        if name != ActionKind.NOTIFY:
            raise ValueError(
                f"only notify takes a value; write `{name}` on its own if it is an "
                f"action ({', '.join(ActionKind)})"
            )
        return {"kind": name, "target": target}
    return item


_ActionInput = PolicyAction | str | dict[str, str]


def _actions_input(value: Any) -> Any:
    items = value if isinstance(value, list | tuple) else [value]
    if not items:
        raise ValueError("a rule needs at least one action; omit 'then' to annotate")
    return tuple(_one_action(item) for item in items)


class Rule(Contract):
    """When findings or checks match, take these actions."""

    name: str = Field(min_length=1, description="Unique within the policy.")
    when: Matcher
    then: Annotated[
        tuple[PolicyAction, ...],
        Field(
            json_schema_extra={"minItems": 1},
            description=(
                "One action or a list. Written `fail`, `{notify: scheme:destination}` "
                "or a list of those; stored as `{kind, target}` entries."
            ),
        ),
        BeforeValidator(
            _actions_input, json_schema_input_type=_ActionInput | list[_ActionInput]
        ),
    ] = (PolicyAction(kind=ActionKind.ANNOTATE),)


def _check_unique_names(names: list[str], what: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"{what} {name!r} appears more than once")
        seen.add(name)


class Policy(Contract):
    """The contents of a `policy.yaml`."""

    version: Literal[1] = 1
    rules: tuple[Rule, ...] = ()

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        _check_unique_names([rule.name for rule in self.rules], "rule name")
        return self


class RuleOutcome(Contract):
    """What one rule matched in a run."""

    rule: Rule
    finding_ids: tuple[str, ...] = Field(
        default=(), description="The findings a finding rule matched."
    )
    check_ids: tuple[str, ...] = Field(
        default=(), description="The checks a health rule matched."
    )

    @property
    def triggered(self) -> bool:
        return bool(self.finding_ids or self.check_ids)

    @model_validator(mode="after")
    def _check_matches(self) -> Self:
        if self.rule.when.selects_checks and self.finding_ids:
            raise ValueError(
                f"rule {self.rule.name!r} selects checks but lists finding ids"
            )
        if not self.rule.when.selects_checks and self.check_ids:
            raise ValueError(
                f"rule {self.rule.name!r} selects findings but lists check ids"
            )
        for ids, what in ((self.finding_ids, "finding"), (self.check_ids, "check")):
            if len(set(ids)) != len(ids):
                raise ValueError(f"rule {self.rule.name!r} lists a {what} twice")
        return self


class PolicyOutcome(Contract):
    """Why a pipeline was or was not failed: every rule and what it matched."""

    rules: tuple[RuleOutcome, ...] = Field(
        description="One entry per policy rule, in policy order, fired or not."
    )
    failed: bool = Field(description="A triggered rule has a `fail` action.")

    @staticmethod
    def _fails(outcomes: tuple[RuleOutcome, ...]) -> bool:
        return any(
            outcome.triggered
            and any(action.kind is ActionKind.FAIL for action in outcome.rule.then)
            for outcome in outcomes
        )

    @classmethod
    def build(cls, rules: tuple[RuleOutcome, ...] | list[RuleOutcome]) -> Self:
        """Make an outcome, deriving `failed` from the rules."""
        outcomes = tuple(rules)
        return cls(rules=outcomes, failed=cls._fails(outcomes))

    @property
    def exit_code(self) -> int:
        """1 when a rule failed the run, else 0."""
        return 1 if self.failed else 0

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        _check_unique_names([o.rule.name for o in self.rules], "rule name")
        if self.failed != self._fails(self.rules):
            raise ValueError("failed does not match the triggered rules' actions")
        return self
