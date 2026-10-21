# dsbf/engine/tips.py
"""Recommendation tips: conditions on a finding's numbers, evaluated without ``eval``.

The library is a YAML file of tips. Each tip names the checks (``checks``) and the
legacy tasks (``applicable_to``) it advises, a ``condition`` over named values, and the
step it recommends. A condition is a comparison, ``{field, op, value}``, or an
``any`` / ``all`` list of conditions. For a check, the values are the metrics of a
finding's evidence; a tip that matches becomes an ``Action`` on that finding, after
the check has returned.
"""

import hashlib
import json
import logging
import operator
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from dsbf.model import Action, Contract, Finding, Guidance, Note, Severity

logger = logging.getLogger(__name__)

Operator = Literal["<", "<=", ">", ">=", "==", "!=", "in"]

_COMPARE: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda value, options: value in options,
}


class Compare(Contract):
    """One comparison of a named value with a constant."""

    field: str = Field(min_length=1)
    op: Operator
    value: bool | int | float | str | tuple[bool | int | float | str, ...]

    @model_validator(mode="after")
    def _value_fits_op(self) -> "Compare":
        if (self.op == "in") != isinstance(self.value, tuple):
            raise ValueError("`in` takes a list of values, and no other operator does")
        return self


class AnyOf(Contract):
    """True when any of the conditions is."""

    any_of: tuple["Condition", ...] = Field(alias="any", min_length=1)


class AllOf(Contract):
    """True when all of the conditions are."""

    all_of: tuple["Condition", ...] = Field(alias="all", min_length=1)


Condition = Annotated[Compare | AnyOf | AllOf, Field(union_mode="left_to_right")]
AnyOf.model_rebuild()
AllOf.model_rebuild()


class Tip(Contract):
    """Advice for a finding whose numbers meet a condition."""

    condition: Condition
    checks: tuple[str, ...] = Field(
        default=(), description="The core checks whose findings it advises."
    )
    codes: tuple[str, ...] = Field(
        default=(),
        description="The finding codes it advises; empty for any code of the check.",
    )
    applicable_to: tuple[str, ...] = Field(
        default=(), description="The legacy tasks it advises, by task name."
    )
    action: str = Field(min_length=1, description="The verb, for example `drop`.")
    method: str | None = None
    when: str | None = Field(
        default=None, description="When the step applies, in words."
    )
    message: str = Field(min_length=1)


def holds(condition: Compare | AnyOf | AllOf, values: Mapping[str, Any]) -> bool:
    """Whether ``condition`` is true of ``values``.

    A value that is missing, null or of a type the comparison does not take makes the
    comparison false: a tip never needs a number a finding lacks.
    """
    if isinstance(condition, AnyOf):
        return any(holds(part, values) for part in condition.any_of)
    if isinstance(condition, AllOf):
        return all(holds(part, values) for part in condition.all_of)
    found = values.get(condition.field)
    if found is None:
        return False
    try:
        return bool(_COMPARE[condition.op](found, condition.value))
    except TypeError:
        return False


class TipLibraryError(Exception):
    """The packaged library cannot be read or is not a valid set of tips."""


def _resource() -> Traversable:
    return files("dsbf").joinpath("static_metadata", "recommendation_library.yaml")


def parse_library(text: str) -> dict[str, Tip]:
    """The tips in ``text``, by id.

    Raises:
        TipLibraryError: ``text`` is not YAML, not a mapping of tips, or a tip is
            invalid. The message names the tip.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise TipLibraryError(f"not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise TipLibraryError("expected a mapping of tip ids to tips")
    library: dict[str, Tip] = {}
    for tip_id, entry in raw.items():
        try:
            library[str(tip_id)] = Tip.model_validate(entry)
        except ValidationError as error:
            raise TipLibraryError(f"tip {tip_id!r} is invalid: {error}") from error
    return library


@cache
def load_library() -> dict[str, Tip]:
    """The packaged library, read once per process.

    Raises:
        TipLibraryError: the file is missing, unreadable or invalid. A failure is not
            cached, so a fixed file is read on the next call.
    """
    try:
        text = _resource().read_text(encoding="utf-8")
    except OSError as error:
        raise TipLibraryError(f"cannot read the library: {error}") from error
    return parse_library(text)


def library_digest() -> str:
    """A digest of the packaged library, or of nothing if it is unusable.

    Part of a result's cache key, because a tip is attached to the result it keeps.
    """
    try:
        library = load_library()
    except TipLibraryError:
        return "unavailable"
    text = json.dumps(
        {name: tip.model_dump(mode="json") for name, tip in library.items()},
        sort_keys=True,
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def legacy_tip(task_name: str, values: Mapping[str, Any]) -> str | None:
    """The message of the first tip for a legacy task that ``values`` meets, if any.

    A library that cannot be used is logged once and gives no tip, because a tip is
    optional enrichment and must never fail an analysis.
    """
    try:
        library = load_library()
    except TipLibraryError as error:
        _warn_once(str(error))
        return None
    for tip in library.values():
        if task_name in tip.applicable_to and holds(tip.condition, values):
            return tip.message
    return None


@cache
def _warn_once(reason: str) -> None:
    logger.warning(f"Recommendation library unavailable: {reason}")


def attach_tips(
    check_id: str, findings: Sequence[Finding]
) -> tuple[tuple[Finding, ...], tuple[Note, ...]]:
    """The findings with the tips that apply to them, and a note if that failed.

    A finding that already has guidance keeps it. A tip that matches is an ``Action``
    in ``ml`` guidance on the finding, on the columns of its subject. Nothing here
    can fail the check: an unusable library, or a lookup that raises, comes back as a
    note and the findings as they were.
    """
    if not findings:
        return (), ()
    try:
        library = load_library()
    except TipLibraryError as error:
        return tuple(findings), (
            Note(
                code="tip_library_unavailable",
                message=f"Recommendation tips were not attached: {error}",
                severity=Severity.NOTICE,
            ),
        )
    tips = [tip for tip in library.values() if check_id in tip.checks]
    if not tips:
        return tuple(findings), ()
    result: list[Finding] = []
    failed = 0
    for finding in findings:
        try:
            result.append(_with_tips(finding, tips))
        except Exception:  # noqa: BLE001 - a tip must never fail its check
            failed += 1
            result.append(finding)
    notes = ()
    if failed:
        notes = (
            Note(
                code="tip_lookup_failed",
                message=(
                    f"Recommendation tips were not attached to {failed} finding(s) "
                    f"of {check_id}."
                ),
                severity=Severity.NOTICE,
                count=failed,
            ),
        )
    return tuple(result), notes


def _with_tips(finding: Finding, tips: Sequence[Tip]) -> Finding:
    if finding.guidance is not None:
        return finding
    actions = tuple(
        Action(
            action=tip.action,
            columns=finding.subject.columns,
            method=tip.method,
            condition=tip.when,
            detail=" ".join(tip.message.split()),
        )
        for tip in tips
        if (not tip.codes or finding.code in tip.codes)
        and holds(tip.condition, finding.evidence.metrics)
    )
    if not actions:
        return finding
    return finding.model_copy(
        update={"guidance": Guidance(phase="ml", actions=actions)}
    )
