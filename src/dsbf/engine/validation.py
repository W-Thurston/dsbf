# dsbf/engine/validation.py
"""Checking the parameters a user supplies against what each check declares."""

import difflib
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from dsbf.engine.check import Check
from dsbf.engine.errors import PlanError
from dsbf.engine.registry import Registry
from dsbf.engine.selection import _hint
from dsbf.model import CheckParams

Params = Mapping[str, Mapping[str, Any]]


def resolve_params(
    registry: Registry, checks: Iterable[Check], params: Params | None
) -> dict[str, CheckParams]:
    """The validated parameters of each of ``checks``, defaults filled in.

    ``params`` may name any registered check, planned or not, so one configuration
    serves any selection; a name the registry doesn't hold is a mistake.

    Raises:
        PlanError: an unknown check id, a key a check doesn't declare, or a value
            outside its type or bounds. Every such mistake is listed.
    """
    params = params or {}
    problems = [
        f"params names unknown check {check_id!r}"
        f"{_hint(check_id, [c.id for c in registry], 'checks')}"
        for check_id in params
        if check_id not in registry
    ]
    resolved: dict[str, CheckParams] = {}
    for check in checks:
        raw = params.get(check.id)
        if raw is not None and not isinstance(raw, Mapping):
            problems.append(
                f"{check.id}: parameters must be a mapping of name to value, "
                f"not {type(raw).__name__}"
            )
            continue
        try:
            resolved[check.id] = check.resolve_params(raw)
        except ValidationError as error:
            problems.extend(_describe(check, error))
    if problems:
        raise PlanError(*problems)
    return resolved


def _describe(check: Check, error: ValidationError) -> list[str]:
    allowed = list(check.spec.params.model_fields)
    problems = []
    for item in error.errors():
        name = ".".join(str(part) for part in item["loc"])
        if item["type"] == "extra_forbidden":
            problems.append(
                f"{check.id}: unknown parameter {name!r}; {_allowed(allowed)}"
                f"{_suggest(str(item['loc'][0]), allowed)}"
            )
        else:
            problems.append(
                f"{check.id}: parameter {name!r}: {item['msg']} "
                f"(got {item.get('input')!r})"
            )
    return problems


def _allowed(allowed: list[str]) -> str:
    if not allowed:
        return "this check takes no parameters"
    return f"allowed: {', '.join(allowed)}"


def _suggest(name: str, allowed: list[str]) -> str:
    close = difflib.get_close_matches(name, allowed, n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""
