# dsbf/policy/evaluate.py
"""Decide which policy rules a run triggers. Nothing here reads or writes anything."""

import operator
from collections.abc import Callable
from typing import Any

from dsbf.model import (
    Finding,
    Matcher,
    MetricComparison,
    Note,
    Operator,
    Policy,
    PolicyOutcome,
    Rule,
    RuleOutcome,
    Run,
)

__all__ = ["evaluate_policy"]

_COMPARE: dict[Operator, Callable[[Any, Any], bool]] = {
    Operator.EQ: operator.eq,
    Operator.NE: operator.ne,
    Operator.GE: operator.ge,
    Operator.GT: operator.gt,
    Operator.LE: operator.le,
    Operator.LT: operator.lt,
}


def evaluate_policy(
    run: Run, policy: Policy, previous: Run | None = None
) -> tuple[PolicyOutcome, tuple[Note, ...]]:
    """What ``policy`` decides about ``run``, and the notes the evaluation leaves.

    Every rule gets an outcome, in policy order, whether it fired or not. A rule
    about findings lists the findings it matched in the run's order; a rule about
    ``health.status`` lists the checks that ended in that status. ``new_since:
    previous_run`` compares finding ids with ``previous``; without one it matches
    nothing and a ``policy_no_previous_run`` note names the rules concerned.
    ``notify`` is recorded in the outcome and never sent.
    """
    known = None if previous is None else {f.id for f in previous.findings}
    outcomes = [_outcome(run, rule, known) for rule in policy.rules]
    waiting = [
        rule.name
        for rule in policy.rules
        if known is None and rule.when.new_since is not None
    ]
    notes = (_no_previous_note(waiting),) if waiting else ()
    return PolicyOutcome.build(outcomes), notes


def _no_previous_note(rules: list[str]) -> Note:
    return Note(
        code="policy_no_previous_run",
        message=(
            f"{', '.join(rules)} look for findings new since the previous run, "
            "and no previous run was given, so they matched nothing."
        ),
        hint="Pass the previous run, or drop `new_since` from the rule.",
    )


def _outcome(run: Run, rule: Rule, known: set[str] | None) -> RuleOutcome:
    when = rule.when
    if when.selects_checks:
        return RuleOutcome(rule=rule, check_ids=_checks(run, when))
    if when.new_since is not None and known is None:
        return RuleOutcome(rule=rule)
    ids = tuple(
        finding.id
        for finding in run.findings
        if _matches(finding, when)
        and (when.new_since is None or finding.id not in (known or ()))
    )
    return RuleOutcome(rule=rule, finding_ids=ids)


def _checks(run: Run, when: Matcher) -> tuple[str, ...]:
    return tuple(
        check_id
        for check_id, result in run.results.items()
        if result.status == when.health_status
        and (when.check is None or check_id == when.check)
    )


def _matches(finding: Finding, when: Matcher) -> bool:
    """True when every condition the matcher gives holds for the finding."""
    return (
        (when.dimension is None or finding.dimension == when.dimension)
        and (when.code is None or finding.code == when.code)
        and (when.check is None or finding.check_id == when.check)
        and (when.column is None or when.column in finding.subject.columns)
        and (when.tag is None or when.tag in finding.tags)
        and (
            when.severity is None
            or _COMPARE[when.severity.op](finding.severity, when.severity.value)
        )
        and (
            when.confidence is None
            or _COMPARE[when.confidence.op](finding.confidence, when.confidence.value)
        )
        and all(
            _metric_holds(finding.evidence.metrics.get(name), comparison)
            for name, comparison in (when.metric or {}).items()
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
