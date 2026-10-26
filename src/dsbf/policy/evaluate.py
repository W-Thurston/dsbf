# dsbf/policy/evaluate.py
"""Decide which policy rules a run triggers. Nothing here reads or writes anything."""

from dsbf.model import (
    Matcher,
    Note,
    Policy,
    PolicyOutcome,
    Rule,
    RuleOutcome,
    Run,
)

__all__ = ["evaluate_policy"]


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
        if when.matches(finding)
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
