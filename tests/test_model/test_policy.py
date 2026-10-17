"""Policy rules parse from the YAML a person writes and fail with precise errors."""

import pytest
import yaml
from pydantic import ValidationError

from dsbf.model import (
    ActionKind,
    Confidence,
    Dimension,
    Matcher,
    Operator,
    Policy,
    PolicyAction,
    PolicyOutcome,
    Rule,
    RuleOutcome,
    Severity,
)

from .builders import round_trip, strict_loads

pytestmark = pytest.mark.unit

EXAMPLE = """
version: 1
rules:
  - name: block-on-leakage
    when: { dimension: leakage, severity: ">=error" }
    then: fail                    # exit code 1, recorded in Run.policy
  - name: sparse-columns
    when: { code: high_null_rate, metric: { null_rate: ">0.3" } }
    then: warn
  - name: check-errors-are-notes
    when: { health.status: error }
    then: annotate                # default; never fails unless a rule says so
  - name: tell-the-team
    when: { severity: error, new_since: previous_run }
    then: { notify: "slack:#data-quality" }
"""


def parse(text):
    return Policy.model_validate(yaml.safe_load(text))


def rule(when, then=None, name="r"):
    fields = {"name": name, "when": when}
    if then is not None:
        fields["then"] = then
    return Rule.model_validate(fields)


def errors(model, data):
    """The (location, message) pairs of a failed validation."""
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    return [(".".join(map(str, e["loc"])), e["msg"]) for e in caught.value.errors()]


def severity(matcher):
    assert matcher.severity is not None
    return matcher.severity


def confidence(matcher):
    assert matcher.confidence is not None
    return matcher.confidence


def metric(matcher, name):
    assert matcher.metric is not None
    return matcher.metric[name]


def messages(model, data):
    return [message for _, message in errors(model, data)]


def test_the_documented_example_parses_into_typed_rules():
    policy = parse(EXAMPLE)

    leakage, sparse, health, notify = policy.rules
    assert [r.name for r in policy.rules] == [
        "block-on-leakage",
        "sparse-columns",
        "check-errors-are-notes",
        "tell-the-team",
    ]
    assert leakage.when.dimension is Dimension.LEAKAGE
    assert severity(leakage.when).op is Operator.GE
    assert severity(leakage.when).value is Severity.ERROR
    assert [a.kind for a in leakage.then] == [ActionKind.FAIL]
    assert sparse.when.code == "high_null_rate"
    assert metric(sparse.when, "null_rate").op is Operator.GT
    assert metric(sparse.when, "null_rate").value == 0.3
    assert health.when.health_status == "error"
    assert health.when.selects_checks
    assert severity(notify.when).op is Operator.EQ
    assert notify.when.new_since == "previous_run"
    assert notify.then == (PolicyAction(kind="notify", target="slack:#data-quality"),)


def test_a_policy_round_trips_through_strict_json_and_keeps_the_file_spelling():
    policy = parse(EXAMPLE)

    assert round_trip(policy) == policy
    dumped = strict_loads(policy.model_dump_json())
    assert dumped["rules"][2]["when"]["health.status"] == "error"
    assert Policy.model_validate(policy.model_dump()) == policy


def test_a_rule_without_then_only_annotates():
    assert [a.kind for a in rule({"code": "x"}).then] == [ActionKind.ANNOTATE]


def test_a_policy_may_have_no_rules_and_only_version_one():
    assert Policy().rules == ()
    assert errors(Policy, {"version": 2}) == [("version", "Input should be 1")]


def test_a_rule_can_take_several_actions():
    both = rule({"code": "x"}, ["fail", {"notify": "email:team@example.com"}])

    assert [a.kind for a in both.then] == [ActionKind.FAIL, ActionKind.NOTIFY]
    assert both.then[1].target == "email:team@example.com"


@pytest.mark.parametrize(
    ("text", "operator", "value"),
    [
        (">=error", Operator.GE, Severity.ERROR),
        (">warn", Operator.GT, Severity.WARN),
        ("<= notice", Operator.LE, Severity.NOTICE),
        ("<warn", Operator.LT, Severity.WARN),
        ("!=info", Operator.NE, Severity.INFO),
        ("==error", Operator.EQ, Severity.ERROR),
        ("error", Operator.EQ, Severity.ERROR),
    ],
)
def test_severity_comparisons_are_read_from_text(text, operator, value):
    comparison = severity(Matcher(severity=text))

    assert (comparison.op, comparison.value) == (operator, value)


def test_confidence_comparisons_use_the_same_syntax():
    comparison = confidence(Matcher(confidence=">=moderate"))

    assert (comparison.op, comparison.value) == (Operator.GE, Confidence.MODERATE)


def test_a_severity_member_is_an_equality():
    assert severity(Matcher(severity=Severity.WARN)).op is Operator.EQ


@pytest.mark.parametrize(
    ("raw", "operator", "value"),
    [
        (">0.3", Operator.GT, 0.3),
        (">=10", Operator.GE, 10),
        ("<-1.5", Operator.LT, -1.5),
        (0.5, Operator.EQ, 0.5),
        (3, Operator.EQ, 3),
        (True, Operator.EQ, True),
        ("true", Operator.EQ, True),
        ("!=false", Operator.NE, False),
        ("==object", Operator.EQ, "object"),
        ("int64", Operator.EQ, "int64"),
    ],
)
def test_metric_comparisons_read_numbers_booleans_and_text(raw, operator, value):
    comparison = metric(Matcher(metric={"m": raw}), "m")

    assert comparison.op is operator
    assert comparison.value == value
    assert type(comparison.value) is type(value)


def test_the_dumped_comparison_form_is_accepted():
    matcher = Matcher(
        severity={"op": ">=", "value": "warn"}, metric={"m": {"op": "<", "value": 2}}
    )

    assert severity(matcher).value is Severity.WARN
    assert metric(matcher, "m").value == 2


@pytest.mark.parametrize(
    ("data", "location", "fragment"),
    [
        (
            {"severity": ">=fatal"},
            "severity.value",
            "'info', 'notice', 'warn' or 'error'",
        ),
        ({"severity": "=>error"}, "severity", "unknown operator"),
        ({"severity": "=error"}, "severity", "unknown operator"),
        ({"severity": ">="}, "severity", "no value after the operator"),
        (
            {"confidence": ">=certain"},
            "confidence.value",
            "'low', 'moderate' or 'high'",
        ),
        ({"metric": {"m": ">high"}}, "metric.m", "compares numbers, not 'high'"),
        ({"metric": {"m": ">=true"}}, "metric.m", "compares numbers, not True"),
        ({"metric": {"m": ">nan"}}, "metric.m", "not a finite number"),
        ({"metric": {"m": "<inf"}}, "metric.m", "not a finite number"),
        ({"metric": {}}, "metric", "at least 1 item"),
        ({"dimension": "vibes"}, "dimension", "Input should be"),
        ({"new_since": "yesterday"}, "new_since", "'previous_run'"),
        ({"code": ""}, "code", "at least 1 character"),
        ({"colunm": "age"}, "colunm", "Extra inputs are not permitted"),
    ],
)
def test_an_invalid_condition_is_reported_where_it_is(data, location, fragment):
    reported = errors(Matcher, data)

    assert any(loc == location and fragment in msg for loc, msg in reported), reported


def test_an_empty_when_is_rejected_so_no_rule_matches_everything():
    ((location, message),) = errors(Matcher, {})

    assert location == ""
    assert "at least one condition" in message


def test_a_check_alone_is_a_condition():
    assert Matcher(check="summarize_nulls").check == "summarize_nulls"


def test_health_status_may_only_be_narrowed_by_check():
    health = Matcher.model_validate(
        {"health.status": "skipped", "check": "ml_readiness"}
    )
    assert health.health_status == "skipped"
    assert Matcher(health_status="error").health_status == "error"

    ((location, message),) = errors(
        Matcher, {"health.status": "error", "severity": "warn", "tag": "good"}
    )
    assert location == ""
    assert "can't be combined with tag, severity" in message


def test_an_unknown_health_status_lists_the_choices():
    ((location, message),) = errors(Matcher, {"health.status": "broken"})

    assert location == "health.status"
    assert "'ok'" in message
    assert "'timeout'" in message


@pytest.mark.parametrize(
    ("then", "fragment"),
    [
        ("notify", "notify needs a target"),
        ({"notify": "slack"}, "scheme:destination"),
        ({"notify": ""}, "scheme:destination"),
        ({"notify": None}, "scheme:destination"),
        ({"fail": "yes"}, "only notify takes a value"),
        ({"notify": "a:b", "fail": True}, "one name or one"),
        ("explode", "Input should be 'annotate', 'warn', 'fail' or 'notify'"),
        ([], "at least one action"),
        ({"kind": "fail", "target": "slack:#x"}, "fail takes no target"),
    ],
)
def test_an_invalid_action_says_how_to_write_it(then, fragment):
    reported = errors(Rule, {"name": "r", "when": {"code": "x"}, "then": then})

    assert any(loc.startswith("then") and fragment in msg for loc, msg in reported), (
        reported
    )


def test_the_error_location_leads_to_the_rule_and_key():
    data = yaml.safe_load(EXAMPLE)
    data["rules"][1]["when"]["metric"]["null_rate"] = ">often"

    reported = errors(Policy, data)

    assert reported[0][0] == "rules.1.when.metric.null_rate"
    assert "compares numbers" in reported[0][1]


def test_every_invalid_rule_is_reported_at_once():
    data = {
        "rules": [
            {"name": "", "when": {"code": "x"}},
            {"name": "b", "when": {"severity": ">=fatal"}, "then": "explode"},
        ]
    }

    locations = {loc for loc, _ in errors(Policy, data)}

    assert {
        "rules.0.name",
        "rules.1.when.severity.value",
        "rules.1.then.0.kind",
    } <= locations


def test_a_rule_needs_a_name_and_a_condition():
    assert ("name", "Field required") in errors(Rule, {"when": {"code": "x"}})
    assert ("when", "Field required") in errors(Rule, {"name": "r"})


def test_duplicate_rule_names_are_named_in_the_error():
    ((location, message),) = errors(
        Policy,
        {
            "rules": [
                {"name": "a", "when": {"code": "x"}},
                {"name": "a", "when": {"tag": "y"}},
            ]
        },
    )

    assert location == ""
    assert "rule name 'a' appears more than once" in message


def test_an_unquoted_operator_is_a_yaml_error_but_a_bare_number_is_an_equality():
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("{ severity: >=error }")

    matcher = Matcher.model_validate(yaml.safe_load("{ metric: { n_unique: 2 } }"))
    assert metric(matcher, "n_unique").op is Operator.EQ
    assert metric(matcher, "n_unique").value == 2


def finding_rule(**when):
    return rule({"severity": ">=error", **when}, "fail", name="gate")


def test_a_triggered_fail_rule_fails_the_run():
    outcome = PolicyOutcome.build(
        [
            RuleOutcome(rule=finding_rule(), finding_ids=("f1", "f2")),
            RuleOutcome(rule=rule({"code": "x"}, "warn", name="quiet")),
        ]
    )

    assert outcome.failed
    assert outcome.exit_code == 1
    assert [o.triggered for o in outcome.rules] == [True, False]
    assert round_trip(outcome) == outcome


def test_rules_that_do_not_fail_leave_the_exit_code_alone():
    outcome = PolicyOutcome.build(
        [
            RuleOutcome(rule=finding_rule()),
            RuleOutcome(
                rule=rule({"code": "x"}, "warn", name="w"), finding_ids=("f1",)
            ),
        ]
    )

    assert not outcome.failed
    assert outcome.exit_code == 0


def test_an_outcome_cannot_claim_a_failure_its_rules_do_not_support():
    quiet = RuleOutcome(rule=finding_rule())

    assert any(
        "failed does not match" in msg
        for _, msg in errors(PolicyOutcome, {"rules": [quiet], "failed": True})
    )
    loud = RuleOutcome(rule=finding_rule(), finding_ids=("f1",))
    assert any(
        "failed does not match" in msg
        for _, msg in errors(PolicyOutcome, {"rules": [loud], "failed": False})
    )


def test_an_outcome_lists_each_rule_once():
    one = RuleOutcome(rule=finding_rule())

    assert any(
        "appears more than once" in msg
        for _, msg in errors(PolicyOutcome, {"rules": [one, one], "failed": False})
    )


def test_a_rule_outcome_matches_findings_or_checks_according_to_its_rule():
    health = rule({"health.status": "error"}, name="h")

    assert RuleOutcome(rule=health, check_ids=("a",)).triggered
    assert any(
        "selects checks but lists finding ids" in m
        for _, m in errors(RuleOutcome, {"rule": health, "finding_ids": ["f1"]})
    )
    assert any(
        "selects findings but lists check ids" in m
        for _, m in errors(RuleOutcome, {"rule": finding_rule(), "check_ids": ["a"]})
    )
    assert any(
        "lists a finding twice" in m
        for _, m in errors(
            RuleOutcome, {"rule": finding_rule(), "finding_ids": ["f", "f"]}
        )
    )
