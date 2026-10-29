"""A run ties the dataset, plan, profile and results together and checks they agree."""

import json
from datetime import timedelta

import pytest
from pydantic import ValidationError

from dsbf.model import (
    SCHEMA_VERSION,
    CategoricalProfile,
    CheckStatus,
    NumericProfile,
    PolicyOutcome,
    Rule,
    RuleOutcome,
    Run,
    RunHealth,
    SamplingInfo,
    Severity,
)

from . import titanic_profile
from .builders import (
    STARTED,
    make_dataset_manifest,
    make_dataset_profile,
    make_finding,
    make_note,
    make_plan,
    make_result,
    make_run,
    round_trip,
    strict_loads,
)

pytestmark = pytest.mark.unit

BUDGET_BYTES = 200_000


def parts(**overrides):
    """The inputs of a one-check run, to be overridden one at a time."""
    run = make_run()
    fields = {
        name: getattr(run, name)
        for name in (
            "run_id",
            "started_at",
            "finished_at",
            "dataset",
            "plan",
            "profile",
        )
    }
    return {"results": list(run.results.values()), **fields, **overrides}


def test_a_hand_built_run_round_trips_through_strict_json():
    run = make_run(
        [
            make_result(),
            make_result(CheckStatus.SKIPPED, check_id="ml_readiness"),
        ]
    )

    restored = Run.model_validate(strict_loads(run.model_dump_json()))

    assert restored == run
    assert restored.schema_version == SCHEMA_VERSION
    assert round_trip(run) == run


def test_the_run_derives_its_findings_and_health_from_the_results():
    run = make_run(
        [
            make_result(
                findings=(
                    make_finding(code="low", severity=Severity.NOTICE),
                    make_finding(code="high", severity=Severity.ERROR),
                    make_finding(code="mid", severity=Severity.WARN),
                )
            ),
            make_result(CheckStatus.ERROR, check_id="fuzzy_duplicates"),
        ]
    )

    assert [f.code for f in run.findings] == ["high", "mid", "low"]
    assert run.health == RunHealth.from_results(run.results)
    assert run.health.findings_at_warn_or_above == 2
    assert run.health.counts.error == 1


def test_findings_are_derived_not_serialised():
    finding = make_finding(code="only")
    run = make_run([make_result(findings=(finding,))])

    data = run.model_dump(mode="json")

    assert "findings" not in data
    assert data["results"]["summarize_nulls"]["findings"][0]["code"] == "only"
    assert run.findings == (finding,)
    assert round_trip(run) == run
    assert round_trip(run).findings == run.findings


def test_equal_severities_are_ordered_by_check_then_id():
    first = make_finding(check_id="a_check", code="z")
    second = make_finding(check_id="b_check", code="a")
    run = make_run(
        [
            make_result(check_id="b_check", findings=(second,)),
            make_result(check_id="a_check", findings=(first,)),
        ]
    )

    assert run.findings == (first, second)


def test_the_header_names_what_the_run_looked_at():
    run = make_run()

    text = run.render_health(pointer="see findings.parquet")

    assert text.splitlines()[0] == (
        "DSBF run 7f3c…  titanic  (10 × 1, full scan)   1 checks"
    )
    assert text.splitlines()[-1].endswith("→ see findings.parquet")


def test_run_notes_are_listed_under_the_check_issues():
    note = make_note(
        code="plugin_failed", message="Could not load a.py.", hint="Fix it."
    )

    run = make_run(notes=(note,))

    lines = run.render_health().splitlines()
    assert "  ! plugin_failed: Could not load a.py. Hint: Fix it." in lines
    assert (
        lines.index("  ! plugin_failed: Could not load a.py. Hint: Fix it.")
        < len(lines) - 1
    )
    assert round_trip(run).notes == (note,)


def test_a_run_without_notes_has_none():
    assert make_run().notes == ()


def test_a_sampled_run_says_so():
    sampling = SamplingInfo(
        strategy="random", original_rows=1000, sampled_rows=10, seed=1
    )

    run = make_run(
        dataset=make_dataset_manifest(n_rows=1000, sampling=sampling),
        profile=make_dataset_profile(n_rows=10),
    )

    assert "(1,000 × 1, sample of 10 rows)" in run.render_health()


def test_two_results_for_one_check_are_refused():
    with pytest.raises(ValueError, match="two results for check"):
        Run.build(**parts(results=[make_result(), make_result()]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0.0", "not readable"),
        ("schema_version", "latest", "major.minor.patch"),
        ("finished_at", STARTED - timedelta(seconds=1), "starts before it finishes"),
        ("profile", make_dataset_profile(n_rows=9), "profile covers 9 rows"),
        ("dataset", make_dataset_manifest(n_columns=2), "1 columns but"),
        ("plan", make_plan("summarize_nulls", "extra"), "no result for"),
        ("plan", make_plan("other"), "unplanned results"),
        ("plan", make_plan("summarize_nulls", version="2"), "planned at version 2"),
    ],
)
def test_the_parts_of_a_run_must_agree(field, value, message):
    with pytest.raises(ValidationError, match=message):
        Run.build(**parts(**{field: value}))


def test_a_minor_schema_version_is_readable():
    assert make_run(schema_version="1.9.3").schema_version == "1.9.3"


def test_health_cannot_be_edited_out_of_step_with_the_results():
    run = make_run()
    data = run.model_dump(mode="json")

    with pytest.raises(ValidationError, match="findings"):
        Run.model_validate({**data, "findings": []})
    with pytest.raises(ValidationError, match="health does not match"):
        Run.model_validate(
            {**data, "health": RunHealth.from_results({}).model_dump(mode="json")}
        )
    with pytest.raises(ValidationError, match="keyed"):
        Run.model_validate(
            {**data, "results": {"wrong": data["results"]["summarize_nulls"]}}
        )


def test_a_run_needs_an_aware_time_and_a_uuid():
    data = make_run().model_dump(mode="json")

    with pytest.raises(ValidationError):
        Run.model_validate({**data, "run_id": "not-a-uuid"})
    with pytest.raises(ValidationError):
        Run.model_validate({**data, "started_at": "2026-10-17T09:00:00"})


def test_a_run_is_immutable():
    run = make_run()

    with pytest.raises(ValidationError):
        run.__setattr__("schema_version", "9.9.9")


def test_the_profile_and_manifests_of_a_real_dataset_fit_the_size_budget():
    frame = titanic_profile.load()
    profile = titanic_profile.build(frame)
    dataset = make_dataset_manifest(n_rows=len(frame), n_columns=len(frame.columns))
    plan = make_plan(*(f"check_{i:02d}" for i in range(61)))

    sizes = {
        "profile": len(profile.model_dump_json()),
        "dataset": len(dataset.model_dump_json()),
        "plan": len(plan.model_dump_json()),
    }

    assert sum(sizes.values()) < BUDGET_BYTES, sizes
    run = Run.build(
        results=[make_result(check_id=c) for c in plan.check_ids],
        run_id="7f3c1a52-4b0e-4d6a-9c1e-2a5b8d0f3e71",
        started_at=STARTED,
        finished_at=STARTED + timedelta(seconds=3),
        dataset=dataset,
        plan=plan,
        profile=profile,
    )
    assert round_trip(run) == run


def test_the_real_dataset_profile_describes_the_data():
    frame = titanic_profile.load()
    profile = titanic_profile.build(frame)

    age = profile.column("age")
    assert (profile.n_rows, profile.n_columns) == (891, 15)
    assert age.semantic_type == "numeric"
    assert age.counts.nulls == int(frame["age"].isna().sum()) == 177
    assert isinstance(age, NumericProfile)
    assert age.histogram is not None
    assert age.histogram.total == 891 - 177
    assert profile.column("deck").counts.nulls == 688
    survived = profile.column("survived")
    assert isinstance(survived, CategoricalProfile)
    assert [(v.value, v.count) for v in survived.top_values] == [(0, 549), (1, 342)]
    assert profile.duplicate_rows == 107
    assert json.loads(profile.model_dump_json())["columns"][0]["name"] == "survived"


def policy_run(*, finding_ids=None, check_ids=("fuzzy_duplicates",), **when):
    """A run with a finding and an errored check, judged by a two-rule policy."""
    finding = make_finding()
    results = [
        make_result(findings=(finding,)),
        make_result(CheckStatus.ERROR, check_id="fuzzy_duplicates"),
    ]
    gate = Rule(name="gate", when={"severity": ">=warn"}, then="fail")
    errored = Rule(name="errored", when={"health.status": "error"})
    outcome = PolicyOutcome.build(
        [
            RuleOutcome(rule=gate, finding_ids=finding_ids or (finding.id,)),
            RuleOutcome(rule=errored, check_ids=check_ids),
        ]
    )
    return make_run(results, policy=outcome)


def test_a_run_without_a_policy_has_no_outcome():
    assert make_run().policy is None


def test_a_policy_outcome_round_trips_inside_the_run():
    run = policy_run()

    restored = Run.model_validate(strict_loads(run.model_dump_json()))

    assert restored == run
    assert restored.policy is not None
    assert restored.policy.failed
    assert restored.policy.exit_code == 1


def test_a_policy_cannot_name_a_finding_the_run_does_not_have():
    with pytest.raises(ValidationError, match=r"rule 'gate' matched unknown findings"):
        policy_run(finding_ids=("0123456789abcdef",))


def test_a_policy_cannot_name_a_check_the_run_does_not_have():
    with pytest.raises(ValidationError, match="rule 'errored' matched unknown check"):
        policy_run(check_ids=("missing",))


def test_a_health_rule_only_matches_checks_with_its_status():
    with pytest.raises(ValidationError, match=r"whose status is ok, not error"):
        policy_run(check_ids=("summarize_nulls",))
