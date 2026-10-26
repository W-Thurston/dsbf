"""The published run schema describes real runs, and changes to it are classified."""

import copy
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from dsbf.model import SCHEMA_VERSION, CheckStatus, Policy, Run
from dsbf.model.schema import (
    Change,
    compare_schemas,
    policy_schema,
    run_schema,
    schema_text,
    schema_version,
    typescript_schema,
    unannounced,
)

from . import titanic_profile
from .builders import (
    make_dataset_manifest,
    make_finding,
    make_plan,
    make_result,
    make_run,
)
from .test_run import policy_run

pytestmark = pytest.mark.unit

COMMITTED = Path(__file__).parents[2] / "schema" / "run.schema.json"


def every_node(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from every_node(value)
    elif isinstance(node, list):
        for item in node:
            yield from every_node(item)


def hand_built_run():
    """A run with findings of several severities, an errored check and a policy."""
    run = policy_run()
    extra = make_result(
        check_id="outliers",
        findings=(make_finding(check_id="outliers", code="many_outliers"),),
    )
    return make_run(
        [*run.results.values(), extra],
        policy=run.policy,
        plan=make_plan(*run.results, "outliers"),
    )


# --- the export -------------------------------------------------------------------


def test_the_committed_schema_is_what_the_models_export():
    assert COMMITTED.read_text(encoding="utf-8") == schema_text(run_schema())


def test_the_committed_policy_schema_is_what_the_models_export():
    committed = COMMITTED.with_name("policy.schema.json")
    assert committed.read_text(encoding="utf-8") == schema_text(policy_schema())


def test_the_policy_schema_accepts_what_a_person_writes():
    validator = Draft202012Validator(policy_schema())
    validator.check_schema(policy_schema())
    written = {
        "version": 1,
        "rules": [
            {"name": "a", "when": {"severity": ">=error"}, "then": "fail"},
            {"name": "b", "when": {"metric": {"null_share": ">0.3"}}},
            {"name": "c", "when": {"health.status": "error"}, "then": ["warn"]},
            {"name": "d", "when": {"code": "x"}, "then": {"notify": "slack:#q"}},
            {"name": "e", "when": {"metric": {"flag": True, "n": 3}}},
        ],
    }
    validator.validate(written)
    Policy.model_validate(written)
    assert list(validator.iter_errors({"rules": [{"name": "a", "when": {"bad": 1}}]}))


def test_the_export_is_deterministic():
    assert schema_text(run_schema()) == schema_text(run_schema())


def test_the_schema_carries_its_version():
    assert schema_version(run_schema()) == SCHEMA_VERSION


def test_the_schema_is_valid_json_schema():
    Draft202012Validator.check_schema(run_schema())


def test_every_object_rejects_unknown_properties():
    schema = run_schema()
    objects = [n for n in every_node(schema) if n.get("type") == "object"]
    open_ones = [
        n
        for n in objects
        if "properties" in n and n.get("additionalProperties") is not False
    ]

    assert objects
    assert not open_ones


def test_findings_are_not_part_of_the_run_schema():
    assert "findings" not in run_schema()["properties"]
    assert "findings" in run_schema()["$defs"]["CheckResult"]["properties"]


def test_properties_carry_no_generated_titles():
    for node in every_node(run_schema()):
        for name, prop in node.get("properties", {}).items():
            if isinstance(prop, dict) and name != "title":
                assert "title" not in prop, name


def test_a_dumped_run_always_has_its_defaulted_fields():
    required = set(run_schema()["required"])

    assert required == set(Run.model_fields)


def test_each_column_profile_is_told_apart_by_a_required_constant():
    defs = run_schema()["$defs"]
    kinds = set()
    for name in (n for n in defs if n.endswith("Profile") and n != "DatasetProfile"):
        kind = defs[name]["properties"]["semantic_type"]
        assert "semantic_type" in defs[name]["required"], name
        kinds.add(kind["const"])

    assert kinds == {
        "numeric",
        "categorical",
        "boolean",
        "datetime",
        "text",
        "identifier",
        "unknown",
    }


# --- the phase exit check ---------------------------------------------------------


def test_a_hand_built_run_validates_round_trips_and_exports():
    run = hand_built_run()
    document = json.loads(run.model_dump_json())

    Draft202012Validator(run_schema()).validate(document)
    assert Run.model_validate_json(run.model_dump_json()) == run
    assert Run.model_validate(document).findings == run.findings
    assert run.policy is not None
    assert "health.status" in document["policy"]["rules"][1]["rule"]["when"]


def test_a_run_with_a_real_profile_validates_against_the_schema():
    frame = titanic_profile.load()
    run = make_run(
        dataset=make_dataset_manifest(n_rows=len(frame), n_columns=len(frame.columns)),
        profile=titanic_profile.build(frame),
    )

    Draft202012Validator(run_schema()).validate(json.loads(run.model_dump_json()))


def test_a_run_missing_a_field_or_with_an_extra_one_does_not_validate():
    document = json.loads(make_run().model_dump_json())
    validator = Draft202012Validator(run_schema())

    with pytest.raises(ValidationError):
        validator.validate({k: v for k, v in document.items() if k != "health"})
    with pytest.raises(ValidationError):
        validator.validate({**document, "findings": []})
    with pytest.raises(ValidationError):
        validator.validate({**document, "run_id": 7})


def test_every_check_status_is_in_the_schema():
    statuses = set(run_schema()["$defs"]["CheckStatus"]["enum"])

    assert statuses == {s.value for s in CheckStatus}


# --- the TypeScript view ----------------------------------------------------------


def test_a_reference_with_its_own_description_is_named_not_inlined():
    schema = {
        "properties": {
            "plain": {"$ref": "#/$defs/Kind"},
            "described": {"$ref": "#/$defs/Kind", "description": "why", "default": "a"},
        },
        "$defs": {"Kind": {"enum": ["a", "b"]}},
    }

    converted = typescript_schema(schema)

    assert converted["properties"]["plain"] == {"$ref": "#/$defs/Kind"}
    assert converted["properties"]["described"] == {
        "tsType": "Kind",
        "description": "why",
    }
    assert converted["$defs"] == schema["$defs"]
    assert schema["properties"]["described"]["default"] == "a"


# --- breaking-change rules --------------------------------------------------------


BASE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a", "b"],
    "properties": {
        "a": {"type": "string", "description": "words", "default": ""},
        "b": {"$ref": "#/$defs/Mode"},
        "c": {
            "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}],
            "default": None,
        },
        "d": {"type": "array", "items": {"$ref": "#/$defs/Mode"}},
    },
    "$defs": {
        "Mode": {"type": "string", "enum": ["fast", "slow"]},
        "Unused": {"type": "string"},
    },
}


def changed(edit):
    new = copy.deepcopy(BASE)
    edit(new)
    return compare_schemas(BASE, new)


def test_identical_schemas_have_no_changes():
    assert compare_schemas(BASE, copy.deepcopy(BASE)) == []
    assert compare_schemas(run_schema(), run_schema()) == []


def test_annotations_and_ordering_are_not_changes():
    def edit(new):
        new["properties"]["a"].update(description="other", title="A", default="x")
        new["$defs"]["Mode"]["enum"] = ["slow", "fast"]
        new["properties"]["c"]["anyOf"].reverse()

    assert changed(edit) == []


@pytest.mark.parametrize(
    ("edit", "path", "text"),
    [
        (lambda s: s["properties"].pop("a"), "#/a", "property removed"),
        (lambda s: s["properties"].pop("d"), "#/d", "property removed"),
        (
            lambda s: s["properties"].update(e={"type": "string"}),
            "#/e",
            "without a default",
        ),
        (
            lambda s: s["required"].remove("a"),
            "#/a",
            "no longer always present",
        ),
        (lambda s: s["properties"]["a"].update(type="integer"), "#/a", "type was"),
        (
            lambda s: s["properties"]["a"].update(maxLength=5),
            "#/a",
            "maxLength was absent",
        ),
        (
            lambda s: s["$defs"]["Mode"]["enum"].append("off"),
            "#/$defs/Mode",
            "enum was",
        ),
        (lambda s: s["$defs"]["Mode"]["enum"].pop(), "#/$defs/Mode", "enum was"),
        (
            lambda s: s["properties"]["b"].update({"$ref": "#/$defs/Unused"}),
            "#/b",
            "$ref",
        ),
        (
            lambda s: s["properties"]["c"]["anyOf"][0].update(minimum=1),
            "#/c",
            "member removed",
        ),
        (
            lambda s: s["properties"]["c"]["anyOf"].append({"type": "string"}),
            "#/c",
            "member added",
        ),
        (
            lambda s: s["properties"]["d"]["items"].update({"$ref": "#/$defs/Unused"}),
            "#/d/items",
            "$ref",
        ),
        (
            lambda s: s.update(additionalProperties=True),
            "#/additionalProperties",
            "became",
        ),
        (lambda s: s["$defs"].pop("Unused"), "#/$defs/Unused", "definition removed"),
    ],
)
def test_changes_that_can_break_a_consumer_are_breaking(edit, path, text):
    result = changed(edit)

    assert result, "expected a change"
    assert all(c.breaking for c in result), result
    assert any(c.path == path and text in c.message for c in result), result


def test_a_nested_change_in_a_definition_is_reported_once_at_the_definition():
    result = changed(lambda s: s["$defs"]["Mode"]["enum"].append("off"))

    assert [c.path for c in result] == ["#/$defs/Mode"]


@pytest.mark.parametrize(
    ("edit", "path"),
    [
        (
            lambda s: s["properties"].update(e={"type": "string", "default": ""}),
            "#/e",
        ),
        (lambda s: s["required"].append("c"), "#/c"),
        (lambda s: s["$defs"].update(Fresh={"type": "string"}), "#/$defs/Fresh"),
    ],
)
def test_a_new_property_with_a_default_or_a_stricter_required_is_additive(edit, path):
    result = changed(edit)

    assert [(c.path, c.breaking) for c in result] == [(path, False)]


def test_a_changed_field_reports_where_and_how():
    result = changed(lambda s: s["properties"]["a"].update(type="integer"))

    assert result == [Change("#/a", True, 'type was "string", now "integer"')]
    assert str(result[0]).startswith("breaking: #/a: ")


def test_deleting_a_field_of_the_real_schema_is_breaking():
    current = run_schema()
    older = copy.deepcopy(current)
    del older["$defs"]["Finding"]["properties"]["code"]["description"]
    smaller = copy.deepcopy(current)
    del smaller["$defs"]["Finding"]["properties"]["code"]

    result = compare_schemas(current, smaller)

    assert [(c.path, c.breaking) for c in result] == [("#/$defs/Finding/code", True)]
    assert compare_schemas(current, older) == []


# --- announcing a change ----------------------------------------------------------

BREAKING = [Change("#/a", True, "property removed")]
ADDITIVE = [Change("#/e", False, "property added")]


@pytest.mark.parametrize(
    ("old", "new", "changes"),
    [
        ("0.1.0", "0.1.0", []),
        ("0.1.0", "0.1.1", []),
        ("0.1.0", "0.2.0", BREAKING),
        ("0.1.0", "1.0.0", BREAKING),
        ("1.4.2", "2.0.0", BREAKING),
        ("0.1.0", "0.2.0", ADDITIVE),
        ("1.4.2", "1.5.0", ADDITIVE),
        ("1.4.2", "2.0.0", ADDITIVE),
        ("0.1.0", "0.2.0", BREAKING + ADDITIVE),
    ],
)
def test_a_version_that_moves_far_enough_announces_the_change(old, new, changes):
    assert unannounced(old, new, changes) == []


@pytest.mark.parametrize(
    ("old", "new", "changes", "message"),
    [
        ("0.1.0", "0.1.0", BREAKING, "breaking change.*need a minor"),
        ("0.1.0", "0.1.1", BREAKING, "need a minor"),
        ("1.4.2", "1.5.0", BREAKING, "need a major"),
        ("1.4.2", "1.4.3", ADDITIVE, "additive change.*need a minor"),
        ("0.1.0", "0.1.0", ADDITIVE, "need a minor"),
        ("0.3.0", "0.2.0", [], "backwards"),
        ("1.0.0", "0.9.0", ADDITIVE, "backwards"),
    ],
)
def test_a_change_the_version_does_not_announce_is_reported(old, new, changes, message):
    problems = unannounced(old, new, changes)

    assert len(problems) == 1
    assert re.search(message, problems[0])


def test_a_malformed_version_is_refused():
    with pytest.raises(ValueError, match="major.minor.patch"):
        unannounced("latest", "0.1.0", [])
