"""A policy loads from a file, a mapping or a Policy, and a bad one names its rule."""

import pytest

from dsbf.config import ConfigError
from dsbf.model import Policy
from dsbf.policy import PolicyError, load_policy
from tests.test_model.test_policy import EXAMPLE

pytestmark = pytest.mark.unit


def test_a_file_loads_the_documented_example(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")
    policy = load_policy(path)
    assert [rule.name for rule in policy.rules] == [
        "block-on-leakage",
        "sparse-columns",
        "check-errors-are-notes",
        "tell-the-team",
    ]
    assert load_policy(str(path)) == policy


def test_a_mapping_and_a_policy_load(tmp_path):
    mapping = {"rules": [{"name": "a", "when": {"code": "x"}, "then": "fail"}]}
    policy = load_policy(mapping)
    assert policy.rules[0].then[0].kind == "fail"
    assert load_policy(policy) is policy


def test_an_empty_file_is_a_policy_without_rules(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("", encoding="utf-8")
    assert load_policy(path) == Policy()


def test_each_problem_is_listed_under_its_rule_name():
    mapping = {
        "rules": [
            {"name": "fine", "when": {"code": "x"}},
            {"name": "sparse", "when": {"metric": {"null_share": ">=high"}}},
            {"name": "typo", "when": {"colum": "a"}, "then": "shout"},
        ]
    }
    with pytest.raises(PolicyError) as caught:
        load_policy(mapping)
    paths = [issue.path for issue in caught.value.issues]
    assert "rules[sparse].when.metric.null_share" in paths
    assert "rules[typo].when.colum" in paths
    assert not any("fine" in path for path in paths)
    assert str(caught.value).startswith("Invalid policy (")
    assert "rules[sparse]" in str(caught.value)


def test_a_rule_without_a_usable_name_is_located_by_position():
    with pytest.raises(PolicyError) as caught:
        load_policy({"rules": [{"when": {"code": "x"}}]})
    assert [issue.path for issue in caught.value.issues] == ["rules[0].name"]


def test_a_policy_error_is_a_config_error():
    with pytest.raises(ConfigError):
        load_policy({"version": 2})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (": : :", "cannot read the policy file"),
        ("- a\n- b\n", "a policy is a mapping"),
    ],
)
def test_an_unreadable_file_is_one_issue(tmp_path, text, message):
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError, match=message) as caught:
        load_policy(path)
    assert caught.value.issues[0].source == f"file {path}"


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(PolicyError, match="does not exist"):
        load_policy(tmp_path / "nope.yaml")
