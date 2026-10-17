# tests/test_config/test_loader.py
"""Layered configuration: merge order, provenance and precise errors."""

import copy
from pathlib import Path

import pytest

from dsbf.config import (
    ConfigError,
    find_project_file,
    load_config,
)
from dsbf.model import CheckParams, Contract

pytestmark = pytest.mark.unit


class OutlierParams(CheckParams):
    method: str = "iqr"
    threshold: float = 3.0


class Engine(Contract):
    max_workers: int = 1
    timeout_s: float | None = None
    packs: tuple[str, ...] = ("core",)
    verbose: bool = False


class Settings(Contract):
    engine: Engine = Engine()
    outliers: OutlierParams = OutlierParams()
    labels: dict[str, str] = {}


def load(**layers):
    layers.setdefault("env", {})
    return load_config(Settings, **layers)


def write_project_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "dsbf.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_with_no_layers_the_model_defaults_apply():
    assert load() == Settings()


def test_each_layer_overrides_the_one_before_it(tmp_path):
    project = write_project_file(tmp_path, "engine:\n  max_workers: 2\n")

    result = load(
        defaults={"engine": {"max_workers": 1}},
        project_file=project,
        env={"DSBF__ENGINE__MAX_WORKERS": "3"},
        cli={"engine": {"max_workers": 4}},
        overrides={"engine": {"max_workers": 5}},
    )
    assert result.engine.max_workers == 5

    layers = ["defaults", "project_file", "env", "cli", "overrides"]
    values = {
        "defaults": {"engine": {"max_workers": 1}},
        "project_file": project,
        "env": {"DSBF__ENGINE__MAX_WORKERS": "3"},
        "cli": {"engine": {"max_workers": 4}},
        "overrides": {"engine": {"max_workers": 5}},
    }
    expected = {"defaults": 1, "project_file": 2, "env": 3, "cli": 4, "overrides": 5}
    for dropped in range(1, 6):
        kept = layers[: 5 - dropped + 1]
        # The highest remaining layer wins.
        assert (
            load(**{name: values[name] for name in kept}).engine.max_workers
            == expected[kept[-1]]
        )


def test_a_partial_layer_keeps_the_sibling_values_of_lower_layers(tmp_path):
    project = write_project_file(
        tmp_path, "engine:\n  max_workers: 2\n  verbose: true\n"
    )

    result = load(project_file=project, overrides={"engine": {"timeout_s": 9}})

    assert result.engine == Engine(max_workers=2, verbose=True, timeout_s=9)
    assert result.outliers == OutlierParams()


def test_lists_and_none_replace_instead_of_merging(tmp_path):
    project = write_project_file(
        tmp_path, "engine:\n  packs: [core, timeseries]\n  timeout_s: 5\n"
    )

    result = load(
        project_file=project, cli={"engine": {"packs": ["ml"], "timeout_s": None}}
    )

    assert result.engine.packs == ("ml",)
    assert result.engine.timeout_s is None


def test_inputs_are_not_modified():
    defaults = {"engine": {"packs": ["a"]}, "labels": {"x": "y"}}
    overrides = {"engine": {"max_workers": 2}}
    before = copy.deepcopy((defaults, overrides))

    result = load(defaults=defaults, overrides=overrides)
    defaults["engine"]["packs"].append("mutated")

    assert (defaults["labels"], overrides) == (before[0]["labels"], before[1])
    assert result.engine.packs == ("a",)


def test_the_project_file_is_found_in_the_given_directory_only(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    path = write_project_file(tmp_path, "")

    assert find_project_file(tmp_path) == path
    assert find_project_file(nested) is None


def test_the_environment_defaults_to_the_process_environment(monkeypatch):
    monkeypatch.setenv("DSBF__ENGINE__MAX_WORKERS", "7")

    assert load_config(Settings).engine.max_workers == 7
    assert load_config(Settings, env={}).engine.max_workers == 1


# --- the environment layer ----------------------------------------------------


def test_environment_names_map_to_nested_lowercase_keys():
    env = {
        "DSBF__ENGINE__MAX_WORKERS": "3",
        "DSBF__ENGINE__VERBOSE": "true",
        "DSBF__OUTLIERS__THRESHOLD": "2.5",
        "DSBF_HOME": "/ignored/single-underscore",
        "PATH": "/ignored",
    }

    result = load(env=env)

    assert result.engine.max_workers == 3
    assert result.engine.verbose is True
    assert result.outliers.threshold == 2.5


def test_environment_values_can_carry_json_lists_and_null():
    env = {
        "DSBF__ENGINE__PACKS": '["core", "text"]',
        "DSBF__ENGINE__TIMEOUT_S": "null",
        "DSBF__LABELS": '{"a": "b"}',
    }

    result = load(defaults={"engine": {"timeout_s": 4}}, env=env)

    assert result.engine.packs == ("core", "text")
    assert result.engine.timeout_s is None
    assert result.labels == {"a": "b"}


@pytest.mark.parametrize(
    "name", ["DSBF__", "DSBF__ENGINE__", "DSBF__A____B", "DSBF____X"]
)
def test_a_malformed_environment_name_is_reported(name):
    with pytest.raises(ConfigError) as caught:
        load(env={name: "1"})

    [issue] = caught.value.issues
    assert issue.path == name
    assert issue.source == f"env {name}"


def test_an_environment_value_and_sub_key_on_one_path_conflict():
    with pytest.raises(ConfigError) as caught:
        load(env={"DSBF__ENGINE": "x", "DSBF__ENGINE__MAX_WORKERS": "2"})

    assert "engine: is set to a value and also has sub-keys" in str(caught.value)


# --- errors -------------------------------------------------------------------


def test_an_unknown_key_names_the_key_the_layer_and_the_allowed_keys(tmp_path):
    project = write_project_file(tmp_path, "engine:\n  max_wrokers: 2\n")

    with pytest.raises(ConfigError) as caught:
        load(project_file=project)

    [issue] = caught.value.issues
    assert issue.path == "engine.max_wrokers"
    assert issue.message == "unknown key"
    assert issue.source == f"file {project}"
    assert issue.allowed == ("max_workers", "timeout_s", "packs", "verbose")
    assert str(issue) == (
        f"engine.max_wrokers: unknown key (from file {project}); "
        "allowed: max_workers, timeout_s, packs, verbose"
    )


def test_an_unknown_top_level_key_lists_the_sections():
    with pytest.raises(ConfigError) as caught:
        load(cli={"outlier": {}})

    [issue] = caught.value.issues
    assert (issue.path, issue.source) == ("outlier", "cli")
    assert issue.allowed == ("engine", "outliers", "labels")


def test_an_unknown_environment_key_names_the_variable():
    with pytest.raises(ConfigError) as caught:
        load(env={"DSBF__ENGINE__WORKERS": "2"})

    [issue] = caught.value.issues
    assert (issue.path, issue.source) == ("engine.workers", "env DSBF__ENGINE__WORKERS")


def test_a_type_error_shows_the_path_value_and_layer():
    with pytest.raises(ConfigError) as caught:
        load(
            env={"DSBF__ENGINE__MAX_WORKERS": "many"},
            overrides={"outliers": {"threshold": "high"}},
        )

    by_path = {issue.path: issue for issue in caught.value.issues}
    assert set(by_path) == {"engine.max_workers", "outliers.threshold"}
    assert by_path["engine.max_workers"].source == "env DSBF__ENGINE__MAX_WORKERS"
    assert "(got 'many')" in by_path["engine.max_workers"].message
    assert by_path["outliers.threshold"].source == "overrides"


def test_the_layer_blamed_is_the_one_that_supplied_the_winning_value(tmp_path):
    project = write_project_file(tmp_path, "engine:\n  max_workers: lots\n")

    with pytest.raises(ConfigError) as caught:
        load(project_file=project, defaults={"engine": {"max_workers": 2}})
    assert caught.value.issues[0].source == f"file {project}"

    # A later valid layer repairs the value, so there is nothing to report.
    assert (
        load(
            project_file=project, cli={"engine": {"max_workers": 2}}
        ).engine.max_workers
        == 2
    )


def test_an_error_inside_a_sequence_gets_an_index():
    class Plan(Contract):
        engines: tuple[Engine, ...] = ()

    with pytest.raises(ConfigError) as caught:
        load_config(
            Plan,
            env={},
            overrides={"engines": [{"max_workers": 1}, {"max_workers": "x"}]},
        )

    [issue] = caught.value.issues
    assert issue.path == "engines[1].max_workers"
    assert issue.source == "overrides"


def test_allowed_keys_are_found_through_optional_and_collection_fields():
    class Deep(Contract):
        maybe: Engine | None = None
        many: list[Engine] = []
        named: dict[str, Engine] = {}

    for layer, path in [
        ({"maybe": {"bogus": 1}}, "maybe.bogus"),
        ({"many": [{"bogus": 1}]}, "many[0].bogus"),
        ({"named": {"x": {"bogus": 1}}}, "named.x.bogus"),
    ]:
        with pytest.raises(ConfigError) as caught:
            load_config(Deep, env={}, overrides=layer)
        [issue] = caught.value.issues
        assert issue.path == path
        assert issue.allowed == ("max_workers", "timeout_s", "packs", "verbose")


def test_every_problem_is_reported_together():
    with pytest.raises(ConfigError) as caught:
        load(
            overrides={"engine": {"bogus": 1, "max_workers": "x"}},
            env={"DSBF__OUTLIERS__THRESHOLD": "y", "DSBF__": "1"},
        )

    # The malformed name stops the run before validation; fix it and the rest appear.
    assert [issue.path for issue in caught.value.issues] == ["DSBF__"]
    with pytest.raises(ConfigError) as caught:
        load(
            overrides={"engine": {"bogus": 1, "max_workers": "x"}},
            env={"DSBF__OUTLIERS__THRESHOLD": "y"},
        )
    assert sorted(issue.path for issue in caught.value.issues) == [
        "engine.bogus",
        "engine.max_workers",
        "outliers.threshold",
    ]
    assert str(caught.value).startswith("Invalid configuration (3 issues):")


# --- the project file ---------------------------------------------------------


def test_an_empty_project_file_is_an_empty_layer(tmp_path):
    assert load(project_file=write_project_file(tmp_path, "# nothing\n")) == Settings()


def test_a_missing_project_file_is_an_error(tmp_path):
    missing = tmp_path / "nope.yaml"

    with pytest.raises(ConfigError) as caught:
        load(project_file=missing)

    [issue] = caught.value.issues
    assert issue.source == f"file {missing}"
    assert "does not exist" in issue.message


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("engine: [unclosed\n", "cannot read"),
        ("- a\n- b\n", "must be a mapping"),
        ("just a string\n", "must be a mapping"),
    ],
)
def test_an_unusable_project_file_is_an_error(tmp_path, text, fragment):
    project = write_project_file(tmp_path, text)

    with pytest.raises(ConfigError) as caught:
        load(project_file=project)

    [issue] = caught.value.issues
    assert fragment in issue.message
    assert issue.source == f"file {project}"
    assert "\n" not in issue.message


def test_a_project_file_that_is_not_utf8_is_an_error(tmp_path):
    project = tmp_path / "dsbf.yaml"
    project.write_bytes(b"\xff\xfe\x00bad")

    with pytest.raises(ConfigError):
        load(project_file=project)
