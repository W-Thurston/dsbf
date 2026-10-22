"""The settings of a profiling run: what ``dsbf.profile()`` accepts and refuses."""

import polars as pl
import pytest
from pydantic import ValidationError

import dsbf
from dsbf.config import (
    DEFAULT_PACKS,
    Backend,
    ConfigError,
    LimitsConfig,
    ProfileConfig,
    SampleConfig,
)
from dsbf.engine import Depth
from dsbf.library import _default_workers, parse_checks
from dsbf.profiling import profile_dataset
from tests.test_engine.toys import declare, make_dataset, plan_for

pytestmark = pytest.mark.unit


class TestChecksSyntax:
    def test_a_plus_or_a_bare_id_adds_and_a_minus_removes(self):
        assert parse_checks(["+a", "b", "-c"]) == (("a", "b"), ("c",))

    def test_one_string_is_one_check(self):
        assert parse_checks("-a") == ((), ("a",))

    def test_spaces_around_the_sign_are_ignored(self):
        assert parse_checks([" + a ", "- b"]) == (("a",), ("b",))

    def test_nothing_is_nothing(self):
        assert parse_checks(()) == ((), ())


class TestProfileConfig:
    def test_the_defaults(self):
        config = ProfileConfig()

        assert config.packs == DEFAULT_PACKS == ("core", "legacy")
        assert config.depth is Depth.STANDARD
        assert config.backend is Backend.PANDAS
        assert config.sample is None
        assert config.workers is None
        assert config.limits == LimitsConfig()

    def test_an_unknown_setting_is_refused(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ProfileConfig.model_validate({"dept": "deep"})

    @pytest.mark.parametrize("workers", [0, -1])
    def test_workers_is_at_least_one(self, workers):
        with pytest.raises(ValidationError):
            ProfileConfig(workers=workers)

    @pytest.mark.parametrize("seconds", [0, -1.0, float("inf"), float("nan")])
    def test_a_limit_is_a_positive_finite_number(self, seconds):
        with pytest.raises(ValidationError):
            LimitsConfig(check_timeout_seconds=seconds)

    def test_a_check_cannot_be_both_included_and_excluded(self):
        with pytest.raises(ValidationError, match="both included and excluded: a"):
            ProfileConfig(include=("a", "b"), exclude=("a",))

    def test_a_sample_has_at_least_one_row(self):
        with pytest.raises(ValidationError):
            SampleConfig.model_validate({"rows": 0})


class TestRefusedSettings:
    @pytest.fixture
    def frame(self):
        return pl.DataFrame({"age": [1.0, 2.0, 3.0], "group": ["a", "b", "a"]})

    def test_every_problem_is_listed_at_once(self, frame):
        with pytest.raises(ConfigError) as caught:
            dsbf.profile(
                frame,
                target="survived",
                time_index="when",
                tasks={"detect_outlier": {"flag_threshold": 0.1}},
            )

        paths = {issue.path for issue in caught.value.issues}
        assert paths == {"target", "time_index", "tasks.detect_outlier"}
        message = str(caught.value)
        assert "'survived' is not a column of the data" in message
        assert "allowed: age, group" in message
        assert "did you mean 'detect_outliers'?" in message

    def test_an_unknown_depth_names_the_setting(self, frame):
        with pytest.raises(ConfigError, match="depth"):
            dsbf.profile(frame, depth="exhaustive")

    def test_a_bad_value_names_where_it_came_from(self, frame):
        with pytest.raises(ConfigError, match=r"workers.*\(from overrides\)"):
            dsbf.profile(frame, workers=0)

    def test_a_settings_file_is_read_only_when_named(
        self, frame, tmp_path, monkeypatch
    ):
        (tmp_path / "dsbf.yaml").write_text("workers: 0\n")
        monkeypatch.chdir(tmp_path)

        # Not named, so not read: only the bad target is reported, not the workers.
        with pytest.raises(ConfigError) as unread:
            dsbf.profile(frame, target="nope")
        with pytest.raises(ConfigError) as named:
            dsbf.profile(frame, config=tmp_path / "dsbf.yaml")

        assert {i.path for i in unread.value.issues} == {"target"}
        assert {i.path for i in named.value.issues} == {"workers"}

    def test_an_argument_overrides_the_settings_file(self, frame, tmp_path):
        path = tmp_path / "settings.yaml"
        path.write_text("depth: nonsense\nworkers: 0\n")

        with pytest.raises(ConfigError) as caught:
            dsbf.profile(frame, config=path, workers=2)

        assert {i.path for i in caught.value.issues} == {"depth"}


class TestDefaultWorkers:
    """A plan with a legacy task runs one check at a time; any other, per core."""

    @staticmethod
    def plan(*checks):
        declared = [
            declare(f"c{i}", packs=packs, kind=kind)
            for i, (packs, kind) in enumerate(checks)
        ]
        return _default_workers(plan_for(declared, profile_dataset(make_dataset())))

    def test_a_plan_of_legacy_tasks_runs_one_at_a_time(self):
        assert self.plan((("legacy",), "analysis"), (("legacy",), "analysis")) == 1

    def test_a_task_a_core_check_replaces_is_still_a_legacy_task(self):
        legacy = (("legacy",), "analysis")
        replaced = (("legacy_superseded",), "analysis")
        assert self.plan(legacy, replaced) == 1

    def test_core_checks_beside_legacy_tasks_do_not_change_that(self):
        legacy = (("legacy",), "analysis")
        assert self.plan(legacy, (("core",), "rule")) == 1
        assert self.plan(legacy, (("core",), "analysis")) == 1

    def test_a_plan_with_no_legacy_task_uses_the_executors_default(self):
        assert self.plan((("core",), "rule")) is None
