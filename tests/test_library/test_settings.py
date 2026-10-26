"""The settings of a profiling run: what ``dsbf.profile()`` accepts and refuses."""

import polars as pl
import pytest
from pydantic import ValidationError

import dsbf
from dsbf.config import (
    DEFAULT_PACKS,
    CacheConfig,
    CheckSettings,
    ConfigError,
    LimitsConfig,
    ProfileConfig,
    SampleConfig,
)
from dsbf.engine import Depth, PlanError
from dsbf.library import parse_checks

pytestmark = pytest.mark.unit

SLEEPY = '''
from dsbf.engine import SemanticTypes, check


@check(
    id="sleepy",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=["validity"],
)
def sleepy(ctx, params):
    """Takes its time."""
    import time

    time.sleep(3)
'''


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

        assert config.packs == DEFAULT_PACKS == ("core",)
        assert config.depth is Depth.STANDARD
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
            )

        paths = {issue.path for issue in caught.value.issues}
        assert paths == {"target", "time_index"}
        message = str(caught.value)
        assert "'survived' is not a column of the data" in message
        assert "allowed: age, group" in message

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


class TestParams:
    def test_a_params_setting_reaches_the_check(self):
        run = dsbf.profile(
            pl.DataFrame({"x": [1.0, None, 3.0, 4.0]}),
            params={"null_rate": {"notice_share": 0.1}},
        )

        assert run.results["null_rate"].params["notice_share"] == 0.1

    def test_params_win_over_the_target_setting(self):
        frame = pl.DataFrame({"a": ["u", "v"] * 20, "b": ["p", "p", "q", "q"] * 10})

        run = dsbf.profile(
            frame, target="a", params={"class_imbalance": {"target": "b"}}
        )

        assert run.results["class_imbalance"].params["target"] == "b"

    def test_a_setting_of_the_removed_engine_is_refused(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ProfileConfig.model_validate({"backend": "polars"})

    def test_an_unknown_check_is_a_planning_error(self):
        with pytest.raises(PlanError, match="unknown check 'nulls'"):
            dsbf.profile(pl.DataFrame({"x": [1, 2]}), params={"nulls": {}})


class TestExecutionSettings:
    @pytest.fixture
    def frame(self):
        return pl.DataFrame({"x": [1.0, None, 3.0, 4.0], "g": ["a", "b", "a", "b"]})

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DSBF_HOME", str(tmp_path / "home"))
        return tmp_path / "home" / "cache" / "results"

    def settings(self, tmp_path, text):
        path = tmp_path / "dsbf.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_the_defaults_leave_the_cache_off_and_capped(self):
        config = ProfileConfig()

        assert config.check_settings == {}
        assert config.cache == CacheConfig(
            enabled=False, max_megabytes=500, max_age_days=90
        )

    @pytest.mark.parametrize("seconds", [0, -1.0, float("inf"), float("nan")])
    def test_a_check_timeout_is_a_positive_finite_number(self, seconds):
        with pytest.raises(ValidationError):
            CheckSettings(timeout_seconds=seconds)

    def test_runs_in_is_thread_or_process(self):
        with pytest.raises(ValidationError):
            CheckSettings.model_validate({"runs_in": "fiber"})

    def test_a_check_timeout_in_the_settings_applies_to_that_check(
        self, frame, tmp_path
    ):
        plugin = tmp_path / "plugin"
        plugin.mkdir()
        (plugin / "sleepy.py").write_text(SLEEPY, encoding="utf-8")
        path = self.settings(
            tmp_path, "check_settings:\n  sleepy: {timeout_seconds: 0.3}\n"
        )

        run = dsbf.profile(
            frame, depth="quick", config=path, plugins=[plugin], checks=["+sleepy"]
        )

        assert run.results["sleepy"].status.value == "timeout"
        assert run.results["sleepy"].notes[0].message == "Did not finish within 0.3s."
        assert run.results["null_rate"].status.value == "ok"

    def test_a_setting_for_a_check_that_does_not_exist_is_refused(
        self, frame, tmp_path
    ):
        path = self.settings(
            tmp_path, "check_settings:\n  null_rat: {timeout_seconds: 5}\n"
        )

        with pytest.raises(ConfigError, match="'null_rat' is not a check") as caught:
            dsbf.profile(frame, config=path)

        assert caught.value.issues[0].path == "check_settings.null_rat"

    def test_the_cache_setting_turns_the_cache_on(self, frame, tmp_path, home):
        path = self.settings(tmp_path, "cache: {enabled: true}\n")

        first = dsbf.profile(frame, depth="quick", config=path)
        second = dsbf.profile(frame, depth="quick", config=path)

        assert any(home.rglob("*.json"))
        assert not any(r.timing.cached for r in first.results.values())
        assert any(r.timing.cached for r in second.results.values())

    def test_the_cache_is_off_without_the_setting(self, frame, home):
        dsbf.profile(frame, depth="quick")

        assert not home.exists()

    def test_the_cache_argument_wins_over_the_setting(self, frame, tmp_path, home):
        path = self.settings(tmp_path, "cache: {enabled: true}\n")

        dsbf.profile(frame, depth="quick", config=path, cache=False)

        assert not home.exists()

    def test_the_size_cap_is_enforced_after_the_run(self, frame, tmp_path, home):
        path = self.settings(tmp_path, "cache: {enabled: true, max_megabytes: 0}\n")

        dsbf.profile(frame, depth="quick", config=path)

        assert list(home.rglob("*.json")) == []
