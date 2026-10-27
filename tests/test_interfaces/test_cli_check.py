"""``dsbf check`` and the settings, policy and verbosity options of ``dsbf profile``."""

import json
import logging
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsbf.interfaces import cli
from dsbf.interfaces.cli import app
from dsbf.utils.logging_utils import STAGE_LEVEL

runner = CliRunner()

# CI renders usage errors with colour; the local terminal does not.
ANSI = re.compile(r"\x1b\[[0-9;]*m")

DOCS = Path(__file__).parents[2] / "docs"
REFERENCE = DOCS / "CLI_REFERENCE.md"
CHECK_REFERENCE = DOCS / "CHECK_REFERENCE.md"

DATA = """\
id,amount,group,note
1,10.5,a,x
2,11.0,b,y
3,,a,z
4,9.5,b,x
5,10.1,a,y
6,10.9,b,z
"""

SLEEPY = '''\
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

FAILING = """\
rules:
  - name: any-missing-values
    when: {code: has_nulls}
    then: fail
  - name: never-matches
    when: {code: no_such_code}
"""
PASSING = """\
rules:
  - name: never-matches
    when: {code: no_such_code}
    then: fail
"""


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    path = tmp_path / "small.csv"
    path.write_text(DATA, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def dsbf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("DSBF_HOME", str(home))
    return home


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def row(text: str, check_id: str) -> str:
    return next(line for line in text.splitlines() if check_id in line)


def run_json(out: Path) -> dict:
    return json.loads((out / "run.json").read_text(encoding="utf-8"))


@pytest.mark.integration
class TestExitCodes:
    def test_a_policy_that_fails_the_run_exits_1_from_check(self, csv, tmp_path):
        policy = write(tmp_path, "policy.yaml", FAILING)
        # Any finding at all: pick the code from a real run rather than guessing.
        first = runner.invoke(app, ["profile", str(csv), "-o", str(tmp_path / "a")])
        code = next(
            f["code"]
            for r in run_json(tmp_path / "a")["results"].values()
            for f in r["findings"]
        )
        policy.write_text(FAILING.replace("has_nulls", code), encoding="utf-8")

        result = runner.invoke(
            app,
            ["check", str(csv), "-o", str(tmp_path / "o"), "--policy", str(policy)],
        )

        assert first.exit_code == 0
        assert result.exit_code == 1, result.output
        assert "policy: any-missing-values matched" in result.stdout
        assert "policy: FAILED" in result.stdout
        assert run_json(tmp_path / "o")["policy"]["failed"] is True

    def test_profile_reports_the_failure_but_exits_0(self, csv, tmp_path):
        first = runner.invoke(app, ["profile", str(csv), "-o", str(tmp_path / "a")])
        code = next(
            f["code"]
            for r in run_json(tmp_path / "a")["results"].values()
            for f in r["findings"]
        )
        policy = write(tmp_path, "policy.yaml", FAILING.replace("has_nulls", code))
        assert first.exit_code == 0

        result = runner.invoke(
            app,
            ["profile", str(csv), "-o", str(tmp_path / "o"), "--policy", str(policy)],
        )

        assert result.exit_code == 0, result.output
        assert "policy: FAILED" in result.stdout

    def test_a_policy_that_passes_exits_0(self, csv, tmp_path):
        policy = write(tmp_path, "policy.yaml", PASSING)

        result = runner.invoke(
            app,
            ["check", str(csv), "-o", str(tmp_path / "o"), "--policy", str(policy)],
        )

        assert result.exit_code == 0, result.output
        assert "policy: passed" in result.stdout

    def test_check_without_a_policy_exits_0_and_prints_no_verdict(self, csv, tmp_path):
        result = runner.invoke(app, ["check", str(csv), "-o", str(tmp_path / "o")])

        assert result.exit_code == 0, result.output
        assert "policy:" not in result.stdout

    @pytest.mark.parametrize(
        "text", ["rules:\n  - name: x\n    when: {colum: a}\n", "[not a policy"]
    )
    def test_an_invalid_policy_exits_2_and_writes_nothing(self, csv, tmp_path, text):
        policy = write(tmp_path, "policy.yaml", text)

        result = runner.invoke(
            app,
            ["check", str(csv), "-o", str(tmp_path / "o"), "--policy", str(policy)],
        )

        assert result.exit_code == 2
        assert "error:" in result.output
        assert not (tmp_path / "o").exists()

    def test_an_unreadable_previous_run_exits_2(self, csv, tmp_path):
        policy = write(tmp_path, "policy.yaml", PASSING)
        previous = write(tmp_path, "run.json", "{}")

        result = runner.invoke(
            app,
            [
                "check",
                str(csv),
                "-o",
                str(tmp_path / "o"),
                "--policy",
                str(policy),
                "--previous",
                str(previous),
            ],
        )

        assert result.exit_code == 2
        assert "error:" in result.output


@pytest.mark.integration
class TestSettings:
    def test_the_options_override_the_file_and_set_overrides_both(self, csv, tmp_path):
        config = write(
            tmp_path,
            "dsbf.yaml",
            "depth: quick\ntarget: group\nlimits:\n  check_timeout_seconds: 30\n",
        )
        out = tmp_path / "o"

        result = runner.invoke(
            app,
            [
                "profile",
                str(csv),
                "-o",
                str(out),
                "--config",
                str(config),
                "--depth",
                "deep",
                "--target",
                "note",
                "--set",
                "target=amount",
            ],
        )

        assert result.exit_code == 0, result.output
        results = run_json(out)["results"]
        assert "bimodality" in results  # a deep check, so --depth beat the file
        assert results["class_imbalance"]["params"]["target"] == "amount"

    def test_the_file_applies_without_options(self, csv, tmp_path):
        config = write(tmp_path, "dsbf.yaml", "target: group\n")
        out = tmp_path / "o"

        result = runner.invoke(
            app, ["profile", str(csv), "-o", str(out), "--config", str(config)]
        )

        assert result.exit_code == 0, result.output
        assert (
            run_json(out)["results"]["class_imbalance"]["params"]["target"] == "group"
        )

    def test_set_reaches_a_nested_setting(self, csv, tmp_path):
        out = tmp_path / "o"

        result = runner.invoke(
            app,
            [
                "profile",
                str(csv),
                "-o",
                str(out),
                "--set",
                "params.null_rate.notice_share=0.2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert run_json(out)["results"]["null_rate"]["params"]["notice_share"] == 0.2

    def test_an_environment_variable_sits_below_the_options(
        self, csv, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("DSBF__TARGET", "note")
        out = tmp_path / "o"

        result = runner.invoke(
            app, ["profile", str(csv), "-o", str(out), "--target", "group"]
        )

        assert result.exit_code == 0, result.output
        assert (
            run_json(out)["results"]["class_imbalance"]["params"]["target"] == "group"
        )

    def test_every_bad_setting_is_listed_and_exits_2(self, csv, tmp_path):
        result = runner.invoke(
            app,
            [
                "profile",
                str(csv),
                "-o",
                str(tmp_path / "o"),
                "--set",
                "workers=0",
                "--set",
                "wokers=1",
            ],
        )

        assert result.exit_code == 2
        assert "workers" in result.output
        assert "wokers: unknown key" in result.output
        assert "cli" in result.output  # the layer that supplied it

    def test_a_missing_config_file_exits_2(self, csv, tmp_path):
        result = runner.invoke(
            app, ["profile", str(csv), "--config", str(tmp_path / "nope.yaml")]
        )

        assert result.exit_code == 2
        assert "does not exist" in result.output

    @pytest.mark.parametrize("assignment", ["workers", "=3", "a..b=1"])
    def test_a_malformed_set_is_a_usage_error(self, csv, assignment):
        result = runner.invoke(app, ["profile", str(csv), "--set", assignment])

        assert result.exit_code == 2
        assert "--set" in ANSI.sub("", result.output)


@pytest.mark.unit
class TestVerbosity:
    def test_an_unknown_verbosity_is_a_usage_error(self, csv):
        result = runner.invoke(app, ["--verbosity", "loud", "profile", str(csv)])

        assert result.exit_code == 2
        assert "loud" in result.output

    @pytest.mark.parametrize(
        ("verbosity", "shown", "hidden"),
        [
            ("debug", ["a stage", "a detail"], []),
            ("stage", ["a stage"], ["a detail"]),
            ("quiet", [], ["a stage", "a detail"]),
        ],
    )
    def test_every_command_prints_the_records_it_asks_for(
        self, csv, tmp_path, monkeypatch, verbosity, shown, hidden
    ):
        real = cli.profile_with

        def noisy(*args, **kwargs):
            logging.getLogger("dsbf.test").log(STAGE_LEVEL, "a stage")
            logging.getLogger("dsbf.test").debug("a detail")
            return real(*args, **kwargs)

        monkeypatch.setattr(cli, "profile_with", noisy)

        result = runner.invoke(
            app,
            ["-v", verbosity, "profile", str(csv), "-o", str(tmp_path / "o")],
        )

        assert result.exit_code == 0, result.output
        for text in shown:
            assert text in result.output
        for text in hidden:
            assert text not in result.output


@pytest.mark.unit
class TestPlanShowsWhatNeedsASetting:
    def test_a_check_that_needs_a_target_is_not_applicable_until_one_is_given(
        self, csv
    ):
        without = runner.invoke(app, ["plan", str(csv)])
        with_target = runner.invoke(
            app, ["plan", str(csv), "--param", "class_imbalance.target=group"]
        )

        line = row(without.stdout, "class_imbalance")
        assert "not applicable" in line
        assert "`target` parameter is not set" in line
        assert "will run" in row(with_target.stdout, "class_imbalance")


@pytest.mark.unit
class TestReference:
    # The reference is generated on Linux, where CI also runs `docs export --check`;
    # its text carries platform-specific defaults.
    @pytest.mark.skipif(sys.platform == "win32", reason="generated on Linux")
    def test_the_checked_in_reference_is_current(self):
        result = runner.invoke(
            app,
            [
                "docs",
                "export",
                "--check",
                "--out",
                str(REFERENCE),
                "--checks-out",
                str(CHECK_REFERENCE),
            ],
        )

        assert result.exit_code == 0, result.output

    def test_a_stale_reference_fails_the_check(self, tmp_path):
        stale = tmp_path / "ref.md"
        stale.write_text("old\n", encoding="utf-8")

        args = ["docs", "export", "--check", "--checks-out", str(CHECK_REFERENCE)]

        result = runner.invoke(app, [*args, "--out", str(stale)])

        assert result.exit_code == 1
        assert "out of date" in result.output
        assert stale.read_text(encoding="utf-8") == "old\n"

    def test_a_stale_check_reference_fails_the_check(self, tmp_path):
        stale = tmp_path / "checks.md"
        stale.write_text("old\n", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "docs",
                "export",
                "--check",
                "--out",
                str(REFERENCE),
                "--checks-out",
                str(stale),
            ],
        )

        assert result.exit_code == 1
        assert str(stale) in result.output

    def test_export_writes_every_command_and_setting(self, tmp_path):
        out = tmp_path / "ref.md"

        result = runner.invoke(
            app,
            [
                "docs",
                "export",
                "--out",
                str(out),
                "--checks-out",
                str(tmp_path / "c.md"),
            ],
        )

        text = out.read_text(encoding="utf-8")
        assert result.exit_code == 0, result.output
        for heading in (
            "dsbf profile",
            "dsbf check",
            "dsbf plan",
            "dsbf schema export",
        ):
            assert f"`{heading}`" in text
        for setting in ("`limits.run_budget_seconds`", "`sample.rows`", "`params`"):
            assert setting in text


@pytest.mark.integration
class TestExecutionOptions:
    def test_the_options_reach_the_settings(self, csv, tmp_path):
        out = tmp_path / "out"

        result = runner.invoke(
            app,
            ["profile", str(csv), "--out", str(out), "--workers", "1"]
            + ["--check-timeout", "30", "--run-budget", "60", "--cache"],
        )

        assert result.exit_code == 0, result.output
        assert any((tmp_path / "home" / "cache" / "results").rglob("*.json"))

    def test_a_second_cached_run_reuses_the_first(self, csv, tmp_path):
        args = ["check", str(csv), "--cache", "--depth", "quick"]

        runner.invoke(app, [*args, "--out", str(tmp_path / "one")])
        result = runner.invoke(app, [*args, "--out", str(tmp_path / "two")])

        assert result.exit_code == 0, result.output
        timings = [r["timing"] for r in run_json(tmp_path / "two")["results"].values()]
        assert any(t.get("cached") for t in timings)

    def test_no_cache_beats_the_settings_file(self, csv, tmp_path):
        config = write(tmp_path, "dsbf.yaml", "cache: {enabled: true}\n")

        result = runner.invoke(
            app,
            ["check", str(csv), "--config", str(config), "--no-cache"]
            + ["--out", str(tmp_path / "out")],
        )

        assert result.exit_code == 0, result.output
        assert not (tmp_path / "home" / "cache").exists()

    def test_a_per_check_timeout_can_be_set_with_set(self, csv, tmp_path):
        plugin = tmp_path / "plugin"
        plugin.mkdir()
        (plugin / "sleepy.py").write_text(SLEEPY, encoding="utf-8")
        out = tmp_path / "out"

        result = runner.invoke(
            app,
            ["profile", str(csv), "--out", str(out), "--include", "sleepy"]
            + ["--plugin", str(plugin)]
            + ["--set", "check_settings.sleepy.timeout_seconds=0.3"],
        )

        assert result.exit_code == 0, result.output
        statuses = {k: v["status"] for k, v in run_json(out)["results"].items()}
        assert statuses["sleepy"] == "timeout"
        assert statuses["null_rate"] == "ok"

    def test_an_unknown_check_in_the_settings_exits_2(self, csv, tmp_path):
        result = runner.invoke(
            app,
            ["profile", str(csv), "--out", str(tmp_path / "out")]
            + ["--set", "check_settings.nope.runs_in=process"],
        )

        assert result.exit_code == 2
        assert "'nope' is not a check" in result.output

    @pytest.mark.parametrize("option", ["--workers", "--check-timeout"])
    def test_a_zero_is_a_usage_error(self, csv, option):
        result = runner.invoke(app, ["profile", str(csv), option, "0"])

        assert result.exit_code == 2


@pytest.mark.integration
class TestCacheClear:
    def test_it_removes_what_a_cached_run_kept(self, csv, tmp_path):
        out = str(tmp_path / "o")
        runner.invoke(app, ["profile", str(csv), "--cache", "--out", out])
        results = tmp_path / "home" / "cache" / "results"
        assert any(results.rglob("*.json"))

        result = runner.invoke(app, ["cache", "clear"])

        assert result.exit_code == 0, result.output
        assert re.search(r"removed \d+ cached results? \([\d,]+ bytes\)", result.output)
        assert not any(results.rglob("*.json"))

    def test_an_empty_cache_clears_to_nothing(self):
        result = runner.invoke(app, ["cache", "clear"])

        assert result.exit_code == 0
        assert "removed 0 cached results" in result.output


@pytest.mark.unit
def test_a_terminal_that_cannot_encode_a_symbol_gets_a_placeholder(monkeypatch):
    import io

    from dsbf.interfaces.cli import tolerate_narrow_encodings

    raw = io.BytesIO()
    narrow = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", narrow)

    tolerate_narrow_encodings()
    print("unavailable ⦸ checks", file=narrow, flush=True)

    assert raw.getvalue().strip() == b"unavailable ? checks"
