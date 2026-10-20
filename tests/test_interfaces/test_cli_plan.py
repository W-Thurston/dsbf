"""``dsbf plan`` shows what would run on a dataset without running anything."""

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dsbf.engine import extras
from dsbf.interfaces.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()
SNAPSHOTS = Path(__file__).parent / "snapshots"

DATA = """\
id,amount,group,note
1,10.5,a,x
2,11.0,b,y
3,,a,z
4,9.5,b,x
5,10.1,a,y
6,10.9,b,z
"""

PLUGIN = '''
from pydantic import Field

from dsbf.engine import Requires, SemanticTypes, check
from dsbf.model import CheckParams, Cost


class SpreadParams(CheckParams):
    min_std: float = Field(0.5, ge=0)


@check(
    id="null_share",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=["completeness"],
    cost=Cost.CHEAP,
    packs=["core"],
)
def null_share(ctx, params):
    """Report columns with nulls."""


@check(
    id="numeric_spread",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=["distribution"],
    params=SpreadParams,
    packs=["core"],
)
def numeric_spread(ctx, params):
    """Spread of each numeric column."""


@check(
    id="stationarity",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=["distribution"],
    extras=["timeseries"],
    cost=Cost.HEAVY,
    packs=["timeseries"],
)
def stationarity(ctx, params):
    """Unit-root test."""


@check(
    id="trend_summary",
    version="1.0.0",
    kind="synthesis",
    applies_to=None,
    dimensions=["usability"],
    requires=Requires(checks=["stationarity"]),
    packs=["timeseries"],
)
def trend_summary(ctx, params):
    """Roll up the trend checks."""


@check(
    id="date_gaps",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("datetime"),
    dimensions=["completeness"],
    packs=["core"],
)
def date_gaps(ctx, params):
    """Gaps between dates."""


@check(
    id="quality_summary",
    version="1.0.0",
    kind="synthesis",
    applies_to=None,
    dimensions=["usability"],
    requires=Requires(checks=["null_share"]),
    after=["numeric_spread"],
    cost=Cost.CHEAP,
    packs=["core"],
)
def quality_summary(ctx, params):
    """Roll up the quality checks."""
'''


@pytest.fixture
def data(tmp_path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(DATA)
    return path


@pytest.fixture
def plugin(tmp_path) -> Path:
    directory = tmp_path / "health_plugin"
    directory.mkdir()
    (directory / "health_checks.py").write_text(textwrap.dedent(PLUGIN))
    return directory


@pytest.fixture
def no_timeseries(monkeypatch):
    real = extras.find_spec
    monkeypatch.setattr(
        extras, "find_spec", lambda m: None if m == "statsmodels" else real(m)
    )


def plan(data, plugin, *options):
    return runner.invoke(
        app, ["plan", str(data), "--plugin", str(plugin), "--no-entry-points", *options]
    )


def assert_snapshot(request, name: str, actual: str) -> None:
    path = SNAPSHOTS / name
    if request.config.getoption("--update-golden"):
        path.write_text(actual + "\n", encoding="utf-8")
    assert path.exists(), f"No snapshot {name}; create it with --update-golden."
    assert actual + "\n" == path.read_text(encoding="utf-8"), (
        f"{name} changed; if intended, rerun with --update-golden and review the diff."
    )


def test_the_plan_for_a_csv_matches_its_snapshot(request, data, plugin, no_timeseries):
    result = plan(data, plugin, "--depth", "deep")

    assert result.exit_code == 0, result.output
    assert_snapshot(request, "plan_deep.txt", result.stdout.rstrip("\n"))


def test_a_missing_extra_is_unavailable_with_an_install_hint(
    data, plugin, no_timeseries
):
    result = plan(data, plugin, "--depth", "deep")

    assert "⦸ unavailable" in result.stdout
    assert "Install it with `pip install 'dsbf[timeseries]'`." in result.stdout
    assert "↷ skipped" in result.stdout


def test_an_unavailable_check_does_not_fail_the_command(data, plugin, no_timeseries):
    assert plan(data, plugin, "--pack", "timeseries").exit_code == 0


def test_a_misspelled_parameter_names_the_check_key_and_allowed_keys(data, plugin):
    result = plan(data, plugin, "--param", "numeric_spread.min_stdd=1")

    assert result.exit_code == 2
    assert (
        "error: numeric_spread: unknown parameter 'min_stdd'; allowed: min_std"
        in result.stderr
    )


def test_every_parameter_mistake_is_reported(data, plugin):
    result = plan(
        data,
        plugin,
        "--param",
        "numeric_spread.min_std=-3",
        "--param",
        "numeric_spred.min_std=1",
    )

    assert result.exit_code == 2
    assert result.stderr.count("error:") == 2


def test_a_parameter_value_is_read_as_yaml_and_shown_when_it_differs(data, plugin):
    result = plan(data, plugin, "--pack", "core", "--param", "numeric_spread.min_std=2")

    assert result.exit_code == 0
    assert "min_std=2.0" in result.stdout


def test_a_malformed_parameter_option_is_a_usage_error(data, plugin):
    result = plan(data, plugin, "--param", "min_std=2")

    assert result.exit_code == 2
    assert "check.key=value" in result.output


def test_selection_options_narrow_the_plan(data, plugin):
    result = plan(
        data,
        plugin,
        *("--pack", "timeseries"),
        *("--depth", "deep"),
        *("--include", "date_gaps"),
        *("--exclude", "trend_summary"),
    )

    assert result.exit_code == 0
    assert "date_gaps" in result.stdout
    assert "stationarity" in result.stdout
    assert "trend_summary" not in result.stdout


def test_an_unknown_pack_is_an_error_with_a_suggestion(data, plugin):
    result = plan(data, plugin, "--pack", "cor")

    assert result.exit_code == 2
    assert "unknown pack 'cor'; did you mean 'core'?" in result.stderr


def test_with_every_check_excluded_the_plan_is_empty_and_says_why(data):
    result = runner.invoke(
        app, ["plan", str(data), "--no-entry-points", "--exclude", "zeros"]
    )

    assert result.exit_code == 0
    assert "nothing to run" in result.stdout
    assert "The selection matched no checks" in result.stdout


def test_a_broken_plugin_is_reported_in_the_plan(data, tmp_path):
    broken = tmp_path / "broken_plugin"
    broken.mkdir()
    (broken / "oops.py").write_text("raise RuntimeError('boom')\n")

    result = runner.invoke(
        app, ["plan", str(data), "--plugin", str(broken), "--no-entry-points"]
    )

    assert result.exit_code == 0
    assert "failed to load" in result.stdout
    assert "boom" in result.stdout


def test_an_unreadable_dataset_is_an_error_with_a_hint(tmp_path):
    result = runner.invoke(app, ["plan", str(tmp_path / "missing.csv")])

    assert result.exit_code == 2
    assert result.stderr.startswith("error:")
