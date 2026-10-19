# tests/test_utils/test_logging_utils.py

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from dsbf.core.context import AnalysisContext
from dsbf.utils.logging_utils import (
    console_logging,
    format_number,
    get_log_fn,
    level_from_name,
    run_log_file,
)

pytestmark = pytest.mark.unit

_IMPORT_ALL = """
import logging
import dsbf, dsbf.engine, dsbf.core.context, dsbf.core.base_engine
import dsbf.core.base_task, dsbf.eda.profile_engine, dsbf.eda.task_registry
import dsbf.interfaces.cli, dsbf.utils.data_loader, dsbf.utils.reco_engine
import dsbf.utils.report_utils, dsbf.utils.task_utils

assert logging.getLoggerClass() is logging.Logger, logging.getLoggerClass()
assert logging.getLevelName(25) == "Level 25", logging.getLevelName(25)
assert logging.getLevelName(15) == "Level 15", logging.getLevelName(15)
handlers = {
    name: [type(h).__name__ for h in logger.handlers]
    for name, logger in logging.Logger.manager.loggerDict.items()
    if isinstance(logger, logging.Logger) and logger.handlers
}
assert handlers == {"dsbf": ["NullHandler"]}, handlers
assert not logging.getLogger().handlers
print("ok")
"""


@pytest.mark.parametrize("name", ["[/x]", "[bold]", "[red]x[/red]", "a[b"])
def test_log_message_with_brackets_is_not_parsed_as_markup(name, capsys) -> None:
    """A column name that looks like Rich markup must be printed, not interpreted."""
    ctx = AnalysisContext(data=None, config={})

    with console_logging("debug"):
        ctx._log(f"Processing column '{name}'", "info", task_name="demo")

    captured = capsys.readouterr()
    assert name in captured.out.replace("\n", "")
    assert "[demo]" in captured.out
    assert "Logging error" not in captured.err


def test_format_number_renders_missing_values_as_na() -> None:
    assert format_number(0.12345, ".4f") == "0.1235"
    assert format_number(None) == "n/a"
    assert format_number(float("nan")) == "n/a"
    assert format_number("text") == "n/a"


def test_importing_dsbf_changes_no_global_logging_state() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", _IMPORT_ALL],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_task_level_messages_reach_the_run_log(tmp_path: Path) -> None:
    ctx = AnalysisContext(data=None, config={})

    with run_log_file(tmp_path, "info"):
        ctx._log("loaded 3 columns", "info", task_name="summarize")
        ctx._log("memory limit exceeded", "warn", task_name="summarize")
        ctx._log("not wanted at info", "debug", task_name="summarize")

    lines = (tmp_path / "run.log").read_text().splitlines()
    assert len(lines) == 2
    assert "[INFO2] dsbf.core.context: [summarize] loaded 3 columns" in lines[0]
    assert "[WARNING] dsbf.core.context: [summarize] memory limit" in lines[1]


def test_run_log_names_the_custom_levels_without_changing_the_global_table(
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("dsbf.test_levels")

    with run_log_file(tmp_path, "stage"):
        get_log_fn(logger, "stage")("starting")

    assert "[STAGE] dsbf.test_levels: starting" in (tmp_path / "run.log").read_text()
    assert logging.getLevelName(25) == "Level 25"


def test_run_log_handler_is_closed_and_the_logger_restored(tmp_path: Path) -> None:
    root = logging.getLogger("dsbf")
    before = (list(root.handlers), root.level)

    with run_log_file(tmp_path, "debug"):
        (handler,) = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert root.level == logging.DEBUG
        assert handler.stream is not None

    assert handler.stream is None
    assert (list(root.handlers), root.level) == before


def test_run_log_handler_is_closed_when_the_run_fails(tmp_path: Path) -> None:
    root = logging.getLogger("dsbf")
    before = list(root.handlers)

    with pytest.raises(RuntimeError), run_log_file(tmp_path, "info"):
        raise RuntimeError("boom")

    assert list(root.handlers) == before


def test_each_run_writes_its_own_log_and_leaves_no_open_file(tmp_path: Path) -> None:
    logger = logging.getLogger("dsbf.test_runs")

    for run in ("a", "b"):
        with run_log_file(tmp_path / run, "info"):
            get_log_fn(logger, "info")(f"run {run}")

    assert "run a" in (tmp_path / "a" / "run.log").read_text()
    assert "run a" not in (tmp_path / "b" / "run.log").read_text()


def test_quiet_run_writes_no_log_file(tmp_path: Path) -> None:
    with run_log_file(tmp_path, "quiet"):
        logging.getLogger("dsbf.test_quiet").error("never written")

    assert not (tmp_path / "run.log").exists()


def test_error_is_a_level_of_its_own(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("dsbf.test_error")

    with caplog.at_level(logging.DEBUG, logger="dsbf"):
        get_log_fn(logger, "error")("failed")
        get_log_fn(logger, "info")("fine")

    assert [r.levelno for r in caplog.records] == [logging.ERROR, 15]
    assert level_from_name("error") == logging.ERROR
    assert level_from_name("nonsense") == 15


def test_console_logging_removes_its_handler_and_restores_the_level() -> None:
    root = logging.getLogger("dsbf")
    before = (list(root.handlers), root.level)

    with console_logging("debug"):
        assert len(root.handlers) == len(before[0]) + 1

    assert (list(root.handlers), root.level) == before
