# tests/test_utils/test_logging_utils.py

import logging
import subprocess
import sys

import pytest

from dsbf.utils.logging_utils import console_logging, level_from_name

pytestmark = pytest.mark.unit

_IMPORT_ALL = """
import logging
import dsbf, dsbf.engine, dsbf.library, dsbf.interfaces.cli

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
    logger = logging.getLogger("dsbf.test_markup")

    with console_logging("debug"):
        logger.warning("Processing column '%s'", name)

    captured = capsys.readouterr()
    assert name in captured.out.replace("\n", "")
    assert "Logging error" not in captured.err


def test_importing_dsbf_changes_no_global_logging_state() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", _IMPORT_ALL],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_error_is_a_level_of_its_own_and_an_unknown_name_is_info() -> None:
    assert level_from_name("error") == logging.ERROR
    assert level_from_name("nonsense") == 15


def test_quiet_console_logging_prints_nothing(capsys) -> None:
    with console_logging("quiet"):
        logging.getLogger("dsbf.test_quiet").error("never printed")

    assert capsys.readouterr().out == ""


def test_console_logging_removes_its_handler_and_restores_the_level() -> None:
    root = logging.getLogger("dsbf")
    before = (list(root.handlers), root.level)

    with console_logging("debug"):
        assert len(root.handlers) == len(before[0]) + 1

    assert (list(root.handlers), root.level) == before
