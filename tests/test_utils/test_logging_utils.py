# tests/test_utils/test_logging_utils.py

import pytest

from dsbf.core.context import AnalysisContext
from dsbf.utils.logging_utils import format_number

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "spec", "expected"),
    [
        (0.12345, ".4f", "0.1235"),
        (3, "d", "3"),
        (None, ".4f", "n/a"),
        (float("nan"), ".4f", "n/a"),
        ("text", ".4f", "n/a"),
    ],
)
def test_format_number_renders_missing_values_as_na(value, spec, expected) -> None:
    assert format_number(value, spec) == expected


@pytest.mark.parametrize("name", ["[/x]", "[bold]", "[red]x[/red]", "a[b"])
def test_log_message_with_brackets_is_not_parsed_as_markup(name, capsys) -> None:
    """A column name that looks like Rich markup must be printed, not interpreted."""
    ctx = AnalysisContext(
        data=None, config={"metadata": {"message_verbosity": "debug"}}
    )

    ctx._log(f"Processing column '{name}'", "info", task_name="demo")

    out = capsys.readouterr().out
    assert name in out.replace("\n", "")
    assert "[demo]" in out
