"""The library example in the README runs as written."""

import re
import warnings
from pathlib import Path

import polars as pl
import pytest

pytestmark = pytest.mark.integration

README = Path(__file__).parents[2] / "README.md"
HEADING = "### Python library"


def example() -> str:
    text = README.read_text(encoding="utf-8")
    section = text[text.index(HEADING) :]
    return re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)  # type: ignore[union-attr]


def test_the_library_example_runs_on_a_csv_called_data_csv(
    tmp_path, monkeypatch, capsys
):
    rows = 120
    pl.DataFrame(
        {
            "survived": [i % 2 for i in range(rows)],
            "age": [20.0 + i % 40 for i in range(rows)],
            "sex": ["f" if i % 3 else "m" for i in range(rows)],
        }
    ).write_csv(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)
    scope: dict = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(example(), str(README), "exec"), scope)  # noqa: S102

    assert scope["run"].dataset.name == "data"
    assert capsys.readouterr().out.startswith("DSBF run ")
    assert isinstance(scope["worst"], list)
    assert {p.name for p in tmp_path.iterdir()} == {"data.csv"}
