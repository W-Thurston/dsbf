"""The HTML sink: escaping, self-containment and size."""

import re
import warnings
from pathlib import Path

import polars as pl
import pytest

import dsbf
from dsbf.model import Run
from dsbf.sinks import known_sinks, render_report

ROOT = Path(__file__).parents[2]
TITANIC = ROOT / "tests" / "golden" / "data" / "titanic.csv"
SIZE_BUDGET = 150_000


@pytest.fixture(scope="module", autouse=True)
def quiet_libraries():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def titanic() -> Run:
    return dsbf.profile(TITANIC)


@pytest.mark.unit
def test_html_is_a_registered_sink():
    assert "html" in known_sinks()


@pytest.mark.integration
def test_the_titanic_report_is_small_offline_and_lists_every_column(titanic, tmp_path):
    path = titanic.write(tmp_path, sinks=["html"])["html"]
    assert path.name == "report.html"
    text = path.read_text(encoding="utf-8")
    assert len(text.encode()) < SIZE_BUDGET
    assert "http://" not in text and "https://" not in text
    assert "<script" not in text
    assert text.count("<div class='card'>") == titanic.profile.n_columns
    assert "<svg" in text


@pytest.mark.integration
def test_hostile_text_is_escaped(tmp_path):
    hostile = "<script>alert(1)</script>"
    frame = pl.DataFrame(
        {
            f'{hostile}"&': [1.0, 2.0, 3.0, 4.0] * 5,
            "'quote'<b>": ["a<i>", "b&c", "a<i>", None] * 5,
        }
    )
    run = dsbf.profile(frame, name="<img src=x onerror=alert(1)>")
    text = render_report(run)
    assert "<script>" not in text
    assert "<img src=x" not in text
    assert "<b>" not in text and "<i>" not in text
    assert re.search(r"&lt;script&gt;alert\(1\)&lt;/script&gt;", text)


@pytest.mark.integration
def test_the_report_shows_the_policy_verdict_only_when_a_policy_ran(titanic):
    assert "<h2>Policy</h2>" not in render_report(titanic)
    rules = {
        "rules": [{"name": "any-nulls", "when": {"code": "has_nulls"}, "then": "fail"}]
    }
    run = dsbf.profile(TITANIC, depth="quick", policy=rules)
    text = render_report(run)
    assert "<h2>Policy</h2>" in text
    assert run.policy is not None
    assert ("Policy failed." in text) is run.policy.failed
