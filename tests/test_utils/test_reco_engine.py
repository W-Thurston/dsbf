# tests/test_utils/test_reco_engine.py
"""The legacy tasks' tip lookup, which reads the engine's library."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from dsbf.engine import tips
from dsbf.utils import reco_engine

pytestmark = pytest.mark.unit


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


@pytest.fixture
def fresh_library(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_RecordingLogger]:
    """Start each test with no cached library and capture the module's warnings."""
    recorder = _RecordingLogger()
    tips.load_library.cache_clear()
    tips._warn_once.cache_clear()
    monkeypatch.setattr(tips, "logger", recorder)
    yield recorder
    tips.load_library.cache_clear()
    tips._warn_once.cache_clear()


def _use_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str | None
) -> None:
    path = tmp_path / "library.yaml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(tips, "_resource", lambda: path)


def test_packaged_library_loads_from_any_working_directory(
    fresh_library, monkeypatch, tmp_path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert not Path("src/dsbf/static_metadata").exists()

    tip = reco_engine.get_recommendation_tip(
        "detect_high_cardinality", {"n_unique": 99}
    )

    assert tip is not None
    assert "high cardinality" in tip
    assert fresh_library.warnings == []


def test_tip_is_none_when_no_condition_matches(fresh_library):
    assert (
        reco_engine.get_recommendation_tip("detect_high_cardinality", {"n_unique": 3})
        is None
    )


def test_a_compound_condition_matches_either_side(fresh_library):
    for skew in (2.5, -2.5):
        assert reco_engine.get_recommendation_tip("detect_skewness", {"skew": skew})
    assert reco_engine.get_recommendation_tip("detect_skewness", {"skew": 0.2}) is None


def test_missing_library_returns_none_and_warns_once(
    fresh_library, monkeypatch, tmp_path
):
    _use_library(monkeypatch, tmp_path, None)

    first = reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0})
    second = reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0})

    assert first is None
    assert second is None
    assert len(fresh_library.warnings) == 1
    assert "unavailable" in fresh_library.warnings[0]


@pytest.mark.parametrize(
    "text",
    [
        "- not\n- a mapping\n",
        "key: [unclosed\n",
        "broken: just a string\n",
        "bad:\n  condition: 'x > 1'\n  action: drop\n  message: m\n",
    ],
    ids=["list", "yaml", "tip-not-a-mapping", "condition-is-code"],
)
def test_unusable_library_returns_none_and_warns(
    fresh_library, monkeypatch, tmp_path, text
):
    _use_library(monkeypatch, tmp_path, text)

    assert reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0}) is None
    assert len(fresh_library.warnings) == 1
