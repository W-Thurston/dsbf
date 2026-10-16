# tests/test_utils/test_reco_engine.py

from pathlib import Path

import pytest

from dsbf.utils import reco_engine

pytestmark = pytest.mark.unit


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


@pytest.fixture
def fresh_cache(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    """Start each test with no cached library and capture the module's warnings."""
    recorder = _RecordingLogger()
    monkeypatch.setattr(reco_engine, "RECO_CACHE", None)
    monkeypatch.setattr(reco_engine, "logger", recorder)
    return recorder


def _use_library(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str | None
) -> None:
    path = tmp_path / "library.yaml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(reco_engine, "_library_resource", lambda: path)


def test_packaged_library_loads_from_any_working_directory(
    fresh_cache, monkeypatch, tmp_path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert not Path("dsbf/static_metadata").exists()

    tip = reco_engine.get_recommendation_tip(
        "detect_high_cardinality", {"n_unique": 99}
    )

    assert tip is not None
    assert "high cardinality" in tip
    assert fresh_cache.warnings == []


def test_tip_is_none_when_no_condition_matches(fresh_cache):
    assert (
        reco_engine.get_recommendation_tip("detect_high_cardinality", {"n_unique": 3})
        is None
    )


def test_missing_library_returns_none_and_warns_once(
    fresh_cache, monkeypatch, tmp_path
):
    _use_library(monkeypatch, tmp_path, None)

    first = reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0})
    second = reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0})

    assert first is None
    assert second is None
    assert len(fresh_cache.warnings) == 1
    assert "unavailable" in fresh_cache.warnings[0]


def test_malformed_library_returns_none_and_warns(fresh_cache, monkeypatch, tmp_path):
    _use_library(monkeypatch, tmp_path, "- not\n- a mapping\n")

    assert reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0}) is None
    assert len(fresh_cache.warnings) == 1


def test_unparseable_yaml_returns_none_and_warns(fresh_cache, monkeypatch, tmp_path):
    _use_library(monkeypatch, tmp_path, "key: [unclosed\n")

    assert reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0}) is None
    assert len(fresh_cache.warnings) == 1


def test_malformed_tip_entry_returns_none_and_warns(fresh_cache, monkeypatch, tmp_path):
    _use_library(monkeypatch, tmp_path, "broken: just a string\n")

    assert reco_engine.get_recommendation_tip("detect_skewness", {"skew": 3.0}) is None
    assert any("Tip lookup failed" in w for w in fresh_cache.warnings)


def test_condition_that_raises_is_skipped_for_the_next_tip(
    fresh_cache, monkeypatch, tmp_path
):
    _use_library(
        monkeypatch,
        tmp_path,
        "first:\n"
        "  condition: 'undefined_name > 1'\n"
        "  applicable_to: [demo]\n"
        "  message: never\n"
        "second:\n"
        "  condition: 'x > 1'\n"
        "  applicable_to: [demo]\n"
        "  message: matched\n",
    )

    assert reco_engine.get_recommendation_tip("demo", {"x": 2}) == "matched"
    assert fresh_cache.warnings == []
