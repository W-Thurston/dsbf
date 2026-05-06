# tests/test_utils/test_plot_factory.py

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas import Series

from dsbf.utils.plot_factory import PlotFactory


@pytest.fixture
def test_series() -> Series:
    return pd.Series([1, 2, 3, 4, 5], name="TestSeries")


@pytest.fixture
def empty_series() -> Series:
    return pd.Series([], dtype=float, name="EmptySeries")


def test_histogram_interactive_returns_plotdata(test_series) -> None:
    result: dict[str, Any] = PlotFactory.plot_histogram_interactive(test_series)
    assert isinstance(result, dict)
    assert result.get("type") == "histogram"
    assert "data" in result and "x" in result["data"]
    assert "config" in result


def test_histogram_interactive_empty_series(empty_series) -> None:
    result: dict[str, Any] = PlotFactory.plot_histogram_interactive(empty_series)
    assert result.get("annotations") == ["Empty series"]


def test_histogram_static_creates_file(tmp_path, test_series) -> None:
    file_path = tmp_path / "histogram.png"
    result: dict[str, Any] = PlotFactory.plot_histogram_static(
        test_series,
        str(file_path),
    )
    # plot_histogram_static saves themed variants (histogram_dark.png,
    # histogram_light.png) rather than the plain path — check at least
    # one themed file was created and the result paths dict is populated.
    assert isinstance(result["path"], dict), "Expected themed paths dict"
    assert len(result["path"]) > 0, "No themed output files were created"
    for theme_path in result["path"].values():
        assert Path(theme_path).exists(), f"Themed file not found: {theme_path}"
    assert "plot_data" in result


def test_plotdata_format_keys_present(test_series) -> None:
    result: dict[str, Any] = PlotFactory.plot_histogram_interactive(test_series)
    assert set(result.keys()).issuperset({"type", "data", "config", "annotations"})
