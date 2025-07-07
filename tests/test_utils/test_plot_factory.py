# tests/test_utils/test_plot_factory.py

import pandas as pd
import pytest

from dsbf.utils.plot_factory import PlotFactory


@pytest.fixture
def test_series():
    return pd.Series([1, 2, 3, 4, 5], name="TestSeries")


@pytest.fixture
def empty_series():
    return pd.Series([], dtype=float, name="EmptySeries")


def test_histogram_interactive_returns_plotdata(test_series):
    result = PlotFactory.plot_histogram_interactive(test_series)
    assert isinstance(result, dict)
    assert result.get("type") == "histogram"
    assert "data" in result and "x" in result["data"]
    assert "config" in result


def test_histogram_interactive_empty_series(empty_series):
    result = PlotFactory.plot_histogram_interactive(empty_series)
    assert result.get("annotations") == ["Empty series"]


def test_histogram_static_creates_file(tmp_path, test_series):
    file_path = tmp_path / "histogram.png"
    result = PlotFactory.plot_histogram_static(test_series, str(file_path))
    assert file_path.exists()
    assert result["path"] == file_path
    assert "plot_data" in result
    file_path.unlink()  # cleanup


def test_plotdata_format_keys_present(test_series):
    result = PlotFactory.plot_histogram_interactive(test_series)
    assert set(result.keys()).issuperset({"type", "data", "config", "annotations"})
