# tests/eda/test_tasks/test_missingness_heatmap.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.missingness_heatmap import MissingnessHeatmap


def test_missingness_heatmap_creates_image(tmp_path):
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})

    context = AnalysisContext(df, output_dir=str(tmp_path))
    result = context.run_task(MissingnessHeatmap())

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "plots" in result.__dict__
    assert result.plots is not None
    assert "missingness_heatmap" in result.plots

    plot_entry = result.plots["missingness_heatmap"]

    # Static plot path
    static_path = plot_entry["static"]
    assert static_path.exists()
    assert static_path.suffix == ".png"

    # Interactive content
    interactive = plot_entry["interactive"]
    assert isinstance(interactive, dict)
    assert interactive["type"] == "matrix"
    assert "annotations" in interactive
    assert "missing" in interactive["config"].get("title", "").lower()


def test_missingness_heatmap_with_no_missing_values(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    context = AnalysisContext(df, output_dir=str(tmp_path))

    result = context.run_task(MissingnessHeatmap())

    assert result.status == "success"
    assert result.data is not None
    assert result.data["missing_cells"] == 0
    assert result.plots is not None
    assert "missingness_heatmap" in result.plots
