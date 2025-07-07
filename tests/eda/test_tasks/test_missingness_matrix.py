# tests/eda/test_tasks/test_missingness_matrix.py

from pathlib import Path

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.tasks.missingness_matrix import MissingnessMatrix


def test_missingness_matrix_runs_successfully(tmp_path):
    """
    Ensure the task runs and produces a plot even with missing values.
    """
    df = pd.DataFrame({"x": [1, None, 3], "y": [None, 2, 3]})
    context = AnalysisContext(df, output_dir=str(tmp_path))
    result = context.run_task(MissingnessMatrix())

    assert result.status == "success"
    assert result.plots is not None
    assert "missingness_matrix" in result.plots


def test_missingness_matrix_handles_all_null(tmp_path):
    """
    Ensure task does not fail on all-null columns.
    """
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    context = AnalysisContext(df, output_dir=str(tmp_path))
    result = context.run_task(MissingnessMatrix())

    assert result.status == "success"
    assert result.plots is not None
    assert "missingness_matrix" in result.plots


def test_missingness_matrix_plot_is_generated(tmp_path):
    """
    Validate that the matrix PNG is saved to disk.
    """
    df = pd.DataFrame({"col1": [1, None, 3], "col2": [None, 2, None]})
    context = AnalysisContext(df, output_dir=str(tmp_path))
    result = context.run_task(MissingnessMatrix())

    assert result.status == "success"
    assert result.plots is not None

    plot_entry = result.plots["missingness_matrix"]
    static_path = plot_entry["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
