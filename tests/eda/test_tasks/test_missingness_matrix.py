# tests/test_tasks/test_missingness_matrix.py

import os

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.missingness_matrix import MissingnessMatrix


def test_missingness_matrix_creates_image(tmp_path):
    df = pd.DataFrame({"x": [1, None, 3], "y": [None, 2, 3]})

    context = AnalysisContext(df, output_dir=str(tmp_path))
    result = context.run_task(MissingnessMatrix())

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "image_path" in result.data
    assert os.path.exists(result.data["image_path"])


def test_missingness_matrix_all_null(tmp_path):
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    context = AnalysisContext(df, output_dir=str(tmp_path))
    result = context.run_task(MissingnessMatrix())
    assert result.status == "success"
    assert result.data is not None
    assert "image_path" in result.data
