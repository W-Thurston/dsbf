# tests/test_tasks/test_missingness_matrix.py

import os

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.missingness_matrix import missingness_matrix


def test_missingness_matrix_creates_image(tmp_path):
    df = pd.DataFrame({"x": [1, None, 3], "y": [None, 2, 3]})

    output_dir = tmp_path / "figs"
    result = missingness_matrix(df, output_dir=str(output_dir))

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "image_path" in result.data
    assert os.path.exists(result.data["image_path"])
