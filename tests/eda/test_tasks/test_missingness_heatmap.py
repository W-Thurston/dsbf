# tests/test_tasks/test_missingness_heatmap.py

import os

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.missingness_heatmap import missingness_heatmap


def test_missingness_heatmap_creates_image(tmp_path):
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})

    output_dir = tmp_path / "figs"
    result = missingness_heatmap(df, output_dir=str(output_dir))

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "image_path" in result.data
    assert os.path.exists(result.data["image_path"])
