# tests/test_tasks/test_missingness_heatmap.py

import os

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
    assert "image_path" in result.data
    assert os.path.exists(result.data["image_path"])
