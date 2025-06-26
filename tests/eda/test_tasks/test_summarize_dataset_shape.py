# tests/test_tasks/test_summarize_dataset_shape.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_dataset_shape import SummarizeDatasetShape


def test_summarize_dataset_shape_expected_output():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})

    task = SummarizeDatasetShape()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert result.data["num_rows"] == 3
    assert result.data["num_columns"] == 2
    assert 0 < result.data["null_cell_percentage"] < 1
    assert "approx_memory_MB" in result.data
