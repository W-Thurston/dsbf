# tests/test_tasks/test_summarize_dataset_shape.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.summarize_dataset_shape import SummarizeDatasetShape
from tests.helpers.context_utils import make_ctx_and_task


def test_summarize_dataset_shape_expected_output():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})

    ctx, task = make_ctx_and_task(
        task_cls=SummarizeDatasetShape,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert result.data["num_rows"] == 3
    assert result.data["num_columns"] == 2
    assert 0 < result.data["null_cell_percentage"] < 1
    assert "approx_memory_MB" in result.data
