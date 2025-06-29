# tests/test_tasks/test_categorical_length_stats.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.categorical_length_stats import CategoricalLengthStats
from tests.helpers.context_utils import make_ctx_and_task


def test_categorical_length_stats_expected_output():
    """
    Test that CategoricalLengthStats correctly computes string length stats
    for text-like columns and skips numeric columns.
    """
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlotte", None],
            "city": ["New York", "Paris", "Berlin", "New York"],
            "age": [25, 30, 35, 40],  # Should be ignored
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=CategoricalLengthStats,
        current_df=df,
    )
    result = ctx.run_task(task)

    # Validate TaskResult structure
    assert result is not None, "Task did not produce an output"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert isinstance(result.data, dict)
    assert "name" in result.data
    assert "city" in result.data
    assert "age" not in result.data  # Not a string-type column

    # Validate content structure
    for col_stats in result.data.values():
        assert {"mean_length", "max_length", "min_length"} <= set(col_stats.keys())
