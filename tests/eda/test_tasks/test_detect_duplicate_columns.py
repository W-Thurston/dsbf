# tests/test_tasks/test_detect_duplicate_columns.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_duplicate_columns import DetectDuplicateColumns
from tests.helpers.context_utils import make_ctx_and_task


def test_detect_duplicate_columns_expected_output():
    """
    Test that DetectDuplicateColumns identifies duplicate column pairs.
    """
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 3],  # Duplicate of 'a'
            "c": [3, 2, 1],
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectDuplicateColumns,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result is not None
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    pairs = result.data.get("duplicate_column_pairs", [])
    assert any(set(pair) == {"a", "b"} for pair in pairs)
