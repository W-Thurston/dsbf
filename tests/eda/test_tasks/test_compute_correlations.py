# tests/test_tasks/test_compute_correlations.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_correlations import compute_correlations


def test_compute_correlations_with_categorical():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "cat1": ["a", "a", "b", "b", "b"],
            "cat2": ["yes", "yes", "no", "no", "no"],
        }
    )

    result = compute_correlations(df)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert any("x|y" in key for key in result.data)
    assert any("cat1|cat2" in key for key in result.data)
    assert 0 <= result.data["cat1|cat2"] <= 1
