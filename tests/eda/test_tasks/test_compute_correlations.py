# tests/test_tasks/test_compute_correlations.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_correlations import ComputeCorrelations


def test_compute_correlations_with_categorical():
    """
    Test that ComputeCorrelations detects Pearson and Cramér’s V correlations
    and returns results in expected flat dictionary format.
    """
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],  # Strong Pearson
            "cat1": ["a", "a", "b", "b", "b"],
            "cat2": ["yes", "yes", "no", "no", "no"],
        }
    )

    task = ComputeCorrelations()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    # Pearson check
    assert any("x|y" in key for key in result.data)
    assert abs(result.data["x|y"] - 1.0) > 0.99  # Perfect correlation

    # Cramér’s V check
    assert "cat1|cat2" in result.data
    v = result.data["cat1|cat2"]
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0
