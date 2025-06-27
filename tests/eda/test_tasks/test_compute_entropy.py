# tests/test_tasks/test_compute_entropy.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_entropy import ComputeEntropy


def test_compute_entropy_expected_output():
    """
    Test that ComputeEntropy returns entropy values for text columns.
    """
    df = pd.DataFrame(
        {
            "cat": ["a", "a", "b", "b", "b", "c", "c", "c", "c"],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Should be ignored
        }
    )

    context = AnalysisContext(df)
    result = context.run_task(ComputeEntropy())

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "cat" in result.data
    assert result.data["cat"] > 0.0
