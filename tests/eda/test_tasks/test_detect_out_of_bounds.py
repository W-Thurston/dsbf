# tests/test_tasks/test_detect_out_of_bounds.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_out_of_bounds import DetectOutOfBounds


def test_detect_out_of_bounds_expected_output():
    df = pd.DataFrame(
        {
            "age": [25, 30, -5, 150],
            "score": [0.8, 0.95, 1.1, 0.5],
            "percent": [50, 110, 20, -10],
            "weight": [150, 180, 200, 175],
        }
    )

    context = AnalysisContext(df)
    result = context.run_task(DetectOutOfBounds())

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "age" in result.data
    assert result.data["age"]["count"] == 2
    assert "score" in result.data
    assert "percent" in result.data
    assert "weight" not in result.data  # no default rule for weight
