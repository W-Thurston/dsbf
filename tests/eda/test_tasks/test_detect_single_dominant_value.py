# tests/test_tasks/test_detect_single_dominant_value.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_single_dominant_value import DetectSingleDominantValue


def test_detect_single_dominant_value_expected_output():
    df = pd.DataFrame(
        {
            "mostly_ones": [1] * 95 + [0] * 5,
            "uniform": [1, 2, 3, 4, 5] * 20,
            "binary": ["yes"] * 96 + ["no"] * 4,
        }
    )

    context = AnalysisContext(df)
    result = context.run_task(
        DetectSingleDominantValue(config={"dominance_threshold": 0.9})
    )

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "mostly_ones" in result.data
    assert "binary" in result.data
    assert "uniform" not in result.data
