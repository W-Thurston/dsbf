# tests/test_tasks/test_detect_skewness.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_skewness import DetectSkewness


def test_detect_skewness_expected_output():
    df = pd.DataFrame(
        {
            "normal": [1, 2, 3, 4, 5],
            "skewed": [1, 1, 1, 2, 100],
        }
    )

    task = DetectSkewness()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    assert "normal" in result.data
    assert "skewed" in result.data
    assert abs(result.data["normal"]) < 1.0
    assert result.data["skewed"] > 1.0
