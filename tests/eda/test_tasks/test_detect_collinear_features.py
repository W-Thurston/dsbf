# tests/test_tasks/test_detect_collinear_features.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_collinear_features import DetectCollinearFeatures


def test_detect_collinear_features_expected_output():
    """
    Test that DetectCollinearFeatures returns expected VIF flags
    for perfectly collinear variables.
    """
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5],
            "x2": [2, 4, 6, 8, 10],  # Strongly collinear with x1
            "x3": [5, 4, 3, 2, 1],
        }
    )

    task = DetectCollinearFeatures(config={"vif_threshold": 5})
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None

    scores = result.data.get("vif_scores", {})
    flagged = result.data.get("collinear_columns", [])

    assert isinstance(scores, dict)
    assert any(v > 5 for v in scores.values())
    assert "x2" in flagged or "x1" in flagged
