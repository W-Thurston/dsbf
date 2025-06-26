# tests/test_tasks/test_check_datetime_consistency.py

import pandas as pd

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.check_datetime_consistency import CheckDatetimeConsistency


def test_check_datetime_consistency_expected_output():
    """
    Ensure CheckDatetimeConsistency correctly parses and flags datetime columns.
    """
    df = pd.DataFrame(
        {
            "dates_good": pd.date_range("2022-01-01", periods=10, freq="D"),
            "dates_bad": ["notadate", "2020-01-01", "???", "2000-12-31"] * 3,
            "non_date": [1, 2, 3, 4] * 3,
        }
    )

    task = CheckDatetimeConsistency()
    task.set_input(df)
    task.run()
    result = task.get_output()

    assert result is not None, "No TaskResult returned"
    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "dates_good" in result.data
    assert isinstance(result.data["dates_good"]["parseable"], bool)
    assert result.data["dates_good"]["parseable"] is True
    assert result.data["dates_good"]["monotonic"] is True
