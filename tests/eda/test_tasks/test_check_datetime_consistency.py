# tests/eda/test_tasks/test_check_datetime_consistency.py

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.check_datetime_consistency import CheckDatetimeConsistency
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_check_datetime_consistency_expected_output(tmp_path):
    """
    Test that CheckDatetimeConsistency returns expected output on
    mixed-validity datetime columns.
    """
    df = pd.DataFrame(
        {
            "valid_dates": ["2020-01-01", "2021-02-02", "2022-03-03"],
            "mixed_dates": ["2020-01-01", "invalid", None],
            "non_date": [123, 456, 789],  # Should be excluded
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CheckDatetimeConsistency,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CheckDatetimeConsistency)

    assert result is not None
    assert isinstance(result.data, dict)
    assert result.status == "success"
    assert result.metadata.get("column_types") is not None

    # Core output checks
    if "valid_dates" in result.data:
        assert result.data["valid_dates"]["percent_valid"] == 100.0
    if "mixed_dates" in result.data:
        assert result.data["mixed_dates"]["percent_valid"] < 100.0
    else:
        # Confirm it was excluded based on semantic type
        excluded = result.metadata.get("excluded_columns", {})
        assert "mixed_dates" in excluded
    assert "non_date" not in result.data

    # Metadata checks
    excluded = result.metadata.get("excluded_columns", {})
    assert "non_date" in excluded

    column_types = result.metadata.get("column_types", {})
    assert "valid_dates" in column_types
    assert "mixed_dates" in column_types
    assert "non_date" in column_types
    assert column_types["valid_dates"]["analysis_intent_dtype"] in [
        "datetime",
        "id",
        "unknown",
    ]
    assert column_types["non_date"]["analysis_intent_dtype"] != "datetime"
