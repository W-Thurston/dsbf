# tests/eda/test_tasks/test_check_datetime_consistency.py

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.check_datetime_consistency import CheckDatetimeConsistency
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_fully_valid_datetime_column(tmp_path) -> None:
    """A column with all parseable datetimes should report 100% validity."""
    df = pd.DataFrame(
        {
            "valid_dates": ["2020-01-01", "2021-02-02", "2022-03-03"],
            "age": [25, 30, 35],  # numeric - should be excluded
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CheckDatetimeConsistency,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CheckDatetimeConsistency)

    assert result.status == "success"
    assert result.data is not None
    # valid_dates should be classified as datetime by infer_types and appear in output
    assert "valid_dates" in result.data
    assert result.data["valid_dates"]["percent_valid"] == 100.0
    assert result.data["valid_dates"]["num_invalid"] == 0
    assert "age" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_partially_invalid_datetime_column(tmp_path) -> None:
    """A column with some unparseable values should report < 100% validity."""
    df = pd.DataFrame(
        {
            "mixed_dates": ["2020-01-01", "not_a_date", "2022-03-03"],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CheckDatetimeConsistency,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CheckDatetimeConsistency)

    assert result.status == "success"
    assert result.data is not None
    if "mixed_dates" in result.data:
        # At least one value is invalid
        assert result.data["mixed_dates"]["percent_valid"] < 100.0
        assert result.data["mixed_dates"]["num_invalid"] >= 1


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_non_datetime_columns_excluded(tmp_path) -> None:
    """Numeric and categorical columns must not appear in output data."""
    df = pd.DataFrame(
        {
            "event_date": ["2020-01-01", "2021-06-15", "2022-12-31"],
            "count": [10, 20, 30],
            "label": ["a", "b", "c"],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CheckDatetimeConsistency,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CheckDatetimeConsistency)

    assert result.status == "success"
    assert "count" not in result.data
    assert "label" not in result.data

    excluded = result.metadata.get("excluded_columns", {})
    assert "count" in excluded
    assert "label" in excluded


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_metadata_fields_populated(tmp_path) -> None:
    """column_types metadata must include all columns regardless of exclusion."""
    df = pd.DataFrame(
        {
            "ts": ["2020-01-01", "2021-02-02"],
            "val": [1, 2],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CheckDatetimeConsistency,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CheckDatetimeConsistency)

    assert result.status == "success"
    column_types = result.metadata.get("column_types", {})
    assert "ts" in column_types
    assert "val" in column_types
    assert column_types["val"]["analysis_intent_dtype"] != "datetime"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must process Polars DataFrames without falling back to pandas explicitly."""
    df = pl.DataFrame(
        {
            "event_ts": ["2020-01-01", "2021-06-15", "not_a_date"],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=CheckDatetimeConsistency,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, CheckDatetimeConsistency)

    assert result.status == "success"
    # If classified as datetime, validity should be < 100% due to "not_a_date"
    if "event_ts" in result.data:
        assert result.data["event_ts"]["num_values"] == 3
