# tests/eda/test_tasks/test_compare_with_reference_dataset.py

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compare_with_reference_dataset import CompareWithReferenceDataset
from tests.helpers.context_utils import make_ctx_and_task


def test_detects_added_and_dropped_columns():
    current = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
    reference = pl.DataFrame({"B": [3, 4], "C": [5, 6]})
    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.summary["added_columns"] == ["A"]
    assert result.summary["dropped_columns"] == ["C"]


def test_type_mismatch_detection():
    current = pl.DataFrame({"col": ["1", "2", "3"]})
    reference = pl.DataFrame({"col": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
    )
    result = ctx.run_task(task)

    assert "col" in result.summary["type_mismatches"]


def test_field_change_flags():
    current = pl.DataFrame({"x": [1.0] * 50 + [10.0] * 50})
    reference = pl.DataFrame({"x": [1.0] * 100})
    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
        task_overrides={
            "missing_pct_threshold": 0.3,
            "unique_count_ratio_threshold": 0.5,
            "minmax_numeric_tolerance": 0.01,
        },
        global_overrides={"message_verbosity": "debug"},
    )
    result = ctx.run_task(task)

    field_info = result.summary["field_changes"]["x"]
    assert field_info["flag_min_diff"] is False
    assert field_info["flag_max_diff"] is True
    assert field_info["flag_missing_diff"] is False
    assert field_info["flag_unique_diff"] is True


def test_skips_without_reference():
    current = pl.DataFrame({"x": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
    )
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "no reference dataset" in result.summary["message"].lower()


def test_handles_column_exception_gracefully():
    # Introduce NaNs that will break min/max calculations
    current = pl.DataFrame({"x": [None, None, None]})
    reference = pl.DataFrame({"x": [1, 2, 3]})
    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert "x" in result.summary["type_mismatches"]
    assert "error" not in result.summary["field_changes"]["x"]
