# tests/eda/test_tasks/test_compare_with_reference_dataset.py

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compare_with_reference_dataset import CompareWithReferenceDataset
from tests.helpers.context_utils import make_ctx_and_task


def test_detects_added_and_dropped_columns(tmp_path) -> None:
    """Added and dropped columns must be correctly identified."""
    current = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
    reference = pl.DataFrame({"B": [3, 4], "C": [5, 6]})

    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.summary["added_columns"] == ["A"]
    assert result.summary["dropped_columns"] == ["C"]


def test_type_mismatch_detection(tmp_path) -> None:
    """Type mismatches between shared columns must be flagged."""
    current = pl.DataFrame({"col": ["1", "2", "3"]})
    reference = pl.DataFrame({"col": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = ctx.run_task(task)

    assert result.status == "success"
    assert "col" in result.summary["type_mismatches"]


def test_field_change_flags(tmp_path) -> None:
    """Numeric range and uniqueness changes must be correctly flagged."""
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
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    field_info = result.data["field_changes"]["x"]
    assert field_info["flag_min_diff"] is False  # same min (1.0)
    assert field_info["flag_max_diff"] is True  # max changed 1.0 → 10.0
    assert field_info["flag_missing_diff"] is False
    assert field_info["flag_unique_diff"] is True  # 1 → 2 unique values


def test_skips_without_reference(tmp_path) -> None:
    """Task must return skipped status when no reference dataset is available."""
    current = pl.DataFrame({"x": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "skipped"
    assert "no reference dataset" in result.summary["message"].lower()


def test_handles_all_null_column_gracefully(tmp_path) -> None:
    """
    An all-null current column must not raise an exception.

    pl.DataFrame({"x": [None, None, None]}) creates Null dtype in Polars,
    which converts to object dtype in pandas. This triggers a type mismatch
    (object vs int64) but no runtime error since no numeric operations are
    attempted on the all-null side.
    """
    current = pl.DataFrame({"x": [None, None, None]})
    reference = pl.DataFrame({"x": [1, 2, 3]})

    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # Null→object vs int64 is a type mismatch
    assert "x" in result.summary["type_mismatches"]
    # No exception-level error should have been recorded for the column
    assert "error" not in result.data["field_changes"]["x"]


def test_no_findings_on_identical_datasets(tmp_path) -> None:
    """Identical current and reference datasets must produce no flags."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=df,
        reference_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["added_columns"] == []
    assert result.summary["dropped_columns"] == []
    assert result.summary["type_mismatches"] == []
    for changes in result.data["field_changes"].values():
        assert changes.get("flag_missing_diff") is False
        assert changes.get("flag_unique_diff") is False


def test_guidance_blurbs_emitted_for_schema_changes(tmp_path) -> None:
    """EDA guidance must be attached for added, dropped, and mismatched columns."""
    current = pl.DataFrame({"new_col": [1, 2], "shared": ["a", "b"]})
    reference = pl.DataFrame({"old_col": [3, 4], "shared": [1, 2]})

    ctx, task = make_ctx_and_task(
        task_cls=CompareWithReferenceDataset,
        current_df=current,
        reference_df=reference,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None

    # Added column guidance
    assert "new_col" in result.guidance
    assert any(g["phase"] == "eda" for g in result.guidance["new_col"]["eda"])

    # Dropped column guidance
    assert "old_col" in result.guidance
    assert any(g["level"] == "warn" for g in result.guidance["old_col"]["eda"])

    # Type mismatch guidance
    assert "shared" in result.guidance
    assert any(
        "mismatch" in g["title"].lower() for g in result.guidance["shared"]["eda"]
    )
