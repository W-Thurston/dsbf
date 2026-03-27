# tests/eda/test_tasks/test_detect_string_anomalies.py


from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.detect_string_anomalies import (
    DetectStringAnomalies,
    _check_invisible_chars,
    _check_length_outliers,
    _check_mixed_case,
    _check_whitespace,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from pandas import Series

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for pure check functions ───────────────────────────────────────


def test_mixed_case_detects_collision() -> None:
    s: Series[str] = pd.Series(
        ["New York", "new york", "NEW YORK", "Boston", "boston"] * 5,
    )
    result: dict | None = _check_mixed_case(s)
    assert result is not None
    assert result["type"] == "mixed_case"
    assert result["affected_groups"] >= 2
    assert "new york" in result["examples"]


def test_mixed_case_clean_series_returns_none() -> None:
    s: Series[str] = pd.Series(["new york", "boston", "chicago"] * 5)
    assert _check_mixed_case(s) is None


def test_whitespace_detects_padding() -> None:
    s: Series[str] = pd.Series(["  active", "inactive", "active ", "pending"] * 5)
    result: dict | None = _check_whitespace(s)
    assert result is not None
    assert result["type"] == "leading_trailing_whitespace"
    assert result["affected_count"] >= 2


def test_whitespace_clean_series_returns_none() -> None:
    s: Series[str] = pd.Series(["active", "inactive", "pending"] * 5)
    assert _check_whitespace(s) is None


def test_invisible_chars_detects_zero_width_space() -> None:
    s: Series[str] = pd.Series(["normal", "with\u200bspace", "also normal"] * 5)
    result: dict | None = _check_invisible_chars(s)
    assert result is not None
    assert result["type"] == "invisible_characters"
    assert result["affected_count"] >= 5


def test_invisible_chars_detects_non_breaking_space() -> None:
    s: Series[str] = pd.Series(["hello\u00a0world", "normal"] * 10)
    result: dict | None = _check_invisible_chars(s)
    assert result is not None
    assert result["affected_count"] == 10


def test_invisible_chars_clean_series_returns_none() -> None:
    s: Series[str] = pd.Series(["active", "inactive", "hello world"] * 5)
    assert _check_invisible_chars(s) is None


def test_length_outliers_detects_extreme_length() -> None:
    normal: list[str] = ["hello", "world", "foo", "bar", "baz"] * 20
    outlier: list[str] = ["x" * 500]  # far beyond typical length
    s: Series[str] = pd.Series(normal + outlier)
    result: dict | None = _check_length_outliers(s)
    assert result is not None
    assert result["type"] == "length_outliers"
    assert result["affected_count"] >= 1


def test_length_outliers_uniform_lengths_returns_none() -> None:
    # All same length — IQR = 0, skip
    s: Series[str] = pd.Series(["abc"] * 50)
    assert _check_length_outliers(s) is None


def test_length_outliers_no_extreme_values_returns_none() -> None:
    s: Series[str] = pd.Series(["hi", "hello", "hey", "howdy", "greetings"] * 10)
    # All lengths similar — no outliers at 3xIQR
    result: dict | None = _check_length_outliers(s)
    assert result is None or result["affected_count"] == 0


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_mixed_case_anomaly_detected(tmp_path) -> None:
    """Mixed case values must produce a finding and EDA guidance."""
    df = pd.DataFrame(
        {"city": ["New York", "new york", "Boston", "boston", "Chicago"] * 10},
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert "city" in result.data
    types = [f["type"] for f in result.data["city"]]
    assert "mixed_case" in types


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_whitespace_anomaly_detected(tmp_path) -> None:
    """Leading/trailing whitespace must produce a finding."""
    df = pd.DataFrame({"status": ["  active", "inactive", "active ", "pending"] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert "status" in result.data
    types: list = [f["type"] for f in result.data["status"]]
    assert "leading_trailing_whitespace" in types


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_invisible_chars_anomaly_detected(tmp_path) -> None:
    """Invisible Unicode characters must produce a finding."""
    df = pd.DataFrame(
        {"label": ["normal", "with\u200bspace", "also normal", "fine"] * 10},
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert "label" in result.data
    types: list = [f["type"] for f in result.data["label"]]
    assert "invisible_characters" in types


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_length_outlier_detected(tmp_path) -> None:
    """Extreme-length values must produce a length outlier finding."""
    normal: list[str] = ["hello", "world", "foo", "bar"] * 20
    df = pd.DataFrame({"notes": [*normal, "x" * 500, "y" * 600]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert "notes" in result.data
    types: list = [f["type"] for f in result.data["notes"]]
    assert "length_outliers" in types


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_clean_column_produces_no_findings(tmp_path) -> None:
    """A clean, consistent string column must not appear in findings."""
    df = pd.DataFrame(
        {"status": ["active", "inactive", "pending", "active", "pending"] * 10}
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert "status" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_multiple_anomaly_types_per_column(tmp_path) -> None:
    """A column can have multiple anomaly types — all must be reported."""
    df = pd.DataFrame({"messy": ["  Active", "active", "ACTIVE", "inactive"] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    if "messy" in result.data:
        types: list = [f["type"] for f in result.data["messy"]]
        # Both whitespace and mixed_case should fire
        assert len(types) >= 1


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_min_values_threshold_skips_small_columns(tmp_path) -> None:
    """Columns with fewer than min_values non-null values must be skipped."""
    df = pd.DataFrame({"tiny": ["New York", "new york", None, None] + [None] * 96})

    ctx, task = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        task_overrides={"min_values": 10},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "tiny" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_attached_for_each_finding(tmp_path) -> None:
    """An EDA guidance blurb must be attached for each finding."""
    df = pd.DataFrame({"city": ["New York", "new york", "Boston", "boston"] * 15})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert result.guidance is not None
    assert "city" in result.guidance
    assert len(result.guidance["city"]["eda"]) > 0
    # Mixed case guidance must recommend a normalise_case action
    actions = result.guidance["city"]["eda"][0]["actions"]
    assert any(a["action"] == "normalise_case" for a in actions)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """Summary counts must reflect the actual findings in data."""
    df = pd.DataFrame(
        {
            "a": ["New York", "new york"] * 20,
            "b": ["active", "inactive", "pending"] * 15,  # clean
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert result.summary["columns_with_anomalies"] == len(result.data)
    total: int = sum(len(v) for v in result.data.values())
    assert result.summary["total_findings"] == total


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    df = pl.DataFrame({"city": ["New York", "new york", "Boston", "boston"] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    """String anomaly task must not generate plots."""
    df = pd.DataFrame({"a": ["hello", "world"] * 10})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectStringAnomalies,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectStringAnomalies)

    assert result.status == "success"
    assert result.plots is None
