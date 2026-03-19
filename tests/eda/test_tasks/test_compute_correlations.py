# tests/eda/test_tasks/test_compute_correlations.py

import warnings

import pandas as pd
import pytest

from dsbf.eda.task_result import TaskResult, WarningDetail
from dsbf.eda.tasks.compute_correlations import ComputeCorrelations
from tests.helpers.context_utils import make_ctx_and_task


def _extract_warning(result: TaskResult, level: str, code: str) -> WarningDetail | None:
    """Helper: extract a specific reliability warning from a TaskResult."""
    return (
        result.reliability_warnings.get(level, {}).get(code)
        if result.reliability_warnings
        else None
    )


def test_pearson_and_cramers_v_computed(tmp_path) -> None:
    """Pearson correlation for numeric pairs and Cramér's V for categorical pairs
    must both be present in output data."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],  # perfect Pearson
            "cat1": ["a", "a", "b", "b", "b"],
            "cat2": ["yes", "yes", "no", "no", "no"],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None

    # Pearson
    assert "x|y" in result.data
    assert abs(result.data["x|y"] - 1.0) < 1e-6

    # Cramér's V
    assert "cat1|cat2" in result.data
    v = result.data["cat1|cat2"]
    assert isinstance(v, float)
    assert 0.0 <= v <= 1.0


def test_all_numeric_pairs_present(tmp_path) -> None:
    """All N*(N-1)/2 numeric pairs must appear in output for a numeric-only dataset."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [2, 3, 4, 5, 6],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "a|b" in result.data
    assert "a|c" in result.data
    assert "b|c" in result.data
    assert all(isinstance(v, float) for v in result.data.values())


def test_warns_on_low_row_count(tmp_path) -> None:
    """A strong_warning must be emitted when N < 30."""
    df = pd.DataFrame({"a": [1, 2], "b": [2, 4]})

    ctx, task = make_ctx_and_task(
        ComputeCorrelations,
        df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "strong_warning", "low_row_count")


def test_warns_on_zero_variance(tmp_path) -> None:
    """A strong_warning must be emitted when a column has near-zero variance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = pd.DataFrame({"a": [1, 1, 1, 1, 1], "b": [2, 3, 4, 5, 6]})
        ctx, task = make_ctx_and_task(
            ComputeCorrelations,
            df,
            global_overrides={"output_dir": str(tmp_path)},
        )
        result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "strong_warning", "zero_variance")


def test_warns_on_extreme_outliers(tmp_path) -> None:
    """A heuristic_caution warning must be emitted when |z| > 3 is detected."""
    df = pd.DataFrame({"a": [1] * 29 + [1000], "b": list(range(30))})

    ctx, task = make_ctx_and_task(
        ComputeCorrelations,
        df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "heuristic_caution", "extreme_outliers")


def test_warns_on_high_skew(tmp_path) -> None:
    """A heuristic_caution warning must be emitted when skewness is high."""
    df = pd.DataFrame(
        {
            "a": [*list(range(1, 25)), 500, 1000, 2000, 3000, 4000, 5000],
            "b": list(range(30)),
        }
    )

    ctx, task = make_ctx_and_task(
        ComputeCorrelations,
        df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "heuristic_caution", "high_skew")


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_no_warnings_on_clean_data(tmp_path) -> None:
    """No reliability warnings should be emitted on clean, large, symmetric data."""
    df = pd.DataFrame({"a": list(range(1, 31)), "b": [x * 2 for x in range(1, 31)]})

    ctx, task = make_ctx_and_task(
        ComputeCorrelations,
        df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    w: dict[str, dict[str, WarningDetail]] | None = result.reliability_warnings
    assert not w or all(not v for v in w.values())


def test_warns_combined_high_skew_low_n(tmp_path) -> None:
    """When both skew and low-N are present the combined warning code must be used."""
    df = pd.DataFrame({"a": [0, 0, 0, 0, 1e6], "b": [1, 2, 3, 4, 5]})

    ctx, task = make_ctx_and_task(
        ComputeCorrelations,
        df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "heuristic_caution", "high_skew_low_n")


def test_warns_combined_outliers_low_n(tmp_path) -> None:
    """When both outliers and low-N are present a combined warning code must be used."""
    df = pd.DataFrame({"a": [0, 0, 0, 0, 1e6], "b": [1, 2, 3, 4, 5]})

    ctx, task = make_ctx_and_task(
        ComputeCorrelations,
        df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "heuristic_caution", "extreme_outliers_low_n")


def test_empty_dataframe(tmp_path) -> None:
    """An empty DataFrame must return a success result with empty data."""
    df = pd.DataFrame()

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data == {}


def test_only_categorical_columns_returns_empty(tmp_path) -> None:
    """
    Known limitation: when no numeric columns are present the task returns early
    before computing Cramér's V for categorical pairs.

    TODO: Fix the early-return guard in compute_correlations to allow Cramér's V
    to run independently of numeric column presence.

    """
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["foo", "bar", "baz"]})

    ctx, task = make_ctx_and_task(
        task_cls=ComputeCorrelations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # Current behavior: empty — Cramér's V skipped due to early return.
    # Once the early-return is fixed this assertion should become:
    # assert "a|b" in result.data
    assert result.data == {}
