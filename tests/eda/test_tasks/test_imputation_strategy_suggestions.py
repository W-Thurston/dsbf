# tests/eda/test_tasks/test_imputation_strategy_suggestions.py


from typing import Any

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.imputation_strategy_suggestions import (
    ImputationStrategySuggestions,
    _select_strategy,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

# ── Unit tests for _select_strategy ───────────────────────────────────────────


def test_severe_missingness_recommends_drop_or_indicator() -> None:
    result: dict = _select_strategy("col", 0.60, "continuous", is_skewed=False)
    assert result["strategy"] == "drop_or_indicator"
    assert result["tier"] == "severe"
    assert result["add_indicator"] is True


def test_significant_skewed_continuous_recommends_median_with_indicator() -> None:
    result: dict = _select_strategy("col", 0.30, "continuous", is_skewed=True)
    assert result["strategy"] == "median_with_indicator"
    assert result["add_indicator"] is True


def test_significant_symmetric_continuous_recommends_mean_or_knn() -> None:
    result: dict = _select_strategy("col", 0.30, "continuous", is_skewed=False)
    assert result["strategy"] == "mean_or_knn_with_indicator"
    assert result["add_indicator"] is True


def test_moderate_skewed_continuous_recommends_median() -> None:
    result: dict = _select_strategy("col", 0.10, "continuous", is_skewed=True)
    assert result["strategy"] == "median"
    assert result["add_indicator"] is False


def test_moderate_symmetric_continuous_recommends_mean() -> None:
    result: dict = _select_strategy("col", 0.10, "continuous", is_skewed=False)
    assert result["strategy"] == "mean"
    assert result["add_indicator"] is False


def test_categorical_moderate_recommends_mode_with_indicator() -> None:
    result: dict = _select_strategy("col", 0.25, "categorical", is_skewed=False)
    assert result["strategy"] == "mode_with_indicator"


def test_categorical_low_recommends_mode() -> None:
    result: dict = _select_strategy("col", 0.03, "categorical", is_skewed=False)
    assert result["strategy"] == "mode"
    assert result["add_indicator"] is False


def test_datetime_recommends_forward_fill() -> None:
    result: dict = _select_strategy("col", 0.10, "datetime", is_skewed=False)
    assert result["strategy"] == "forward_fill"


def test_low_missingness_any_method_acceptable() -> None:
    result: dict = _select_strategy("col", 0.02, "continuous", is_skewed=False)
    assert result["tier"] == "low"
    assert result["add_indicator"] is False


def test_result_always_has_required_keys() -> None:
    for intent in ("continuous", "categorical", "datetime"):
        for pct in (0.02, 0.10, 0.30, 0.60):
            result: dict = _select_strategy("col", pct, intent, is_skewed=False)
            for key in ("strategy", "method", "tier", "add_indicator", "rationale"):
                assert (
                    key in result
                ), f"Missing key '{key}' for intent={intent}, pct={pct}"


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_suggestions_generated_for_missing_columns(tmp_path) -> None:
    """Columns with missing values must receive imputation suggestions."""
    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0, None, 5.0] * 20,
            "b": ["x", "y", None, "x", "y"] * 20,
            "c": list(range(100)),  # no nulls
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ImputationStrategySuggestions)

    assert result.status == "success"
    suggestions = result.data["suggestions"]
    assert "a" in suggestions
    assert "b" in suggestions
    assert "c" not in suggestions  # no nulls → no suggestion


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_severe_missingness_flagged(tmp_path) -> None:
    """A column with >50% nulls must receive a drop_or_indicator suggestion."""
    df = pd.DataFrame({"sparse": [None] * 70 + list(range(30))})

    ctx, _ = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ImputationStrategySuggestions)

    assert result.status == "success"
    assert "sparse" in result.data["suggestions"]
    assert result.data["suggestions"]["sparse"]["strategy"] == "drop_or_indicator"
    assert result.summary["severe_count"] == 1


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_column_gets_mode_strategy(tmp_path) -> None:
    """A categorical column with moderate nulls must get mode imputation."""
    vals: list[str | None] = ["active", "inactive", "pending", None] * 25
    df = pd.DataFrame({"status": vals})

    ctx, task = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"status": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    strategy = result.data["suggestions"]["status"]["strategy"]
    assert "mode" in strategy


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_skewed_column_gets_median_strategy(tmp_path) -> None:
    """A skewed continuous column must prefer median over mean."""
    # Highly right-skewed: most values near 1, a few extreme outliers
    vals: list[float | None] = (
        [1.0] * 70
        + [None] * 15
        + [
            100.0,
            200.0,
            500.0,
            1000.0,
            2000.0,
            5000.0,
            10000.0,
            20000.0,
            50000.0,
            100000.0,
            200000.0,
            500000.0,
            1000000.0,
            2000000.0,
            5000000.0,
        ]
    )
    df = pd.DataFrame({"income": vals})

    ctx, task = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"income": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    strategy = result.data["suggestions"]["income"]["strategy"]
    assert "median" in strategy


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_reads_null_percentages_from_summarize_nulls_context(tmp_path) -> None:
    """Task must read null percentages from summarize_nulls result when available."""
    df = pd.DataFrame({"x": [1.0, None, 3.0] * 20})

    ctx, task = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )

    # Inject a summarize_nulls result directly into context
    mock_nulls = TaskResult(
        name="summarize_nulls",
        status="success",
        summary={"message": "mock"},
        data={
            "null_percentages": {"x": 0.35},
            "null_counts": {"x": 21},
            "high_null_columns": [],
            "null_patterns": {},
        },
    )
    ctx.results["summarize_nulls"] = mock_nulls
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # 35% null → significant tier → mean_or_knn_with_indicator (symmetric)
    assert "x" in result.data["suggestions"]
    assert result.data["suggestions"]["x"]["null_pct"] == 0.35


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_min_null_pct_threshold_respected(tmp_path) -> None:
    """Columns below min_null_pct must not receive suggestions."""
    # 1/100 = 1% null - below default 1% threshold only if exactly 0
    df = pd.DataFrame({"almost_clean": [None, *list(range(999))]})

    ctx, task = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        task_overrides={"min_null_pct": 0.05},  # 5% threshold
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # 1/1000 = 0.1% < 5% → no suggestion
    assert "almost_clean" not in result.data["suggestions"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_attached_for_each_suggestion(tmp_path) -> None:
    """ML guidance must be attached for each column with a suggestion."""
    df = pd.DataFrame({"a": [None] * 30 + list(range(70))})

    ctx, _ = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ImputationStrategySuggestions)

    assert result.status == "success"
    assert result.guidance is not None
    assert "a" in result.guidance
    assert len(result.guidance["a"]["ml"]) > 0
    assert len(result.guidance["a"]["ml"][0]["actions"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_severe_column_guidance_is_warn_level(tmp_path) -> None:
    """Severe missingness guidance must be emitted at warn level."""
    df = pd.DataFrame({"sparse": [None] * 60 + list(range(40))})

    ctx, _ = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ImputationStrategySuggestions)

    assert result.status == "success"
    guidance: dict[str, Any] = result.guidance["sparse"]["ml"][0]
    assert guidance["level"] == "warn"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_clean_dataset_produces_no_suggestions(tmp_path) -> None:
    """A completely non-null dataset must produce no suggestions."""
    df = pd.DataFrame({"a": [1, 2, 3] * 20, "b": ["x", "y", "z"] * 20})

    ctx, _ = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ImputationStrategySuggestions)

    assert result.status == "success"
    assert result.data["suggestions"] == {}
    assert result.summary["suggestion_count"] == 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    df = pl.DataFrame({"x": [1.0, None, 3.0, None, 5.0] * 20})

    ctx, _ = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ImputationStrategySuggestions)

    assert result.status == "success"
    assert "x" in result.data["suggestions"]


def test_no_plots_generated(tmp_path) -> None:
    """Imputation suggestion task must not generate plots."""
    df = pd.DataFrame({"a": [1.0, None, 3.0] * 20})

    ctx, _ = make_ctx_and_task(
        task_cls=ImputationStrategySuggestions,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ImputationStrategySuggestions)

    assert result.status == "success"
    assert result.plots is None
