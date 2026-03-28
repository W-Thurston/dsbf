# tests/eda/test_tasks/test_compute_pairwise_associations.py

import warnings
from typing import Literal

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.compute_pairwise_associations import (
    ComputePairwiseAssociations,
    _kendalls_tau,
    _pearson_r,
    _spearman_r,
    _strength,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

# ── Helper ────────────────────────────────────────────────────────────────────


def _extract_warning(result: TaskResult, level: str, code: str):
    return (
        result.reliability_warnings.get(level, {}).get(code)
        if result.reliability_warnings
        else None
    )


# ── Unit tests for pure helper functions ──────────────────────────────────────


def test_pearson_r_perfect_positive() -> None:
    a: pd.Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b: pd.Series = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    assert abs(_pearson_r(a, b) - 1.0) < 1e-6


def test_spearman_r_monotonic_non_linear() -> None:
    """Spearman must detect a monotonic non-linear relationship that Pearson misses."""
    a: pd.Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b: pd.Series = pd.Series(
        [1.0, 4.0, 9.0, 16.0, 25.0]
    )  # y = x², perfect Spearman, non-linear Pearson
    sp: float | None = _spearman_r(a, b)
    pe: float | None = _pearson_r(a, b)
    assert sp is not None
    assert abs(sp - 1.0) < 1e-6  # perfect monotonic
    assert pe is not None
    assert pe < 1.0  # Pearson is not 1.0 for non-linear


def test_spearman_r_returns_none_for_small_sample() -> None:
    a: pd.Series = pd.Series([1.0, 2.0])
    b: pd.Series = pd.Series([2.0, 4.0])
    assert _spearman_r(a, b) is None


def test_kendalls_tau_perfect_monotonic() -> None:
    a: pd.Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b: pd.Series = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    result: float | None = _kendalls_tau(a, b)
    assert result is not None
    assert abs(result - 1.0) < 1e-6


def test_kendalls_tau_returns_none_for_small_sample() -> None:
    a: pd.Series = pd.Series([1.0, 2.0])
    b: pd.Series = pd.Series([2.0, 4.0])
    assert _kendalls_tau(a, b) is None


def test_kendalls_tau_negative_monotonic() -> None:
    a: pd.Series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b: pd.Series = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
    result: float | None = _kendalls_tau(a, b)
    assert result is not None
    assert abs(result - (-1.0)) < 1e-6


def test_strength_labels() -> None:
    assert _strength(0.8, "pearson_r") == "strong"
    assert _strength(0.5, "pearson_r") == "moderate"
    assert _strength(0.3, "pearson_r") == "weak"
    assert _strength(0.1, "pearson_r") == "negligible"
    assert _strength(0.6, "cramers_v") == "strong"
    assert _strength(0.4, "cramers_v") == "moderate"
    assert _strength(0.2, "cramers_v") == "weak"
    assert _strength(0.05, "cramers_v") == "negligible"
    assert _strength(0.15, "eta_squared") == "strong"
    assert _strength(0.07, "eta_squared") == "moderate"
    assert _strength(0.02, "eta_squared") == "weak"
    assert _strength(0.005, "eta_squared") == "negligible"
    assert _strength(0.8, "spearman_r") == "strong"
    assert _strength(0.8, "kendalls_tau") == "strong"
    assert _strength(0.3, "kendalls_tau") == "weak"
    assert _strength(0.1, "kendalls_tau") == "negligible"


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_pearson_r_for_continuous_pair(tmp_path) -> None:
    """Pearson r must be computed for two continuous columns."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "y": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert "x|y" in result.data
    entry = result.data["x|y"]
    assert entry["metric_type"] == "pearson_r"
    assert abs(entry["metric"] - 1.0) < 1e-4
    assert entry["strength"] == "strong"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_spearman_r_computed_when_method_spearman(tmp_path) -> None:
    """Spearman r must be used for continuous pairs when method='spearman'."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "y": [1.0, 4.0, 9.0, 16.0, 25.0] * 10,  # y=x², perfect Spearman
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "spearman"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x|y" in result.data
    entry = result.data["x|y"]
    assert entry["metric_type"] == "spearman_r"
    assert abs(entry["metric"] - 1.0) < 1e-4


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_both_method_stores_pearson_and_spearman(tmp_path) -> None:
    """method='both' must store Pearson as primary and Spearman as secondary."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "y": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "both"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["x|y"]
    assert entry["metric_type"] == "pearson_r"
    assert "spearman_r" in entry
    assert abs(entry["spearman_r"] - 1.0) < 1e-4


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_spearman_detects_non_linear_relationship(tmp_path) -> None:
    """Spearman must score > than Pearson for a non-linear monotonic relationship."""
    df = pd.DataFrame(
        {
            "x": list(range(1, 51)),
            "y": [i**2 for i in range(1, 51)],
        },
    )

    ctx_sp, task_sp = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "spearman", "min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx_sp.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result_sp: TaskResult = ctx_sp.run_task(task_sp)

    ctx_pe, task_pe = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "pearson", "min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx_pe.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result_pe: TaskResult = ctx_pe.run_task(task_pe)

    sp_val = result_sp.data["x|y"]["metric"]
    pe_val = result_pe.data["x|y"]["metric"]
    assert abs(sp_val) > abs(pe_val)  # Spearman closer to 1.0 for y=x²


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_cramers_v_for_categorical_pair(tmp_path) -> None:
    """Cramér's V must be computed for two categorical columns."""
    df = pd.DataFrame(
        {
            "color": ["red", "red", "blue", "blue", "green"] * 10,
            "shape": ["circle", "circle", "square", "square", "triangle"] * 10,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert "color|shape" in result.data
    entry = result.data["color|shape"]
    assert entry["metric_type"] == "cramers_v"
    assert 0.0 <= entry["metric"] <= 1.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_eta_squared_for_continuous_categorical_pair(tmp_path) -> None:
    """Eta squared must be computed for continuous x multi-level categorical pairs."""
    df = pd.DataFrame(
        {
            "score": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
            "group": ["A", "A", "B", "B", "C"] * 10,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    key: Literal["score|group", "group|score"] = (
        "score|group" if "score|group" in result.data else "group|score"
    )
    assert key in result.data
    assert result.data[key]["metric_type"] == "eta_squared"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_id_and_datetime_columns_skipped(tmp_path) -> None:
    """Columns typed as id or datetime must not appear in associations."""
    df = pd.DataFrame(
        {"id_col": range(50), "value": range(50), "label": ["a", "b"] * 25},
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "id_col": "id",
            "value": "continuous",
            "label": "categorical",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    for key in result.data:
        if key == "__correlation_matrix__":
            continue
        col_a, col_b = key.split("|")
        assert col_a != "id_col"
        assert col_b != "id_col"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correlation_matrix_present_in_data(tmp_path) -> None:
    """__correlation_matrix__ must be present in data when numeric pairs exist."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "b": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
            "c": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert "__correlation_matrix__" in result.data
    matrix = result.data["__correlation_matrix__"]
    assert "a" in matrix
    assert "b" in matrix["a"]
    # Diagonal must be 1.0
    assert abs(matrix["a"]["a"] - 1.0) < 1e-6


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_correlation_matrix_absent_for_no_numeric_pairs(tmp_path) -> None:
    """__correlation_matrix__ must be absent when there are no numeric pairs."""
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "green"] * 10,
            "shape": ["circle", "square", "triangle"] * 10,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert "__correlation_matrix__" not in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_strength_labels_present(tmp_path) -> None:
    """Every association entry must have a strength label."""
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10, "b": [5.0, 4.0, 3.0, 2.0, 1.0] * 10},
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    valid_strengths: set[str] = {"strong", "moderate", "weak", "negligible"}
    for key, entry in result.data.items():
        if key == "__correlation_matrix__":
            continue
        assert entry["strength"] in valid_strengths


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_match_data(tmp_path) -> None:
    """Summary pair_count must equal the num of association entries (excl. matrix)."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "y": [2.0, 4.0, 6.0, 8.0, 10.0] * 10,
            "z": [5.0, 4.0, 3.0, 2.0, 1.0] * 10,
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    pair_entries: dict = {
        k: v for k, v in result.data.items() if k != "__correlation_matrix__"
    }
    assert result.summary["pair_count"] == len(pair_entries)
    total_by_strength: int = sum(result.summary["strength_counts"].values())
    assert total_by_strength == len(pair_entries)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames by converting to pandas internally."""
    df = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10, "b": [2.0, 4.0, 6.0, 8.0, 10.0] * 10},
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert len(result.data) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_reliability_warning_on_low_n(tmp_path) -> None:
    """A strong_warning must be emitted when N < 30."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert _extract_warning(result, "strong_warning", "low_row_count")


def test_reliability_warning_on_zero_variance(tmp_path) -> None:
    """A strong_warning must be emitted when a column has near-zero variance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = pd.DataFrame({"a": [1, 1, 1, 1, 1], "b": [2, 3, 4, 5, 6]})
        ctx, task = make_ctx_and_task(
            task_cls=ComputePairwiseAssociations,
            current_df=df,
            task_overrides={"min_sample_size": 2},
            global_overrides={"output_dir": str(tmp_path)},
        )
        result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "strong_warning", "zero_variance")


def test_reliability_warning_on_extreme_outliers(tmp_path) -> None:
    """A heuristic_caution warning must be emitted when |z| > 3 is detected."""
    df = pd.DataFrame({"a": [1] * 29 + [1000], "b": list(range(30))})
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "heuristic_caution", "extreme_outliers")


def test_reliability_warning_on_high_skew(tmp_path) -> None:
    """A heuristic_caution warning must be emitted when skewness is high."""
    df = pd.DataFrame(
        {
            "a": [*list(range(1, 25)), 500, 1000, 2000, 3000, 4000, 5000],
            "b": list(range(30)),
        },
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "heuristic_caution", "high_skew")


def test_combined_skew_low_n_warning(tmp_path) -> None:
    """Combined high_skew_low_n code must fire when both conditions are present."""
    df = pd.DataFrame({"a": [0, 0, 0, 0, 1e6], "b": [1, 2, 3, 4, 5]})
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert _extract_warning(result, "heuristic_caution", "high_skew_low_n")


def test_no_warnings_on_clean_data(tmp_path) -> None:
    """No reliability warnings must be emitted on clean, large, symmetric data."""
    df = pd.DataFrame({"a": list(range(1, 31)), "b": [x * 2 for x in range(1, 31)]})
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    w = result.reliability_warnings
    assert not w or all(not v for v in w.values())


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_auto_routes_to_tau_for_small_samples(tmp_path) -> None:
    """method='auto' must use Kendall's tau when n_valid_pairs < 30."""
    df = pd.DataFrame(
        {
            "x": list(range(1, 16)),
            "y": list(range(1, 16)),
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "auto", "min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "x|y" in result.data
    assert result.data["x|y"]["metric_type"] == "kendalls_tau"
    assert abs(result.data["x|y"]["metric"] - 1.0) < 1e-4


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_auto_routes_to_pearson_for_large_samples(tmp_path) -> None:
    """method='auto' must use Pearson when n_valid_pairs >= 30."""
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, 51)],
            "y": [float(i) * 2 for i in range(1, 51)],
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "auto", "min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["x|y"]["metric_type"] == "pearson_r"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_method_kendall_always_uses_tau(tmp_path) -> None:
    """method='kendall' must use Kendall's tau regardless of sample size."""
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, 51)],
            "y": [float(i) * 2 for i in range(1, 51)],
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "kendall", "min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["x|y"]["metric_type"] == "kendalls_tau"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_tau_in_correlation_matrix(tmp_path) -> None:
    """Kendall's tau values must be included in the correlation matrix."""
    df = pd.DataFrame(
        {
            "x": list(range(1, 16)),
            "y": list(range(1, 16)),
            "z": list(range(15, 0, -1)),
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "auto", "min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types", {"x": "continuous", "y": "continuous", "z": "continuous"}
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "__correlation_matrix__" in result.data
    matrix = result.data["__correlation_matrix__"]
    # tau values must populate the matrix for small-sample pairs
    assert "x" in matrix
    assert "y" in matrix["x"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_tau_negative_correlation(tmp_path) -> None:
    """Kendall's tau must produce negative values for inverse relationships."""
    df = pd.DataFrame(
        {
            "x": list(range(1, 16)),
            "y": list(range(15, 0, -1)),
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"method": "kendall", "min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["x|y"]["metric"] < 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_auto_default_is_used_when_no_method_override(tmp_path) -> None:
    """When no method is configured, 'auto' must be the default."""
    df = pd.DataFrame(
        {
            "x": list(range(1, 16)),
            "y": list(range(1, 16)),
        }
    )
    ctx, task = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        task_overrides={"min_sample_size": 2},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"x": "continuous", "y": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # n=15 < 30 → auto should route to tau
    assert result.data["x|y"]["metric_type"] == "kendalls_tau"
    assert result.metadata["method"] == "auto"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_plots_generated(tmp_path) -> None:
    """Pairwise associations task must not generate plots."""
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10, "b": [2.0, 4.0, 6.0, 8.0, 10.0] * 10},
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputePairwiseAssociations,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputePairwiseAssociations)

    assert result.status == "success"
    assert result.plots is None
