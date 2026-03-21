# tests/eda/test_tasks/test_compute_kurtosis.py

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import kurtosis as scipy_kurtosis

from dsbf.eda.tasks.compute_kurtosis import (
    ComputeKurtosis,
    _classify_kurtosis,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for _classify_kurtosis ─────────────────────────────────────────


def test_classify_strongly_leptokurtic() -> None:
    assert _classify_kurtosis(3.0) == "strongly_leptokurtic"
    assert _classify_kurtosis(10.0) == "strongly_leptokurtic"


def test_classify_mildly_leptokurtic() -> None:
    assert _classify_kurtosis(1.0) == "mildly_leptokurtic"
    assert _classify_kurtosis(2.9) == "mildly_leptokurtic"


def test_classify_mesokurtic() -> None:
    assert _classify_kurtosis(0.0) == "mesokurtic"
    assert _classify_kurtosis(0.5) == "mesokurtic"
    assert _classify_kurtosis(-0.9) == "mesokurtic"


def test_classify_platykurtic() -> None:
    assert _classify_kurtosis(-1.0) == "platykurtic"
    assert _classify_kurtosis(-2.0) == "platykurtic"


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_kurtosis_computed_for_numeric_columns(tmp_path) -> None:
    """Kurtosis must be computed for all continuous numeric columns."""
    df = pd.DataFrame(
        {
            "a": list(range(1, 101)),
            "b": [float(x) for x in range(1, 101)],
        },
    )
    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert "a" in result.data
    assert "b" in result.data
    assert "kurtosis" in result.data["a"]
    assert "classification" in result.data["a"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_normal_distribution_near_zero_kurtosis(tmp_path) -> None:
    """A normally distributed sample must have excess kurtosis close to 0."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal": rng.normal(0, 1, 10_000)})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    k = result.data["normal"]["kurtosis"]
    assert abs(k) < 0.5  # large sample — excess kurtosis should be near 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_fat_tailed_distribution_high_kurtosis(tmp_path) -> None:
    """A fat-tailed distribution must produce strongly leptokurtic classification."""
    # t-distribution with df=3 has very heavy tails (excess kurtosis = 6)
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"heavy": rng.standard_t(df=3, size=5_000)})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert result.data["heavy"]["classification"] == "strongly_leptokurtic"
    assert result.data["heavy"]["kurtosis"] > 3.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_uniform_distribution_negative_kurtosis(tmp_path) -> None:
    """A uniform distribution produces platykurtic classification (kurtosis ≈ -1.2)."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"uniform": rng.uniform(0, 1, 10_000)})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert result.data["uniform"]["classification"] == "platykurtic"
    assert result.data["uniform"]["kurtosis"] < -1.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_kurtosis_values_match_scipy_directly(tmp_path) -> None:
    """Kurtosis values must match scipy.stats.kurtosis(fisher=True, bias=False)."""
    rng: Generator = np.random.default_rng(0)
    data = rng.exponential(scale=2.0, size=500)
    df = pd.DataFrame({"exp": data})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    expected: float = round(float(scipy_kurtosis(data, fisher=True, bias=False)), 4)
    assert result.status == "success"
    assert abs(result.data["exp"]["kurtosis"] - expected) < 1e-4


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_column_with_fewer_than_4_values_skipped(tmp_path) -> None:
    """Columns with fewer than 4 non-null values must be skipped without error."""
    df = pd.DataFrame(
        {
            "tiny": [1.0, 2.0, 3.0]
            + [None] * 97,  # 3 non-null values — below threshold
            "normal": list(range(100)),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert "tiny" not in result.data
    assert "normal" in result.data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_leptokurtic_column(tmp_path) -> None:
    """EDA and ML guidance must be emitted for strongly leptokurtic columns."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"heavy": rng.standard_t(df=3, size=2_000)})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert result.guidance is not None
    assert "heavy" in result.guidance
    assert len(result.guidance["heavy"]["eda"]) > 0
    assert len(result.guidance["heavy"]["ml"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_platykurtic_column(tmp_path) -> None:
    """EDA guidance must be emitted for platykurtic columns."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"uniform": rng.uniform(0, 1, 5_000)})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert result.guidance is not None
    assert "uniform" in result.guidance
    assert len(result.guidance["uniform"]["eda"]) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_guidance_for_mesokurtic_column(tmp_path) -> None:
    """No guidance must be emitted for mesokurtic (normal-like) columns."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"normal": rng.normal(0, 1, 10_000)})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    # Normal distribution should be classified mesokurtic — no guidance
    classification = result.data["normal"]["classification"]
    if classification == "mesokurtic":
        assert result.guidance is None or "normal" not in (result.guidance or {})


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_notable_count_correct(tmp_path) -> None:
    """notable_count in summary must equal cols with non-mesokurtic classification."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "heavy": rng.standard_t(df=3, size=2_000),  # strongly leptokurtic
            "uniform": rng.uniform(0, 1, 2_000),  # platykurtic
            "normal": rng.normal(0, 1, 2_000),  # mesokurtic
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    notable_in_data: int = sum(
        1 for v in result.data.values() if v["classification"] != "mesokurtic"
    )
    assert result.summary["notable_count"] == notable_in_data


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    df = pl.DataFrame({"x": list(range(1, 101))})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert "x" in result.data


def test_no_plots_generated(tmp_path) -> None:
    """Kurtosis task must not generate plots."""
    df = pd.DataFrame({"a": list(range(100))})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeKurtosis,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeKurtosis)

    assert result.status == "success"
    assert result.plots is None
