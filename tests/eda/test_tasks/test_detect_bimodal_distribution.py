# tests/eda/test_tasks/test_detect_bimodal_distribution.py

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from dsbf.eda.tasks.detect_bimodal_distribution import DetectBimodalDistribution
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from collections.abc import Generator

    from dsbf.eda.task_result import TaskResult


def test_bimodal_column_is_flagged(tmp_path) -> None:
    """A clearly bimodal column must be flagged True in bimodal_flags."""
    rng: Generator = np.random.default_rng(42)

    x1 = rng.normal(0, 1, 100)
    x2 = rng.normal(5, 1, 100)
    df = pd.DataFrame(
        {
            "bimodal": np.concatenate([x1, x2]),
            "uniform": rng.uniform(0, 1, 200),
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        task_overrides={"bic_threshold": 5.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    assert result.data is not None

    flags = result.data["bimodal_flags"]
    scores = result.data["bic_scores"]

    assert "bimodal" in flags
    assert flags["bimodal"] is True
    assert isinstance(scores["bimodal"]["bic_1_component"], float)
    assert isinstance(scores["bimodal"]["bic_2_components"], float)
    assert scores["bimodal"]["delta"] > 0


def test_unimodal_column_not_flagged(tmp_path) -> None:
    """A clearly unimodal column must not be flagged."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"unimodal": rng.normal(5, 1, 200)})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    flags = result.data["bimodal_flags"]
    if "unimodal" in flags:
        assert flags["unimodal"] is False


def test_bic_scores_structure(tmp_path) -> None:
    """BIC scores dict must contain the expected keys for processed columns."""
    rng: Generator = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 100)
    x2 = rng.normal(5, 1, 100)
    df = pd.DataFrame({"bimodal": np.concatenate([x1, x2])})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        task_overrides={"bic_threshold": 5.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    scores = result.data["bic_scores"]
    assert "bimodal" in scores
    entry = scores["bimodal"]
    assert "bic_1_component" in entry
    assert "bic_2_components" in entry
    assert "delta" in entry
    assert "relative_improvement" in entry


def test_guidance_attached_for_flagged_column(tmp_path) -> None:
    """Bimodal guidance blurbs must be attached for flagged columns."""
    rng: Generator = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 100)
    x2 = rng.normal(5, 1, 100)
    df = pd.DataFrame({"bimodal": np.concatenate([x1, x2])})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        task_overrides={"bic_threshold": 5.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    assert result.data["bimodal_flags"].get("bimodal") is True
    assert result.guidance is not None
    assert "bimodal" in result.guidance
    assert len(result.guidance["bimodal"]["eda"]) > 0
    assert len(result.guidance["bimodal"]["ml"]) > 0


def test_skips_low_sample_column(tmp_path) -> None:
    """Columns with fewer than 10 non-null values must be skipped."""
    df = pd.DataFrame({"tiny": [1.0, 2.0, 3.0] * 3})  # 9 rows

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    assert "tiny" not in result.data["bimodal_flags"]


def test_skips_all_null_column(tmp_path) -> None:
    """Columns with all null values must be skipped without error."""
    df = pd.DataFrame({"null_col": [None] * 50})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    assert "null_col" not in result.data["bimodal_flags"]


def test_skips_constant_column(tmp_path) -> None:
    """Constant columns (zero variance) must be skipped before GMM fitting."""
    df = pd.DataFrame({"constant": [42.0] * 100})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    assert "constant" not in result.data["bimodal_flags"]


def test_no_plots_generated(tmp_path) -> None:
    """Bimodal detection task must not generate plots."""
    rng: Generator = np.random.default_rng(42)
    df = pd.DataFrame({"col": rng.normal(0, 1, 100)})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectBimodalDistribution,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, DetectBimodalDistribution)

    assert result.status == "success"
    assert result.plots is None
