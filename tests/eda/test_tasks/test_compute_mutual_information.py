# tests/eda/test_tasks/test_compute_mutual_information.py

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.compute_mutual_information import (
    ComputeMutualInformation,
    _strength_label,
)
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult

# ── Unit tests for _strength_label ────────────────────────────────────────────


def test_strength_high() -> None:
    assert _strength_label(0.20) == "high"
    assert _strength_label(0.15) == "high"


def test_strength_moderate() -> None:
    assert _strength_label(0.10) == "moderate"
    assert _strength_label(0.05) == "moderate"


def test_strength_low() -> None:
    assert _strength_label(0.03) == "low"
    assert _strength_label(0.01) == "low"


def test_strength_negligible() -> None:
    assert _strength_label(0.005) == "negligible"
    assert _strength_label(0.0) == "negligible"


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_target_column_returns_empty(tmp_path) -> None:
    """Without a target_column config, task must succeed with empty mi_scores."""
    df = pd.DataFrame({"a": range(100), "b": range(100)})

    ctx, _ = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, ComputeMutualInformation)

    assert result.status == "success"
    assert result.data["mi_scores"] == {}
    assert result.summary["feature_count"] == 0
    assert (
        "target_column" in result.summary["message"].lower()
        or "no target" in result.summary["message"].lower()
    )


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_continuous_target_computes_mi(tmp_path) -> None:
    """MI must be computed for all features against a continuous target."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 300)
    df = pd.DataFrame(
        {
            "feature_a": x,
            "feature_b": rng.normal(0, 1, 300),
            "target": x * 2 + rng.normal(0, 0.1, 300),  # strongly correlated
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "feature_a": "continuous",
            "feature_b": "continuous",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    scores = result.data["mi_scores"]
    assert "feature_a" in scores
    assert "feature_b" in scores
    assert "target" not in scores  # target excluded from features
    # feature_a should have higher MI than random feature_b
    assert scores["feature_a"]["mi_score"] > scores["feature_b"]["mi_score"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_target_uses_classif(tmp_path) -> None:
    """A categorical target must use mutual_info_classif."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    # Binary target: 1 if x > 0, else 0
    target = (x > 0).astype(int).astype(str)
    df = pd.DataFrame({"feature": x, "label": target})

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "label"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "feature": "continuous",
            "label": "categorical",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "feature" in result.data["mi_scores"]
    # Feature perfectly separates the classes - must have high MI
    assert result.data["mi_scores"]["feature"]["mi_score"] > 0.0
    assert result.metadata["target_intent"] == "categorical"


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_random_feature_has_low_mi(tmp_path) -> None:
    """A feature unrelated to the target must have near-zero MI."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "random": rng.normal(0, 1, n),
            "target": rng.normal(0, 1, n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "random": "continuous",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["mi_scores"]["random"]["strength"] in ("negligible", "low")


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_mi_scores_sorted_descending(tmp_path) -> None:
    """mi_scores must be sorted by MI score in descending order."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "strong": x,
            "weak": rng.normal(0, 1, n),
            "target": x + rng.normal(0, 0.1, n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "strong": "continuous",
            "weak": "continuous",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    scores = list(result.data["mi_scores"].values())
    for i in range(len(scores) - 1):
        assert scores[i]["mi_score"] >= scores[i + 1]["mi_score"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_normalised_score_stored(tmp_path) -> None:
    """Each entry must contain both mi_score and mi_normalised."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 200),
            "target": rng.normal(0, 1, 200),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous", "target": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    entry = result.data["mi_scores"]["a"]
    assert "mi_score" in entry
    assert "mi_normalised" in entry
    assert "strength" in entry
    assert "is_discrete" in entry
    # mi_normalised must be <= mi_score / log(2) approximately (non-negative)
    assert entry["mi_score"] >= 0.0
    assert entry["mi_normalised"] >= 0.0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_categorical_feature_marked_discrete(tmp_path) -> None:
    """Categorical features must be flagged as discrete in the result."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "cat_feature": ["A", "B", "C"] * (n // 3) + ["A"] * (n % 3),
            "target": rng.normal(0, 1, n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "cat_feature": "categorical",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.data["mi_scores"]["cat_feature"]["is_discrete"] is True


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_id_columns_excluded(tmp_path) -> None:
    """ID and datetime columns must not appear in MI scores."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "id": range(n),
            "feature": rng.normal(0, 1, n),
            "target": rng.normal(0, 1, n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "id": "id",
            "feature": "continuous",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "id" not in result.data["mi_scores"]
    assert "feature" in result.data["mi_scores"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_emitted_for_high_mi_features(tmp_path) -> None:
    """EDA and ML guidance must be attached for features above min_mi threshold."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "strong": x,
            "target": x + rng.normal(0, 0.05, n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target", "min_mi_for_guidance": 0.0},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "strong": "continuous",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "strong" in result.guidance
    assert len(result.guidance["strong"]["eda"]) > 0
    assert len(result.guidance["strong"]["ml"]) > 0
    ml_actions = result.guidance["strong"]["ml"][0]["actions"]
    assert any(a["action"] == "include_in_model" for a in ml_actions)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_counts_correct(tmp_path) -> None:
    """Summary feature_count must equal number of entries in mi_scores."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "target": rng.normal(0, 1, n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types", {"a": "continuous", "b": "continuous", "target": "continuous"}
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["feature_count"] == len(result.data["mi_scores"])


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_columns_with_nulls_handled(tmp_path) -> None:
    """Features with null values must be imputed and not cause failure."""
    rng = np.random.default_rng(42)
    n = 200
    feature = rng.normal(0, 1, n).tolist()
    feature[::10] = [None] * len(feature[::10])  # inject nulls
    df = pd.DataFrame(
        {
            "feature_with_nulls": feature,
            "target": rng.normal(0, 1, n),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "feature_with_nulls": "continuous",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "feature_with_nulls" in result.data["mi_scores"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    rng = np.random.default_rng(42)
    n = 200
    df = pl.DataFrame(
        {
            "feature": rng.normal(0, 1, n).tolist(),
            "target": rng.normal(0, 1, n).tolist(),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {
            "feature": "continuous",
            "target": "continuous",
        },
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert "feature" in result.data["mi_scores"]


def test_no_plots_generated(tmp_path) -> None:
    """MI task must not generate plots."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 100),
            "target": rng.normal(0, 1, 100),
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=ComputeMutualInformation,
        current_df=df,
        task_overrides={"target_column": "target"},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous", "target": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
