# tests/eda/test_tasks/test_data_quality_scorer.py

from pathlib import Path

import pytest

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.data_quality_scorer import DataQualityScorer
from tests.helpers.context_utils import make_ctx_and_task


@pytest.fixture
def dummy_context(tmp_path):
    """
    Simulate a realistic profiling context with synthetic task
    results and reliability flags.
    """
    # Use make_ctx_and_task to load default config
    ctx, _ = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        global_overrides={"output_dir": str(tmp_path)},
    )

    # Inject reliability flags
    ctx.reliability_flags = {
        "skew_vals": {"col1": 3.5, "col2": -2.8},
        "zero_variance_cols": ["col3"],
        "extreme_outliers": True,
    }

    # Inject mock task results
    ctx.results = {
        "null_summary": TaskResult(
            name="null_summary",
            status="success",
            data={
                "missingness": {
                    "col1": {"percent_missing": 0.1},
                    "col4": {"percent_missing": 0.4},
                }
            },
        ),
        "detect_mixed_type_columns": TaskResult(
            name="detect_mixed_type_columns",
            status="success",
            data={"columns_with_issues": ["col5", "col6"]},
            recommendations=["Convert col5 to consistent type."],
        ),
        "regex_format_violations": TaskResult(
            name="regex_format_violations",
            status="success",
            data={"columns_with_issues": ["col7"]},
            recommendations=["Fix date format in col7."],
        ),
        "detect_collinear_features": TaskResult(
            name="detect_collinear_features",
            status="success",
            data={"redundant_pairs": [("col1", "col2"), ("col3", "col4")]},
            recommendations=["Drop one of each collinear pair."],
        ),
        "detect_feature_drift": TaskResult(
            name="detect_feature_drift",
            status="success",
            data={"drifted_features": ["col2", "col8"]},
            recommendations=["Reevaluate features with significant drift."],
        ),
    }

    return ctx


def test_data_quality_scorer_smoke(tmp_path, dummy_context):
    ctx = dummy_context
    ctx, scorer = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        global_overrides={"output_dir": str(tmp_path)},
    )
    scorer.context = ctx
    scorer.context.results = dummy_context.results
    scorer.context.reliability_flags = dummy_context.reliability_flags
    scorer.run()
    result = scorer.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert "overall_score" in result.summary
    assert "category_breakdown" in result.summary
    assert isinstance(result.recommendations, list)


def test_category_breakdown_keys(tmp_path, dummy_context):
    ctx, scorer = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        global_overrides={"output_dir": str(tmp_path)},
    )
    scorer.context = ctx
    scorer.context.results = dummy_context.results
    scorer.context.reliability_flags = dummy_context.reliability_flags
    scorer.run()
    categories = scorer.get_output().summary["category_breakdown"]
    expected = {"completeness", "consistency", "distribution", "redundancy", "drift"}
    assert set(categories.keys()) == expected


def test_recommendations_present(tmp_path, dummy_context):
    ctx, scorer = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        global_overrides={"output_dir": str(tmp_path)},
    )
    scorer.context = ctx
    scorer.context.results = dummy_context.results
    scorer.context.reliability_flags = dummy_context.reliability_flags
    scorer.run()
    recs = scorer.get_output().recommendations
    assert "Convert col5 to consistent type." in recs
    assert "Fix date format in col7." in recs
    assert "Drop one of each collinear pair." in recs
    assert "Reevaluate features with significant drift." in recs


def test_category_weights_in_summary(tmp_path, dummy_context):
    ctx, scorer = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        global_overrides={"output_dir": str(tmp_path)},
    )
    scorer.context = ctx
    scorer.context.results = dummy_context.results
    scorer.context.reliability_flags = dummy_context.reliability_flags
    scorer.run()

    summary = scorer.get_output().summary
    assert "category_weights" in summary
    assert isinstance(summary["category_weights"], dict)
    assert all(k in summary["category_weights"] for k in summary["category_breakdown"])


def test_weight_override_affects_score(tmp_path, dummy_context):
    custom_weights = {
        "completeness": 0.0,
        "consistency": 0.0,
        "distribution": 0.0,
        "redundancy": 0.0,
        "drift": 1.0,
    }
    ctx, scorer = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        task_overrides={"weights": custom_weights},
        global_overrides={"output_dir": str(tmp_path)},
    )
    scorer.context = ctx
    scorer.context.results = dummy_context.results
    scorer.context.reliability_flags = dummy_context.reliability_flags
    scorer.run()

    summary = scorer.get_output().summary
    # Drift penalty was 2 * 5 = 10 → 100 - 10 = 90
    assert summary["overall_score"] == 90


def test_data_quality_score_plot_generated(tmp_path, dummy_context):
    """
    Check that a barplot is generated for overall + category scores.
    """
    ctx, scorer = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        global_overrides={"output_dir": str(tmp_path)},
    )

    scorer.context = ctx
    scorer.context.results = dummy_context.results
    scorer.context.reliability_flags = dummy_context.reliability_flags

    scorer.run()
    result = scorer.get_output()

    assert result.status == "success"
    assert result.plots is not None
    assert "data_quality_scores" in result.plots

    plot_entry = result.plots["data_quality_scores"]
    static_path = plot_entry["static"]
    interactive = plot_entry["interactive"]

    assert isinstance(static_path, Path)
    assert static_path.exists()
    assert static_path.suffix == ".png"

    assert interactive["type"] == "bar"
    assert "annotations" in interactive
    assert any("overall" in a.lower() for a in interactive["annotations"])


def test_score_with_empty_context(tmp_path):
    """
    If no issues or reliability flags are present, the score should default to 100.
    """
    ctx, scorer = make_ctx_and_task(
        DataQualityScorer,
        current_df=None,
        global_overrides={"output_dir": str(tmp_path)},
    )
    scorer.context = ctx
    scorer.run()
    result = scorer.get_output()

    assert result.status == "success"
    assert result.summary["overall_score"] == 100
    assert all(v == 100 for v in result.summary["category_breakdown"].values())
