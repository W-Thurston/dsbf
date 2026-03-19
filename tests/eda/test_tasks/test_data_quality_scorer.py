# tests/eda/test_tasks/test_data_quality_scorer.py

import pandas as pd

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.data_quality_scorer import DataQualityScorer
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def _make_scorer_context(tmp_path, upstream_results: dict | None = None):
    """Helper: create a context with optional injected upstream task results."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    if upstream_results:
        ctx.results = upstream_results
    return ctx


def test_scorer_runs_with_no_upstream_results(tmp_path):
    """Scorer must succeed and default all dimensions to green when no source
    tasks have run."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    scorer = DataQualityScorer()
    scorer.set_input(df)
    scorer.context = ctx
    scorer.run()
    result = scorer.get_output()

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.summary is not None
    for dimension in ("completeness", "validity", "usability", "redundancy", "leakage"):
        assert dimension in result.summary
        assert result.summary[dimension] == "green"


def test_summary_keys_are_five_dimensions(tmp_path):
    """Summary must contain exactly the five data-health dimension keys."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    scorer = DataQualityScorer()
    scorer.set_input(df)
    scorer.context = ctx
    scorer.run()

    assert set(scorer.get_output().summary.keys()) == {
        "completeness",
        "validity",
        "usability",
        "redundancy",
        "leakage",
    }


def test_data_field_contains_categories_and_totals(tmp_path):
    """Data field must contain total_columns, all_columns, and categories."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))
    scorer = DataQualityScorer()
    scorer.set_input(df)
    scorer.context = ctx
    scorer.run()

    data = scorer.get_output().data
    assert "total_columns" in data
    assert "all_columns" in data
    assert "categories" in data
    for dim in ("completeness", "validity", "usability", "redundancy", "leakage"):
        assert dim in data["categories"]
        cat = data["categories"][dim]
        assert "affected_columns" in cat
        assert "pct_affected" in cat
        assert "level" in cat
        assert "findings" in cat
        assert cat["level"] in ("green", "amber", "red")


def test_null_columns_raise_completeness_level(tmp_path):
    """Columns with high missingness must appear in completeness findings."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))

    # Inject summarize_nulls result with a column at 30% missing
    ctx.results["summarize_nulls"] = TaskResult(
        name="summarize_nulls",
        status="success",
        data={"null_percentages": {"a": 0.3}},
    )
    # Inject infer_types so total_columns resolves
    ctx.results["infer_types"] = TaskResult(
        name="infer_types",
        status="success",
        data={"a": {"inferred_dtype": "int64", "analysis_intent_dtype": "continuous"}},
    )

    scorer = DataQualityScorer()
    scorer.set_input(df)
    scorer.context = ctx
    scorer.run()

    result = scorer.get_output()
    completeness = result.data["categories"]["completeness"]
    assert "a" in completeness["affected_columns"]
    assert any(f["column"] == "a" for f in completeness["findings"])
    assert completeness["level"] in ("amber", "red")


def test_constant_columns_raise_validity_level(tmp_path):
    """Constant columns from detect_constant_columns must appear in validity."""
    df = pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))

    ctx.results["infer_types"] = TaskResult(
        name="infer_types",
        status="success",
        data={
            "x": {"inferred_dtype": "int64", "analysis_intent_dtype": "continuous"},
            "y": {"inferred_dtype": "int64", "analysis_intent_dtype": "continuous"},
        },
    )
    ctx.results["detect_constant_columns"] = TaskResult(
        name="detect_constant_columns",
        status="success",
        data={"constant_columns": ["x"]},
    )

    scorer = DataQualityScorer()
    scorer.set_input(df)
    scorer.context = ctx
    scorer.run()

    result = scorer.get_output()
    validity = result.data["categories"]["validity"]
    assert "x" in validity["affected_columns"]
    assert any(
        f["column"] == "x" and f["issue"] == "constant_column"
        for f in validity["findings"]
    )


def test_collinear_columns_raise_redundancy_level(tmp_path):
    """Columns flagged by VIF must appear in redundancy findings."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))

    ctx.results["infer_types"] = TaskResult(
        name="infer_types",
        status="success",
        data={
            "a": {"inferred_dtype": "int64", "analysis_intent_dtype": "continuous"},
            "b": {"inferred_dtype": "int64", "analysis_intent_dtype": "continuous"},
        },
    )
    ctx.results["detect_collinear_features"] = TaskResult(
        name="detect_collinear_features",
        status="success",
        data={
            "vif_scores": {"a": 25.0, "b": 25.0},
            "collinear_columns": ["a", "b"],
        },
    )

    scorer = DataQualityScorer()
    scorer.set_input(df)
    scorer.context = ctx
    scorer.run()

    result = scorer.get_output()
    redundancy = result.data["categories"]["redundancy"]
    assert (
        "a" in redundancy["affected_columns"] or "b" in redundancy["affected_columns"]
    )
    assert any(f["issue"] == "high_vif" for f in redundancy["findings"])


def test_leakage_pairs_raise_leakage_level(tmp_path):
    """Leakage pairs must appear in leakage findings with error severity."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    ctx = AnalysisContext(df, output_dir=str(tmp_path))

    ctx.results["infer_types"] = TaskResult(
        name="infer_types",
        status="success",
        data={
            "a": {"inferred_dtype": "int64", "analysis_intent_dtype": "continuous"},
            "b": {"inferred_dtype": "int64", "analysis_intent_dtype": "continuous"},
        },
    )
    ctx.results["detect_data_leakage"] = TaskResult(
        name="detect_data_leakage",
        status="success",
        data={"leakage_pairs": {"a|b": 0.99}},
    )

    scorer = DataQualityScorer()
    scorer.set_input(df)
    scorer.context = ctx
    scorer.run()

    result = scorer.get_output()
    leakage = result.data["categories"]["leakage"]
    assert "a" in leakage["affected_columns"]
    assert "b" in leakage["affected_columns"]
    assert any(
        f["issue"] == "leakage_pair" and f["severity"] == "error"
        for f in leakage["findings"]
    )


def test_scorer_via_run_task_with_dependencies(tmp_path):
    """Smoke test: scorer must pass run_task_with_dependencies without error."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": ["x", "y", "z", "x", "y"],
        }
    )
    ctx, _ = make_ctx_and_task(
        task_cls=DataQualityScorer,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = run_task_with_dependencies(ctx, DataQualityScorer)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.data is not None
    assert "categories" in result.data
