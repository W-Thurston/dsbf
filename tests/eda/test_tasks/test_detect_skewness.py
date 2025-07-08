# tests/eda/test_tasks/test_detect_skewness.py

import warnings
from pathlib import Path

import pandas as pd

from dsbf.eda.tasks.detect_skewness import DetectSkewness
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_detect_skewness_expected_output(tmp_path):
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning, module="seaborn"
    )

    df = pd.DataFrame(
        {
            "normal": [1, 2, 3, 4, 5],
            "skewed": [1, 1, 1, 2, 100],
        }
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = run_task_with_dependencies(ctx, DetectSkewness)

    # Semantic type check
    semantic_types = ctx.get_metadata("semantic_types")
    assert semantic_types is not None
    assert semantic_types["normal"] == "continuous"
    assert semantic_types["skewed"] == "continuous"

    # Result checks
    assert result.status == "success"
    assert result.data is not None
    assert "normal" in result.data
    assert "skewed" in result.data
    assert abs(result.data["normal"]) < 1.0
    assert result.data["skewed"] > 1.0

    # Column types check
    column_types = result.metadata.get("column_types", {})
    assert "normal" in column_types
    assert "skewed" in column_types
    assert column_types["normal"]["inferred_dtype"] in {"int64", "float64"}
    assert column_types["normal"]["analysis_intent_dtype"] == "continuous"
    assert column_types["skewed"]["analysis_intent_dtype"] == "continuous"

    # Plot checks
    assert result.plots is not None
    assert "skewed" in result.plots
    static_path = result.plots["skewed"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()

    interactive = result.plots["skewed"]["interactive"]
    assert isinstance(interactive, dict)
    assert "annotations" in interactive
    assert any("Skewness:" in a for a in interactive["annotations"])


def test_detect_skewness_all_nulls(tmp_path):
    df = pd.DataFrame({"a": [None, None, None], "b": [float("nan")] * 3})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    assert result.data == {}
    assert result.plots == {}
    excluded = result.metadata.get("excluded_columns", {})
    assert set(excluded.keys()) == {"a", "b"}


def test_detect_skewness_constant_column(tmp_path):
    df = pd.DataFrame({"const": [5, 5, 5, 5, 5]})

    ctx, _ = make_ctx_and_task(
        task_cls=DetectSkewness,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result = run_task_with_dependencies(ctx, DetectSkewness)

    assert result.status == "success"
    assert result.data is not None
    assert abs(result.data["const"]) < 1e-9
    assert result.plots is not None
    assert "const" in result.plots
    static_path = result.plots["const"]["static"]
    assert isinstance(static_path, Path)
    assert static_path.exists()
    static_path.unlink()

    excluded = result.metadata.get("excluded_columns")
    assert excluded is not None
    assert "const" not in excluded
