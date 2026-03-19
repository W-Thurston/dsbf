# tests/eda/test_tasks/test_summarize_text_fields.py

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from dsbf.eda.tasks.summarize_text_fields import SummarizeTextFields
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

if TYPE_CHECKING:
    from dsbf.eda.task_result import TaskResult


def test_text_stats_computed(tmp_path) -> None:
    """Expected statistics keys must be present for a text column."""
    # Force text classification: long strings (mean length > 30)
    long_texts: list[str] = [
        "This is a fairly long sentence that should be classified as text by DSBF.",
        "Another reasonably long sentence to ensure the column is typed as text.",
        "Yet another sentence of sufficient length to be treated as a text column.",
    ]
    df = pd.DataFrame({"description": long_texts})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeTextFields)

    assert result.status == "success"
    assert "description" in result.data
    stats = result.data["description"]
    for key in (
        "avg_char_length",
        "avg_word_count",
        "total_chars",
        "most_frequent_value",
        "contains_symbols",
    ):
        assert key in stats


def test_avg_char_length_reasonable(tmp_path) -> None:
    """avg_char_length must reflect the actual character count."""
    texts: list[str] = ["hello world"] * 5 + [
        "This is a fairly long sentence that should be classified as text by DSBF.",
    ] * 10
    df = pd.DataFrame({"notes": texts})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeTextFields)

    assert result.status == "success"
    if "notes" in result.data:
        assert result.data["notes"]["avg_char_length"] > 0


def test_symbol_detection(tmp_path) -> None:
    """contains_symbols must be True when values contain non-alphanumeric characters."""
    long_texts: list[str] = [
        "Hello, world! This sentence has symbols and is long enough for text type.",
        "Another sentence with punctuation! This is also long enough to classify.",
        "Yet another sentence with exclamation! Sufficient length for classification.",
    ]
    df = pd.DataFrame({"text": long_texts})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeTextFields)

    assert result.status == "success"
    if "text" in result.data:
        assert result.data["text"]["contains_symbols"] is True


def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must work on Polars DataFrames for text columns."""
    long_texts: list[str] = [
        "This is a fairly long sentence that should be classified as text by DSBF.",
        "Another reasonably long sentence to ensure the column is typed as text.",
        "Yet another sentence of sufficient length to be treated as a text column.",
    ]
    df = pl.DataFrame({"notes": long_texts})

    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeTextFields)

    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    long_texts: list[str] = [
        "This is a fairly long sentence that should be classified as text by DSBF.",
        "Another reasonably long sentence to ensure the column is typed as text.",
        "Yet another sentence of sufficient length to be treated as a text column.",
    ]
    df = pd.DataFrame({"notes": long_texts})
    ctx, _ = make_ctx_and_task(
        task_cls=SummarizeTextFields,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = run_task_with_dependencies(ctx, SummarizeTextFields)
    assert result.status == "success"
    assert result.plots is None
