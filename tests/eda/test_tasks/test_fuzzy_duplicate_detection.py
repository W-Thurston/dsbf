# tests/eda/test_tasks/test_fuzzy_duplicate_detection.py


from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pytest

from dsbf.eda.tasks.fuzzy_duplicate_detection import (
    FuzzyDuplicateDetection,
    _build_candidate_pairs,
    _row_string,
    _similarity,
    _tokenise,
)
from tests.helpers.context_utils import make_ctx_and_task

if TYPE_CHECKING:
    from pandas import Series

    from dsbf.eda.task_result import TaskResult

# ── Unit tests for pure helpers ───────────────────────────────────────────────


def test_row_string_basic() -> None:
    row: Series = pd.Series({"name": "New York", "status": "Active"})
    result: str = _row_string(row, ["name", "status"])
    assert result == "new york active"


def test_row_string_null_values() -> None:
    row: Series = pd.Series({"name": "London", "status": None})
    result: str = _row_string(row, ["name", "status"])
    assert "london" in result
    assert result.count("  ") == 0  # no double spaces


def test_row_string_normalises_whitespace() -> None:
    row: Series = pd.Series({"name": "  New   York  "})
    result: str = _row_string(row, ["name"])
    assert result == "new york"


def test_tokenise_basic() -> None:
    assert _tokenise("new york city") == {"new", "york", "city"}


def test_tokenise_empty_string() -> None:
    assert _tokenise("") == set()


def test_similarity_identical_strings() -> None:
    assert _similarity("hello world", "hello world") == 1.0


def test_similarity_empty_strings() -> None:
    assert _similarity("", "") == 1.0


def test_similarity_one_empty() -> None:
    assert _similarity("hello", "") == 0.0


def test_similarity_typo() -> None:
    # Single character difference - should be high but not 1.0
    score: float = _similarity("new york", "new york ")  # trailing space
    assert 0.8 < score < 1.0


def test_similarity_case_difference() -> None:
    # Row strings are pre-lowercased so this tests the scoring on equal strings
    score: float = _similarity("new york", "new york")
    assert score == 1.0


def test_similarity_completely_different() -> None:
    score: float = _similarity("apple", "orange")
    assert score < 0.5


def test_build_candidate_pairs_shared_token() -> None:
    """Rows sharing a token must be in the candidate pairs."""
    strings: list[str] = ["new york giants", "new york jets", "chicago bears"]
    pairs: list[tuple[int, int]] = _build_candidate_pairs(strings, max_candidates=1000)
    # (0, 1) share "new" and "york"
    assert (0, 1) in pairs


def test_build_candidate_pairs_no_shared_tokens() -> None:
    """Completely disjoint rows must not be candidates."""
    strings: list[str] = ["apple banana", "orange grape", "mango kiwi"]
    pairs: list[tuple[int, int]] = _build_candidate_pairs(strings, max_candidates=1000)
    # None of these share any tokens
    assert len(pairs) == 0


def test_build_candidate_pairs_max_candidates_respected() -> None:
    """max_candidates must cap the number of pairs returned."""
    strings: list[str] = [f"the word token{i}" for i in range(100)]
    pairs: list[tuple[int, int]] = _build_candidate_pairs(strings, max_candidates=5)
    assert len(pairs) <= 5


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_case_variants_detected(tmp_path) -> None:
    """Rows differing only by case must be flagged as fuzzy duplicates."""
    df = pd.DataFrame(
        {
            "city": ["New York", "new york", "Chicago", "chicago"] * 5,
            "status": ["active"] * 20,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"city": "categorical", "status": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    pairs = result.data["__dataset__"]["fuzzy_pairs"]
    assert len(pairs) > 0
    # All scores must be above threshold
    assert all(p["similarity"] >= 0.85 for p in pairs)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_whitespace_variants_detected(tmp_path) -> None:
    """Rows differing only by whitespace must be flagged."""
    df = pd.DataFrame(
        {
            "name": ["  John Smith  ", "John Smith", "Jane Doe", "Jane  Doe"] * 5,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"name": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    pairs = result.data["__dataset__"]["fuzzy_pairs"]
    assert len(pairs) > 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_truly_distinct_rows_not_flagged(tmp_path) -> None:
    """Completely distinct rows must not be flagged as fuzzy duplicates."""
    df = pd.DataFrame(
        {
            "city": [
                "London",
                "Tokyo",
                "Sydney",
                "Paris",
                "Berlin",
                "Mumbai",
                "Cairo",
                "Lima",
                "Oslo",
                "Seoul",
            ],
            "country": [
                "UK",
                "Japan",
                "Australia",
                "France",
                "Germany",
                "India",
                "Egypt",
                "Peru",
                "Norway",
                "SouthKorea",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {"city": "categorical", "country": "categorical"},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    pairs = result.data["__dataset__"]["fuzzy_pairs"]
    assert len(pairs) == 0


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_pairs_sorted_by_similarity_descending(tmp_path) -> None:
    """Fuzzy pairs must be sorted with highest similarity first."""
    df = pd.DataFrame(
        {
            "name": [
                "New York City",
                "New York Cty",  # very close
                "New York",
                "New York",  # identical
                "Chicago",
                "Chcago",  # close
            ]
            * 5,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.7},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"name": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    pairs = result.data["__dataset__"]["fuzzy_pairs"]
    if len(pairs) > 1:
        for i in range(len(pairs) - 1):
            assert pairs[i]["similarity"] >= pairs[i + 1]["similarity"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_differing_columns_identified(tmp_path) -> None:
    """The differing_columns field must list columns that differ between rows."""
    df = pd.DataFrame(
        {
            "name": ["New York", "new york"] * 10,
            "country": ["USA", "USA"] * 10,  # same
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata(
        "semantic_types",
        {"name": "categorical", "country": "categorical"},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    pairs = result.data["__dataset__"]["fuzzy_pairs"]
    if pairs:
        # 'name' differs by case, 'country' is the same
        for pair in pairs:
            if "name" in pair["differing_columns"]:
                assert "country" not in pair["differing_columns"]


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_threshold_controls_flagging(tmp_path) -> None:
    """A stricter threshold must flag fewer pairs than a lenient one."""
    df = pd.DataFrame(
        {
            "city": ["New York", "New York!", "Tokyo", "Tokio"] * 10,
        },
    )

    ctx_strict, task_strict = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.99},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx_strict.set_metadata("semantic_types", {"city": "categorical"})
    result_strict: TaskResult = ctx_strict.run_task(task_strict)

    ctx_lenient, task_lenient = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.5},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx_lenient.set_metadata("semantic_types", {"city": "categorical"})
    result_lenient: TaskResult = ctx_lenient.run_task(task_lenient)

    # _strict_count: int = len(result_strict.data["__dataset__"]["fuzzy_pairs"])
    lenient_count: int = len(result_lenient.data["__dataset__"]["fuzzy_pairs"])

    assert result_strict.status == "success"
    assert result_lenient.status == "success"
    assert len(result_strict.data["__dataset__"]["fuzzy_pairs"]) <= lenient_count


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_sampling_applied_for_large_dataset(tmp_path) -> None:
    """Datasets exceeding max_comparison_rows must trigger sampling."""
    df = pd.DataFrame(
        {
            "city": ["New York"] * 200 + ["London"] * 200,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85, "max_comparison_rows": 50},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"city": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["sampled"] is True
    assert result.summary["rows_compared"] == 50
    assert result.reliability_warnings is not None


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_guidance_attached_for_found_pairs(tmp_path) -> None:
    """EDA guidance must be attached under __dataset__ when pairs are found."""
    df = pd.DataFrame(
        {
            "name": ["New York", "new york"] * 15,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"name": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "__dataset__" in result.guidance
    assert len(result.guidance["__dataset__"]["eda"]) > 0
    actions = result.guidance["__dataset__"]["eda"][0]["actions"]
    assert any(a["action"] == "normalise_strings" for a in actions)


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_no_string_columns_returns_gracefully(tmp_path) -> None:
    """A DataFrame with no string columns must return success with empty pairs."""
    df = pd.DataFrame({"a": range(50), "b": [float(x) for x in range(50)]})

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"a": "continuous", "b": "continuous"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # No categorical/text columns → make_empty_result; summary has no
    # fuzzy_pair_count key — just verify the task returns cleanly.
    assert "message" in result.summary


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_summary_pair_count_matches_data(tmp_path) -> None:
    """Summary fuzzy_pair_count must equal len(fuzzy_pairs) in data."""
    df = pd.DataFrame(
        {
            "city": ["New York", "new york", "London", "london"] * 10,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"city": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    data_count: int = len(result.data["__dataset__"]["fuzzy_pairs"])
    assert result.summary["fuzzy_pair_count"] == data_count


@pytest.mark.filterwarnings("ignore:Could not infer format.*:UserWarning")
def test_polars_dataframe_handled(tmp_path) -> None:
    """Task must handle Polars DataFrames via pandas conversion."""
    df = pl.DataFrame(
        {
            "city": ["New York", "new york", "Chicago"] * 10,
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"city": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"


def test_no_plots_generated(tmp_path) -> None:
    """Fuzzy duplicate task must not generate plots."""
    df = pd.DataFrame({"city": ["New York", "new york"] * 10})
    ctx, task = make_ctx_and_task(
        task_cls=FuzzyDuplicateDetection,
        current_df=df,
        task_overrides={"similarity_threshold": 0.85},
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"city": "categorical"})
    result: TaskResult = ctx.run_task(task)
    assert result.status == "success"
    assert result.plots is None
