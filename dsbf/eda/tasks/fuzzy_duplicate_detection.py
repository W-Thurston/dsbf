# dsbf/eda/tasks/fuzzy_duplicate_detection.py
#
# Detects near-duplicate rows using token-based blocking and character-level
# similarity scoring via difflib.SequenceMatcher.
#
# Algorithm:
#   1. Select string/categorical columns as the comparison surface.
#   2. Represent each row as a normalised string (lowercased, whitespace-normalised).
#   3. Apply a blocking step: only compare rows that share at least one token
#      in their string representations. This reduces the O(n²) pair space to a
#      tractable subset without missing most genuine near-duplicates.
#   4. Score each candidate pair with SequenceMatcher.ratio() in [0, 1].
#   5. Flag pairs above the similarity threshold.
#
# For datasets larger than max_comparison_rows, a random sample is taken and
# a warning is attached to the result.

import difflib
import re
from collections import defaultdict
from typing import Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import (
    TaskResult,
    add_reliability_warning,
    make_failure_result,
)

# ── Row representation ────────────────────────────────────────────────────────


def _row_string(row: pd.Series, cols: list[str]) -> str:
    """
    Produce a normalised string representation of a row for similarity scoring.

    Args:
        row: A pandas Series representing one DataFrame row.
        cols: Column names to include in the representation.

    Returns:
        Lowercase, whitespace-normalised string of all selected column values.

    """
    parts: list[str] = []
    for col in cols:
        val = row[col]
        if pd.isna(val):
            parts.append("")
        else:
            parts.append(str(val).lower().strip())
    combined: str = " ".join(parts)
    return re.sub(r"\s+", " ", combined).strip()


def _tokenise(s: str) -> set[str]:
    """
    Split a row string into a set of word tokens for blocking.

    Args:
        s: Normalised row string.

    Returns:
        Set of non-empty whitespace-separated tokens.

    """
    return {t for t in s.split() if t}


# ── Blocking ──────────────────────────────────────────────────────────────────


def _build_candidate_pairs(
    row_strings: list[str],
    max_candidates: int,
) -> list[tuple[int, int]]:
    """
    Build candidate pairs using token-overlap blocking.

    Two rows are candidates if they share at least one token. This filters
    the O(n²) comparison space to rows that have any lexical overlap,
    catching most genuine near-duplicates while skipping totally dissimilar rows.

    Args:
        row_strings: List of normalised row strings, indexed by row position.
        max_candidates: Maximum candidate pairs to return (caps runtime).

    Returns:
        List of (i, j) index pairs where i < j.

    """
    # Build inverted index: token → set of row indices containing that token
    token_index: dict[str, set[int]] = defaultdict(set)
    for idx, s in enumerate(row_strings):
        for token in _tokenise(s):
            token_index[token].add(idx)

    # Collect pairs that share at least one token
    candidate_pairs: set[tuple[int, int]] = set()
    for token, indices in token_index.items():
        idx_list: list[int] = sorted(indices)
        for a in range(len(idx_list)):
            for b in range(a + 1, len(idx_list)):
                pair: tuple[int, int] = (idx_list[a], idx_list[b])
                candidate_pairs.add(pair)
                if len(candidate_pairs) >= max_candidates:
                    return list(candidate_pairs)

    return list(candidate_pairs)


# ── Similarity scoring ────────────────────────────────────────────────────────


def _similarity(a: str, b: str) -> float:
    """
    Compute character-level similarity between two strings.

    Uses difflib.SequenceMatcher.ratio(), which returns a value in [0, 1]
    where 1.0 means identical and 0.0 means completely dissimilar.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity ratio in [0.0, 1.0].

    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="fuzzy_duplicate_detection",
    display_name="Fuzzy Duplicate Detection",
    description=(
        "Detects near-duplicate rows using token-based blocking and "
        "character-level similarity scoring. Catches rows that differ by "
        "case, whitespace, typos, or minor formatting variations that exact "
        "duplicate detection misses."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="moderate",
    tags=["duplicates", "rows", "fuzzy", "quality", "string"],
    expected_semantic_types=["categorical", "text"],
)
class FuzzyDuplicateDetection(BaseTask):
    """
    Detect near-duplicate rows using approximate string similarity.

    Exact duplicate detection (``detect_duplicates``) misses rows that differ
    by a single character, case difference, or minor formatting variation.
    This task catches those near-duplicates using a two-phase approach:

    **Phase 1 - Blocking (token overlap):**
    Only row pairs that share at least one word token in their string
    representations are treated as candidates. This reduces the O(n²) search
    space to a tractable subset without missing most genuine near-duplicates.

    **Phase 2 - Scoring (character similarity):**
    Each candidate pair is scored with ``difflib.SequenceMatcher.ratio()``,
    which compares character sequences and returns a similarity in [0, 1].
    Pairs above ``similarity_threshold`` (default: 0.85) are flagged.

    **Comparison surface:**
    Only string/categorical columns are included. Numeric and datetime columns
    are excluded - small numeric differences are handled by ``detect_outliers``,
    not by string similarity.

    **Large dataset handling:**
    When the dataset exceeds ``max_comparison_rows`` (default: 5000), a random
    sample is taken and a reliability warning is attached noting that results
    are not exhaustive.

    **Output structure:**
    Findings are stored under the ``__dataset__`` sentinel key (row-level
    findings cannot be attributed to a single column). Each finding includes
    the row indices, similarity score, and the column values that drove it.

    Configurable parameters (via config["tasks"]["fuzzy_duplicate_detection"]):
        similarity_threshold (float): Minimum similarity to flag a pair.
            Default: 0.85
        max_comparison_rows (int): Row count above which sampling is used.
            Default: 5000
        max_candidates (int): Maximum candidate pairs from blocking step.
            Default: 50000
        min_string_cols (int): Minimum string columns required to run.
            Default: 1
    """

    def run(self) -> None:
        """
        Execute fuzzy duplicate detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df, matched_cols, excluded = self.setup_run()

            if not matched_cols:
                self.output = self.make_empty_result(
                    "No eligible columns found — fuzzy duplicate detection skipped.",
                    excluded,
                )
                return

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'categorical/text' column(s)",
                "debug",
            )

            threshold_raw: Any | None = self.get_task_param("similarity_threshold")
            threshold: float = (
                float(threshold_raw) if threshold_raw is not None else 0.85
            )

            max_rows_raw: Any | None = self.get_task_param("max_comparison_rows")
            max_comparison_rows: int = (
                int(max_rows_raw) if max_rows_raw is not None else 5_000
            )

            max_cand_raw: Any | None = self.get_task_param("max_candidates")
            max_candidates: int = (
                int(max_cand_raw) if max_cand_raw is not None else 50_000
            )

            min_str_raw: Any | None = self.get_task_param("min_string_cols")
            min_string_cols: int = int(min_str_raw) if min_str_raw is not None else 1

            # --- Select string/categorical columns ---
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            string_cols: list[str] = [
                col
                for col in df.columns
                if semantic_types.get(col, "") in ("categorical", "text")
                or df[col].dtype == object
            ]
            # Exclude columns that are entirely null
            string_cols = [col for col in string_cols if df[col].notna().any()]

            if len(string_cols) < min_string_cols:
                self._log(
                    f"    Fewer than {min_string_cols} string column(s) found - "
                    "skipping fuzzy duplicate detection.",
                    "debug",
                )
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": (
                            f"No string columns available for comparison "
                            f"(min_string_cols={min_string_cols})."
                        ),
                        "fuzzy_pair_count": 0,
                    },
                    data={"__dataset__": {"fuzzy_pairs": []}},
                    metadata={
                        "string_cols_used": [],
                        "threshold": threshold,
                        "suggested_viz_type": "table",
                        "recommended_section": "Quality",
                        "display_priority": "medium",
                        "excluded_columns": excluded,
                        "column_types": self.get_column_type_info(
                            matched_cols + list(excluded.keys()),
                        ),
                    },
                )
                return

            # --- Sample if dataset is too large ---
            sampled = False
            working_df = df
            if len(df) > max_comparison_rows:
                working_df = df.sample(
                    n=max_comparison_rows,
                    random_state=42,
                ).reset_index(drop=True)
                sampled = True
                self._log(
                    f"    Dataset has {len(df):,} rows - sampling "
                    f"{max_comparison_rows:,} for comparison.",
                    "debug",
                )

            n_rows: int = len(working_df)
            self._log(
                f"    Comparing {n_rows:,} rows across "
                f"{len(string_cols)} column(s): {string_cols}",
                "debug",
            )

            # --- Build row string representations ---
            row_strings: list[str] = [
                _row_string(working_df.iloc[i], string_cols) for i in range(n_rows)
            ]

            # Drop rows that produce an empty string (all-null or empty values)
            valid_indices: list[int] = [i for i, s in enumerate(row_strings) if s]
            if len(valid_indices) < 2:
                self._log(
                    "    Fewer than 2 non-empty rows - nothing to compare.",
                    "debug",
                )
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={
                        "message": "Insufficient non-empty rows.",
                        "fuzzy_pair_count": 0,
                    },
                    data={"__dataset__": {"fuzzy_pairs": []}},
                    metadata={
                        "string_cols_used": string_cols,
                        "threshold": threshold,
                        "suggested_viz_type": "table",
                        "recommended_section": "Quality",
                        "display_priority": "medium",
                        "excluded_columns": excluded,
                        "column_types": self.get_column_type_info(
                            matched_cols + list(excluded.keys()),
                        ),
                    },
                )
                return

            valid_strings: list[str] = [row_strings[i] for i in valid_indices]

            # --- Blocking: find candidate pairs ---
            candidates: list[tuple[int, int]] = _build_candidate_pairs(
                valid_strings,
                max_candidates,
            )
            self._log(
                f"    {len(candidates):,} candidate pair(s) from blocking step.",
                "debug",
            )

            # --- Score candidates ---
            fuzzy_pairs: list[dict[str, Any]] = []

            for local_i, local_j in candidates:
                real_i: int = valid_indices[local_i]
                real_j: int = valid_indices[local_j]

                score: float = _similarity(
                    valid_strings[local_i], valid_strings[local_j]
                )

                if score >= threshold:
                    # Collect the column values that drove the similarity
                    row_i_vals: dict = {
                        col: working_df.iloc[real_i][col] for col in string_cols
                    }
                    row_j_vals: dict = {
                        col: working_df.iloc[real_j][col] for col in string_cols
                    }
                    # Identify differing columns
                    differing: list[str] = [
                        col
                        for col in string_cols
                        if str(row_i_vals.get(col, "")).lower().strip()
                        != str(row_j_vals.get(col, "")).lower().strip()
                    ]

                    fuzzy_pairs.append(
                        {
                            "row_i": int(real_i),
                            "row_j": int(real_j),
                            "similarity": round(score, 4),
                            "differing_columns": differing,
                            "row_i_values": {
                                col: str(v) for col, v in row_i_vals.items()
                            },
                            "row_j_values": {
                                col: str(v) for col, v in row_j_vals.items()
                            },
                        },
                    )

            # Sort by similarity descending
            fuzzy_pairs.sort(key=lambda x: -x["similarity"])

            self._log(
                f"    {len(fuzzy_pairs)} fuzzy duplicate pair(s) found "
                f"at threshold={threshold}.",
                "debug",
            )

            row_info: int | str = (
                f"sampled {max_comparison_rows}" if sampled else n_rows
            )
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Found {len(fuzzy_pairs)} near-duplicate row pair(s) "
                        f"at similarity ≥ {threshold} "
                        f"({row_info} rows)."
                    ),
                    "fuzzy_pair_count": len(fuzzy_pairs),
                    "threshold": threshold,
                    "rows_compared": n_rows,
                    "sampled": sampled,
                },
                data={"__dataset__": {"fuzzy_pairs": fuzzy_pairs}},
                metadata={
                    "string_cols_used": string_cols,
                    "threshold": threshold,
                    "max_comparison_rows": max_comparison_rows,
                    "max_candidates": max_candidates,
                    "sampled": sampled,
                    "suggested_viz_type": "table",
                    "recommended_section": "Quality",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            if sampled:
                add_reliability_warning(
                    self.output,
                    level="heuristic_caution",
                    code="sampled_comparison",
                    description=(
                        f"Dataset has {len(df):,} rows. Fuzzy comparison was run "
                        f"on a random sample of {max_comparison_rows:,} rows - "
                        f"near-duplicates outside the sample are not detected."
                    ),
                    recommendation=(
                        "Set max_comparison_rows higher or filter to a known "
                        "problematic subset before running this task."
                    ),
                )

            if fuzzy_pairs:
                self._attach_guidance(fuzzy_pairs, threshold, string_cols, sampled)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(
        self,
        fuzzy_pairs: list[dict[str, Any]],
        threshold: float,
        string_cols: list[str],
        sampled: bool,
    ) -> None:
        """
        Generate EDA guidance for detected near-duplicate pairs.

        Guidance is attached under the ``__dataset__`` sentinel key since
        near-duplicate findings are row-level, not column-level.

        Args:
            fuzzy_pairs: List of fuzzy pair dicts sorted by similarity.
            threshold: Similarity threshold used.
            string_cols: String columns used for comparison.
            sampled: Whether a sampling step was applied.

        """
        n: int = len(fuzzy_pairs)
        top: list[dict[str, Any]] = fuzzy_pairs[:3]

        example_lines: list[str] = []
        for pair in top:
            i, j = pair["row_i"], pair["row_j"]
            score = pair["similarity"]
            diff_cols = pair["differing_columns"]
            example_lines.append(
                f"  rows {i} & {j}: similarity={score:.3f}"
                + (
                    f", differs on: {diff_cols}"
                    if diff_cols
                    else " (identical after normalisation)"
                ),
            )
        examples_str: str = "\n".join(example_lines)

        sample_note: str = (
            " Note: results are from a sampled subset - actual count may be higher."
            if sampled
            else ""
        )

        body: str = (
            f"Found {n} near-duplicate row pair(s) at similarity ≥ {threshold} "
            f"across columns: {string_cols}.{sample_note}\n\n"
            f"Top examples:\n{examples_str}\n\n"
            f"Near-duplicates that differ by case, whitespace, or minor typos "
            f"are invisible to exact duplicate detection but inflate counts, "
            f"corrupt value distributions, and cause join operations to fail "
            f"silently. Common causes: data entry variations, inconsistent "
            f"normalisation across sources, or records merged from multiple "
            f"systems with different formatting conventions."
        )

        self.add_guidance(
            result=self.output,
            column="__dataset__",
            phase="eda",
            level="warn" if n >= 10 else "info",
            title=f"Near-Duplicate Rows: {n} pair(s) at similarity ≥ {threshold}",
            body=body.strip(),
            actions=[
                {
                    "action": "normalise_strings",
                    "detail": (
                        "Apply str.lower().strip() to string columns before "
                        "deduplication to collapse case/whitespace variants"
                    ),
                },
                {
                    "action": "review_and_deduplicate",
                    "detail": (
                        f"Review the {n} flagged pair(s) and remove or merge "
                        "confirmed duplicates"
                    ),
                },
            ],
            metric={
                "fuzzy_pair_count": n,
                "threshold": threshold,
                "sampled": sampled,
            },
        )
