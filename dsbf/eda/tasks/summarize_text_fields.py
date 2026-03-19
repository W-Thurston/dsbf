# dsbf/eda/tasks/summarize_text_fields.py

import re
from collections import Counter
from typing import Any, Literal

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars


@register_task(
    display_name="Summarize Text Fields",
    description=(
        "Summarizes content of text columns (length, frequency, symbols, etc.)."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["text", "summary"],
    expected_semantic_types=["text"],
)
class SummarizeTextFields(BaseTask):
    """
    Summarize structural characteristics of text-typed columns.

    For each column classified as ``text`` by ``infer_types``, computes:

    - ``avg_char_length``: mean character count per value
    - ``avg_word_count``: mean whitespace-delimited token count
    - ``avg_word_length``: total characters / total words
    - ``total_chars``: sum of character counts across all non-null values
    - ``most_frequent_value``: the single most common string value
    - ``contains_symbols``: True if any value contains non-alphanumeric characters

    Supports both Polars (via ``is_text_polars``) and Pandas (via ``is_text_pandas``).
    Columns that are object dtype but not string-like are skipped silently.
    """

    def run(self) -> None:  # noqa: C901
        """
        Execute text field summarization and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} 'text' column(s)", "debug")

            results: dict[str, dict[str, Any]] = {}

            if is_polars(df):
                for col in df.columns:
                    if not is_text_polars(df[col]):
                        continue
                    try:
                        strings = df[col].drop_nulls().to_list()
                        if not strings:
                            continue

                        char_counts: list[int] = [len(s) for s in strings]
                        word_counts: list[int] = [len(s.split()) for s in strings]
                        total_chars: Literal[0] | int = sum(char_counts)
                        total_words: Literal[0] | int = sum(word_counts)
                        avg_word_len: float = (
                            total_chars / total_words if total_words else 0.0
                        )

                        most_common = Counter(strings).most_common(1)
                        top_value = most_common[0][0] if most_common else None
                        has_symbols = any(re.search(r"[^\w\s]", s) for s in strings)

                        self._log(f"    Summarized text column: '{col}'", "debug")
                        results[col] = {
                            "avg_char_length": sum(char_counts) / len(char_counts),
                            "avg_word_count": sum(word_counts) / len(word_counts),
                            "avg_word_length": avg_word_len,
                            "total_chars": total_chars,
                            "most_frequent_value": top_value,
                            "contains_symbols": has_symbols,
                        }
                    except Exception:  # noqa: BLE001, S112
                        continue
            else:
                for col in df.columns:
                    if not is_text_pandas(df[col]):
                        continue
                    try:
                        texts = df[col].dropna().astype(str)
                        char_counts = texts.map(len)
                        word_counts = texts.map(lambda s: len(s.split()))
                        total_chars = int(char_counts.sum())
                        total_words = int(word_counts.sum())
                        avg_word_len = total_chars / total_words if total_words else 0.0

                        most_common = Counter(texts).most_common(1)
                        top_value = most_common[0][0] if most_common else None
                        has_symbols = any(re.search(r"[^\w\s]", s) for s in texts)

                        self._log(f"    Summarized text column: '{col}'", "debug")
                        results[col] = {
                            "avg_char_length": float(char_counts.mean()),
                            "avg_word_count": float(word_counts.mean()),
                            "avg_word_length": avg_word_len,
                            "total_chars": total_chars,
                            "most_frequent_value": top_value,
                            "contains_symbols": has_symbols,
                        }
                    except Exception:  # noqa: BLE001, S112
                        continue

            self._log(f"    Processed {len(results)} text columns", "debug")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Summarized {len(results)} text column(s)."},
                data=results,
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Text Features",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
