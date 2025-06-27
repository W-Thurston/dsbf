# dsbf/eda/tasks/summarize_text_fields.py

import re
from collections import Counter
from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars


@register_task(
    display_name="Summarize Text Fields",
    description=(
        "Summarizes content of text columns" "(length, frequency, symbols, etc.)."
    ),
    depends_on=["infer_types"],
    stage="cleaned",
    tags=["text", "summary"],
)
class SummarizeTextFields(BaseTask):
    """
    Summarizes text-based columns, computing:
    - Average character length
    - Average word count
    - Average word length
    - Total characters
    - Most frequent string
    - Symbol presence flag
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data
            results: Dict[str, Dict[str, Any]] = {}

            if is_polars(df):
                for col in df.columns:
                    if is_text_polars(df[col]):
                        try:
                            strings = df[col].drop_nulls().to_list()
                            if not strings:
                                continue

                            char_counts = [len(s) for s in strings]
                            word_counts = [len(s.split()) for s in strings]
                            total_chars = sum(char_counts)
                            total_words = sum(word_counts)
                            avg_word_len = (
                                total_chars / total_words if total_words else 0
                            )

                            most_common = Counter(strings).most_common(1)
                            top_value = most_common[0][0] if most_common else None
                            has_symbols = any(re.search(r"[^\w\s]", s) for s in strings)

                            results[col] = {
                                "avg_char_length": sum(char_counts) / len(char_counts),
                                "avg_word_count": sum(word_counts) / len(word_counts),
                                "avg_word_length": avg_word_len,
                                "total_chars": total_chars,
                                "most_frequent_value": top_value,
                                "contains_symbols": has_symbols,
                            }
                        except Exception:
                            continue
            else:
                for col in df.columns:
                    if is_text_pandas(df[col]):
                        try:
                            texts = df[col].dropna().astype(str)
                            char_counts = texts.map(len)
                            word_counts = texts.map(lambda s: len(s.split()))
                            total_chars = char_counts.sum()
                            total_words = word_counts.sum()
                            avg_word_len = (
                                total_chars / total_words if total_words else 0
                            )

                            most_common = Counter(texts).most_common(1)
                            top_value = most_common[0][0] if most_common else None
                            has_symbols = any(re.search(r"[^\w\s]", s) for s in texts)

                            results[col] = {
                                "avg_char_length": char_counts.mean(),
                                "avg_word_count": word_counts.mean(),
                                "avg_word_length": avg_word_len,
                                "total_chars": total_chars,
                                "most_frequent_value": top_value,
                                "contains_symbols": has_symbols,
                            }
                        except Exception:
                            continue

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Summarized {len(results)} text column(s).",
                data=results,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during text field summarization: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
