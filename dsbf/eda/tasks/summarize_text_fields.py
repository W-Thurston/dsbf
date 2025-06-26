# dsbf/eda/tasks/summarize_text_fields.py

import re
from collections import Counter
from typing import Any

from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars


def summarize_text_fields(df: Any) -> TaskResult:
    """
    Summarizes text columns with average char length, word count, word length,
    total characters, most frequent string, and presence of non-alphanumeric symbols.

    Args:
        df (DataFrame): Input Pandas or Polars DataFrame.

    Returns:
        TaskResult: Dictionary with per-column text statistics.
    """
    results = {}
    try:
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
                        avg_word_len = total_chars / total_words if total_words else 0

                        most_common = Counter(strings).most_common(1)
                        top_value = most_common[0][0] if most_common else None
                        has_symbols = any(re.search(r"[^\\w\\s]", s) for s in strings)

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
                        avg_word_len = total_chars / total_words if total_words else 0

                        most_common = Counter(texts).most_common(1)
                        top_value = most_common[0][0] if most_common else None
                        has_symbols = any(re.search(r"[^\\w\\s]", s) for s in texts)

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

        return TaskResult(
            name="summarize_text_fields",
            status="success",
            summary=f"Summarized {len(results)} text column(s).",
            data=results,
        )

    except Exception as e:
        return TaskResult(
            name="summarize_text_fields",
            status="failed",
            summary=f"Error during text field summarization: {e}",
            data=None,
            metadata={"exception": type(e).__name__},
        )
