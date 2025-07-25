# dsbf/eda/tasks/summarize_text_fields.py

import re
from collections import Counter
from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars
from dsbf.utils.plot_factory import PlotFactory


@register_task(
    display_name="Summarize Text Fields",
    description=(
        "Summarizes content of text columns" "(length, frequency, symbols, etc.)."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["text", "summary"],
    expected_semantic_types=["text"],
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

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_col)} 'text' column(s)", "debug")

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

                            self._log(f"    Summarized text column: {col}", "debug")
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

                            self._log(f"    Summarized text column: {col}", "debug")
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

            plots: dict[str, dict[str, Any]] = {}

            if is_polars(df):
                df = df.to_pandas()

            for col in results:
                series = df[col].dropna().astype(str).map(len)
                if series.empty:
                    continue

                save_path = self.get_output_path(f"{col}_text_length.png")
                static = PlotFactory.plot_histogram_static(series, save_path)
                interactive = PlotFactory.plot_histogram_interactive(series)

                avg_len = results[col].get("avg_char_length", 0)
                max_len = series.max()
                interactive["annotations"] = [f"Avg: {avg_len:.1f}, Max: {max_len}"]

                plots[col] = {
                    "static": static["path"],
                    "interactive": interactive,
                }

            self._log(f"    Processed {len(results)} text columns", "debug")
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": (f"Summarized {len(results)} text column(s).")},
                data=results,
                plots=plots,
                metadata={
                    "suggested_viz_type": "histogram",
                    "recommended_section": "Text Features",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} — {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
