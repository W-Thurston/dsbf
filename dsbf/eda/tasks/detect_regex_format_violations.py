# dsbf/eda/tasks/detect_regex_format_violations.py

import re
from re import Pattern

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_text_polars


@register_task(
    name="detect_regex_format_violations",
    display_name="Detect Regex Format Violations",
    description=(
        "Detects string columns whose values do not conform to "
        "user-specified regex patterns."
    ),
    depends_on=["infer_types"],
    tags=["format", "regex", "validation", "anomaly"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    expected_semantic_types=["text"],
)
class DetectRegexFormatViolations(BaseTask):
    r"""
    Validates string columns against user-supplied regex patterns.

    For each column name present in the ``custom_patterns`` config dict, checks
    whether every non-null value matches the specified regex via ``re.fullmatch``.
    Columns not listed in ``custom_patterns`` are silently skipped — this task
    is a no-op unless patterns are configured.

    Only Polars String/Utf8 columns are checked (via ``is_text_polars``). Non-text
    columns are skipped.

    EDA guidance blurbs are emitted for each column with at least one violation,
    describing the pattern, violation count, and sample non-conforming values.

    Config format::

        tasks:
          detect_regex_format_violations:
            custom_patterns:
              email: "^[\\w.+-]+@[\\w-]+\\.[a-zA-Z]{2,}$"
              phone: "^\\+?[0-9]{7,15}$"
            max_violations: 5

    Configurable parameters (via config["tasks"]["detect_regex_format_violations"]):
        custom_patterns (dict): Mapping of column name → regex pattern string.
            Default: {} (task is a no-op with no patterns configured)
        max_violations (int): Maximum number of violating values stored per column
            in the summary. Default: 5
    """

    def run(self) -> None:
        """
        Execute regex format validation and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} 'text' column(s)", "debug")

            patterns: dict[str, str] = dict(
                self.get_task_param("custom_patterns") or {},
            )
            max_violations = int(self.get_task_param("max_violations") or 5)

            summary: dict = {}
            data: dict = {}
            recs: list[str] = []

            for col_name, pattern in patterns.items():
                if col_name not in df.columns:
                    continue

                col = df[col_name]
                if not is_text_polars(col):
                    self._log(
                        f"    [{self.name}] Skipping '{col_name}': not a text column.",
                        "debug",
                    )
                    continue

                try:
                    regex: Pattern[str] = re.compile(pattern)
                except re.error as e:
                    self._log(
                        f"    [{self.name}] Invalid regex for '{col_name}': {e}",
                        "debug",
                    )
                    continue

                values = col.drop_nulls().to_list()
                violations: list = [v for v in values if not regex.fullmatch(str(v))]
                num_violations: int = len(violations)

                if num_violations > 0:
                    summary[col_name] = {
                        "num_violations": num_violations,
                        "pattern": pattern,
                        "sample_violations": violations[:max_violations],
                    }
                    data[col_name] = violations
                    recs.append(
                        f"'{col_name}' has {num_violations} value(s) that do not "
                        f"match the expected format pattern: {pattern}",
                    )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "num_columns_with_violations": len(summary),
                    "columns": list(summary.keys()),
                    "violations": summary,
                },
                data=data,
                recommendations=recs,
                metadata={
                    "suggested_viz_type": "none",
                    "recommended_section": "Format",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col_name, col_summary in summary.items():
                self._attach_guidance(col_name, col_summary)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_summary: dict) -> None:
        """
        Generate EDA guidance for a column with regex format violations.

        Args:
            col: Column name.
            col_summary: Violation summary dict containing ``num_violations``,
                ``pattern``, and ``sample_violations``.

        """
        num = col_summary["num_violations"]
        pattern = col_summary["pattern"]
        samples = col_summary.get("sample_violations", [])
        sample_str: str = ", ".join(repr(v) for v in samples[:3])

        eda_body: str = (
            f"'{col}' has {num} value(s) that do not match the expected format "
            f"pattern ``{pattern}``. Sample non-conforming values: {sample_str}. "
            f"Format violations typically indicate data entry inconsistencies, "
            f"upstream pipeline changes, or free-text values in a structured field. "
            f"Investigate whether violations are correctable by normalisation "
            f"(e.g. stripping whitespace, lowercasing) or require manual review."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="warn",
            title=f"Regex Format Violations ({num} values)",
            body=eda_body.strip(),
            actions=[],
            metric={
                "num_violations": num,
                "pattern": pattern,
            },
        )
