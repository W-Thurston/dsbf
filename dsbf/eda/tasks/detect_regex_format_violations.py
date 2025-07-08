import re

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_text_polars


@register_task(
    name="detect_regex_format_violations",
    display_name="Detect Regex Format Violations",
    description=(
        "Detects string columns whose values do not "
        "conform to user-specified regex patterns."
    ),
    depends_on=["infer_types"],
    tags=["format", "regex", "validation", "anomaly"],
    profiling_depth="full",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    inputs=["dataframe"],
    outputs=["TaskResult"],
    expected_semantic_types=["text"],
)
class DetectRegexFormatViolations(BaseTask):
    """
    Detects string columns that fail to match specified regex formats.

    Config format:
    tasks:
      detect_regex_format_violations:
        patterns:
          column_name: regex_pattern
        max_violations_to_store: 5
    """

    def run(self):
        try:

            # ctx = self.context
            df = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} 'text' column(s)", "debug")

            patterns = dict(self.get_task_param("custom_patterns") or {})
            max_violations = int(self.get_task_param("max_violations") or 5)

            summary = {}
            data = {}
            recs = []

            for col_name, pattern in patterns.items():
                if col_name not in df.columns:
                    continue
                col = df[col_name]
                if not is_text_polars(col):
                    continue

                try:
                    regex = re.compile(pattern)
                except re.error:
                    continue  # Invalid regex pattern

                values = col.drop_nulls().to_list()
                violations = [v for v in values if not regex.fullmatch(str(v))]
                num_violations = len(violations)

                if num_violations > 0:
                    summary[col_name] = {
                        "num_violations": num_violations,
                        "sample_violations": violations[:max_violations],
                    }
                    data[col_name] = violations
                    recs.append(
                        f"`{col_name}` has {num_violations} value(s)"
                        f" that do not match the expected format."
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
                    "suggested_viz_type": "None",
                    "recommended_section": "Format",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )
        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
