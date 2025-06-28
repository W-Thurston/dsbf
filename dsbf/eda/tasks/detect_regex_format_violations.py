import re

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
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
    stage="cleaned",
    inputs=["dataframe"],
    outputs=["TaskResult"],
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
            df = self.input_data
            cfg = self.config or {}
            patterns = cfg.get("patterns", {})
            max_violations = cfg.get("max_violations_to_store", 5)

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
            )
        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": f"Error during Regex Format Detection: {e}"},
                data=None,
                metadata={"exception": type(e).__name__},
            )
