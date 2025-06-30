# dsbf/eda/tasks/detect_mixed_type_columns.py

from collections import Counter, defaultdict

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


@register_task(
    name="detect_mixed_type_columns",
    display_name="Detect Mixed-Type Columns",
    description=(
        "Flags columns that contain multiple" " Python data types (e.g., str + float)."
    ),
    depends_on=["infer_types"],
    tags=["type", "format", "anomaly"],
    profiling_depth="standard",
    stage="cleaned",
    inputs=["dataframe"],
    outputs=["TaskResult"],
)
class DetectMixedTypeColumns(BaseTask):
    """
    Detects columns that contain more than one data type,
        such as a mix of floats and strings.
    Focuses on object-like columns where dtype is not strongly enforced.
    """

    def run(self) -> None:

        try:

            # ctx = self.context
            df = self.input_data

            min_ratio = float(self.get_task_param("min_ratio") or 0.05)
            ignore_null_type = bool(self.get_task_param("ignore_null_type") or True)

            flagged_columns = []
            details = {}
            recommendations = []

            for col in df.columns:
                # Skip strictly typed Polars columns unless they're pl.Object
                if is_polars(df):
                    if df[col].dtype != pl.Object:
                        continue

                try:
                    values = df[col].to_numpy()
                except Exception:
                    continue  # Skip columns that fail conversion

                # Type counter (excluding None if requested)
                type_counter = Counter()
                for v in values:
                    if v is None:
                        if not ignore_null_type:
                            type_counter["NoneType"] += 1
                    else:
                        type_counter[type(v).__name__] += 1

                if len(type_counter) <= 1:
                    continue  # Only one type, skip

                total = sum(type_counter.values())
                if total == 0:
                    continue  # Column is entirely null or empty

                minority_types = {
                    t: count
                    for t, count in type_counter.items()
                    if count / total >= min_ratio
                    and count != max(type_counter.values())
                }

                if not minority_types:
                    continue

                # Collect sample values from minority types
                samples_by_type = defaultdict(list)
                for v in values:
                    if v is None and ignore_null_type:
                        continue
                    tname = type(v).__name__
                    if tname in minority_types and len(samples_by_type[tname]) < 5:
                        samples_by_type[tname].append(repr(v))

                flagged_columns.append(col)
                details[col] = {
                    "type_counts": dict(type_counter),
                    "sample_values": dict(samples_by_type),
                }
                recommendations.append(
                    f"Column '{col}' contains multiple data types"
                    f" ({', '.join(type_counter)}). "
                    f"Consider cleaning or coercing values."
                )

            summary = {
                "num_mixed_type_columns": len(flagged_columns),
                "columns": flagged_columns,
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data=details,
                recommendations=recommendations,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
