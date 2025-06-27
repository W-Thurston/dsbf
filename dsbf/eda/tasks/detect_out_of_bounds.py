# dsbf/eda/tasks/detect_out_of_bounds.py

from typing import Any, Dict, Tuple

import numpy as np

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars

DEFAULT_RULES: Dict[str, Tuple[float, float]] = {
    "age": (0, 120),
    "temperature": (-100, 150),
    "percent": (0, 100),
    "score": (0, 1),
}


@register_task(
    display_name="Detect Out of Bounds",
    description="Detects numeric values outside expected or logical ranges.",
    depends_on=["infer_types"],
    stage="cleaned",
    tags=["bounds", "validation"],
)
class DetectOutOfBounds(BaseTask):
    """
    Detects numeric columns with values outside expected or domain-specific bounds.
    """

    def run(self) -> None:
        try:
            df: Any = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            bounds = {**DEFAULT_RULES, **self.config.get("custom_bounds", {})}
            flagged: Dict[str, Dict[str, Any]] = {}

            for col in df.select_dtypes(include=np.number).columns:
                if col in bounds:
                    lower, upper = bounds[col]
                    series = df[col].dropna()
                    violations = series[(series < lower) | (series > upper)]

                    if not violations.empty:
                        flagged[col] = {
                            "count": int(violations.count()),
                            "min_violation": float(violations.min()),
                            "max_violation": float(violations.max()),
                            "expected_range": (float(lower), float(upper)),
                        }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=f"Detected {len(flagged)} column(s) with out-of-bounds values.",
                data=flagged,
                metadata={"rule_columns": list(bounds.keys())},
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary=f"Error during bounds checking: {e}",
                data=None,
                metadata={"exception": type(e).__name__},
            )
