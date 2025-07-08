# dsbf/eda/tasks/detect_high_cardinality.py

from typing import Any, Dict

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    display_name="Detect High Cardinality",
    description="Detects columns with too many unique values.",
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "cardinality"],
    expected_semantic_types=["categorical"],
)
class DetectHighCardinality(BaseTask):
    """
    Detects columns with a number of unique values greater than a threshold.
    """

    def run(self) -> None:
        """
        Execute the high-cardinality detection task and store the results in
            `self.output`.
        """
        try:

            # ctx = self.context
            df: Any = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} 'categorical' column(s)", "debug")

            cardinality_threshold = float(
                self.get_task_param("cardinality_threshold") or 50
            )

            results: Dict[str, int] = {}

            if is_polars(df):
                for col in df.columns:
                    try:
                        n_unique = df[col].n_unique()
                        if n_unique > cardinality_threshold:
                            results[col] = n_unique
                            self._log(f"{col} has {n_unique} unique values", "debug")
                    except Exception:
                        continue
            else:
                for col in df.columns:
                    try:
                        n_unique = df[col].nunique()
                        if n_unique > cardinality_threshold:
                            results[col] = n_unique
                            self._log(f"{col} has {n_unique} unique values", "debug")
                    except Exception:
                        continue

            # Plotting
            plots: dict[str, dict[str, Any]] = {}

            if is_polars(df):
                df = df.to_pandas()

            for col in results:
                series = df[col].dropna()
                counts = series.value_counts().head(10)
                counts.name = col

                n_unique = results[col]
                annotation = [f"Detected {n_unique} unique values"]

                save_path = self.get_output_path(f"{col}_high_cardinality.png")
                static = PlotFactory.plot_barplot_static(counts, save_path)
                interactive = PlotFactory.plot_barplot_interactive(counts)
                interactive["annotations"] = annotation

                plots[col] = {
                    "static": static["path"],
                    "interactive": interactive,
                }

            # Build TaskResult
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (f"Detected {len(results)} high-cardinality column(s).")
                },
                data=results,
                plots=plots,
                metadata={
                    "cardinality_threshold": cardinality_threshold,
                    "suggested_viz_type": "bar",
                    "recommended_section": "Cardinality",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and results:
                col = next(iter(results))  # First offending column
                n_unique = results[col]
                result = self.output
                if result:
                    tip = get_recommendation_tip(self.name, {"n_unique": n_unique})
                    self.set_ml_signals(
                        result=result,
                        score=0.7,
                        tags=["transform", "monitor"],
                        recommendation=tip
                        or (
                            f"Column '{col}' has high cardinality "
                            f"({n_unique} unique values). "
                            "Consider frequency encoding, bucketing, or"
                            " dimensionality reduction."
                        ),
                    )
                    result.summary["column"] = col

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
