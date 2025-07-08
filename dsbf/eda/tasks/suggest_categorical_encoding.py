# dsbf/eda/tasks/suggest_categorical_encoding.py

from typing import Any, Optional, cast

import pandas as pd
import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.plot_factory import PlotFactory
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    name="suggest_categorical_encoding",
    display_name="Suggest Categorical Encoding",
    description=(
        "Recommends encoding strategies (e.g., one-hot, frequency, target) "
        "based on column cardinality and optional target correlation."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="modeling",
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "encoding", "ml_readiness"],
    expected_semantic_types=["categorical"],
)
class SuggestCategoricalEncoding(BaseTask):
    """
    Suggests categorical encoding strategies based on:
    - Cardinality thresholds
    - Optional numeric target correlation (if target is provided)

    Strategies:
      - One-hot encoding: cardinality <= low_threshold
      - Frequency encoding: low < cardinality <= high_threshold
      - Target encoding: numeric target and correlated

    Supports both Polars and Pandas backends.
    """

    def run(self) -> None:
        try:
            df = self.input_data

            # Use semantic typing to select relevant columns
            matched_col, excluded = self.get_columns_by_intent()
            self._log(f"Processing {len(matched_col)} 'categorical' column(s)", "debug")

            low_threshold = int(self.get_task_param("low_cardinality_threshold") or 10)
            high_threshold = int(
                self.get_task_param("high_cardinality_threshold") or 50
            )
            corr_threshold = float(self.get_task_param("correlation_threshold") or 0.3)
            target_col: Optional[str] = self.get_task_param("target_column")

            if is_polars(df):
                categorical_cols = [
                    col
                    for col in df.columns
                    if df[col].dtype in (pl.Utf8, pl.Categorical)
                ]
            else:
                categorical_cols = [
                    col
                    for col in df.select_dtypes(include=["object", "category"]).columns
                ]

            suggestions = {}

            for col in categorical_cols:
                # Get cardinality
                try:
                    n_unique = (
                        df[col].n_unique()
                        if is_polars(df)
                        else df[col].nunique(dropna=True)
                    )
                except Exception:
                    continue

                # Suggest encoding
                if n_unique <= low_threshold:
                    strategy = "one-hot"
                elif n_unique <= high_threshold:
                    strategy = "frequency"
                else:
                    strategy = "frequency (high-cardinality)"

                # If numeric target provided and available, suggest target encoding
                if target_col and target_col in df.columns:
                    try:
                        if is_polars(df):
                            if df[target_col].dtype.is_numeric():
                                unique_vals = df[col].unique().to_list()
                                category_to_int = {
                                    v: i for i, v in enumerate(unique_vals)
                                }

                                # Add encoded column using replace()
                                df_encoded = df.with_columns(
                                    [
                                        pl.col(col)
                                        .replace(category_to_int)
                                        .cast(pl.Int64)
                                        .alias("encoded_cat")
                                    ]
                                )

                                # Compute correlation
                                corr_df = df_encoded.select(
                                    ["encoded_cat", target_col]
                                ).drop_nulls()
                                corr_val = corr_df.select(
                                    pl.corr("encoded_cat", target_col)
                                )[0, 0]
                                corr = (
                                    abs(corr_val)
                                    if corr_val is not None
                                    and not pl.Series([corr_val]).is_nan().any()
                                    else 0
                                )
                            else:
                                corr = 0
                        else:
                            if df[target_col].dtype.kind in "iuf":
                                from sklearn.preprocessing import LabelEncoder

                                encoded = LabelEncoder().fit_transform(
                                    df[col].astype(str)
                                )
                                corr_matrix = (
                                    pl.DataFrame(
                                        {"encoded": encoded, "target": df[target_col]}
                                    )
                                    .to_pandas()
                                    .corr()
                                )

                                raw_corr = corr_matrix.iloc[0, 1]
                                corr: float = (
                                    abs(cast(float, raw_corr))
                                    if raw_corr is not None
                                    else 0.0
                                )
                            else:
                                corr = 0

                        if corr > corr_threshold:
                            strategy = f"{strategy} + target encoding"

                    except Exception as e:
                        if self.context:
                            raise
                        self.output = make_failure_result(self.name, e)

                suggestions[col] = {
                    "cardinality": n_unique,
                    "suggested_encoding": strategy,
                }

            summary = {
                "message": (
                    f"Encoding suggestions generated for {len(suggestions)}"
                    " categorical columns."
                )
            }

            plots: dict[str, dict[str, Any]] = {}

            try:
                # Build barplot of cardinality across all categorical columns
                card_series = pd.Series(
                    {col: val["cardinality"] for col, val in suggestions.items()},
                    name="Cardinality",
                ).sort_values(ascending=False)

                if not card_series.empty:
                    save_path = self.get_output_path("categorical_cardinality.png")
                    static = PlotFactory.plot_barplot_static(
                        card_series,
                        save_path=save_path,
                        title="Categorical Cardinalities",
                    )
                    interactive = PlotFactory.plot_barplot_interactive(
                        card_series,
                        title="Categorical Cardinalities",
                        annotations=["Used for encoding strategy suggestions"],
                    )

                    plots["cardinality_distribution"] = {
                        "static": static["path"],
                        "interactive": interactive,
                    }
            except Exception as e:
                self._log(
                    f"[PlotFactory] Skipped categorical cardinality plot: {e}",
                    level="debug",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data={"encoding_suggestions": suggestions},
                recommendations=[
                    "Apply appropriate encoding based on cardinality. "
                    "Use target encoding for high-cardinality columns with"
                    " numeric correlation."
                ],
                plots=plots,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Encoding",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys())
                    ),
                },
            )

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and suggestions:
                col = next(iter(suggestions))
                strategy = suggestions[col]["suggested_encoding"]
                score = 0.8 if "target encoding" in strategy else 0.6
                tip = get_recommendation_tip(self.name, {"strategy": strategy})
                self.set_ml_signals(
                    result=self.output,
                    score=score,
                    tags=["transform"],
                    recommendation=tip
                    or (
                        f"Column '{col}' is best encoded using: {strategy}. "
                        "This improves modeling of categorical variables."
                    ),
                )
                self.output.summary["column"] = col

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
