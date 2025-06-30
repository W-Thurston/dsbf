# dsbf/eda/tasks/suggest_categorical_encoding.py

from typing import Optional, cast

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars


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
    tags=["categorical", "encoding", "ml_readiness"],
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
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
