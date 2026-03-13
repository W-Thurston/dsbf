# dsbf/eda/tasks/suggest_categorical_encoding.py

from typing import cast

import polars as pl
from sklearn.preprocessing import LabelEncoder

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
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
            self._log(
                f"    Processing {len(matched_col)} 'categorical' column(s)",
                "debug",
            )

            low_threshold = int(self.get_task_param("low_cardinality_threshold") or 10)
            high_threshold = int(
                self.get_task_param("high_cardinality_threshold") or 50,
            )
            corr_threshold = float(self.get_task_param("correlation_threshold") or 0.3)
            target_col: str | None = self.get_task_param("target_column")

            if is_polars(df):
                categorical_cols: list = [
                    col
                    for col in df.columns
                    if df[col].dtype in (pl.Utf8, pl.Categorical)
                ]
            else:
                categorical_cols = list(
                    df.select_dtypes(include=["object", "category"]).columns,
                )

            suggestions: dict = {}

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
                                        .alias("encoded_cat"),
                                    ],
                                )

                                # Compute correlation
                                corr_df = df_encoded.select(
                                    ["encoded_cat", target_col],
                                ).drop_nulls()
                                corr_val = corr_df.select(
                                    pl.corr("encoded_cat", target_col),
                                )[0, 0]
                                corr = (
                                    abs(corr_val)
                                    if corr_val is not None
                                    and not pl.Series([corr_val]).is_nan().any()
                                    else 0
                                )
                            else:
                                corr = 0
                        elif df[target_col].dtype.kind in "iuf":
                            encoded = LabelEncoder().fit_transform(df[col].astype(str))
                            corr_matrix = (
                                pl.DataFrame(
                                    {"encoded": encoded, "target": df[target_col]},
                                )
                                .to_pandas()
                                .corr()
                            )

                            raw_corr = corr_matrix.iloc[0, 1]
                            corr: float = (
                                abs(cast("float", raw_corr))
                                if raw_corr is not None
                                else 0.0
                            )
                        else:
                            corr = 0

                        if corr > corr_threshold:
                            strategy: str = f"{strategy} + target encoding"

                    except Exception as e:
                        if self.context:
                            raise
                        self.output = make_failure_result(self.name, e)

                suggestions[col] = {
                    "cardinality": n_unique,
                    "suggested_encoding": strategy,
                }

            summary: dict[str, str] = {
                "message": (
                    f"Encoding suggestions generated for {len(suggestions)}"
                    " categorical columns."
                ),
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data={"encoding_suggestions": suggestions},
                recommendations=[
                    "Apply appropriate encoding based on cardinality. "
                    "Use target encoding for high-cardinality columns with"
                    " numeric correlation.",
                ],
                plots={},
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Encoding",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_col + list(excluded.keys()),
                    ),
                },
            )

            for col, col_data in suggestions.items():
                self._attach_guidance(
                    col,
                    col_data["cardinality"],
                    col_data["suggested_encoding"],
                )

            # Apply ML scoring to self.output
            if self.get_engine_param("enable_impact_scoring", True) and suggestions:
                col = next(iter(suggestions))
                strategy = suggestions[col]["suggested_encoding"]
                score: float = 0.8 if "target encoding" in strategy else 0.6
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"strategy": strategy},
                )
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
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, cardinality: int, strategy: str) -> None:
        """
        Generate EDA + ML guidance for a categorical column's encoding posture.

        Strategy families (from the task's logic):
        - one-hot                      - cardinality ≤ low_threshold (default 10)
        - frequency                    - low < cardinality ≤ high_threshold (default 50)
        - frequency (high-cardinality) - cardinality > high_threshold
        - any of the above + target encoding - when numeric target correlation found
        """
        has_target_encoding: bool = "target encoding" in strategy
        base_strategy: str = strategy.replace(" + target encoding", "").strip()

        # --- EDA blurb: describe the cardinality tier ---
        if base_strategy == "one-hot":
            eda_level = "good"
            eda_title: str = f"Low Cardinality ({cardinality} values)"
            eda_body: str = (
                f"{col} has {cardinality} unique values - a manageable number of "
                f"distinct categories. Frequency distributions are easy to read and "
                f"group comparisons are statistically tractable. This is the simplest "
                f"cardinality tier to analyse; bar charts and grouped summaries will "
                f"give a clear picture of how values are distributed."
            )
        elif base_strategy == "frequency":
            eda_level = "info"
            eda_title = f"Moderate Cardinality ({cardinality} values)"
            eda_body = (
                f"{col} has {cardinality} unique values - enough categories that "
                "individual bars will be small but the column is still comprehensible. "
                "Focus on the top 10-15 most frequent values first to understand where "
                "the majority of observations sit. Check whether the long tail of rare "
                "categories represents genuine diversity or sparse / miscoded data."
            )
        else:
            # frequency (high-cardinality)
            eda_level = "warn"
            eda_title = f"High Cardinality ({cardinality} values)"
            eda_body = (
                f"{col} has {cardinality} unique values - too many to analyse "
                f"category-by-category. Standard frequency plots will be unreadable "
                f"at this scale. Focus on the top-N most frequent values, the "
                f"distribution of frequency counts themselves (how many categories "
                f"appear only once?), and whether the column is a genuine categorical "
                f"feature or effectively an identifier."
            )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level=eda_level,
            title=eda_title,
            body=eda_body.strip(),
            actions=[],
            metric={"cardinality": cardinality, "suggested_encoding": strategy},
        )

        # --- ML blurb: prescribe encoding with tradeoffs ---
        if base_strategy == "one-hot":
            ml_level = "good"
            ml_title: str = f"One-Hot Encoding Recommended ({cardinality} values)"
            ml_body: str = (
                f"{col} has {cardinality} unique values - one-hot encoding is the "
                f"standard choice. It adds {cardinality} binary features, which is "
                f"compact at this cardinality. Works with all model families. "
                f"Drop one category to avoid perfect multicollinearity in linear "
                f"models (use drop='first' or 'if_binary')."
            )
            if has_target_encoding:
                ml_body += (
                    f" Target correlation was detected - target encoding is also "
                    f"viable if you want a single ordinal feature rather than "
                    f"{cardinality} binary columns."
                )
            ml_actions: list[dict[str, str]] = [
                {
                    "action": "encode",
                    "method": "one_hot",
                    "column": col,
                    "detail": "Standard for low-cardinality categoricals",
                },
                {
                    "action": "drop_first",
                    "column": col,
                    "detail": "Avoid dummy variable trap in linear models",
                },
            ]

        elif base_strategy == "frequency":
            ml_level = "info"
            ml_title = f"Frequency Encoding Recommended ({cardinality} values)"
            ml_body = (
                f"{col} has {cardinality} unique values - one-hot would produce "
                f"{cardinality} features, which is manageable but adds noise for "
                f"rare categories. Frequency encoding replaces each category with "
                f"its count (or proportion), preserving ordinality of popularity "
                f"in a single feature. Group rare categories into 'Other' before "
                f"encoding to reduce noise from singletons."
            )
            if has_target_encoding:
                ml_body += (
                    " Target correlation was detected - target encoding may "
                    "outperform frequency encoding; use with cross-validation "
                    "folds to prevent target leakage."
                )
            ml_actions = [
                {
                    "action": "encode",
                    "method": "frequency_encoding",
                    "column": col,
                    "detail": "Replaces category with its row count or proportion",
                },
                {
                    "action": "group_rare",
                    "column": col,
                    "detail": "Collapse low-frequency categories into 'Other' first",
                },
            ]
            if has_target_encoding:
                ml_actions.append(
                    {
                        "action": "encode",
                        "method": "target_encoding",
                        "column": col,
                        "detail": "Use within CV folds to prevent leakage",
                    },
                )

        else:
            # frequency (high-cardinality)
            ml_level = "warn"
            ml_title = f"High-Cardinality Encoding Required ({cardinality} values)"
            ml_body = (
                f"{col} has {cardinality} unique values. One-hot encoding would "
                f"create {cardinality} sparse binary features - almost certainly "
                f"too many. Frequency encoding or hashing are the practical "
                f"defaults. If a numeric target is available, target encoding "
                f"(mean of target per category) often gives the best signal in "
                f"a single feature but must be applied within cross-validation "
                f"folds to prevent leakage. Verify this is a genuine feature "
                f"and not an identifier before encoding."
            )
            if has_target_encoding:
                ml_body = (
                    f"{col} has {cardinality} unique values and target correlation "
                    f"was detected. Target encoding is the recommended strategy - "
                    f"it distils the predictive relationship between category and "
                    f"target into a single numeric feature. Apply strictly within "
                    f"CV folds; fitting on the full training set causes target leakage."
                )
            ml_actions = [
                {
                    "action": "encode",
                    "method": "target_encoding",
                    "column": col,
                    "detail": "Best signal for high-cardinality with a numeric target;"
                    " use within CV folds",
                },
                {
                    "action": "encode",
                    "method": "frequency_encoding",
                    "column": col,
                    "detail": "No target needed; encodes popularity as a numeric "
                    "signal",
                },
                {
                    "action": "encode",
                    "method": "hash_encoding",
                    "column": col,
                    "detail": "Fixed output dimensionality; useful under memory "
                    "constraints",
                },
                {
                    "action": "drop",
                    "column": col,
                    "detail": "If confirmed to be an identifier rather than a feature",
                },
            ]

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level=ml_level,
            title=ml_title,
            body=ml_body.strip(),
            actions=ml_actions,
            metric={"cardinality": cardinality, "suggested_encoding": strategy},
        )
