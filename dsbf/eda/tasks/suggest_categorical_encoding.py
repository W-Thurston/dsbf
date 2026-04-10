# dsbf/eda/tasks/suggest_categorical_encoding.py

from typing import cast

import polars as pl
from polars import DataFrame
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
    phase="ml_readiness",
    domain="core",
    runtime_estimate="fast",
    tags=["categorical", "encoding", "ml_readiness"],
    expected_semantic_types=["categorical"],
)
class SuggestCategoricalEncoding(BaseTask):
    """
    Recommend encoding strategies for categorical columns.

    Determines the optimal encoding approach for each categorical column
    based on cardinality:

    - **one-hot**: cardinality ≤ ``low_cardinality_threshold`` (default 10)
    - **frequency**: low < cardinality ≤ ``high_cardinality_threshold`` (default 50)
    - **frequency (high-cardinality)**: cardinality > high threshold

    If a numeric target column is configured, the task also checks whether
    label-encoded values correlate with the target and upgrades the strategy
    to include target encoding when correlation exceeds ``correlation_threshold``.

    Supports both Polars and Pandas DataFrames.

    Configurable parameters (via config["tasks"]["suggest_categorical_encoding"]):
        low_cardinality_threshold (int): Max unique values for one-hot. Default: 10
        high_cardinality_threshold (int): Max unique values for frequency. Default: 50
        correlation_threshold (float): Min abs target correlation for target
            encoding upgrade. Default: 0.3
        target_column (str): Optional name of the target column.
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute encoding strategy suggestion and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'categorical' column(s)",
                "debug",
            )

            low_threshold = int(self.get_task_param("low_cardinality_threshold") or 10)
            high_threshold = int(
                self.get_task_param("high_cardinality_threshold") or 50,
            )
            corr_threshold = float(self.get_task_param("correlation_threshold") or 0.3)
            target_col: str | None = self.get_task_param("target_column")

            if is_polars(df):
                # pl.Utf8 is a deprecated alias for pl.String in modern Polars.
                categorical_cols: list[str] = [
                    col
                    for col in df.columns
                    if df[col].dtype in (pl.String, pl.Utf8, pl.Categorical)
                ]
            else:
                categorical_cols = list(
                    df.select_dtypes(include=["object", "category"]).columns,
                )

            suggestions: dict[str, dict] = {}

            for col in categorical_cols:
                try:
                    n_unique = (
                        df[col].n_unique()
                        if is_polars(df)
                        else df[col].nunique(dropna=True)
                    )
                except Exception:  # noqa: BLE001, S112
                    continue

                if n_unique <= low_threshold:
                    strategy = "one-hot"
                elif n_unique <= high_threshold:
                    strategy = "frequency"
                else:
                    strategy = "frequency (high-cardinality)"

                # Optionally upgrade to target encoding if a numeric target is
                # available and the encoded column correlates with it.
                if target_col and target_col in df.columns:
                    try:
                        if is_polars(df):
                            if df[target_col].dtype.is_numeric():
                                unique_vals = df[col].unique().to_list()
                                category_to_int: dict[int, int] = {
                                    v: i for i, v in enumerate(unique_vals)
                                }
                                df_encoded = df.with_columns(
                                    pl.col(col)
                                    .replace(category_to_int)
                                    .cast(pl.Int64)
                                    .alias("encoded_cat"),
                                )
                                corr_df = df_encoded.select(
                                    ["encoded_cat", target_col],
                                ).drop_nulls()
                                corr_val = corr_df.select(
                                    pl.corr("encoded_cat", target_col),
                                )[0, 0]
                                corr: float = (
                                    abs(corr_val)
                                    if corr_val is not None
                                    and not pl.Series([corr_val]).is_nan().any()
                                    else 0.0
                                )
                            else:
                                corr = 0.0
                        elif df[target_col].dtype.kind in "iuf":
                            encoded = LabelEncoder().fit_transform(df[col].astype(str))
                            corr_matrix: DataFrame = (
                                pl.DataFrame(
                                    {"encoded": encoded, "target": df[target_col]},
                                )
                                .to_pandas()
                                .corr()
                            )
                            raw_corr = corr_matrix.iloc[0, 1]
                            corr = (
                                abs(cast("float", raw_corr))
                                if raw_corr is not None
                                else 0.0
                            )
                        else:
                            corr = 0.0

                        if corr > corr_threshold:
                            strategy: str = f"{strategy} + target encoding"

                    except Exception as e:  # noqa: BLE001
                        # Log and skip target encoding for this column - do not
                        # abort the entire task or corrupt self.output.
                        self._log(
                            f"    [{self.name}] Target correlation failed for "
                            f"'{col}': {type(e).__name__} - {e}",
                            "debug",
                        )

                # Track whether this column is a raw string dtype.
                # Raw strings (object / pl.String) will cause sklearn's
                # fit() to raise a ValueError; category dtype is tolerated
                # by some frameworks. This signal drives the ml_level below.
                if is_polars(df):
                    is_raw_string: bool = df[col].dtype in (pl.String, pl.Utf8)
                else:
                    is_raw_string = str(df[col].dtype) == "object"

                suggestions[col] = {
                    "cardinality": n_unique,
                    "suggested_encoding": strategy,
                    "is_raw_string": is_raw_string,
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Encoding suggestions generated for {len(suggestions)} "
                        "categorical columns."
                    ),
                },
                data={"encoding_suggestions": suggestions},
                recommendations=[
                    "Apply appropriate encoding based on cardinality. "
                    "Use target encoding for high-cardinality columns with "
                    "numeric correlation.",
                ],
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Encoding",
                    "display_priority": "high",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, col_data in suggestions.items():
                self._attach_guidance(
                    col,
                    col_data["cardinality"],
                    col_data["suggested_encoding"],
                    col_data["is_raw_string"],
                )

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and suggestions:
                top_col: str = next(iter(suggestions))
                top_strategy = suggestions[top_col]["suggested_encoding"]
                score: float = 0.8 if "target encoding" in top_strategy else 0.6
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"strategy": top_strategy},
                )
                self.set_ml_signals(
                    result=self.output,
                    score=score,
                    tags=["transform"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' is best encoded using: {top_strategy}. "
                        "This improves modeling of categorical variables."
                    ),
                )
                self.output.summary["column"] = top_col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(
        self,
        col: str,
        cardinality: int,
        strategy: str,
        is_raw_string: bool,  # noqa: FBT001
    ) -> None:
        """
        Generate EDA and ML guidance for a categorical column's encoding posture.

        The ML-phase severity depends on both *cardinality* and *dtype*:

        - Raw ``object``/``String`` columns will cause ``sklearn``'s ``fit()``
          to raise a ``ValueError`` — these always emit at least ``"warn"`` for
          the ML phase, regardless of cardinality, because they are not
          model-ready as-is.
        - ``category``-dtype columns are tolerated by some frameworks (e.g.
          LightGBM, CatBoost) and can receive ``"good"`` or ``"info"`` levels
          since the urgency depends on the downstream modeling stack.

        Strategy families:

        - ``one-hot`` — cardinality ≤ low_threshold (default 10)
        - ``frequency`` — low < cardinality ≤ high_threshold (default 50)
        - ``frequency (high-cardinality)`` — cardinality > high_threshold
        - any of the above ``+ target encoding`` — numeric target correlation found

        Args:
            col: Column name.
            cardinality: Number of unique non-null values.
            strategy: Encoding strategy string from the suggestion dict.
            is_raw_string: True when the column dtype is ``object`` (pandas) or
                ``String``/``Utf8`` (Polars) — i.e. a plain text column that
                sklearn cannot ingest without encoding.

        """
        has_target_encoding: bool = "target encoding" in strategy
        base_strategy: str = strategy.replace(" + target encoding", "").strip()

        if base_strategy == "one-hot":
            eda_level = "good"
            eda_title: str = f"Low Cardinality ({cardinality} values)"
            eda_body: str = (
                f"'{col}' has {cardinality} unique values - a manageable number of "
                f"distinct categories. Frequency distributions are easy to read and "
                f"group comparisons are statistically tractable. Bar charts and "
                f"grouped summaries will give a clear picture of value distribution."
            )
        elif base_strategy == "frequency":
            eda_level = "info"
            eda_title = f"Moderate Cardinality ({cardinality} values)"
            eda_body = (
                f"'{col}' has {cardinality} unique values - enough categories that "
                f"individual bars will be small but the column is still "
                f"comprehensible. Focus on the top 10-15 most frequent values first. "
                f"Check whether the long tail of rare categories represents genuine "
                f"diversity or sparse / miscoded data."
            )
        else:
            eda_level = "warn"
            eda_title = f"High Cardinality ({cardinality} values)"
            eda_body = (
                f"'{col}' has {cardinality} unique values - too many to analyse "
                f"category-by-category. Standard frequency plots will be unreadable "
                f"at this scale. Focus on the top-N most frequent values, the "
                f"distribution of frequency counts (how many categories appear only "
                f"once?), and whether this is a genuine feature or an identifier."
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

        if base_strategy == "one-hot":
            # Raw string columns must be encoded before sklearn can ingest them;
            # this is a concrete pre-modeling requirement, not just a suggestion.
            # category-dtype columns are framework-dependent — flagged as good.
            ml_level = "warn" if is_raw_string else "good"
            ml_title: str = (
                f"Encoding Required — One-Hot ({cardinality} values)"
                if is_raw_string
                else f"One-Hot Encoding Recommended ({cardinality} values)"
            )
            ml_body: str = (
                f"'{col}' is a raw string column with {cardinality} unique values. "
                f"sklearn (and most ML frameworks) cannot ingest string columns "
                f"directly — this column must be encoded before calling fit(). "
                f"One-hot encoding is the standard choice at this cardinality, "
                f"adding {cardinality} binary features. Drop one category to avoid "
                f"perfect multicollinearity in linear models "
                f"(drop='first' or 'if_binary')."
                if is_raw_string
                else f"'{col}' has {cardinality} unique values — one-hot encoding is "
                f"the standard choice. It adds {cardinality} binary features, compact"
                f" at this cardinality. Drop one category to avoid perfect "
                f"multicollinearity in linear models (drop='first' or 'if_binary')."
            )
            if has_target_encoding:
                ml_body += (
                    " Target correlation was detected — target encoding is also "
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
            ml_level = "warn" if is_raw_string else "info"
            ml_title = (
                f"Encoding Required — Frequency ({cardinality} values)"
                if is_raw_string
                else f"Frequency Encoding Recommended ({cardinality} values)"
            )
            ml_body = (
                f"'{col}' is a raw string column with {cardinality} unique values. "
                f"This column must be encoded before calling fit(). One-hot would "
                f"produce {cardinality} features — manageable but noisy for rare "
                f"categories. Frequency encoding replaces each category with its "
                f"count, reducing to a single numeric feature. Group rare categories "
                f"into 'Other' before encoding."
                if is_raw_string
                else f"'{col}' has {cardinality} unique values — one-hot would produce "
                f"{cardinality} features, manageable but noisy for rare categories. "
                f"Frequency encoding replaces each category with its count, "
                f"preserving ordinality of popularity in a single feature. Group "
                f"rare categories into 'Other' before encoding."
            )
            if has_target_encoding:
                ml_body += (
                    " Target correlation detected - target encoding may outperform "
                    "frequency encoding; use within cross-validation folds to "
                    "prevent leakage."
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
            ml_level = "warn"
            ml_title = f"High-Cardinality Encoding Required ({cardinality} values)"
            if has_target_encoding:
                ml_body = (
                    f"'{col}' has {cardinality} unique values and target correlation "
                    f"was detected. Target encoding is the recommended strategy - it "
                    f"distils the predictive relationship into a single numeric "
                    f"feature. Apply strictly within CV folds; fitting on the full "
                    f"training set causes target leakage."
                )
            else:
                ml_body = (
                    f"'{col}' has {cardinality} unique values. One-hot would create "
                    f"{cardinality} sparse binary features - almost certainly too "
                    f"many. Frequency encoding or hashing are the practical defaults. "
                    f"If a numeric target is available, target encoding often gives "
                    f"the best signal in a single feature but must be applied within "
                    f"CV folds. Confirm this is a genuine feature, not an identifier."
                )
            ml_actions = [
                {
                    "action": "encode",
                    "method": "target_encoding",
                    "column": col,
                    "detail": "Best for high-cardinality with numeric target; "
                    "use within CV folds",
                },
                {
                    "action": "encode",
                    "method": "frequency_encoding",
                    "column": col,
                    "detail": "No target needed; encodes popularity as numeric signal",
                },
                {
                    "action": "encode",
                    "method": "hash_encoding",
                    "column": col,
                    "detail": (
                        "Fixed output dimensionality; useful under memory constraints"
                    ),
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
