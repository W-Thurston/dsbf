# dsbf/eda/tasks/extract_datetime_features.py

from typing import Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

# ── Feature catalogue ─────────────────────────────────────────────────────────
#
# Each entry describes one extractable feature. The task computes the actual
# distribution for EDA (value counts / range), and emits the list as action
# chips for the ML guidance blurb.
#
# Cyclic features use sin/cos encoding to preserve the circular nature of
# time components (hour 23 is adjacent to hour 0).

_ORDINAL_FEATURES: list[dict] = [
    {
        "name": "year",
        "extractor": lambda s: s.dt.year,
        "description": "Calendar year",
        "ml_note": "Captures long-term trend. Use with caution — may cause leakage "
        "if the target shifts over time.",
    },
    {
        "name": "month",
        "extractor": lambda s: s.dt.month,
        "description": "Month of year (1-12)",
        "ml_note": "Captures seasonal patterns. Pair with cyclic encoding for "
        "circular distance (January is adjacent to December).",
    },
    {
        "name": "day_of_month",
        "extractor": lambda s: s.dt.day,
        "description": "Day of month (1-31)",
        "ml_note": "Useful when within-month patterns exist (e.g. payroll dates).",
    },
    {
        "name": "day_of_week",
        "extractor": lambda s: s.dt.dayofweek,
        "description": "Day of week (0=Monday, 6=Sunday)",
        "ml_note": "Captures weekly seasonality. Pair with is_weekend for simpler "
        "weekly patterns.",
    },
    {
        "name": "hour",
        "extractor": lambda s: s.dt.hour,
        "description": "Hour of day (0-23)",
        "ml_note": "Captures intra-day patterns. Use cyclic encoding for models "
        "sensitive to distance (hour 23 ≈ hour 0).",
    },
    {
        "name": "quarter",
        "extractor": lambda s: s.dt.quarter,
        "description": "Calendar quarter (1-4)",
        "ml_note": "Coarser seasonal signal than month. Useful when quarterly "
        "business cycles drive the target.",
    },
    {
        "name": "week_of_year",
        "extractor": lambda s: s.dt.isocalendar().week.astype(int),
        "description": "ISO week of year (1-53)",
        "ml_note": "Higher granularity than month for weekly seasonal patterns.",
    },
    {
        "name": "days_since_epoch",
        "extractor": lambda s: (s - pd.Timestamp("1970-01-01")).dt.days,
        "description": "Days elapsed since 1970-01-01",
        "ml_note": "Ordinal representation of the full datetime. Captures long-term "
        "trend without calendar structure. Compatible with all model types.",
    },
]

_BINARY_FEATURES: list[dict] = [
    {
        "name": "is_weekend",
        "extractor": lambda s: s.dt.dayofweek >= 5,
        "description": "True if Saturday or Sunday",
        "ml_note": "Simple binary weekend flag. Useful when weekend/weekday "
        "behaviour differs substantially.",
    },
    {
        "name": "is_month_start",
        "extractor": lambda s: s.dt.is_month_start,
        "description": "True if the first day of a month",
        "ml_note": "Useful for datasets with month-start events (e.g. billing cycles).",
    },
    {
        "name": "is_month_end",
        "extractor": lambda s: s.dt.is_month_end,
        "description": "True if the last day of a month",
        "ml_note": "Captures end-of-month patterns common in financial datasets.",
    },
]

_CYCLIC_FEATURES: list[dict] = [
    {
        "name": "month_sin_cos",
        "period": 12,
        "ordinal_name": "month",
        "extractor": lambda s: s.dt.month,
        "description": "Cyclic sin/cos encoding of month (preserves Jan≈Dec adjacency)",
        "ml_note": "Replace raw month with sin(2πxmonth/12) and cos(2πxmonth/12) "
        "for distance-based models and neural networks.",
    },
    {
        "name": "day_of_week_sin_cos",
        "period": 7,
        "ordinal_name": "day_of_week",
        "extractor": lambda s: s.dt.dayofweek,
        "description": (
            "Cyclic sin/cos encoding of day of week (preserves Mon≈Sun adjacency)"
        ),
        "ml_note": "Replace raw day_of_week with sin(2πxdow/7) and cos(2πxdow/7).",
    },
    {
        "name": "hour_sin_cos",
        "period": 24,
        "ordinal_name": "hour",
        "extractor": lambda s: s.dt.hour,
        "description": (
            "Cyclic sin/cos encoding of hour (preserves 23:00≈00:00 adjacency)"
        ),
        "ml_note": "Replace raw hour with sin(2πxhour/24) and cos(2πxhour/24) "
        "for any model sensitive to distance.",
    },
]


def _temporal_summary(series: pd.Series) -> dict[str, Any]:
    """
    Compute a summary of the temporal distribution of a datetime column.

    Args:
        series: Non-null datetime Series.

    Returns:
        Dict with ``min``, ``max``, ``range_days``, ``n_unique_dates``,
        ``has_time_component``, and ``dominant_components`` keys.

    """
    min_dt = series.min()
    max_dt = series.max()
    range_days = (max_dt - min_dt).days

    has_time = bool((series.dt.hour != 0).any() or (series.dt.minute != 0).any())
    n_unique = series.nunique()

    # Identify which time components have meaningful variation (> 1 distinct value)
    dominant: list[str] = []
    for component, extractor in [
        ("year", lambda s: s.dt.year),
        ("month", lambda s: s.dt.month),
        ("day_of_week", lambda s: s.dt.dayofweek),
        ("hour", lambda s: s.dt.hour),
    ]:
        try:
            array = extractor(series).to_numpy()
            if array.shape[0] != 0 or (array[0] != array).all():
                dominant.append(component)
        except Exception:  # noqa: BLE001, PERF203, S110
            pass

    return {
        "min": str(min_dt),
        "max": str(max_dt),
        "range_days": range_days,
        "n_unique_dates": int(n_unique),
        "has_time_component": has_time,
        "dominant_components": dominant,
    }


def _relevant_features(summary: dict[str, Any]) -> list[str]:
    """
    Determine which features are relevant given the temporal summary.

    Features that have no variation (e.g. hour features when all timestamps
    are midnight) are excluded to avoid recommending useless extractions.

    Args:
        summary: Output of ``_temporal_summary``.

    Returns:
        List of feature names to recommend.

    """
    dominant: set = set(summary["dominant_components"])
    has_time = summary["has_time_component"]
    range_days = summary["range_days"]

    recommended: list[str] = [
        "days_since_epoch",
    ]  # always useful as ordinal representation

    if range_days > 365:  # noqa: PLR2004
        recommended.append("year")
    if "month" in dominant:
        recommended += ["month", "quarter", "week_of_year", "month_sin_cos"]
    if "day_of_week" in dominant:
        recommended += ["day_of_week", "is_weekend", "day_of_week_sin_cos"]
    if "day_of_month" in dominant or range_days > 28:  # noqa: PLR2004
        recommended += ["day_of_month", "is_month_start", "is_month_end"]
    if has_time and "hour" in dominant:
        recommended += ["hour", "hour_sin_cos"]

    # Deduplicate while preserving order
    seen: set[str] = set()
    return [f for f in recommended if not (f in seen or seen.add(f))]


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="extract_datetime_features",
    display_name="Extract Datetime Features",
    description=(
        "Analyses datetime columns and recommends derived features: "
        "year, month, day-of-week, hour, is_weekend, quarter, cyclic "
        "sin/cos encodings, and days-since-epoch."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["datetime", "feature_engineering", "ml_readiness"],
    expected_semantic_types=["datetime"],
)
class ExtractDatetimeFeatures(BaseTask):
    """
    Analyse datetime columns and recommend derived features for modeling.

    For each column classified as ``datetime`` by ``infer_types``, computes:

    **EDA profile:**
    - Date range (min, max, span in days)
    - Number of unique timestamps
    - Whether a time-of-day component is present
    - Which time components show meaningful variation (year, month,
      day-of-week, hour)

    **ML feature recommendations** (as structured action chips):
    - Ordinal features: year, month, day_of_month, day_of_week, hour,
      quarter, week_of_year, days_since_epoch
    - Binary flags: is_weekend, is_month_start, is_month_end
    - Cyclic encodings: month_sin_cos, day_of_week_sin_cos, hour_sin_cos

    Features are filtered to those relevant for the column's actual data —
    hour features are not recommended when all timestamps are midnight, year
    is not recommended for sub-annual datasets, etc.

    This task is purely advisory — it never modifies the DataFrame. Raw
    datetime columns are not usable as model inputs directly; their value
    lies entirely in the extracted components.

    Polars DataFrames are converted to pandas for datetime accessor support.
    """

    def run(self) -> None:  # noqa: C901, PLR0912
        """
        Execute datetime feature analysis and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} 'datetime' column(s)",
                "debug",
            )

            # Identify datetime columns: prefer semantic types from infer_types,
            # fall back to pandas datetime dtype detection.
            semantic_types: dict[str, str] = {}
            if self.context:
                semantic_types = self.context.get_metadata("semantic_types") or {}

            datetime_cols: list[str] = []
            for col in df.columns:
                intent: str = semantic_types.get(col, "")
                if intent == "datetime" or (
                    not intent and pd.api.types.is_datetime64_any_dtype(df[col])
                ):
                    datetime_cols.append(col)

            if not datetime_cols:
                # Attempt to parse object columns that look like dates
                for col in df.select_dtypes(include=["object"]).columns:
                    try:
                        parsed = pd.to_datetime(
                            df[col],
                            infer_datetime_format=True,
                            errors="coerce",
                        )
                        if parsed.notna().mean() > 0.9:
                            df[col] = parsed
                            datetime_cols.append(col)
                            self._log(
                                f"    '{col}' inferred as datetime from object dtype.",
                                "debug",
                            )
                    except Exception:  # noqa: BLE001, PERF203, S110
                        pass

            results: dict[str, dict] = {}

            for col in datetime_cols:
                series = df[col].dropna()
                if len(series) < 2:
                    self._log(
                        f"    '{col}' skipped: fewer than 2 non-null values.",
                        "debug",
                    )
                    continue

                # Ensure datetime dtype
                if not pd.api.types.is_datetime64_any_dtype(series):
                    try:
                        series = pd.to_datetime(series, errors="coerce").dropna()
                    except Exception as e:
                        self._log(
                            f"    '{col}' could not be parsed as datetime: {e}",
                            "debug",
                        )
                        continue

                try:
                    summary: dict[str, Any] = _temporal_summary(series)
                except Exception as e:
                    self._log(f"    '{col}' temporal summary failed: {e}", "debug")
                    continue

                recommended: list[str] = _relevant_features(summary)
                self._log(
                    f"    '{col}': range={summary['range_days']}d, "
                    f"{len(recommended)} features recommended.",
                    "debug",
                )

                results[col] = {
                    "temporal_summary": summary,
                    "recommended_features": recommended,
                }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"Analysed {len(results)} datetime column(s); "
                        f"{
                            sum(
                                len(v['recommended_features']) for v in results.values()
                            )
                        } "
                        f"total feature extraction(s) recommended."
                    ),
                    "datetime_columns_found": len(results),
                    "total_features_recommended": sum(
                        len(v["recommended_features"]) for v in results.values()
                    ),
                },
                data=results,
                metadata={
                    "suggested_viz_type": "table",
                    "recommended_section": "Distributions",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, col_data in results.items():
                self._attach_guidance(col, col_data)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_data: dict) -> None:
        """
        Generate EDA and ML guidance for a datetime column.

        Args:
            col: Column name.
            col_data: Result dict containing ``temporal_summary`` and
                ``recommended_features``.

        """
        summary = col_data["temporal_summary"]
        recommended = col_data["recommended_features"]
        range_days = summary["range_days"]
        has_time = summary["has_time_component"]
        dominant = summary["dominant_components"]

        # --- EDA blurb ---
        components_str: str = ", ".join(dominant) if dominant else "limited"
        eda_body: str = (
            f"'{col}' spans {range_days:,} days "
            f"({summary['min'][:10]} to {summary['max'][:10]}) "
            f"with {summary['n_unique_dates']:,} unique timestamp(s). "
            f"Varying components: {components_str}. "
            f"{'A time-of-day component is present. ' if has_time else ''}"
            f"Raw datetime values are not interpretable by most models — "
            f"their predictive signal lives in extracted components such as "
            f"day-of-week, month, or hour. Inspect the distribution of each "
            f"recommended component to understand seasonal and cyclical patterns."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info",
            title=f"Datetime Column — {len(recommended)} Feature(s) Suggested",
            body=eda_body.strip(),
            actions=[],
            metric={
                "range_days": range_days,
                "n_unique_dates": summary["n_unique_dates"],
                "has_time_component": has_time,
                "dominant_components": dominant,
            },
        )

        # --- ML blurb: one action chip per recommended feature ---
        # Build lookup dicts from all feature catalogues
        all_features: dict[str, dict] = {}
        for feat in _ORDINAL_FEATURES + _BINARY_FEATURES + _CYCLIC_FEATURES:
            all_features[feat["name"]] = feat

        actions: list = []
        for feat_name in recommended:
            feat: dict = all_features.get(feat_name)
            if feat is None:
                continue
            actions.append(
                {
                    "action": "extract_feature",
                    "feature": feat_name,
                    "column": col,
                    "description": feat["description"],
                    "ml_note": feat["ml_note"],
                },
            )

        if not actions:
            return

        ml_body: str = (
            f"'{col}' cannot be used as a raw model input — datetime values "
            f"must be decomposed into numeric features. "
            f"{len(actions)} extraction(s) recommended based on the column's "
            f"temporal range and variation. Cyclic encodings (sin/cos) are "
            f"recommended for distance-based models and neural networks; "
            f"ordinal extractions are suitable for tree-based models. "
            f"Always extract from the full dataset before train/test splitting "
            f"to prevent look-ahead bias."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="ml",
            level="info",
            title=f"Datetime Feature Extraction Required ({len(actions)} feature(s))",
            body=ml_body.strip(),
            actions=actions,
            metric={
                "recommended_features": recommended,
                "feature_count": len(recommended),
            },
        )
