# dsbf/eda/tasks/suggest_dtype_optimizations.py

import numpy as np
import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

# ── Downcast rules ────────────────────────────────────────────────────────────
#
# Each rule is a tuple of (condition_fn, suggested_dtype, explanation).
# Rules are evaluated in order; the first match wins per column.
#
# condition_fn(series, inferred_dtype, analysis_intent_dtype) -> bool


def _fits_bool(series: pd.Series, inferred: str, intent: str) -> bool:
    """Int/object column with exactly two distinct values that map to True/False."""
    if inferred not in ("int64", "int32", "object"):
        return False
    vals = set(series.dropna().unique())
    return (
        vals <= {0, 1}
        or vals <= {True, False}
        or vals <= {"true", "false", "True", "False", "yes", "no", "Yes", "No"}
    )


def _fits_int8(series: pd.Series, inferred: str, intent: str) -> bool:
    """int64/int32 column whose range fits in int8 [-128, 127]."""
    if inferred not in ("int64", "int32", "int16") or intent != "continuous":
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(non_null.min() >= -128 and non_null.max() <= 127)


def _fits_int16(series: pd.Series, inferred: str, intent: str) -> bool:
    """int64/int32 column whose range fits in int16 [-32768, 32767]."""
    if inferred not in ("int64", "int32") or intent != "continuous":
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(non_null.min() >= -32_768 and non_null.max() <= 32_767)


def _fits_int32(series: pd.Series, inferred: str, intent: str) -> bool:
    """int64 column whose range fits in int32 [-2^31, 2^31-1]."""
    if inferred != "int64" or intent != "continuous":
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(non_null.min() >= -(2**31) and non_null.max() <= 2**31 - 1)


def _fits_float32(series: pd.Series, inferred: str, intent: str) -> bool:
    """float64 column where float32 precision is sufficient (no extreme values)."""
    if inferred != "float64" or intent != "continuous":
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    # float32 range: ~±3.4e38. Flag columns with values safely inside that range
    # and no values so small they would underflow to zero in float32.
    abs_vals = non_null.abs()
    max_val = abs_vals.max()
    min_nonzero = abs_vals[abs_vals > 0].min() if (abs_vals > 0).any() else np.inf
    return bool(max_val < 1e37 and (min_nonzero > 1e-37 or np.isinf(min_nonzero)))


def _fits_category(series: pd.Series, inferred: str, intent: str) -> bool:
    """Object column classified as categorical - category encoding saves memory."""
    return inferred == "object" and intent == "categorical"


# Ordered list of (check_fn, target_dtype, savings_note).
# bool check runs before int checks since bool takes priority.
_DOWNCAST_RULES: list[tuple] = [
    (_fits_bool, "bool", "2 values stored as int/object - bool uses 1 byte per row"),
    (_fits_int8, "int8", "range fits in int8 (1 byte vs 8 bytes per row)"),
    (_fits_int16, "int16", "range fits in int16 (2 bytes vs 8 bytes per row)"),
    (_fits_int32, "int32", "range fits in int32 (4 bytes vs 8 bytes per row)"),
    (
        _fits_float32,
        "float32",
        "float32 precision sufficient (4 bytes vs 8 bytes per row)",
    ),
    (
        _fits_category,
        "category",
        "low-cardinality string - category encoding stores one int per row",
    ),
]


def _estimate_savings_bytes(
    series: pd.Series,
    current_dtype: str,
    suggested_dtype: str,
) -> int:
    """
    Estimate memory saved per row x n_rows for the suggested downcast.

    Returns 0 when the saving cannot be estimated (e.g. category encoding
    depends on the number of unique values, which varies).
    """
    dtype_sizes: dict[str, int] = {
        "int64": 8,
        "int32": 4,
        "int16": 2,
        "int8": 1,
        "float64": 8,
        "float32": 4,
        "bool": 1,
    }
    current_size = dtype_sizes.get(current_dtype, 0)
    suggested_size = dtype_sizes.get(suggested_dtype, 0)
    if current_size == 0 or suggested_size == 0:
        return 0
    return (current_size - suggested_size) * len(series)


# ── Task ───────────────────────────────────────────────────────────────────────


@register_task(
    name="suggest_dtype_optimizations",
    display_name="Suggest Dtype Optimizations",
    description=(
        "Recommends dtype downcasts that reduce memory usage without losing "
        "information: int64→int8/16/32, float64→float32, object→category, "
        "int→bool. Reads infer_types and summarize_dataset_shape outputs."
    ),
    depends_on=["infer_types", "summarize_dataset_shape"],
    profiling_depth="standard",
    stage="raw",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["memory", "dtype", "optimization", "overview"],
    expected_semantic_types=["any"],
)
class SuggestDtypeOptimizations(BaseTask):
    """
    Recommend dtype downcasts that reduce memory without information loss.

    Evaluates each column against a set of downcast rules:

    - ``int64 / int32`` → ``int8``, ``int16``, or ``int32`` when value range fits
    - ``float64`` → ``float32`` when values are safely within float32 range
    - ``object`` (categorical intent) → ``category`` encoding
    - ``int / object`` with only 2 distinct values → ``bool``

    For each column where a smaller dtype is safe, an EDA guidance blurb is
    emitted with the suggested dtype, estimated byte savings, and a note on
    when the downcast is appropriate.

    This task is purely advisory - it never modifies the DataFrame. The
    optimization should be applied by the user before training or storage,
    not automatically by the profiling engine.

    Reads per-column memory from ``summarize_dataset_shape`` if available in
    context, to surface exact savings figures. Falls back to estimating from
    the raw series if the shape task has not run.

    Configurable parameters (via config["tasks"]["suggest_dtype_optimizations"]):
        min_savings_bytes (int): Minimum estimated byte savings required to emit
            a suggestion. Default: 1024 (1 KB). Prevents noise for tiny columns.
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute dtype optimization analysis and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data
            if is_polars(df):
                df = df.to_pandas()

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(f"    Processing {len(matched_cols)} column(s)", "debug")

            param = self.get_task_param("min_savings_bytes")
            min_savings = int(param) if param is not None else 1024

            # Read inferred dtypes from context (populated by infer_types).
            inferred_dtypes: dict[str, str] = {}
            analysis_intents: dict[str, str] = {}
            if self.context:
                inferred_dtypes = self.context.get_metadata("inferred_dtypes") or {}
                analysis_intents = self.context.get_metadata("semantic_types") or {}

            # Read per-column memory from summarize_dataset_shape if available.
            current_memory_bytes: dict[str, int] = {}
            if self.context:
                shape_result = self.context.results.get("summarize_dataset_shape")
                if shape_result and shape_result.status == "success":
                    current_memory_bytes = (
                        shape_result.data.get("column_memory_bytes") or {}
                    )

            suggestions: dict[str, dict] = {}
            total_potential_savings: int = 0

            for col in df.columns:
                series = df[col]
                inferred = inferred_dtypes.get(col, str(series.dtype))
                intent = analysis_intents.get(col, "unknown")

                for check_fn, target_dtype, savings_note in _DOWNCAST_RULES:
                    try:
                        if not check_fn(series, inferred, intent):
                            continue
                    except Exception:
                        continue

                    # Estimate savings from shape task data or from series size.
                    current_bytes = current_memory_bytes.get(col)
                    if current_bytes is not None:
                        estimated_savings: int = _estimate_savings_bytes(
                            series,
                            inferred,
                            target_dtype,
                        )
                        # If we can't estimate the savings (returns 0 from helper),
                        # use the raw current_bytes as the upper-bound savings
                        # for category (object → category can't be size-estimated
                        # without knowing n_unique).
                        if estimated_savings <= 0 and target_dtype == "category":
                            n_unique = series.nunique()
                            # category stores one int8 per row + a lookup table
                            estimated_savings = current_bytes - (
                                len(series) * 1 + n_unique * 50
                            )
                    else:
                        estimated_savings = _estimate_savings_bytes(
                            series,
                            inferred,
                            target_dtype,
                        )

                    if estimated_savings < min_savings:
                        self._log(
                            f"    '{col}' → {target_dtype} skipped: "
                            f"estimated savings {estimated_savings}B < {min_savings}B",
                            "debug",
                        )
                        continue

                    suggestions[col] = {
                        "current_dtype": inferred,
                        "suggested_dtype": target_dtype,
                        "estimated_savings_bytes": max(0, estimated_savings),
                        "estimated_savings_MB": round(
                            max(0, estimated_savings) / 1_048_576, 4
                        ),
                        "savings_note": savings_note,
                        "analysis_intent": intent,
                    }
                    total_potential_savings += max(0, estimated_savings)
                    self._log(
                        f"    '{col}': {inferred} → {target_dtype} "
                        f"(~{estimated_savings / 1024:.1f} KB saved)",
                        "debug",
                    )
                    break  # first matching rule wins

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "message": (
                        f"{len(suggestions)} dtype optimization(s) suggested, "
                        f"~{total_potential_savings / 1_048_576:.2f} MB potential "
                        "savings."
                    ),
                    "suggestion_count": len(suggestions),
                    "total_potential_savings_MB": round(
                        total_potential_savings / 1_048_576,
                        4,
                    ),
                },
                data={"suggestions": suggestions},
                metadata={
                    "min_savings_bytes": min_savings,
                    "suggested_viz_type": "table",
                    "recommended_section": "Overview",
                    "display_priority": "low",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col, info in suggestions.items():
                self._attach_guidance(col, info)

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, info: dict) -> None:
        """
        Generate an EDA guidance blurb for a column with a dtype optimization.

        Args:
            col: Column name.
            info: Suggestion dict containing ``current_dtype``, ``suggested_dtype``,
                ``estimated_savings_bytes``, and ``savings_note``.

        """
        current = info["current_dtype"]
        suggested = info["suggested_dtype"]
        savings_mb = info["estimated_savings_MB"]
        note = info["savings_note"]

        if suggested == "category":
            body: str = (
                f"'{col}' is stored as ``object`` dtype but has low cardinality "
                f"(classified as categorical). Converting to ``category`` stores one "
                f"integer per row and a compact lookup table instead of a full Python "
                f"string object per row. Estimated saving: ~{savings_mb:.4f} MB. "
                f"Apply with: ``df['{col}'] = df['{col}'].astype('category')``. "
                f"Note: category dtype sorts by code order rather than alphabetically "
                f"unless categories are explicitly ordered."
            )
        elif suggested == "bool":
            body = (
                f"'{col}' is stored as ``{current}`` but contains only 2 distinct "
                f"values. Converting to ``bool`` uses 1 byte per row vs "
                f"{note.split('-')[0].strip()}. Estimated saving: ~{savings_mb:.4f}"
                f" MB. Apply with: ``df['{col}'] = df['{col}'].astype(bool)``. "
                "Verify that the two values map correctly to True/False before"
                "converting."
            )
        elif suggested in ("int8", "int16", "int32"):
            body = (
                f"'{col}' is stored as ``{current}`` but its value range fits in "
                f"``{suggested}`` ({note}). "
                f"Estimated saving: ~{savings_mb:.4f} MB. "
                f"Apply with: ``df['{col}'] = df['{col}'].astype('{suggested}')``. "
                f"Caution: if new data arrives with values outside the current range, "
                f"the smaller dtype will overflow silently - only downcast when the "
                f"range is known to be stable."
            )
        else:  # float32
            body = (
                f"'{col}' is stored as ``float64`` but ``float32`` precision is "
                f"sufficient given the observed value range. {note}. "
                f"Estimated saving: ~{savings_mb:.4f} MB. "
                f"Apply with: ``df['{col}'] = df['{col}'].astype('float32')``. "
                f"Float32 has ~7 significant decimal digits vs float64's ~15. "
                f"Avoid downcasting columns used in high-precision financial or "
                f"scientific calculations."
            )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info",
            title=(
                f"Dtype Optimization: {current} → "
                f"{suggested} (~{savings_mb:.4f} MB saved)"
            ),
            body=body.strip(),
            actions=[
                {
                    "action": "downcast",
                    "column": col,
                    "from_dtype": current,
                    "to_dtype": suggested,
                    "detail": note,
                }
            ],
            metric={
                "current_dtype": current,
                "suggested_dtype": suggested,
                "estimated_savings_bytes": info["estimated_savings_bytes"],
                "estimated_savings_MB": savings_mb,
            },
        )
