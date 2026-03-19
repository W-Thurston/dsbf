# dsbf/eda/tasks/compare_with_reference_dataset.py

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat

    from dsbf.core.context import AnalysisContext


@register_task(
    name="compare_with_reference_dataset",
    display_name="Compare with Reference Dataset",
    description=(
        "Compares schema and structural differences"
        " between the current and reference dataset."
    ),
    depends_on=["infer_types"],
    profiling_depth="basic",
    stage="raw",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["schema", "drift", "comparison"],
    expected_semantic_types=["any"],
)
class CompareWithReferenceDataset(BaseTask):
    """
    Compares the current dataset against a reference dataset for structural drift.

    Checks for added or dropped columns, type mismatches, and per-column changes
    in missingness rate, unique count, and numeric range. Emits EDA guidance blurbs
    for any detected schema or distributional changes.

    Both DataFrames are normalized to pandas at entry since this task performs
    structural metadata comparison rather than heavy numerical computation,
    and the reference dataset is always loaded as pandas via pd.read_csv in
    profile_engine.py.

    Requires that a reference dataset is available via ``ctx.reference_data``.
    If none is provided, the task returns a skipped result.

    Configurable parameters (via config["tasks"]["compare_with_reference_dataset"]):
        missing_pct_threshold (float): Missingness difference that triggers a flag.
            Default: 0.3
        unique_count_ratio_threshold (float): Unique count ratio change that triggers
            a flag. Default: 0.5
        minmax_numeric_tolerance (float): Relative change in min/max that triggers
            a flag. Default: 0.01
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute the reference dataset comparison and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        ctx: AnalysisContext | None = self.context
        current_df = self.input_data
        reference_df: Any | None = getattr(ctx, "reference_data", None)

        if reference_df is None:
            self.output = TaskResult(
                name=self.name,
                status="skipped",
                summary={"message": "[SKIPPED] No reference dataset provided."},
                data={},
                recommendations=[],
            )
            return

        try:
            # Normalize both DataFrames to pandas for structural comparison.
            # The reference is always pandas (pd.read_csv); normalize current
            # to match and avoid Polars/pandas API mismatches throughout.
            if is_polars(current_df):
                current_pd: pd.DataFrame = current_df.to_pandas()
            else:
                current_pd = current_df

            if isinstance(reference_df, pl.DataFrame):
                reference_pd: pd.DataFrame = reference_df.to_pandas()
            else:
                reference_pd = reference_df

            missing_threshold = float(
                self.get_task_param("missing_pct_threshold") or 0.3,
            )
            unique_threshold = float(
                self.get_task_param("unique_count_ratio_threshold") or 0.5,
            )
            minmax_tol = float(self.get_task_param("minmax_numeric_tolerance") or 0.01)

            current_cols: set[str] = set(current_pd.columns)
            reference_cols: set[str] = set(reference_pd.columns)

            added: list[str] = sorted(current_cols - reference_cols)
            dropped: list[str] = sorted(reference_cols - current_cols)
            shared: list[str] = sorted(current_cols & reference_cols)

            type_mismatches: list[str] = []
            field_changes: dict[str, dict[str, Any]] = {}

            for col in shared:
                try:
                    cur_dtype = current_pd[col].dtype
                    ref_dtype = reference_pd[col].dtype

                    if cur_dtype != ref_dtype:
                        type_mismatches.append(col)

                    # Numeric range comparison — only meaningful when types match
                    # and both are numeric.
                    is_numeric: bool = pd.api.types.is_numeric_dtype(cur_dtype)
                    if is_numeric and cur_dtype == ref_dtype:
                        cur_min = float(current_pd[col].min())
                        cur_max = float(current_pd[col].max())
                        ref_min = float(reference_pd[col].min())
                        ref_max = float(reference_pd[col].max())

                        rel_min_diff: float = abs(cur_min - ref_min) / (
                            abs(ref_min) + 1e-6
                        )
                        rel_max_diff: float = abs(cur_max - ref_max) / (
                            abs(ref_max) + 1e-6
                        )
                    else:
                        cur_min = cur_max = ref_min = ref_max = None
                        rel_min_diff = rel_max_diff = None

                    cur_missing: float = current_pd[col].isna().mean()
                    ref_missing: float = reference_pd[col].isna().mean()
                    missing_diff: ConvertibleToFloat = abs(cur_missing - ref_missing)

                    cur_nunique = int(current_pd[col].nunique())
                    ref_nunique = int(reference_pd[col].nunique())
                    unique_diff: int = abs(cur_nunique - ref_nunique)
                    unique_ratio_diff: float = unique_diff / max(ref_nunique, 1)

                    field_changes[col] = {
                        "missing_pct_current": round(float(cur_missing), 4),
                        "missing_pct_reference": round(float(ref_missing), 4),
                        "missing_pct_diff": round(float(missing_diff), 4),
                        "flag_missing_diff": bool(missing_diff > missing_threshold),
                        "unique_count_current": cur_nunique,
                        "unique_count_reference": ref_nunique,
                        "unique_count_diff": unique_diff,
                        "unique_count_diff_ratio": round(float(unique_ratio_diff), 4),
                        "flag_unique_diff": bool(unique_ratio_diff > unique_threshold),
                        "min_current": cur_min,
                        "max_current": cur_max,
                        "min_reference": ref_min,
                        "max_reference": ref_max,
                        "flag_min_diff": rel_min_diff is not None
                        and bool(rel_min_diff > minmax_tol),
                        "flag_max_diff": rel_max_diff is not None
                        and bool(rel_max_diff > minmax_tol),
                    }

                except Exception as e:  # noqa: BLE001, PERF203
                    field_changes[col] = {"error": str(e)}

            data: dict[str, Any] = {
                "added_columns": added,
                "dropped_columns": dropped,
                "type_mismatches": sorted(type_mismatches),
                "field_changes": field_changes,
            }

            recommendations: list[str] = []
            if added:
                recommendations.append(
                    f"New columns detected since reference: {', '.join(added)}",
                )
            if dropped:
                recommendations.append(
                    f"Columns dropped since reference: {', '.join(dropped)}",
                )
            if type_mismatches:
                recommendations.append(
                    f"Type mismatches vs. reference in: {', '.join(type_mismatches)}",
                )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "added_columns": added,
                    "dropped_columns": dropped,
                    "type_mismatches": sorted(type_mismatches),
                    "message": (
                        f"{len(added)} added, {len(dropped)} dropped, "
                        f"{len(type_mismatches)} type mismatches vs. reference."
                    ),
                },
                data=data,
                recommendations=recommendations,
            )

            # Emit EDA guidance blurbs for meaningful schema changes.
            for col in added:
                self.add_guidance(
                    result=self.output,
                    column=col,
                    phase="eda",
                    level="info",
                    title="New column vs. reference dataset",
                    body=(
                        f"'{col}' is present in the current dataset but absent from "
                        "the reference. Verify this is intentional and not an "
                        "upstream schema change."
                    ),
                    actions=[],
                    metric={"status": "added"},
                )

            for col in dropped:
                self.add_guidance(
                    result=self.output,
                    column=col,
                    phase="eda",
                    level="warn",
                    title="Column dropped vs. reference dataset",
                    body=(
                        f"'{col}' was present in the reference dataset but is absent "
                        "from the current dataset. This may indicate a pipeline or "
                        "schema change that affects downstream analysis."
                    ),
                    actions=[],
                    metric={"status": "dropped"},
                )

            for col in type_mismatches:
                cur = str(current_pd[col].dtype)
                ref = str(reference_pd[col].dtype)
                self.add_guidance(
                    result=self.output,
                    column=col,
                    phase="eda",
                    level="warn",
                    title=f"Type mismatch vs. reference ({ref} → {cur})",
                    body=(
                        f"'{col}' changed dtype from {ref} (reference) to {cur} "
                        "(current). This may cause join failures, aggregation errors, "
                        "or silent coercion depending on downstream usage."
                    ),
                    actions=[],
                    metric={"current_dtype": cur, "reference_dtype": ref},
                )

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)
