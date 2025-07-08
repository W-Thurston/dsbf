# dsbf/eda/tasks/compare_with_reference_dataset.py

from typing import Any, Dict

import pandas as pd
import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.plot_factory import PlotFactory


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
    tags=["schema", "drift", "comparison"],
    expected_semantic_types=["any"],
)
class CompareWithReferenceDataset(BaseTask):
    """
    Compares the current dataset with the reference
    dataset to identify schema and structural differences.

    Checks for:
      - Added or dropped columns
      - Type mismatches
      - Changes in min/max, missing %, and unique counts for shared columns

    Requires that a reference dataset is available in `ctx.reference_data`.
    """

    def run(self) -> None:
        ctx = self.context
        current_df = self.input_data
        reference_df = getattr(ctx, "reference_data", None)

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
            missing_threshold = float(
                self.get_task_param("missing_pct_threshold") or 0.3
            )
            unique_threshold = float(
                self.get_task_param("unique_count_ratio_threshold") or 0.5
            )
            minmax_tol = float(self.get_task_param("minmax_numeric_tolerance") or 0.01)

            added = list(set(current_df.columns) - set(reference_df.columns))
            dropped = list(set(reference_df.columns) - set(current_df.columns))
            shared = list(set(current_df.columns) & set(reference_df.columns))

            type_mismatches = []
            field_changes: Dict[str, Dict[str, Any]] = {}

            for col in shared:
                try:
                    cur_dtype = current_df.schema[col]
                    ref_dtype = reference_df.schema[col]
                    if cur_dtype != ref_dtype:
                        type_mismatches.append(col)

                    if cur_dtype.is_numeric() and cur_dtype == ref_dtype:
                        cur_min, cur_max = current_df.select(
                            [
                                pl.col(col).min().alias("min_val"),
                                pl.col(col).max().alias("max_val"),
                            ]
                        ).row(0)

                        ref_min, ref_max = reference_df.select(
                            [
                                pl.col(col).min().alias("min_val"),
                                pl.col(col).max().alias("max_val"),
                            ]
                        ).row(0)

                        rel_min_diff = (
                            abs(cur_min - ref_min) / (abs(ref_min) + 1e-6)
                            if ref_min is not None
                            else None
                        )
                        rel_max_diff = (
                            abs(cur_max - ref_max) / (abs(ref_max) + 1e-6)
                            if ref_max is not None
                            else None
                        )
                    else:
                        cur_min = cur_max = ref_min = ref_max = None
                        rel_min_diff = rel_max_diff = None

                    cur_missing = (
                        current_df.select(pl.col(col).is_null().sum()).item()
                        / current_df.height
                    )
                    ref_missing = (
                        reference_df.select(pl.col(col).is_null().sum()).item()
                        / reference_df.height
                    )
                    missing_diff = abs(cur_missing - ref_missing)

                    cur_nunique = current_df.select(pl.col(col).n_unique()).item()
                    ref_nunique = reference_df.select(pl.col(col).n_unique()).item()
                    unique_diff = abs(cur_nunique - ref_nunique)
                    unique_ratio_diff = unique_diff / max(ref_nunique, 1)

                    field_changes[col] = {
                        "missing_pct_current": round(cur_missing, 4),
                        "missing_pct_reference": round(ref_missing, 4),
                        "missing_pct_diff": round(missing_diff, 4),
                        "flag_missing_diff": missing_diff > missing_threshold,
                        "unique_count_current": cur_nunique,
                        "unique_count_reference": ref_nunique,
                        "unique_count_diff": unique_diff,
                        "unique_count_diff_ratio": round(unique_ratio_diff, 4),
                        "flag_unique_diff": unique_ratio_diff > unique_threshold,
                        "min_current": cur_min,
                        "max_current": cur_max,
                        "min_reference": ref_min,
                        "max_reference": ref_max,
                        "flag_min_diff": rel_min_diff is not None
                        and rel_min_diff > minmax_tol,
                        "flag_max_diff": rel_max_diff is not None
                        and rel_max_diff > minmax_tol,
                    }
                except Exception as e:
                    field_changes[col] = {"error": str(e)}

            summary = {
                "added_columns": sorted(added),
                "dropped_columns": sorted(dropped),
                "type_mismatches": sorted(type_mismatches),
                "field_changes": field_changes,
            }

            recommendations = []
            if added:
                recommendations.append(f"New columns detected: {', '.join(added)}")
            if dropped:
                recommendations.append(f"Dropped columns: {', '.join(dropped)}")
            if type_mismatches:
                recommendations.append(
                    f"Type mismatches in: {', '.join(type_mismatches)}"
                )

            # Plotting
            plots: dict[str, dict[str, Any]] = {}

            try:
                drift_counts = {
                    col: sum(
                        1 for k in data if k.startswith("flag_") and data[k] is True
                    )
                    for col, data in field_changes.items()
                    if isinstance(data, dict)
                }

                drift_series = pd.Series(drift_counts, name="Drift Flags").sort_values(
                    ascending=False
                )

                if not drift_series.empty:
                    save_path = self.get_output_path("reference_drift_flags.png")
                    static = PlotFactory.plot_barplot_static(
                        drift_series,
                        save_path=save_path,
                        title="Reference Drift Flags per Column",
                    )
                    interactive = PlotFactory.plot_barplot_interactive(
                        drift_series,
                        title="Reference Drift Flags per Column",
                        annotations=["Flags: missing %, unique count, min/max"],
                    )

                    plots["reference_drift_flags"] = {
                        "static": static["path"],
                        "interactive": interactive,
                    }

            except Exception as e:
                self._log(f"[PlotFactory] Skipped drift barplot: {e}", level="debug")

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data=summary,
                recommendations=recommendations,
                plots=plots,
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
