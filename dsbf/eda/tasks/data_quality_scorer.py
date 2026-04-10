# dsbf/eda/tasks/data_quality_scorer.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult


def _level(pct_affected: float, any_affected: bool, max_severity: str = "info") -> str:
    """
    Map a proportion of affected columns to a traffic-light level.

    With a severity floor so that error-level findings never produce a green result.

    Thresholds are proportional to dataset size so that e.g. 3 affected
    columns means something very different in a 10-column dataset vs a
    1000-column one.

    Severity floor rules:
        - ``"error"`` findings → minimum level is ``"amber"`` (green → amber;
          amber and red unchanged). A dimension cannot be green if any finding
          is error-severity.
        - ``"warn"`` / ``"info"`` → no floor; proportion alone determines level.

    Args:
        pct_affected: Fraction of total columns affected (0.0 - 1.0).
        any_affected: True if at least one column is affected. Used to
            return ``"green"`` cleanly when the count is zero, avoiding
            edge cases when total_columns is very small.
        max_severity: The highest severity among all findings in the dimension.
            Defaults to ``"info"`` (no floor).

    Returns:
        One of ``"green"``, ``"amber"``, or ``"red"``.

    """
    if not any_affected:
        return "green"

    # Proportion-based level
    if pct_affected <= 0.05:
        prop_level = "green"
    elif pct_affected <= 0.15:
        prop_level = "amber"
    else:
        prop_level = "red"

    # Severity floor: error findings can never be green
    _rank: dict[str, int] = {"green": 0, "amber": 1, "red": 2}
    floor_level: str = "amber" if max_severity == "error" else "green"

    return max(prop_level, floor_level, key=lambda lv: _rank[lv])


def _category_block(
    affected_columns: list[str],
    total_columns: int,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the standard per-category output dict for a data-health dimension.

    Args:
        affected_columns: Columns flagged by any source task for this dimension.
        total_columns: Total column count in the dataset (denominator for pct).
        findings: List of per-column finding dicts for this dimension.

    Returns:
        Dict with keys: ``affected_columns``, ``affected_count``,
        ``total_columns``, ``pct_affected``, ``level``, and ``findings``.

    """
    affected_count: int = len(affected_columns)
    pct: float = affected_count / total_columns if total_columns else 0.0

    _sev_rank: dict[str, int] = {"error": 2, "warn": 1, "info": 0}
    max_severity: str = max(
        (f.get("severity", "info") for f in findings),
        key=lambda s: _sev_rank.get(s, 0),
        default="info",
    )

    return {
        "label": "",  # populated by caller if needed
        "affected_columns": affected_columns,
        "affected_count": affected_count,
        "total_columns": total_columns,
        "pct_affected": round(pct, 4),
        "level": _level(pct, bool(affected_columns), max_severity),
        "findings": findings,
    }


@register_task(
    name="data_quality_scorer",
    display_name="Data Quality Scorer",
    description=(
        "Aggregates EDA profiling results into five data-health dimensions: "
        "Completeness, Validity, Usability, Redundancy, and Leakage. "
        "Each dimension reports affected column counts and a proportional "
        "traffic-light level (green / amber / red) for the header bar."
    ),
    tags=["scoring", "summary", "meta"],
    stage="report",
    domain="core",
    profiling_depth="basic",
    runtime_estimate="fast",
    phase="diagnostic",
    expected_semantic_types=["any"],
    # Explicit depends_on is critical: without it the topological sort places
    # this task at level 0, causing it to run before any source tasks have
    # written their results to context.results - producing all-green output.
    depends_on=[
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        # detect_out_of_bounds is intentionally excluded from hard dependencies:
        # it can fail when config bounds are defined for column names that don't
        # exist in the dataset, or when dtype casting issues occur. The scorer
        # handles a missing or failed result gracefully via the null-safe guard in
        # the validity block - validity simply shows no out-of-bounds findings.
        # Including it as a hard dependency causes the entire scorer (and
        # ml_readiness_scorer) to be skipped when one bounds check fails.
        "detect_constant_columns",
        "detect_zeros",
        "detect_id_columns",
        "detect_single_dominant_value",
        "detect_high_cardinality",
        # detect_collinear_features is intentionally excluded: it can fail on
        # datasets with constant or near-constant numeric columns (zero-size
        # array error in VIF computation). The scorer handles a missing or
        # failed result gracefully - Redundancy simply shows no findings.
        # Including it as a hard dependency would cause the scorer to be
        # skipped whenever VIF fails, producing no health data at all.
        "detect_data_leakage",
    ],
)
class DataQualityScorer(BaseTask):
    """
    Aggregates completed EDA task outputs into a structured data-health summary.

    Reads ``context.results`` after all source tasks have run and produces a
    five-dimension summary used to populate the Data Health Bar and Quality tab.

    Categories and source tasks:

    - **Completeness** - ``summarize_nulls``: columns where ≥ 5% of values are
      missing.
    - **Validity** - ``detect_out_of_bounds`` (domain violations),
      ``detect_constant_columns`` (zero-information columns),
      ``detect_zeros`` (> 95% zeros - structural empties).
    - **Usability** - ``detect_id_columns`` (IDs masquerading as features),
      ``detect_single_dominant_value`` (≥ 95% single value),
      ``detect_high_cardinality`` (near-unique categoricals).
    - **Redundancy** - ``detect_collinear_features`` (VIF > 10).
    - **Leakage** - ``detect_data_leakage`` (near-perfectly correlated pairs).

    Output data shape::

        {
            "total_columns": int,
            "all_columns": [str, ...],
            "categories": {
                "<dimension>": {
                    "affected_columns": [str, ...],
                    "affected_count":   int,
                    "total_columns":    int,
                    "pct_affected":     float,
                    "level":            "green" | "amber" | "red",
                    "findings":         [{"column": str, "issue": str, ...}, ...]
                },
                ...
            }
        }

    The ``summary`` field mirrors category levels for fast API access::

        {"completeness": "green", "validity": "amber", ...}
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Aggregate source task results into the data-health summary.

        Raises:
            RuntimeError: If no AnalysisContext is set.

        """
        if self.context is None:
            raise RuntimeError("AnalysisContext is not set.")

        results: dict[str, TaskResult] = self.context.results

        # ── Total column count ────────────────────────────────────────────────
        # Primary source: summarize_dataset_shape.
        # Fallback: count keys from infer_types (always runs).
        total_columns: int = 0
        shape_task: TaskResult | None = results.get("summarize_dataset_shape")
        if shape_task and shape_task.status == "success" and shape_task.data:
            total_columns = int(shape_task.data.get("num_columns") or 0)

        if not total_columns:
            types_task: TaskResult | None = results.get("infer_types")
            if types_task and types_task.status == "success" and types_task.data:
                total_columns = len(types_task.data)

        if not total_columns:
            for task in results.values():
                if task.status == "success" and isinstance(task.data, dict):
                    candidate: int = len(task.data)
                    total_columns = max(total_columns, candidate)

        # Full ordered column list - used by the frontend to derive clean columns.
        all_columns: list[str] = []
        types_task = results.get("infer_types")
        if types_task and types_task.status == "success" and types_task.data:
            all_columns = list(types_task.data.keys())

        # ── COMPLETENESS ─────────────────────────────────────────────────────
        # Source: summarize_nulls → data.null_percentages {col: float 0-1}
        # Threshold: ≥ 5% missing (matches EDA guidance blurb lower bound).
        completeness_cols: list[str] = []
        completeness_findings: list[dict[str, Any]] = []

        nulls_task: TaskResult | None = results.get("summarize_nulls")
        if nulls_task and nulls_task.status == "success" and nulls_task.data:
            null_pcts: dict[str, float] = nulls_task.data.get("null_percentages") or {}
            for col, pct in null_pcts.items():
                if pct >= 0.05:
                    completeness_cols.append(col)
                    if pct >= 0.50:
                        severity = "error"
                    elif pct >= 0.20:
                        severity = "warn"
                    else:
                        severity = "info"
                    completeness_findings.append(
                        {
                            "column": col,
                            "issue": "missing_values",
                            "pct_null": round(pct, 4),
                            "severity": severity,
                        },
                    )

        completeness_findings.sort(key=lambda f: f["pct_null"], reverse=True)

        # ── VALIDITY ─────────────────────────────────────────────────────────
        # Sources:
        #   detect_out_of_bounds    → data {col: {count, min_violation, …}}
        #   detect_constant_columns → data.constant_columns [col, …]
        #   detect_zeros            → data.zero_flags {col: bool}
        #                             (task threshold is > 95% zeros - structural)
        validity_cols: list[str] = []
        validity_findings: list[dict[str, Any]] = []

        oob_task: TaskResult | None = results.get("detect_out_of_bounds")
        if oob_task and oob_task.status == "success" and oob_task.data:
            for col, info in oob_task.data.items():
                if col not in validity_cols:
                    validity_cols.append(col)
                validity_findings.append(
                    {
                        "column": col,
                        "issue": "out_of_bounds",
                        "violation_count": info.get("count"),
                        "severity": "warn",
                    },
                )

        const_task: TaskResult | None = results.get("detect_constant_columns")
        if const_task and const_task.status == "success" and const_task.data:
            for col in const_task.data.get("constant_columns") or []:
                if col not in validity_cols:
                    validity_cols.append(col)
                validity_findings.append(
                    {
                        "column": col,
                        "issue": "constant_column",
                        "severity": "error",
                    },
                )

        zeros_task: TaskResult | None = results.get("detect_zeros")
        if zeros_task and zeros_task.status == "success" and zeros_task.data:
            zero_flags: dict[str, bool] = zeros_task.data.get("zero_flags") or {}
            zero_pcts: dict[str, float] = zeros_task.data.get("zero_percentages") or {}
            for col, flagged in zero_flags.items():
                if flagged and col not in validity_cols:
                    validity_cols.append(col)
                    validity_findings.append(
                        {
                            "column": col,
                            "issue": "structural_zeros",
                            "pct_zero": round(zero_pcts.get(col, 0.0), 4),
                            "severity": "warn",
                        },
                    )

        # ── USABILITY ────────────────────────────────────────────────────────
        # Sources:
        #   detect_id_columns            → data {col: "N unique (likely ID)"}
        #   detect_single_dominant_value → data {col: {mode_proportion, …}}
        #   detect_high_cardinality      → data {col: n_unique}
        usability_cols: list[str] = []
        usability_findings: list[dict[str, Any]] = []

        # Build a set of columns confirmed as numeric by summarize_numeric.
        # detect_id_columns can incorrectly flag float columns with high
        # cardinality when infer_types misclassifies them. Any column present
        # in summarize_numeric is genuinely numeric and should never be
        # reported as a likely ID.
        numeric_cols: set[str] = set()
        num_task: TaskResult | None = results.get("summarize_numeric")
        if num_task and num_task.status == "success" and num_task.data:
            numeric_cols = set(num_task.data.keys())

        id_task: TaskResult | None = results.get("detect_id_columns")
        if id_task and id_task.status == "success" and id_task.data:
            for col in id_task.data:
                if col in numeric_cols:
                    continue
                if col not in usability_cols:
                    usability_cols.append(col)
                usability_findings.append(
                    {
                        "column": col,
                        "issue": "likely_id",
                        "severity": "warn",
                    },
                )

        dom_task: TaskResult | None = results.get("detect_single_dominant_value")
        if dom_task and dom_task.status == "success" and dom_task.data:
            dom_threshold: float = float(
                (dom_task.metadata or {}).get("dominance_threshold") or 0.95,
            )
            for col, info in dom_task.data.items():
                if not isinstance(info, dict):
                    continue
                mode_prop: float = info.get("mode_proportion") or 0.0
                if mode_prop < dom_threshold:
                    continue
                if col not in usability_cols:
                    usability_cols.append(col)
                usability_findings.append(
                    {
                        "column": col,
                        "issue": "dominant_value",
                        "mode_proportion": round(mode_prop, 4),
                        "severity": "warn",
                    },
                )

        card_task: TaskResult | None = results.get("detect_high_cardinality")
        if card_task and card_task.status == "success" and card_task.data:
            for col, n_unique in card_task.data.items():
                if col in numeric_cols:
                    continue
                if col not in usability_cols:
                    usability_cols.append(col)
                usability_findings.append(
                    {
                        "column": col,
                        "issue": "high_cardinality",
                        "n_unique": n_unique,
                        "severity": "warn",
                    },
                )

        # ── REDUNDANCY ───────────────────────────────────────────────────────
        # Source: detect_collinear_features → data.collinear_columns [col]
        redundancy_cols: list[str] = []
        redundancy_findings: list[dict[str, Any]] = []

        colin_task: TaskResult | None = results.get("detect_collinear_features")
        if colin_task and colin_task.status == "success" and colin_task.data:
            vif_scores: dict[str, float] = colin_task.data.get("vif_scores") or {}
            collinear: list[str] = colin_task.data.get("collinear_columns") or []
            for col in collinear:
                redundancy_cols.append(col)
                redundancy_findings.append(
                    {
                        "column": col,
                        "issue": "high_vif",
                        "vif_score": round(vif_scores.get(col, 0.0), 2),
                        "severity": "warn",
                    },
                )
            redundancy_findings.sort(key=lambda f: f["vif_score"], reverse=True)

        # ── LEAKAGE ──────────────────────────────────────────────────────────
        # Source: detect_data_leakage → data.leakage_pairs {"col_a|col_b": float}
        # Leakage is kept separate from redundancy: redundancy is inefficiency,
        # leakage is a correctness failure.
        leakage_cols: list[str] = []
        leakage_findings: list[dict[str, Any]] = []

        leak_task: TaskResult | None = results.get("detect_data_leakage")
        if leak_task and leak_task.status == "success" and leak_task.data:
            pairs: dict[str, float] = leak_task.data.get("leakage_pairs") or {}
            for pair_key, corr in pairs.items():
                if "|" not in pair_key:
                    continue
                col_a, col_b = pair_key.split("|", 1)
                for col in (col_a, col_b):
                    if col not in leakage_cols:
                        leakage_cols.append(col)
                leakage_findings.append(
                    {
                        "col_a": col_a,
                        "col_b": col_b,
                        "correlation": round(corr, 4),
                        "issue": "leakage_pair",
                        "severity": "error",
                    },
                )
            leakage_findings.sort(key=lambda f: abs(f["correlation"]), reverse=True)

        # ── Assemble output ───────────────────────────────────────────────────
        categories: dict[str, dict[str, Any]] = {
            "completeness": _category_block(
                completeness_cols,
                total_columns,
                completeness_findings,
            ),
            "validity": _category_block(
                validity_cols,
                total_columns,
                validity_findings,
            ),
            "usability": _category_block(
                usability_cols,
                total_columns,
                usability_findings,
            ),
            "redundancy": _category_block(
                redundancy_cols,
                total_columns,
                redundancy_findings,
            ),
            "leakage": _category_block(leakage_cols, total_columns, leakage_findings),
        }

        # Flat level summary - convenient for the API and Data Health Bar.
        level_summary: dict[str, str] = {
            name: cat["level"] for name, cat in categories.items()
        }

        self.output = TaskResult(
            name=self.name,
            status="success",
            summary=level_summary,
            data={
                "total_columns": total_columns,
                "all_columns": all_columns,
                "categories": categories,
            },
            metadata={
                "scoring_method": "proportional_traffic_light",
                "thresholds": {
                    "green": "0 - 5% of columns affected",
                    "amber": "5 - 15% of columns affected",
                    "red": "> 15% of columns affected",
                },
                "suggested_viz_type": "status_bar",
                "recommended_section": "Summary",
                "display_priority": "high",
            },
        )
