# dsbf/eda/tasks/ml_readiness_scorer.py

from typing import Any

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult

# ── Dimension → source task mapping ──────────────────────────────────────────
# Each task's ML blurbs are routed to exactly one dimension.
# A column can appear in multiple dimensions if multiple tasks flag it —
# that is intentional and honest (e.g. a high-cardinality ID column is
# both Unusable and has Encoding implications).

_TASK_DIMENSION: dict[str, str] = {
    # What preparation does the data need?
    "detect_skewness": "transformations",
    "detect_outliers": "transformations",
    "suggest_numerical_binning": "transformations",
    "detect_bimodal_distribution": "transformations",
    # How do categorical/boolean features need to be represented?
    "detect_high_cardinality": "encoding",
    "suggest_categorical_encoding": "encoding",
    "summarize_boolean_fields": "encoding",
    # How does missingness affect model training?
    "summarize_nulls": "missingness",
    "summarize_numeric": "missingness",  # near-zero variance blurbs
    # Are any features likely to leak target information?
    "detect_data_leakage": "leakage",
    # Are any features structurally unsuitable for modeling?
    "detect_constant_columns": "unusable",
    "detect_id_columns": "unusable",
    "detect_single_dominant_value": "unusable",
    "detect_zeros": "unusable",
}

_DIMENSIONS: list[str] = [
    "transformations",
    "encoding",
    "missingness",
    "leakage",
    "unusable",
]

_DIMENSION_LABELS: dict[str, str] = {
    "transformations": "Transformations Needed",
    "encoding": "Encoding Required",
    "missingness": "Missingness Impact",
    "leakage": "Leakage Risk",
    "unusable": "Unusable Features",
}

_SEVERITY_RANK: dict[str, int] = {
    "error": 3,
    "warn": 2,
    "info": 1,
    "good": 0,
}


def _traffic_light(
    pct_affected: float, has_error: bool, has_warn: bool
) -> str:  # noqa: FBT001
    """
    Map dimension metrics to a proportional traffic-light level.

    Args:
        pct_affected: Proportion of total columns with findings in this dimension.
        has_error: True if any finding in the dimension has level ``"error"``.
        has_warn: True if any finding in the dimension has level ``"warn"``.

    Returns:
        ``"red"`` if error-level findings exist or > 15% of columns are affected;
        ``"amber"`` if warn-level findings exist or 5-15% affected;
        ``"green"`` otherwise.

    """
    if has_error or pct_affected > 0.15:  # noqa: PLR2004
        return "red"
    if has_warn or pct_affected > 0.05:  # noqa: PLR2004
        return "amber"
    return "green"


def _gate(categories: dict[str, dict]) -> str:
    """
    Derive an overall readiness gate from dimension traffic-light levels.

    Args:
        categories: Dimension output dicts, each with a ``"level"`` key.

    Returns:
        ``"not_ready"`` if any dimension is red;
        ``"needs_work"`` if any dimension is amber and none are red;
        ``"ready"`` if all dimensions are green.

    """
    levels: list = [cat["level"] for cat in categories.values()]
    if "red" in levels:
        return "not_ready"
    if "amber" in levels:
        return "needs_work"
    return "ready"


@register_task(
    name="ml_readiness_scorer",
    display_name="ML Readiness Scorer",
    description=(
        "Aggregates ML-phase guidance blurbs from all EDA tasks into five "
        "preparation-focused dimensions: Transformations, Encoding, Missingness, "
        "Leakage Risk, and Unusable Features. Each dimension gets a proportional "
        "traffic-light level. An overall gate (ready / needs_work / not_ready) "
        "is derived from the dimension levels."
    ),
    tags=["ml_readiness", "scoring", "summary", "meta"],
    stage="report",
    phase="ml_readiness",
    domain="core",
    profiling_depth="standard",
    runtime_estimate="fast",
    expected_semantic_types=["any"],
    depends_on=[
        "infer_types",
        "summarize_nulls",
        "summarize_numeric",
        "summarize_boolean_fields",
        "detect_skewness",
        "detect_outliers",
        "detect_zeros",
        "detect_constant_columns",
        "detect_id_columns",
        "detect_single_dominant_value",
        "detect_high_cardinality",
        "detect_bimodal_distribution",
        "detect_data_leakage",
        "suggest_categorical_encoding",
        "data_quality_scorer",
    ],
)
class MlReadinessScorer(BaseTask):
    """
    Produce a structured ML readiness report by aggregating ML-phase guidance.

    Routes ML guidance blurbs emitted by EDA tasks into five preparation-focused
    dimensions. Unlike a numeric score, the output is organized around what a
    data scientist needs to *do* before modeling — not how "bad" the data is.

    The gate (not_ready / needs_work / ready) is the primary signal; dimension-level
    traffic lights show where the work is.

    Output shape (``data`` field)::

        {
            "readiness_gate":  "ready" | "needs_work" | "not_ready",
            "total_columns":   int,
            "all_columns":     [str, ...],
            "clean_columns":   [str, ...],
            "categories": {
                "<dimension>": {
                    "label":            str,
                    "affected_columns": [str, ...],
                    "affected_count":   int,
                    "total_columns":    int,
                    "pct_affected":     float,
                    "level":            "green" | "amber" | "red",
                    "findings": [
                        {
                            "column":  str,
                            "level":   str,
                            "title":   str,
                            "body":    str,
                            "actions": [...],
                            "metric":  {...},
                            "task":    str,
                        },
                        ...
                    ]
                },
                ...
            }
        }

    The summary field provides the fast gate for the API::

        {"readiness_gate": str}
    """

    def run(self) -> None:  # noqa: C901
        """
        Aggregate ML guidance from context results and populate self.output.

        Raises:
            RuntimeError: If no analysis context is attached.

        """
        if self.context is None:
            raise RuntimeError("AnalysisContext is not set.")

        results: dict[str, TaskResult] = self.context.results

        # ── Total column count + ID-intent column set ────────────────────────
        all_columns: list[str] = []
        id_intent_cols: set[str] = set()
        types_task: TaskResult | None = results.get("infer_types")
        if types_task and types_task.status == "success" and types_task.data:
            all_columns = list(types_task.data.keys())
            id_intent_cols = {
                col
                for col, info in types_task.data.items()
                if isinstance(info, dict) and info.get("analysis_intent_dtype") == "id"
            }
        total_columns: int = len(all_columns)

        # ── Collect findings per dimension ────────────────────────────────────
        dim_findings: dict[str, list[dict[str, Any]]] = {d: [] for d in _DIMENSIONS}
        dim_cols: dict[str, set[str]] = {d: set() for d in _DIMENSIONS}

        for task_name, dimension in _TASK_DIMENSION.items():
            task_result: TaskResult | None = results.get(task_name)
            if not task_result or task_result.status != "success":
                continue
            guidance: dict[Any, Any] | Any = (
                getattr(task_result, "guidance", None) or {}
            )
            if not isinstance(guidance, dict):
                continue

            for col, phases in guidance.items():
                if not isinstance(phases, dict):
                    continue
                # Encoding suggestions are not meaningful for identifier columns —
                # ID-intent columns are better routed to the Unusable dimension.
                if dimension == "encoding" and col in id_intent_cols:
                    continue
                ml_blurbs: list[dict[str, Any]] = phases.get("ml") or []
                for blurb in ml_blurbs:
                    lvl = blurb.get("level", "info")
                    dim_findings[dimension].append(
                        {
                            "column": col,
                            "level": lvl,
                            "title": blurb.get("title", ""),
                            "body": blurb.get("body", ""),
                            "actions": blurb.get("actions") or [],
                            "metric": blurb.get("metric") or {},
                            "task": task_name,
                        },
                    )
                    dim_cols[dimension].add(col)

        # Sort findings within each dimension: worst level first, then column name.
        for dim in _DIMENSIONS:
            dim_findings[dim].sort(
                key=lambda f: (-_SEVERITY_RANK.get(f["level"], 0), f["column"]),
            )

        # ── Build category blocks ─────────────────────────────────────────────
        categories: dict[str, dict[str, Any]] = {}

        for dim in _DIMENSIONS:
            findings: list[dict[str, Any]] = dim_findings[dim]
            cols: list[str] = sorted(dim_cols[dim])
            count: int = len(cols)
            pct: float = count / total_columns if total_columns else 0.0
            has_error: bool = any(f["level"] == "error" for f in findings)
            has_warn: bool = any(f["level"] == "warn" for f in findings)

            categories[dim] = {
                "label": _DIMENSION_LABELS[dim],
                "affected_columns": cols,
                "affected_count": count,
                "total_columns": total_columns,
                "pct_affected": round(pct, 4),
                "level": _traffic_light(pct, has_error, has_warn),
                "findings": findings,
            }

        gate: str = _gate(categories)

        # Columns with zero ML findings across all dimensions.
        all_affected: set[str] = set()
        for cols in dim_cols.values():
            all_affected.update(cols)
        clean_columns: list[str] = sorted(
            c for c in all_columns if c not in all_affected
        )

        self.output = TaskResult(
            name=self.name,
            status="success",
            summary={"readiness_gate": gate},
            data={
                "readiness_gate": gate,
                "total_columns": total_columns,
                "all_columns": all_columns,
                "clean_columns": clean_columns,
                "categories": categories,
            },
            metadata={
                "dimensions": _DIMENSIONS,
                "dimension_labels": _DIMENSION_LABELS,
                "gate_logic": {
                    "not_ready": "any dimension is red",
                    "needs_work": "any dimension is amber, none red",
                    "ready": "all dimensions green",
                },
                "traffic_light_thresholds": {
                    "red": "error-level findings present OR >15% cols affected",
                    "amber": "warn-level findings present OR 5-15% cols affected",
                    "green": "info/good only AND ≤5% cols affected",
                },
                "suggested_viz_type": "dimension_grid",
                "recommended_section": "ML Readiness",
                "display_priority": "high",
            },
        )
