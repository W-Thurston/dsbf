# tests/validation/assertions/high_missingness_dataset.py
"""
Assertions for the high-missingness dataset (2,000 rows, 9 columns).

This dataset exercises DSBF's handling of extreme and structured missing
data.  Each null pattern is deliberately different:

  age            ~8%  MCAR — just above amber threshold
  income         ~25% MCAR — moderate dropout
  device_type    ~41% MAR  — null when channel == "web" (P=0.95)
  premium_score  ~41% MAR  — null when plan_type == "Basic" (P=0.90)
  notes          ~75% sparse — extreme sparsity edge case
  region/channel/plan_type/is_churned  — 0% null, clean baseline

Primary assertions:
  - summarize_nulls measures every null rate correctly
  - Completeness dimension is red (multiple columns above 20% threshold)
  - ML Readiness Missingness dimension is non-green
  - MAR columns have null rates that reflect their structured patterns
  - No task fails due to high null rates (graceful degradation)
"""

from __future__ import annotations

from typing import Any

EXPECTED_SKIPS: set[str] = {
    "compare_with_reference_dataset",
    "detect_class_imbalance",
    "detect_feature_drift",
    "detect_target_drift",
    "schema_validation",
}

# Expected null rate ranges per column (lower, upper).
# Ranges are generous (+/-5pp) to be robust to minor RNG variation.
EXPECTED_NULL_RANGES: dict[str, tuple[float, float]] = {
    "age": (0.05, 0.12),  # seeded ~7.5%
    "income": (0.20, 0.30),  # seeded ~24.6%
    "device_type": (0.35, 0.50),  # seeded ~41.2%  (MAR via channel)
    "premium_score": (0.35, 0.50),  # seeded ~40.6%  (MAR via plan_type)
    "notes": (0.70, 0.80),  # seeded ~75.0%
    "region": (0.0, 0.0),
    "channel": (0.0, 0.0),
    "plan_type": (0.0, 0.0),
    "is_churned": (0.0, 0.0),
}

# Columns whose null rate is above the 20% red threshold
HIGH_NULL_COLUMNS: set[str] = {"device_type", "premium_score", "notes"}


def validate(report: dict[str, Any]) -> None:
    """
    Run all assertions for the high-missingness dataset.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _no_unexpected_failures(report)
    _core_tasks_succeeded(report)
    _shape_correct(report)
    _null_rates_correct(report)
    _completeness_is_red(report)
    _high_null_columns_flagged(report)
    _clean_columns_have_no_nulls(report)
    _ml_readiness_missingness_non_green(report)
    _mar_null_rates_reflect_structure(report)
    print("✓  high_missingness dataset: all assertions passed")


# ── Checks ────────────────────────────────────────────────────────────────────


def _no_unexpected_failures(report: dict) -> None:
    """
    No task should fail due to high null rates.

    Tasks must succeed (possibly with reliability warnings) or skip
    intentionally — not error because most column values are missing.
    """
    for task_name, task in report.get("results", {}).items():
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "failed":
            msg: str = (
                f"Task '{task_name}' failed on high-missingness data.\n"
                f"High null rates should degrade gracefully, not cause failures.\n"
                f"  error_metadata: {task.get('error_metadata')}"
            )
            raise AssertionError(msg)
        if status == "skipped" and task_name not in EXPECTED_SKIPS:
            msg = (
                f"Task '{task_name}' was skipped unexpectedly — add to "
                f"EXPECTED_SKIPS if intentional."
            )
            raise AssertionError(msg)


def _core_tasks_succeeded(report: dict) -> None:
    """Core tasks must succeed even with high null rates throughout."""
    must_succeed: set[str] = {
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        "data_quality_scorer",
    }
    results = report.get("results", {})
    for task_name in must_succeed:
        task = results.get(task_name)
        assert isinstance(task, dict), f"'{task_name}' missing from results"
        assert task.get("status") == "success", (
            f"'{task_name}' must succeed on high-missingness data, "
            f"got status='{task.get('status')}'.\n"
            f"  error_metadata: {task.get('error_metadata')}"
        )


def _shape_correct(report: dict) -> None:
    """Shape task should report the correct dimensions."""
    shape = report["results"].get("summarize_dataset_shape", {})
    if shape.get("status") != "success":
        return
    data = shape.get("data", {})
    assert data.get("num_rows") == 2000, (
        f"Expected 2000 rows, got {data.get('num_rows')}"
    )
    assert data.get("num_columns") == 9, (
        f"Expected 9 columns, got {data.get('num_columns')}"
    )


def _null_rates_correct(report: dict) -> None:
    """
    summarize_nulls must report null rates within expected ranges.

    Ranges are generous to be robust to minor RNG variation, but tight
    enough to catch a column being silently dropped or miscounted.
    """
    nulls = report["results"].get("summarize_nulls", {})
    if nulls.get("status") != "success":
        return
    null_pcts = nulls.get("data", {}).get("null_percentages", {})

    for col, (lo, hi) in EXPECTED_NULL_RANGES.items():
        pct = null_pcts.get(col)
        assert pct is not None, (
            f"summarize_nulls did not report a null percentage for '{col}'"
        )
        assert lo <= pct <= hi, (
            f"Column '{col}' null rate {pct:.1%} outside expected "
            f"range [{lo:.0%}, {hi:.0%}]"
        )

    print(f"  ✓  Null rates in range for all {len(EXPECTED_NULL_RANGES)} columns")


def _completeness_is_red(report: dict) -> None:
    """
    The Completeness dimension must be red.

    device_type (~41%), premium_score (~41%), and notes (~75%) all exceed
    the 20% red threshold.
    """
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        print("  ℹ  data_quality_scorer unavailable — completeness check skipped")
        return
    cats = dq.get("data", {}).get("categories", {})
    completeness = cats.get("completeness", {})
    level = completeness.get("level")
    assert level == "red", (
        f"Completeness dimension should be red (device_type/premium_score/"
        f"notes all above 20% null), got '{level}'.\n"
        f"  findings: {completeness.get('findings', [])}"
    )
    affected = completeness.get("affected_count", 0)
    print(f"  ✓  Completeness: red ({affected} column(s) flagged)")


def _high_null_columns_flagged(report: dict) -> None:
    """Columns with >= 20% nulls must appear in Completeness affected_columns."""
    dq = report["results"].get("data_quality_scorer", {})
    if not isinstance(dq, dict) or dq.get("status") != "success":
        return
    cats = dq.get("data", {}).get("categories", {})
    affected_cols: list[str] = cats.get("completeness", {}).get("affected_columns", [])
    for col in HIGH_NULL_COLUMNS:
        assert col in affected_cols, (
            f"'{col}' should be in Completeness affected_columns "
            f"(null rate > 20%), but affected_columns = {affected_cols}"
        )


def _clean_columns_have_no_nulls(report: dict) -> None:
    """Clean baseline columns must have exactly 0% nulls."""
    nulls = report["results"].get("summarize_nulls", {})
    if nulls.get("status") != "success":
        return
    null_pcts = nulls.get("data", {}).get("null_percentages", {})
    for col in ("region", "channel", "plan_type", "is_churned"):
        pct = null_pcts.get(col)
        if pct is None:
            continue
        assert pct == 0.0, f"Clean column '{col}' should have 0% nulls, got {pct:.1%}"


def _ml_readiness_missingness_non_green(report: dict) -> None:
    """
    ML Readiness Missingness dimension must be red and gate must be not_ready.

    notes at 75% null crosses the error-level threshold — passing it to
    sklearn would drop 75% of training rows.  That is a genuine modeling
    blocker, so the gate should be not_ready (error-level finding present).
    """
    scorer = report["results"].get("ml_readiness_scorer", {})
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        print("  ℹ  ml_readiness_scorer unavailable — missingness check skipped")
        return
    cats = scorer.get("data", {}).get("categories", {})
    missingness = cats.get("missingness", {})
    level = missingness.get("level")
    gate = scorer.get("data", {}).get("readiness_gate")

    assert level == "red", (
        f"ML Readiness Missingness should be red (notes 75% null is "
        f"error-level), got '{level}'"
    )
    assert gate == "not_ready", (
        f"ML gate should be not_ready (error-level missingness finding for "
        f"notes), got '{gate}'"
    )
    print(f"  ✓  ML Missingness dimension: {level}  |  gate: {gate}")


def _mar_null_rates_reflect_structure(report: dict) -> None:
    """
    MAR columns must be substantially more null than MCAR columns.

    device_type (null when channel=="web") and premium_score (null when
    plan_type=="Basic") should both be much more null than the MCAR columns
    age and income.  This verifies the structured pattern was applied
    correctly, even though we cannot directly assert on the mechanism
    analysis output (which uses hedged confidence language).
    """
    nulls = report["results"].get("summarize_nulls", {})
    if nulls.get("status") != "success":
        return
    null_pcts = nulls.get("data", {}).get("null_percentages", {})

    device_null = null_pcts.get("device_type", 0)
    premium_null = null_pcts.get("premium_score", 0)
    income_null = null_pcts.get("income", 0)

    assert device_null > income_null + 0.10, (
        f"device_type ({device_null:.1%}) should be substantially more null "
        f"than income ({income_null:.1%}) given the MAR web-channel structure"
    )
    assert premium_null > income_null + 0.10, (
        f"premium_score ({premium_null:.1%}) should be substantially more "
        f"null than income ({income_null:.1%}) given the MAR Basic-plan structure"
    )
    print(
        f"  ✓  MAR null rates structurally distinct from MCAR: "
        f"device_type={device_null:.1%}, premium_score={premium_null:.1%} "
        f"vs income={income_null:.1%}",
    )
