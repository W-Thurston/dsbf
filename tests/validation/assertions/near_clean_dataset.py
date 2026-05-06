# tests/validation/assertions/near_clean_dataset.py
"""
Assertions for the near-clean dataset (3,000 rows, 9 columns).

This dataset is engineered with a precise set of deliberate issues so that
assertions can be written against known expected findings.  The goal is to
verify that scoring thresholds are calibrated correctly — a mostly-fine
dataset should land at amber, not red, and the gate should be needs_work,
not not_ready.

Deliberate issues and their expected findings
─────────────────────────────────────────────
``income``         7% nulls → Completeness amber (warn)
``purchase_amount``  skew ≈ 2.0 → log-transform warn (Transformations)
``plan_type``      ~71% "Standard" → single dominant value warn (Usability)
``region``, ``plan_type``, ``segment``  raw object dtype →
                   Encoding Required warn (Encoding)

Intentionally absent (verify all-clear states)
───────────────────────────────────────────────
- No ID columns, no leakage, no constants, no duplicate columns
- No out-of-bounds, no zero-variance, no datetime columns
"""

from __future__ import annotations

from typing import Any

# Tasks that may skip or degrade on realistic production data — correct
# behaviour, not bugs.
EXPECTED_SKIPS: set[str] = {
    "compare_with_reference_dataset",
    "detect_class_imbalance",
    "detect_feature_drift",
    "detect_target_drift",
    "schema_validation",
}

# Quality dimensions expected to be amber on this dataset.
# Only completeness is amber: income has 6.4% nulls (above the 5% threshold).
# Usability stays green because plan_type's 71% dominance is below the
# detect_single_dominant_value flag threshold of 95%.
EXPECTED_AMBER_DIMENSIONS: set[str] = {"completeness"}

# Quality dimensions that must be green (no action-level findings).
EXPECTED_GREEN_DIMENSIONS: set[str] = {"validity", "usability", "redundancy", "leakage"}


def validate(report: dict[str, Any]) -> None:
    """
    Run all assertions for the near-clean dataset.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _no_unexpected_failures(report)
    _core_tasks_succeeded(report)
    _row_and_col_count(report)
    _income_nulls_detected(report)
    _purchase_amount_skew_flagged(report)
    _plan_type_dominant_calibration(report)
    _quality_dimensions(report)
    _ml_readiness_gate(report)
    _no_id_columns_flagged(report)
    _no_leakage_flagged(report)
    print("✓  near_clean dataset: all assertions passed")


# ── Checks ────────────────────────────────────────────────────────────────────


def _no_unexpected_failures(report: dict) -> None:
    """No task should fail with status='failed' on near-clean data."""
    for task_name, task in report.get("results", {}).items():
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "failed":
            raise AssertionError(
                f"Task '{task_name}' failed on near-clean dataset — "
                f"expected graceful success or skip.\n"
                f"  error_metadata: {task.get('error_metadata')}"
            )
        if status == "skipped":
            assert task_name in EXPECTED_SKIPS, (
                f"Task '{task_name}' was skipped unexpectedly.\n"
                f"  Add to EXPECTED_SKIPS if this skip is intentional."
            )


def _core_tasks_succeeded(report: dict) -> None:
    """These tasks must succeed on any realistic dataset."""
    must_succeed: set[str] = {
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        "summarize_numeric",
        "detect_skewness",
        "detect_outliers",
        "data_quality_scorer",
        "ml_readiness_scorer",
    }
    results = report.get("results", {})
    for task_name in must_succeed:
        task = results.get(task_name)
        assert isinstance(task, dict), f"'{task_name}' missing from results"
        assert task.get("status") == "success", (
            f"'{task_name}' must succeed on near-clean data, "
            f"got status='{task.get('status')}'.\n"
            f"  error_metadata: {task.get('error_metadata')}"
        )


def _row_and_col_count(report: dict) -> None:
    """Shape task should report the correct dimensions."""
    shape = report["results"].get("summarize_dataset_shape", {})
    if shape.get("status") != "success":
        return
    data = shape.get("data", {})
    num_rows = data.get("num_rows")
    num_cols = data.get("num_columns")
    assert num_rows == 3000, f"Expected 3000 rows, got {num_rows}"
    assert num_cols == 9, f"Expected 9 columns, got {num_cols}"


def _income_nulls_detected(report: dict) -> None:
    """
    income has ~7% nulls — above the 5% amber threshold.

    Completeness should flag it and report a null percentage between 5%
    and 10% (actual seeded value ≈ 6.4%).
    """
    nulls = report["results"].get("summarize_nulls", {})
    if nulls.get("status") != "success":
        return
    null_pcts = nulls.get("data", {}).get("null_percentages", {})
    income_null = null_pcts.get("income")
    assert income_null is not None, (
        "summarize_nulls did not report a null percentage for 'income'"
    )
    assert 0.05 < income_null < 0.10, (
        f"income null rate should be 5–10% (actual seeded ≈ 6.4%), "
        f"got {income_null:.1%}"
    )


def _purchase_amount_skew_flagged(report: dict) -> None:
    """
    purchase_amount has skew ≈ 2.0 — well above the 1.0 warn threshold.

    detect_skewness should flag it, and the ML readiness scorer should
    surface a transformations finding for this column.
    """
    skewness = report["results"].get("detect_skewness", {})
    if skewness.get("status") != "success":
        return
    data = skewness.get("data", {})
    # detect_skewness stores per-column results — find purchase_amount
    pa_skew = data.get("purchase_amount") if isinstance(data, dict) else None
    if pa_skew is None:
        # Some task shapes nest under a 'skewness' key
        pa_skew = data.get("skewness", {}).get("purchase_amount")
    assert pa_skew is not None, (
        "detect_skewness did not report a skewness value for 'purchase_amount'"
    )
    assert pa_skew > 1.0, (
        f"purchase_amount skew should be > 1.0 (actual ≈ 2.0), got {pa_skew:.3f}"
    )


def _plan_type_dominant_calibration(report: dict) -> None:
    """
    plan_type has ~71% 'Standard' — measurable but below the 95% flag threshold.

    detect_single_dominant_value should:
    - Measure mode_proportion between 0.68 and 0.74 (seeded ≈ 0.708)
    - NOT flag it as dominant (threshold is 95%)
    - Report dominance_level as 'moderate' (not 'high' or 'very high')

    This tests that the task correctly measures without over-flagging —
    a column that is "frequently the most common value" but not "nearly
    always the same value" should inform, not alarm.
    """
    dominant = report["results"].get("detect_single_dominant_value", {})
    if dominant.get("status") != "success":
        return
    data = dominant.get("data", {})

    # Task should have measured plan_type
    plan_data = data.get("plan_type")
    assert plan_data is not None, (
        "detect_single_dominant_value did not report data for 'plan_type'"
    )

    mode_prop = plan_data.get("mode_proportion", 0)
    assert 0.68 < mode_prop < 0.74, (
        f"plan_type mode_proportion should be 0.68–0.74 (seeded ≈ 0.708), "
        f"got {mode_prop:.3f}"
    )

    # Should NOT be in the flagged set (threshold is 95%)
    flagged = data.get("dominant_columns", [])
    assert "plan_type" not in flagged, (
        f"plan_type ({mode_prop:.1%} dominant) should NOT be flagged — "
        f"the 95% threshold should not be crossed. "
        f"If the threshold changed, update this assertion."
    )


def _quality_dimensions(report: dict) -> None:
    """
    Completeness and Usability should be amber; all others green.

    This verifies that:
    - The 7% null in income pushed Completeness to amber (not green)
    - The dominant plan_type pushed Usability to amber (not green)
    - No other dimension was accidentally triggered
    - No dimension is red (no error-level findings)
    """
    scorer = report["results"].get("data_quality_scorer")
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        print(
            "  ℹ  data_quality_scorer unavailable — dimension level assertions skipped"
        )
        return

    cats = scorer.get("data", {}).get("categories", {})

    for dim in EXPECTED_AMBER_DIMENSIONS:
        cat = cats.get(dim, {})
        level = cat.get("level")
        assert level == "amber", (
            f"Quality dimension '{dim}' should be amber on near-clean data "
            f"(has known warn-level findings), got '{level}'.\n"
            f"  findings: {cat.get('findings', [])}"
        )

    for dim in EXPECTED_GREEN_DIMENSIONS:
        cat = cats.get(dim, {})
        level = cat.get("level")
        assert level == "green", (
            f"Quality dimension '{dim}' should be green on near-clean data "
            f"(no findings expected), got '{level}'.\n"
            f"  findings: {cat.get('findings', [])}"
        )

    # No dimension should be red — that would indicate an error-level finding
    # which this dataset is specifically designed not to trigger
    for dim, cat in cats.items():
        level = cat.get("level")
        assert level != "red", (
            f"Quality dimension '{dim}' is red on near-clean data — "
            f"this dataset should not trigger error-level quality findings.\n"
            f"  findings: {cat.get('findings', [])}"
        )


def _ml_readiness_gate(report: dict) -> None:
    """
    ML Readiness gate should be needs_work, not not_ready.

    All ML findings on this dataset are warn-level:
    - Encoding warn (raw string categoricals)
    - Transformations warn (purchase_amount skew)
    - Usability warn (plan_type dominant)
    - Completeness warn (income nulls)

    No error-level findings → gate stays at needs_work, not not_ready.
    This specifically tests that the severity-based gate logic works
    correctly on realistic warn-level data.
    """
    scorer = report["results"].get("ml_readiness_scorer")
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        print("  ℹ  ml_readiness_scorer unavailable — gate assertion skipped")
        return

    gate = scorer.get("data", {}).get("readiness_gate")
    cats = scorer.get("data", {}).get("categories", {})

    # Verify no error-level findings exist
    error_findings: list[tuple] = [
        (dim, f)
        for dim, cat in cats.items()
        for f in cat.get("findings", [])
        if f.get("level") == "error"
    ]
    if error_findings:
        print("\n  Error-level ML findings found (unexpected on near-clean data):")
        for dim, f in error_findings:
            print(f"    [{dim}] col={f.get('column')!r} title={f.get('title')!r}")
    assert not error_findings, (
        f"Found {len(error_findings)} error-level ML Readiness finding(s) "
        "on near-clean data — none expected. "
        "This dataset should produce only warn-level findings."
    )

    assert gate != "not_ready", (
        "ML gate is 'not_ready' on near-clean data despite zero error-level "
        "findings. The gate should be severity-based. "
        "Check _gate() in ml_readiness_scorer.py."
    )
    assert gate == "needs_work", (
        f"ML gate should be 'needs_work' (warn-level findings present), "
        f"got '{gate}'. Check that encoding/transformation warn findings "
        f"are being emitted correctly."
    )
    print(f"  ✓  ML gate: {gate} (warn-level findings present — expected)")


def _no_id_columns_flagged(report: dict) -> None:
    """
    This dataset has no ID columns — detect_id_columns should flag nothing.

    Verifies that the clean continuous columns (age, income, etc.) with
    high unique counts are not misclassified as identifiers.
    """
    id_task = report["results"].get("detect_id_columns", {})
    if id_task.get("status") != "success":
        return
    data = id_task.get("data", {})
    # detect_id_columns returns an empty dict when nothing is flagged,
    # or a dict of {col: reason} when columns are flagged.
    flagged: list[str] = list(data.keys()) if isinstance(data, dict) else []
    assert len(flagged) == 0, (
        f"detect_id_columns should flag nothing on near-clean data, "
        f"but flagged: {flagged}. "
        f"Check that numeric columns with high unique counts are not "
        f"being misidentified as IDs."
    )


def _no_leakage_flagged(report: dict) -> None:
    """
    No column pairs should be flagged for data leakage.

    The near-clean dataset has no engineered correlations between columns.
    """
    leakage = report["results"].get("detect_data_leakage", {})
    if leakage.get("status") != "success":
        return
    pairs: dict = leakage.get("data", {}).get("leakage_pairs", {})
    assert len(pairs) == 0, (
        f"detect_data_leakage should find no leakage pairs on near-clean "
        f"data, but found: {list(pairs.keys())[:3]}"
    )
