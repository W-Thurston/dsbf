# tests/validation/assertions/single_column_dataset.py
"""
Assertions for the single-column dataset (500 rows, 1 column).

A single continuous log-normal column named ``value`` (skew ~2.1).

Primary purpose: verify that every task requiring two or more columns
degrades cleanly — returning status='success' with empty output rather
than raising an unhandled exception.

Tasks verified to degrade gracefully:
  detect_collinear_features   "Not enough numeric features" (< 2 cols guard)
  compute_pairwise_associations  0 pairs computed (empty loop)
  detect_duplicate_columns    0 pairs to compare
  detect_data_leakage         no pairs to check
  generate_dataset_summary_plots  correlation matrix skipped (< 2 cols)

Tasks expected to produce real findings:
  detect_skewness         skew ~2.1 triggers log-transform warn
  detect_outliers         log-normal has extreme right-tail values
  summarize_numeric       mean, std, percentiles all computed
  normality_tests         runs on the single column
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

# Tasks that operate on column pairs — must succeed with empty output
PAIR_TASKS: set[str] = {
    "detect_collinear_features",
    "compute_pairwise_associations",
    "detect_duplicate_columns",
    "detect_data_leakage",
}


def validate(report: dict[str, Any]) -> None:
    """
    Run all assertions for the single-column dataset.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _no_unexpected_failures(report)
    _core_tasks_succeeded(report)
    _shape_correct(report)
    _single_column_is_continuous(report)
    _pair_tasks_degrade_cleanly(report)
    _collinear_features_empty_output(report)
    _associations_empty(report)
    _skewness_flagged(report)
    _no_nulls(report)
    print("✓  single_column dataset: all assertions passed")


# ── Checks ────────────────────────────────────────────────────────────────────


def _no_unexpected_failures(report: dict) -> None:
    """No task should fail — single-column data must degrade gracefully."""
    for task_name, task in report.get("results", {}).items():
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "failed":
            raise AssertionError(
                f"Task '{task_name}' failed on single-column data.\n"
                f"Tasks must degrade gracefully when there is only one column.\n"
                f"  error_metadata: {task.get('error_metadata')}"
            )
        if status == "skipped" and task_name not in EXPECTED_SKIPS:
            raise AssertionError(
                f"Task '{task_name}' was skipped unexpectedly — add to "
                f"EXPECTED_SKIPS if intentional."
            )


def _core_tasks_succeeded(report: dict) -> None:
    """Core per-column tasks must succeed even with only one column."""
    must_succeed: set[str] = {
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        "summarize_numeric",
        "detect_skewness",
    }
    results = report.get("results", {})
    for task_name in must_succeed:
        task = results.get(task_name)
        assert isinstance(task, dict), f"'{task_name}' missing from results"
        assert task.get("status") == "success", (
            f"'{task_name}' must succeed on single-column data, "
            f"got status='{task.get('status')}'.\n"
            f"  error_metadata: {task.get('error_metadata')}"
        )


def _shape_correct(report: dict) -> None:
    """Shape task should report exactly 1 column and 500 rows."""
    shape = report["results"].get("summarize_dataset_shape", {})
    if shape.get("status") != "success":
        return
    data = shape.get("data", {})
    assert data.get("num_rows") == 500, f"Expected 500 rows, got {data.get('num_rows')}"
    assert data.get("num_columns") == 1, (
        f"Expected 1 column, got {data.get('num_columns')}"
    )


def _single_column_is_continuous(report: dict) -> None:
    """
    infer_types must classify 'value' as continuous.

    This is load-bearing: if the column is misclassified, the per-column
    task assertions below become vacuous.
    """
    types = report["results"].get("infer_types", {})
    if types.get("status") != "success":
        return
    data = types.get("data", {})
    intent = data.get("value", {}).get("analysis_intent_dtype")
    assert intent == "continuous", (
        f"'value' should be classified as continuous, got '{intent}'"
    )


def _pair_tasks_degrade_cleanly(report: dict) -> None:
    """
    Tasks that operate on column pairs must complete with status='success'.

    With only one column there are zero pairs to process — the correct
    behavior is an empty result, not a crash.
    """
    results = report.get("results", {})
    failures: list[str] = []

    for task_name in PAIR_TASKS:
        task = results.get(task_name)
        if task is None:
            continue  # not run at this profiling depth — acceptable
        status = task.get("status")
        if status == "failed":
            failures.append(
                f"  {task_name}: status='failed' — "
                f"{task.get('error_metadata', {}).get('trace_summary', '?')}"
            )
        elif status not in {"success", "skipped"}:
            failures.append(f"  {task_name}: unexpected status='{status}'")

    if failures:
        raise AssertionError(
            "Pair-based tasks failed on single-column data "
            "(expected graceful empty output):\n" + "\n".join(failures)
        )

    ran = sorted(t for t in PAIR_TASKS if results.get(t) is not None)
    print(f"  ✓  {len(ran)} pair task(s) degraded cleanly: {ran}")


def _collinear_features_empty_output(report: dict) -> None:
    """
    detect_collinear_features must return empty scores, not error.

    The task has an explicit guard: if fewer than 2 continuous columns
    are present, it returns immediately with empty vif_scores and an
    informational message.  This verifies the guard fires correctly.
    """
    task = report["results"].get("detect_collinear_features", {})
    if task is None or task.get("status") != "success":
        return
    data = task.get("data", {})
    vif_scores: dict = data.get("vif_scores", {})
    collinear: list = data.get("collinear_columns", [])
    assert len(vif_scores) == 0, (
        f"detect_collinear_features should produce empty vif_scores on a "
        f"single-column dataset, got: {vif_scores}"
    )
    assert len(collinear) == 0, (
        f"detect_collinear_features should produce no collinear_columns on a "
        f"single-column dataset, got: {collinear}"
    )
    print("  ✓  detect_collinear_features: empty output (< 2 cols guard fired)")


def _associations_empty(report: dict) -> None:
    """
    compute_pairwise_associations must return 0 associations.

    With one column the inner loop `for col_b in eligible[i+1:]` produces
    an empty iteration — no crash, just an empty result dict.
    """
    task = report["results"].get("compute_pairwise_associations", {})
    if task is None or task.get("status") != "success":
        return
    data = task.get("data", {})
    # Associations stored as {pair_key: {...}} — should be empty
    assoc_count = len([k for k in data.keys() if k != "__correlation_matrix__"])
    assert assoc_count == 0, (
        f"compute_pairwise_associations should produce 0 pairs on a "
        f"single-column dataset, got {assoc_count} pair(s)"
    )
    print("  ✓  compute_pairwise_associations: 0 pairs (correct)")


def _skewness_flagged(report: dict) -> None:
    """
    detect_skewness must flag 'value' — log-normal with sigma=0.6
    has skew ~2.1, well above the 1.0 warn threshold.

    This verifies that per-column tasks still produce real findings
    on a single-column dataset — the dataset is not degenerate.
    """
    skewness = report["results"].get("detect_skewness", {})
    if skewness.get("status") != "success":
        return
    data = skewness.get("data", {})
    skew_val = data.get("value")
    assert skew_val is not None, (
        "detect_skewness did not report a skewness value for 'value'"
    )
    assert skew_val > 1.0, (
        f"'value' skewness should be > 1.0 (log-normal, seeded ~2.1), "
        f"got {skew_val:.3f}"
    )
    print(f"  ✓  detect_skewness: 'value' skew={skew_val:.2f} (flagged correctly)")


def _no_nulls(report: dict) -> None:
    """Single column has no missing values."""
    nulls = report["results"].get("summarize_nulls", {})
    if nulls.get("status") != "success":
        return
    pct = nulls.get("data", {}).get("null_percentages", {}).get("value", None)
    assert pct == 0.0, f"'value' should have 0% nulls, got {pct}"
