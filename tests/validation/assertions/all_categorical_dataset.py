# tests/validation/assertions/all_categorical_dataset.py
"""
Assertions for the all-categorical dataset (2,000 rows, 9 columns).

Zero numeric columns. Primary purpose: verify that every continuous-only
task degrades cleanly (status='success' with empty/skipped output) when
it finds no columns to process, rather than raising an exception.

Column inventory:
  color, size, material, region, category  — balanced low-cardinality
  tag                                       — ~575 unique values (high-cardinality)
  dominant                                  — 95.6% "Standard" (error-level dominant)
  is_active, is_premium                     — booleans

Expected findings:
  Usability error:  dominant column (>= 95% threshold)
  Encoding warn:    tag (high-cardinality) + raw string encoding on all object cols
  All others:       green or advisory only
"""

from __future__ import annotations

from typing import Any

# Tasks that process only continuous columns — should all succeed with
# empty output rather than erroring on a zero-continuous-column dataset.
CONTINUOUS_ONLY_TASKS: set[str] = {
    "detect_outliers",
    "detect_skewness",
    "detect_bimodal_distribution",
    "detect_near_zero_variance",
    "detect_collinear_features",
    "normality_tests",
    "normality_qq_plots",
    "suggest_numerical_binning",
    "summarize_numeric",
    "compute_kurtosis",
}

EXPECTED_SKIPS: set[str] = {
    "compare_with_reference_dataset",
    "detect_class_imbalance",
    "detect_feature_drift",
    "detect_target_drift",
    "schema_validation",
}


def validate(report: dict[str, Any]) -> None:
    """
    Run all assertions for the all-categorical dataset.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _no_unexpected_failures(report)
    _core_tasks_succeeded(report)
    _shape_correct(report)
    _no_numeric_columns_inferred(report)
    _continuous_tasks_degrade_cleanly(report)
    _dominant_column_flagged(report)
    _tag_high_cardinality_flagged(report)
    _no_nulls(report)
    _ml_readiness_no_continuous_transform_findings(report)
    print("✓  all_categorical dataset: all assertions passed")


# ── Checks ────────────────────────────────────────────────────────────────────


def _no_unexpected_failures(report: dict) -> None:
    """No task should fail on this dataset — only expected skips are allowed."""
    for task_name, task in report.get("results", {}).items():
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "failed":
            raise AssertionError(
                f"Task '{task_name}' failed on all-categorical data.\n"
                f"  error_metadata: {task.get('error_metadata')}"
            )
        if status == "skipped" and task_name not in EXPECTED_SKIPS:
            raise AssertionError(
                f"Task '{task_name}' was skipped unexpectedly — add to "
                f"EXPECTED_SKIPS if intentional."
            )


def _core_tasks_succeeded(report: dict) -> None:
    """Core pipeline tasks must succeed regardless of column types."""
    must_succeed: set[str] = {
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        "data_quality_scorer",
        "suggest_categorical_encoding",
    }
    results = report.get("results", {})
    for task_name in must_succeed:
        task = results.get(task_name)
        assert isinstance(task, dict), f"'{task_name}' missing from results"
        assert task.get("status") == "success", (
            f"'{task_name}' must succeed on all-categorical data, "
            f"got status='{task.get('status')}'.\n"
            f"  error_metadata: {task.get('error_metadata')}"
        )


def _shape_correct(report: dict) -> None:
    """Shape task should report correct dimensions."""
    shape = report["results"].get("summarize_dataset_shape", {})
    if shape.get("status") != "success":
        return
    data = shape.get("data", {})
    assert (
        data.get("num_rows") == 2000
    ), f"Expected 2000 rows, got {data.get('num_rows')}"
    assert (
        data.get("num_columns") == 9
    ), f"Expected 9 columns, got {data.get('num_columns')}"


def _no_numeric_columns_inferred(report: dict) -> None:
    """
    infer_types must not classify any column as 'continuous'.

    This is the foundational check — if any column is mistyped as
    continuous, it would undermine all the continuous-task empty-state
    assertions below.
    """
    types = report["results"].get("infer_types", {})
    if types.get("status") != "success":
        return
    data = types.get("data", {})
    continuous_cols: list[str] = [
        col
        for col, info in data.items()
        if info.get("analysis_intent_dtype") == "continuous"
    ]
    assert len(continuous_cols) == 0, (
        f"infer_types classified {len(continuous_cols)} column(s) as continuous "
        f"on an all-categorical dataset: {continuous_cols}. "
        f"This would cause false findings in continuous-only tasks."
    )


def _continuous_tasks_degrade_cleanly(report: dict) -> None:
    """
    Tasks that only operate on continuous columns should complete with
    status='success' and produce empty/minimal output — not error or crash.

    This is the central purpose of the all-categorical dataset: verifying
    that 'no columns to process' is handled gracefully everywhere.
    """
    results = report.get("results", {})
    failures: list[str] = []

    for task_name in CONTINUOUS_ONLY_TASKS:
        task = results.get(task_name)
        if task is None:
            # Task may not run at all on this profiling depth — acceptable
            continue
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
            "Continuous-only tasks failed on all-categorical data "
            "(expected graceful empty output):\n" + "\n".join(failures),
        )

    # Report which tasks ran vs were absent
    ran = [t for t in CONTINUOUS_ONLY_TASKS if results.get(t) is not None]
    print(
        f"  ✓  {len(ran)} continuous-only task(s) degraded cleanly: "
        f"{sorted(ran)[:4]}{'...' if len(ran) > 4 else ''}"
    )


def _dominant_column_flagged(report: dict) -> None:
    """
    'dominant' column has 95.6% single value — above the 95% error threshold.

    detect_single_dominant_value should flag it. This is the only
    error-level finding expected on this dataset.
    """
    task = report["results"].get("detect_single_dominant_value", {})
    if task.get("status") != "success":
        return
    data = task.get("data", {})

    # Check the dominant column was measured at ≥ 95%
    col_data = data.get("dominant")
    assert (
        col_data is not None
    ), "detect_single_dominant_value did not report data for 'dominant' column"
    mode_prop = col_data.get("mode_proportion", 0)
    assert (
        mode_prop >= 0.95
    ), f"'dominant' column mode_proportion should be ≥ 0.95, got {mode_prop:.3f}"

    # Verify the data quality scorer surfaced this as a Usability finding
    dq = report["results"].get("data_quality_scorer", {})
    if dq.get("status") != "success":
        return
    usability = dq.get("data", {}).get("categories", {}).get("usability", {})
    usability_level = usability.get("level")
    assert usability_level in {"amber", "red"}, (
        f"Usability dimension should be amber or red (dominant column present), "
        f"got '{usability_level}'"
    )
    print(f"  ✓  Usability dimension: {usability_level} (dominant column flagged)")


def _tag_high_cardinality_flagged(report: dict) -> None:
    """
    'tag' has ~575 unique values — above the default high-cardinality threshold (50).

    detect_high_cardinality should flag it, and suggest_categorical_encoding
    should recommend frequency encoding rather than one-hot.
    """
    hc_task = report["results"].get("detect_high_cardinality", {})
    if hc_task.get("status") != "success":
        return
    data = hc_task.get("data", {})
    # detect_high_cardinality returns {col: cardinality_count} for flagged
    # columns — the presence of a column name as a key means it was flagged.
    flagged: list[str] = list(data.keys())
    assert "tag" in flagged, (
        f"detect_high_cardinality should flag 'tag' (~575 unique values, "
        f"threshold=50), but flagged columns were: {flagged}"
    )

    # Encoding suggestion for tag should be frequency-based, not one-hot
    enc_task = report["results"].get("suggest_categorical_encoding", {})
    if enc_task.get("status") != "success":
        return
    enc_data = enc_task.get("data", {}).get("encoding_suggestions", {})
    tag_enc = enc_data.get("tag", {}).get("suggested_encoding", "")
    assert "frequency" in tag_enc.lower(), (
        f"suggest_categorical_encoding should recommend frequency encoding "
        f"for high-cardinality 'tag', got: '{tag_enc}'"
    )
    print(f"  ✓  tag: flagged high-cardinality, encoding='{tag_enc}'")


def _no_nulls(report: dict) -> None:
    """All columns are fully populated — no null percentages above 0."""
    nulls = report["results"].get("summarize_nulls", {})
    if nulls.get("status") != "success":
        return
    null_pcts = nulls.get("data", {}).get("null_percentages", {})
    for col, pct in null_pcts.items():
        assert pct == 0.0, (
            f"Column '{col}' shows {pct:.1%} nulls on all-categorical dataset "
            f"— dataset was designed with no missing values"
        )


def _ml_readiness_no_continuous_transform_findings(report: dict) -> None:
    """
    The Transformations dimension in ML Readiness should have no findings
    sourced from continuous-only tasks (outliers, skewness, binning).

    Since there are no continuous columns, these tasks have nothing to
    emit. Any finding in Transformations must come from another source
    (e.g. bimodal detection on categoricals — unlikely) or is a bug.
    """
    scorer = report["results"].get("ml_readiness_scorer", {})
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        print(
            "  ℹ  ml_readiness_scorer unavailable — "
            "transformation finding check skipped"
        )
        return

    cats = scorer.get("data", {}).get("categories", {})
    transform_findings = cats.get("transformations", {}).get("findings", [])

    # Continuous-only tasks that should produce zero ML findings here
    continuous_sources: set[str] = {
        "detect_outliers",
        "detect_skewness",
        "suggest_numerical_binning",
        "detect_bimodal_distribution",
    }
    spurious: list[dict] = [
        f for f in transform_findings if f.get("task") in continuous_sources
    ]
    assert len(spurious) == 0, (
        f"Transformations dimension has {len(spurious)} finding(s) from "
        f"continuous-only tasks on an all-categorical dataset:\n"
        + "\n".join(
            f"  [{f.get('task')}] col={f.get('column')!r} title={f.get('title')!r}"
            for f in spurious
        )
    )
    print(
        f"  ✓  Transformations dimension: no continuous-task findings "
        f"({len(transform_findings)} total finding(s) in dimension)",
    )
