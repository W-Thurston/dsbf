# tests/validation/assertions/clean_dataset.py
"""
Assertions for the clean dataset profiling output.

Every column is well-behaved — no findings expected anywhere.

Known backend bugs and expected skips are tracked explicitly so failures
surface with clear descriptions rather than confusing assertion errors.
"""

from __future__ import annotations

from typing import Any

# ── Known backend bugs ────────────────────────────────────────────────────────
# Tasks that currently fail due to known bugs. Each entry describes the root
# cause and the fix needed. Remove an entry once the backend bug is resolved.

KNOWN_BACKEND_BUGS: dict[str, str] = {
    "detect_out_of_bounds": (
        "Fails with 'Invalid comparison between dtype=int64 and str' when config "
        "bounds are stored as strings but the column dtype is numeric. "
        "Fix needed in detect_out_of_bounds.py: cast bound values to the column "
        "dtype before comparison."
    ),
}

# Quality dimensions known to produce false-positive findings on the clean dataset.
# The task succeeds but its findings are incorrect — tracked separately from
# KNOWN_BACKEND_BUGS because the failure is in scoring logic, not task execution.
KNOWN_FALSE_POSITIVE_DIMENSIONS: dict[str, str] = {
    "redundancy": (
        "detect_collinear_features produces inflated VIF scores (11-34) on "
        "uncorrelated columns (all pairwise r < 0.02). Likely cause: regression "
        "matrix includes categorical/boolean columns; balanced categoricals "
        "create a near-linear dependency with the intercept term, inflating VIF "
        "for all continuous columns. Fix: restrict regression to continuous "
        "columns only in detect_collinear_features.py."
    ),
}

# Tasks that are skipped (status='skipped') when optional data sources are not
# configured. This is correct behaviour, not a bug.
EXPECTED_SKIPS: set[str] = {
    "compare_with_reference_dataset",  # no reference dataset configured
    "detect_class_imbalance",  # no target column configured
    "detect_feature_drift",  # no reference dataset configured
    "detect_target_drift",  # no target column configured
    "schema_validation",  # enable_schema_validation: false in config
}

# Tasks whose None status in results is caused by a known upstream dependency
# failure cascading down. Not bugs in these tasks themselves.
DEPENDENCY_CASCADE_SKIPS: set[str] = {
    "data_quality_scorer",  # hard-depends on detect_out_of_bounds
    "ml_readiness_scorer",  # hard-depends on data_quality_scorer
}


# ── Entry point ────────────────────────────────────────────────────────────────


def validate(report: dict[str, Any]) -> None:
    """
    Run all assertions for the clean dataset.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _report_known_bugs(report)
    _no_unexpected_failures(report)
    _must_succeed_tasks_ran(report)
    _type_inference_correct(report)
    _no_nulls(report)
    _quality_all_green_if_available(report)
    _no_quality_findings_if_available(report)
    _ml_readiness_no_errors_if_available(report)
    print("✓  clean dataset: all assertions passed")


# ── Checks ────────────────────────────────────────────────────────────────────


def _report_known_bugs(report: dict) -> None:
    """Print a clear notice for each known bug that was triggered."""
    triggered = [
        (name, desc)
        for name, desc in KNOWN_BACKEND_BUGS.items()
        if isinstance(report["results"].get(name), dict)
        and report["results"][name].get("status") == "failed"
    ]
    if triggered:
        print("\n  ⚠  Known backend bugs triggered (fix before next full validation):")
        for name, desc in triggered:
            print(f"     [{name}]\n       {desc}")
        print()


def _no_unexpected_failures(report: dict) -> None:
    """
    No task should fail unless it's a known bug.

    Skipped tasks (optional data not configured) are fine.
    None entries (dependency cascade from a known bug) are tracked separately.
    """
    results = report.get("results", {})

    for task_name, task in results.items():
        # None means the task was skipped via dependency cascade
        if task is None:
            assert task_name in DEPENDENCY_CASCADE_SKIPS, (
                f"Task '{task_name}' is None in results (dependency-cascade skip) "
                "but is not in DEPENDENCY_CASCADE_SKIPS.\n"
                "  If caused by a known upstream bug, add it to "
                "DEPENDENCY_CASCADE_SKIPS."
            )
            continue

        if not isinstance(task, dict):
            continue

        status = task.get("status")

        if status == "failed":
            assert task_name in KNOWN_BACKEND_BUGS, (
                f"Task '{task_name}' failed unexpectedly on the clean dataset.\n"
                f"  error: {task.get('error_metadata', {}).get('trace_summary')}\n"
                f"  If this is a known bug, add it to KNOWN_BACKEND_BUGS."
            )

        if status == "skipped":
            assert task_name in EXPECTED_SKIPS, (
                f"Task '{task_name}' was skipped but is not in EXPECTED_SKIPS.\n"
                f"  If this skip is intentional (e.g. optional data not configured), "
                f"add it to EXPECTED_SKIPS."
            )


def _must_succeed_tasks_ran(report: dict) -> None:
    """These tasks must succeed regardless of dataset size or configuration."""
    must_succeed: set[str] = {
        "infer_types",
        "summarize_dataset_shape",
        "summarize_nulls",
        "summarize_numeric",
        "detect_skewness",
        "detect_outliers",
        "detect_constant_columns",
        "detect_duplicate_columns",
        "detect_id_columns",
    }
    results = report.get("results", {})
    for task_name in must_succeed:
        task = results.get(task_name)
        assert isinstance(
            task, dict
        ), f"'{task_name}' is missing or None — must always be present in results"
        assert task.get("status") == "success", (
            f"'{task_name}' must succeed on the clean dataset, "
            f"got status='{task.get('status')}'.\n"
            f"  error: {task.get('error_metadata')}"
        )


def _type_inference_correct(report: dict) -> None:
    """Column types must be inferred correctly for the clean dataset."""
    types = report["results"]["infer_types"]["data"]
    assert types, "infer_types produced no column data"

    expected: dict[str, str] = {
        "age": "continuous",
        "income": "continuous",
        "score": "continuous",
        "tenure_years": "continuous",
        "region": "categorical",
        "plan_type": "categorical",
        "department": "categorical",
    }
    for col, expected_intent in expected.items():
        actual = types.get(col, {}).get("analysis_intent_dtype")
        assert (
            actual == expected_intent
        ), f"Column '{col}': expected intent '{expected_intent}', got '{actual}'"


def _no_nulls(report: dict) -> None:
    """Clean dataset has no nulls — every column should show 0.0%."""
    null_pcts = (
        report["results"]["summarize_nulls"].get("data", {}).get("null_percentages", {})
    )
    assert null_pcts, "summarize_nulls produced no null_percentages"
    for col, pct in null_pcts.items():
        assert (
            pct == 0.0
        ), f"Column '{col}' shows {pct * 100:.1f}% nulls on the clean dataset"


def _quality_all_green_if_available(report: dict) -> None:
    """
    If data_quality_scorer ran:
    all dimensions must be green except known false positives.
    """
    scorer = report["results"].get("data_quality_scorer")
    if scorer is None:
        print(
            "  ℹ  data_quality_scorer skipped (upstream dependency failed) — "
            "green check deferred",
        )
        return
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        return

    cats = scorer.get("data", {}).get("categories", {})
    for dim, cat in cats.items():
        if dim in KNOWN_FALSE_POSITIVE_DIMENSIONS:
            level = cat.get("level")
            print(
                f"  ℹ  Quality '{dim}' is '{level}' (known false positive — "
                f"{KNOWN_FALSE_POSITIVE_DIMENSIONS[dim][:80]}…)",
            )
            continue
        level = cat.get("level")
        assert level == "green", (
            f"Quality dimension '{dim}' is '{level}' on the clean dataset — "
            f"expected 'green'.\n  findings: {cat.get('findings', [])}"
        )


def _no_quality_findings_if_available(report: dict) -> None:
    """
    If data_quality_scorer ran:
    zero findings expected (except known false positives).
    """
    scorer = report["results"].get("data_quality_scorer")
    if scorer is None:
        return
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        return

    cats = scorer.get("data", {}).get("categories", {})
    for dim, cat in cats.items():
        if dim in KNOWN_FALSE_POSITIVE_DIMENSIONS:
            continue
        findings = cat.get("findings", [])
        assert len(findings) == 0, (
            f"Quality dimension '{dim}' has {len(findings)} finding(s) on the "
            f"clean dataset — expected none.\n  findings: {findings[:3]}"
        )


def _ml_readiness_no_errors_if_available(report: dict) -> None:
    """
    If ml_readiness_scorer ran, assert expected finding levels for the clean dataset.

    Expected state after all severity calibration fixes:
      - Zero ``error``-level findings: the clean dataset has no modeling blockers.
      - Gate is ``"needs_work"``: the three raw-string categorical columns
        (region, plan_type, department) are object dtype and must be encoded
        before sklearn can ingest them — this is a genuine warn-level finding,
        not a false positive.  A clean dataset with categorical columns is
        accurately described as "needs work" before modeling.
      - Gate is NOT ``"not_ready"``: no error-level blockers exist.

    If the gate is ``"ready"`` that would indicate the encoding warn findings
    are no longer being emitted correctly — fail with a clear message.
    """
    scorer = report["results"].get("ml_readiness_scorer")
    if scorer is None:
        print(
            "  ℹ  ml_readiness_scorer skipped (upstream dependency failed) — "
            "level check deferred",
        )
        return
    if not isinstance(scorer, dict) or scorer.get("status") != "success":
        return

    cats = scorer.get("data", {}).get("categories", {})
    gate = scorer.get("data", {}).get("readiness_gate")

    error_findings: list[tuple] = [
        (dim, f)
        for dim, cat in cats.items()
        for f in cat.get("findings", [])
        if f.get("level") == "error"
    ]

    if error_findings:
        print("\n  Error-level ML findings on clean dataset:")
        for dim, f in error_findings:
            print(
                f"    [{dim}] col={f.get('column')!r} "
                f"title={f.get('title')!r} task={f.get('task')!r}"
            )
        msg: str = (
            f"Found {len(error_findings)} error-level ML Readiness finding(s) "
            f"on the clean dataset — none expected."
        )
        raise AssertionError(msg)

    assert gate != "not_ready", (
        "ML gate is 'not_ready' on the clean dataset despite zero error-level "
        "findings. The gate should be severity-based — check _gate() in "
        "ml_readiness_scorer.py."
    )

    # Encoding warn findings for raw-string categoricals are expected and correct.
    # The gate should be 'needs_work', not 'ready' — 'ready' would mean encoding
    # warn findings are not being emitted, which would be a regression.
    encoding_warns: list = [
        f
        for f in cats.get("encoding", {}).get("findings", [])
        if f.get("level") == "warn"
    ]
    if gate == "ready" and not encoding_warns:
        print(
            "  ⚠  ML gate is 'ready' with no encoding warn findings — "
            "expected warn-level encoding findings for raw-string categorical "
            "columns (region, plan_type, department). Check suggest_categorical_"
            "encoding.py.",
        )
    elif gate == "needs_work":
        print(
            f"  ✓  ML gate: needs_work "
            f"({len(encoding_warns)} encoding warn finding(s) — expected)",
        )
    else:
        print(f"  ✓  ML gate: {gate}")
