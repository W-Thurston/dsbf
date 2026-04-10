# tests/validation/assertions/tiny_dataset.py
"""
Assertions for the tiny dataset (25 rows) profiling output.

The key property: tasks may skip or produce limited output,
but none should crash with status='error'.
"""

from __future__ import annotations

from typing import Any

# Tasks that are expected to skip or produce empty output on 25 rows.
# Any task NOT in this list that errors is a genuine bug.
EXPECTED_TO_SKIP_OR_LIMIT = {
    "detect_bimodal_distribution",  # BIC fit may skip on tiny n
    "compute_pairwise_associations",  # may produce 0 pairs
    "normality_tests",  # low power, may mark as unreliable
    "normality_qq_plots",  # depends on normality_tests
    "compute_mutual_information",  # may skip
    "detect_collinear_features",  # VIF unstable with tiny n
}

# Tasks that must succeed regardless of dataset size
MUST_SUCCEED = {
    "infer_types",
    "summarize_dataset_shape",
    "summarize_nulls",
    "data_quality_scorer",
}


def validate(report: dict[str, Any]) -> None:
    _no_unexpected_errors(report)
    _must_succeed_tasks_ran(report)
    _row_count_correct(report)
    _sample_size_context(report)
    print("✓  tiny dataset: all assertions passed")


# ── Individual checks ─────────────────────────────────────────────────────────


def _no_unexpected_errors(report: dict) -> None:
    """
    Tasks may skip gracefully on tiny n, but none should error unless
    they're in the explicitly expected-to-skip set.
    """
    for task_name, task in report.get("results", {}).items():
        if not isinstance(task, dict):
            continue
        status = task.get("status")
        if status == "error" and task_name not in EXPECTED_TO_SKIP_OR_LIMIT:
            msg: str = (
                f"Task '{task_name}' errored on a tiny dataset - this should degrade "
                f"gracefully (status='skipped' or produce empty output), not crash.\n"
                f"  error_metadata: {task.get('error_metadata')}"
            )
            raise AssertionError(msg)


def _must_succeed_tasks_ran(report: dict) -> None:
    """Core tasks must succeed even on 25 rows."""
    results = report.get("results", {})
    for task_name in MUST_SUCCEED:
        task = results.get(task_name)
        assert task is not None, f"'{task_name}' is missing from results entirely"
        status = task.get("status")
        assert status == "success", (
            f"'{task_name}' must succeed on any dataset size, "
            f"but got status='{status}'.\n"
            f"  error_metadata: {task.get('error_metadata')}"
        )


def _row_count_correct(report: dict) -> None:
    """The shape task should report the correct row count."""
    shape = report.get("results", {}).get("summarize_dataset_shape", {})
    if shape.get("status") != "success":
        return
    num_rows = shape.get("data", {}).get("num_rows")
    assert (
        num_rows == 25
    ), f"Expected 25 rows in tiny dataset, but shape task reports {num_rows}"


def _sample_size_context(report: dict) -> None:
    """
    The shape task or summary should note that sample size is small.

    We check that it at least ran - the dashboard Overview tab shows
    a sample size adequacy metric which is where the warning surfaces visually.
    """
    shape = report.get("results", {}).get("summarize_dataset_shape", {})
    assert shape.get("status") == "success", (
        "summarize_dataset_shape must succeed to enable sample size "
        "adequacy in dashboard"
    )
    num_rows = shape.get("data", {}).get("num_rows", 0)
    assert num_rows < 100, (
        f"Expected a tiny dataset (< 100 rows) but got {num_rows} - "
        "wrong dataset may have been profiled"
    )
