# tests/eda/test_tasks/test_detect_regex_format_violations.py

import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_regex_format_violations import DetectRegexFormatViolations
from tests.helpers.context_utils import make_ctx_and_task


def test_detects_email_format_violations(tmp_path) -> None:
    """Values not matching the email pattern must be flagged."""
    df = pl.DataFrame(
        {
            "email": [
                "user@example.com",
                "bad-email",
                "another@example.org",
                "oops_at_domain.com",
                "wrong@site",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={
            "custom_patterns": {"email": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
            "max_violations": 2,
        },
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert "email" in result.summary["columns"]
    v = result.summary["violations"]["email"]
    assert v["num_violations"] == 3
    assert len(v["sample_violations"]) == 2  # capped at max_violations=2


def test_ignores_columns_not_in_patterns(tmp_path) -> None:
    """Columns not listed in custom_patterns must produce no violations."""
    df = pl.DataFrame({"some_col": ["123", "456"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={"custom_patterns": {}},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_columns_with_violations"] == 0


def test_no_violations_on_conforming_data(tmp_path) -> None:
    """A column where all values match the pattern must produce no findings."""
    df = pl.DataFrame({"phone": ["+12345678901", "+447911123456", "+33612345678"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={"custom_patterns": {"phone": r"^\+?[0-9]{7,15}$"}},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_columns_with_violations"] == 0


def test_guidance_attached_for_violating_columns(tmp_path) -> None:
    """EDA guidance blurbs must be attached for each column with violations."""
    df = pl.DataFrame({"code": ["ABC-123", "XYZ-456", "bad", "123-DEF"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={"custom_patterns": {"code": r"^[A-Z]{3}-[0-9]{3}$"}},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_columns_with_violations"] == 1
    assert result.guidance is not None
    assert "code" in result.guidance
    assert len(result.guidance["code"]["eda"]) > 0
    assert result.guidance["code"]["eda"][0]["level"] == "warn"


def test_invalid_regex_skipped_gracefully(tmp_path) -> None:
    """An invalid regex pattern must be skipped without crashing."""
    df = pl.DataFrame({"col": ["value1", "value2"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={"custom_patterns": {"col": "[invalid(regex"}},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_columns_with_violations"] == 0


def test_no_plots_generated(tmp_path) -> None:
    """Regex violation detection must not generate plots."""
    df = pl.DataFrame({"email": ["bad-email", "also@bad"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={"custom_patterns": {"email": r"^[\w\.-]+@[\w\.-]+\.\w+$"}},
        global_overrides={"output_dir": str(tmp_path)},
    )
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.plots is None
