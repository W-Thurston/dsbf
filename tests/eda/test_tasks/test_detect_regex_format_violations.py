import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_regex_format_violations import DetectRegexFormatViolations
from tests.helpers.context_utils import make_ctx_and_task


def test_detects_regex_format_violations():
    df = pl.DataFrame(
        {
            "email": [
                "user@example.com",
                "bad-email",
                "another@example.org",
                "oops_at_domain.com",
                "wrong@site",
            ]
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={
            "custom_patterns": {"email": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
            "max_violations": 2,
        },
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result is not None
    assert result.status == "success"
    assert "email" in result.summary["columns"]
    v = result.summary["violations"]["email"]
    assert v["num_violations"] == 3
    assert len(v["sample_violations"]) == 2


def test_ignores_columns_not_in_config():
    df = pl.DataFrame({"some_col": ["123", "456"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectRegexFormatViolations,
        current_df=df,
        task_overrides={"custom_patterns": {}},
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result is not None
    assert result.status == "success"
    assert result.summary["num_columns_with_violations"] == 0
