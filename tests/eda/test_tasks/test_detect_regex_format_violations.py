import polars as pl

from dsbf.core.context import AnalysisContext
from dsbf.eda.tasks.detect_regex_format_violations import DetectRegexFormatViolations


def run_task(df, config=None):
    context = AnalysisContext(data=df, config=config or {})
    task = DetectRegexFormatViolations(
        name="detect_regex_format_violations",
        config=(config or {})
        .get("tasks", {})
        .get("detect_regex_format_violations", {}),
    )
    task.set_input(df)
    task.context = context
    task.run()
    return task.get_output()


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
    config = {
        "tasks": {
            "detect_regex_format_violations": {
                "patterns": {"email": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
                "max_violations_to_store": 2,
            }
        }
    }
    result = run_task(df, config)
    assert result is not None
    assert result.status == "success"
    assert "email" in result.summary["columns"]
    v = result.summary["violations"]["email"]
    assert v["num_violations"] == 3
    assert len(v["sample_violations"]) == 2


def test_ignores_columns_not_in_config():
    df = pl.DataFrame({"some_col": ["123", "456"]})
    config = {"tasks": {"detect_regex_format_violations": {"patterns": {}}}}
    result = run_task(df, config)
    assert result is not None
    assert result.status == "success"
    assert result.summary["num_columns_with_violations"] == 0
