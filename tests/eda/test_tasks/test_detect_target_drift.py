import polars as pl

from dsbf.core.context import AnalysisContext
from dsbf.eda.tasks.detect_target_drift import DetectTargetDrift


def make_ctx_and_task(data, ref_data, target, config_overrides=None):
    config = {
        "target": target,
        "target_drift_thresholds": {
            "psi": 0.1,
            "ks_pvalue": 0.05,
            "chi2_pvalue": 0.05,
            "entropy_delta": 0.5,
        },
    }
    if config_overrides:
        config.update(config_overrides)

    ctx = AnalysisContext(data=data, config=config)
    ctx.reference_data = ref_data
    task = DetectTargetDrift(name="detect_target_drift", config=config)
    return ctx, task


def test_numeric_target_drift_detected():
    current = pl.DataFrame({"target": [1.0] * 50 + [10.0] * 50})
    reference = pl.DataFrame({"target": [1.0] * 100})
    ctx, task = make_ctx_and_task(current, reference, target="target")
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["target_type"] == "numerical"
    assert result.data["psi"] > 0.1
    assert result.data["drift_rating"] in {"moderate", "significant"}
    assert result.recommendations is not None
    assert "retrain" in result.recommendations[0].lower()


def test_categorical_target_drift_detected():
    current = pl.DataFrame({"target": ["A"] * 10 + ["B"] * 90})
    reference = pl.DataFrame({"target": ["A"] * 50 + ["B"] * 50})
    ctx, task = make_ctx_and_task(current, reference, target="target")
    result = ctx.run_task(task)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["target_type"] == "categorical"
    assert result.data["tvd"] > 0.1
    assert result.data["drift_rating"] in {"moderate", "significant"}
    assert "drift" in result.summary["message"].lower()


def test_missing_reference_skips():
    current = pl.DataFrame({"target": [1, 2, 3]})
    ctx = AnalysisContext(data=current, config={"target": "target"})
    task = DetectTargetDrift(name="detect_target_drift", config={"target": "target"})
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "skipped" in result.summary["message"].lower()


def test_missing_target_column_skips():
    current = pl.DataFrame({"x": [1, 2, 3]})
    ref = pl.DataFrame({"x": [1, 2, 3]})
    ctx, task = make_ctx_and_task(current, ref, target="y")
    result = ctx.run_task(task)

    assert result.status == "skipped"
    assert "missing" in result.summary["message"].lower()


def test_error_handling_on_invalid_input():
    current = pl.DataFrame({"target": []})
    reference = pl.DataFrame({"target": [1, 2, 3]})
    ctx, task = make_ctx_and_task(current, reference, target="target")
    result = ctx.run_task(task)

    assert result.status in {"error", "skipped", "failed"}
    assert (
        "error" in result.summary["message"].lower()
        or "skipped" in result.summary["message"].lower()
    )
