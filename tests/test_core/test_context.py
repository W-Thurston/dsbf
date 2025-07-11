# tests/test_core/test_context.py

import pandas as pd
import polars as pl
import pytest

from dsbf.core.base_task import BaseTask
from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult


class DummyTask(BaseTask):
    def __init__(self, name="dummy_task", output=None):
        super().__init__(name=name)
        self.output = output

    def run(self):
        pass

    def get_output(self):
        return self.output


def test_validate_with_none_data_raises():
    ctx = AnalysisContext(data=None)
    with pytest.raises(ValueError):
        ctx.validate()


def test_validate_with_invalid_data_type_raises():
    ctx = AnalysisContext(data="not_a_dataframe")
    with pytest.raises(TypeError):
        ctx.validate()


def test_get_and_set_metadata():
    ctx = AnalysisContext(data=pd.DataFrame())
    ctx.set_metadata("example_key", 123)
    assert ctx.get_metadata("example_key") == 123
    assert ctx.get_metadata("nonexistent_key", default="fallback") == "fallback"


def test_get_and_set_result():
    ctx = AnalysisContext(data=pd.DataFrame())
    result = TaskResult(name="test", summary={"message": "ok"})
    ctx.set_result("test", result)
    assert ctx.has_result("test") is True
    assert ctx.get_result("test") == result
    assert ctx.get_result("nonexistent") is None


def test_run_task_validates_and_stores_result():
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = AnalysisContext(data=df)
    result = TaskResult(name="dummy_task", summary={"message": "ok"})
    task = DummyTask(output=result)

    returned = ctx.run_task(task)

    assert returned is result
    assert ctx.get_result("dummy_task") == result


def test_run_task_raises_on_none_result():
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = AnalysisContext(data=df)
    task = DummyTask(output=None)
    with pytest.raises(RuntimeError):
        ctx.run_task(task)


def test_repr_method():
    ctx = AnalysisContext(data=pd.DataFrame(), config={"a": 1})
    assert "AnalysisContext" in repr(ctx)
    assert "config_keys" in repr(ctx)


def test_compute_reliability_flags_caches_result():
    df = pd.DataFrame({"x": [1, 2, 1000]})
    ctx = AnalysisContext(data=df)
    assert ctx.reliability_flags == {}
    ctx.compute_reliability_flags(df)
    assert "n_rows" in ctx.reliability_flags

    # Should not recompute
    cached = ctx.reliability_flags
    ctx.compute_reliability_flags(df)
    assert ctx.reliability_flags is cached


def test_compute_reliability_flags_works_with_polars():
    df = pl.DataFrame({"x": [1, 2, 3, 1000]})
    ctx = AnalysisContext(data=df)
    ctx.compute_reliability_flags(df)
    assert isinstance(ctx.reliability_flags, dict)
    assert "n_rows" in ctx.reliability_flags
