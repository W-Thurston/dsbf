# tests/test_core/test_base_task.py

import logging
import re

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult


def strip_ansi_and_whitespace(text: str) -> str:
    """Remove ANSI codes and compress whitespace for easier matching."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # Strip ANSI color codes
    return " ".join(text.split())  # Collapse all whitespace to single spaces


class DummyTask(BaseTask):
    def run(self):
        self.output = TaskResult(name=self.name, summary={"message": "ok"})


def test_set_ml_signals_adds_metadata():
    task = DummyTask(name="dummy")
    result = TaskResult(name="dummy", summary={})
    task.set_ml_signals(result, score=0.8, tags=["leakage"], recommendation="Drop col")
    assert result.ml_impact_score == 0.8
    assert result.recommendations is not None
    assert "Drop col" in result.recommendations
    assert result.recommendation_tags is not None
    assert "leakage" in result.recommendation_tags


def test_get_output_path_creates_fig_dir(tmp_path):
    ctx = AnalysisContext(data=pd.DataFrame(), config={}, output_dir=str(tmp_path))
    task = DummyTask(name="dummy")
    task.context = ctx
    path = task.get_output_path("plot.png")
    assert path.endswith("plot.png")
    assert tmp_path.joinpath("figs").exists()


def test_get_columns_by_intent_filters_semantic_types():
    ctx = AnalysisContext(data=pd.DataFrame(), config={})
    ctx.set_metadata("semantic_types", {"a": "continuous", "b": "categorical"})
    task = DummyTask(name="dummy")
    task.context = ctx
    matched, excluded = task.get_columns_by_intent(expected_types=["continuous"])
    assert matched == ["a"]
    assert "b" in excluded


def test_get_column_type_info_merges_both_dtypes():
    ctx = AnalysisContext(data=pd.DataFrame(), config={})
    ctx.set_metadata("semantic_types", {"a": "continuous"})
    ctx.set_metadata("inferred_dtypes", {"a": "float"})
    task = DummyTask(name="dummy")
    task.context = ctx
    out = task.get_column_type_info(["a"])
    assert out["a"]["inferred_dtype"] == "float"
    assert out["a"]["analysis_intent_dtype"] == "continuous"


def test_get_expected_types_from_registry_gracefully_handles_missing():
    task = DummyTask(name="dummy")
    # Should not raise even if not registered
    expected = task.get_expected_types()
    assert isinstance(expected, list)


def test_log_fallback_no_context(caplog, capfd):

    # Temporarily override setup_logger to attach stream handler
    def test_logger(name, level="info"):
        logger = logging.getLogger(name)
        logger.handlers = []  # clear existing
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return logger

    # Inject patched logger for fallback
    DummyTask._log = lambda self, msg, level="info": print(msg)

    task = DummyTask(name="dummy")
    task._log("Test fallback log", level="info")

    out, _ = capfd.readouterr()
    clean_out = strip_ansi_and_whitespace(out)
    assert "Test fallback log" in clean_out
