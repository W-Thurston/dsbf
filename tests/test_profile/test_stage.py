"""Stage inference reads the profile alone, so the data's library cannot matter."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.config import load_default_config
from dsbf.data import load
from dsbf.model import DatasetProfile, Stage
from dsbf.profile import StageThresholds, infer_stage, profile_dataset

pytestmark = pytest.mark.unit


def _stage(frame, thresholds=None) -> Stage:
    return infer_stage(profile_dataset(load(frame)), thresholds)


def _numeric(n=60, nulls=0):
    values = np.random.default_rng(0).normal(size=n)
    values = values.astype(object)
    values[:nulls] = None
    return pl.Series(values.tolist(), dtype=pl.Float64)


def test_the_same_numeric_frame_has_one_stage_from_pandas_and_polars():
    # The 4×2 numeric frame the legacy inference called model_ready from pandas
    # and cleaned from Polars.
    data = {"a": [1.5, 2.5, 3.5, 4.5], "b": [10.0, 20.0, 30.0, 40.0]}

    from_pandas = _stage(pd.DataFrame(data))
    from_polars = _stage(pl.DataFrame(data))

    assert from_pandas == from_polars == Stage.MODELING


def test_pandas_and_polars_agree_on_a_mixed_frame_with_nulls():
    data = {
        "x": [1.0, 2.0, None, 4.0] * 25,
        "label": ["a", "b", "a", None] * 25,
        "n": list(range(100)),
    }

    assert _stage(pd.DataFrame(data)) == _stage(pl.DataFrame(data))


def test_mostly_missing_data_is_raw():
    frame = pl.DataFrame({"a": _numeric(nulls=30), "b": _numeric(nulls=30)})

    assert _stage(frame) == Stage.RAW


def test_a_mean_null_ratio_at_the_raw_threshold_is_not_raw():
    frame = pl.DataFrame({"a": _numeric(n=10, nulls=4), "b": _numeric(n=10)})
    # The mean null ratio is 0.2.
    lenient = {"max_null_ratio_cleaned": 0.5}

    at = StageThresholds(null_ratio_raw_threshold=0.2, **lenient)
    below = StageThresholds(null_ratio_raw_threshold=0.19, **lenient)

    assert _stage(frame, at) == Stage.CLEANED
    assert _stage(frame, below) == Stage.RAW


def test_complete_mostly_numeric_data_is_modeling():
    frame = pl.DataFrame(
        {"a": _numeric(), "b": _numeric(), "c": ["x", "y"] * 30, "d": _numeric()}
    )

    assert _stage(frame) == Stage.MODELING


def test_modeling_needs_the_numeric_share():
    frame = pl.DataFrame({"a": _numeric(), "b": ["x", "y"] * 30, "c": ["p", "q"] * 30})

    assert _stage(frame) == Stage.CLEANED


def test_a_few_nulls_keep_numeric_data_from_modeling():
    frame = pl.DataFrame({"a": _numeric(n=100, nulls=8), "b": _numeric(n=100)})
    # mean null ratio 0.04 is under the 0.05 limit, so make it 0.06
    frame = pl.DataFrame({"a": _numeric(n=100, nulls=12), "b": _numeric(n=100)})

    assert _stage(frame) == Stage.CLEANED


def test_a_high_cardinality_categorical_column_is_raw():
    values = [f"v{i % 60}" for i in range(600)]
    frame = pl.DataFrame({"a": values, "b": ["x", "y"] * 300})

    assert _stage(frame) == Stage.RAW


def test_high_cardinality_is_counted_on_categorical_columns_only():
    # A continuous column has far more than 50 distinct values and is not a cleaning
    # problem; neither is an identifier.
    frame = pl.DataFrame(
        {
            "a": _numeric(n=200),
            "id": [f"id-{i}" for i in range(200)],
            "c": ["x", "y"] * 100,
        }
    )

    assert _stage(frame) == Stage.CLEANED


def test_the_high_cardinality_threshold_is_configurable():
    values = [f"v{i % 60}" for i in range(600)]
    frame = pl.DataFrame({"a": values, "b": ["x", "y"] * 300})

    stage = _stage(frame, StageThresholds(high_cardinality_threshold=100))

    assert stage == Stage.CLEANED


def test_a_profile_with_no_rows_or_no_columns_is_raw():
    assert _stage(pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})) == Stage.RAW
    assert infer_stage(DatasetProfile(n_rows=5, columns=())) == Stage.RAW


def test_inference_returns_only_stages_a_dataset_can_be_at():
    frames = [
        pl.DataFrame({"a": _numeric()}),
        pl.DataFrame({"a": ["x", "y"] * 30}),
        pl.DataFrame({"a": _numeric(nulls=40)}),
    ]

    assert {_stage(f) for f in frames} <= {Stage.RAW, Stage.CLEANED, Stage.MODELING}


def test_the_default_config_section_loads_as_the_default_thresholds():
    assert StageThresholds.from_config(load_default_config()) == StageThresholds()


def test_thresholds_are_read_from_the_engine_section():
    config = {"engine": {"stage_inference": {"high_cardinality_threshold": 7}}}

    assert StageThresholds.from_config(config).high_cardinality_threshold == 7


def test_a_config_without_the_section_gives_the_defaults():
    assert StageThresholds.from_config({}) == StageThresholds()
    assert StageThresholds.from_config({"engine": None}) == StageThresholds()


def test_a_misplaced_or_misspelt_threshold_is_an_error():
    with pytest.raises(ValidationError):
        StageThresholds.from_config({"engine": {"stage_inference": {"nulls": 1}}})
    with pytest.raises(ValidationError):
        StageThresholds(null_ratio_raw_threshold=1.5)


def test_the_vocabulary_has_the_task_stages():
    assert [s.value for s in Stage] == ["raw", "cleaned", "modeling", "report", "any"]
