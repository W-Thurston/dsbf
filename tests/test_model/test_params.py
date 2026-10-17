# tests/test_model/test_params.py
"""Check parameters are declared, defaulted and strict."""

from typing import Literal

import pytest
from pydantic import Field, ValidationError

from dsbf.model import CheckParams

pytestmark = pytest.mark.unit


class OutlierParams(CheckParams):
    method: Literal["all", "iqr", "zscore"] = "all"
    flag_threshold: float = Field(
        0.01, ge=0, le=1, description="Share of outliers that raises a finding."
    )


def test_a_check_runs_with_no_configuration():
    assert OutlierParams() == OutlierParams(method="all", flag_threshold=0.01)


def test_unknown_parameters_are_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        OutlierParams.model_validate({"flag_treshold": 0.5})


def test_declared_bounds_and_choices_are_enforced():
    with pytest.raises(ValidationError):
        OutlierParams.model_validate({"flag_threshold": 2})
    with pytest.raises(ValidationError):
        OutlierParams.model_validate({"method": "mad"})


def test_parameters_are_immutable_and_hashable_by_value():
    params = OutlierParams()

    with pytest.raises(ValidationError):
        params.method = "iqr"  # type: ignore[misc]
    assert params.model_copy(update={"method": "iqr"}) != params


def test_a_parameter_without_a_default_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="without a default: window"):

        class Bad(CheckParams):
            window: int
            step: int = 1


def test_descriptions_reach_the_json_schema():
    properties = OutlierParams.model_json_schema()["properties"]

    assert properties["flag_threshold"]["description"].startswith("Share of outliers")
    assert properties["flag_threshold"]["default"] == 0.01
