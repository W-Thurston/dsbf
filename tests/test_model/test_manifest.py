"""The dataset and plan manifests identify what a run saw and was asked to do."""

import math

import pytest
from pydantic import ValidationError

from dsbf.model import (
    DatasetManifest,
    PlanManifest,
    PlannedCheck,
    SamplingInfo,
    SourceRef,
)

from .builders import make_dataset_manifest, round_trip, strict_loads

pytestmark = pytest.mark.unit


def test_an_in_memory_frame_has_no_location_or_format():
    manifest = make_dataset_manifest(source=SourceRef(kind="dataframe"))

    assert manifest.source.location is None
    assert round_trip(manifest) == manifest


def test_a_source_kind_is_one_of_the_known_kinds():
    with pytest.raises(ValidationError):
        SourceRef.model_validate({"kind": "carrier pigeon"})


def test_a_sample_must_describe_the_same_source():
    sampling = SamplingInfo(
        strategy="random", original_rows=1000, sampled_rows=100, seed=7
    )

    manifest = make_dataset_manifest(n_rows=1000, sampling=sampling)

    assert manifest.analysed_rows == 100
    assert make_dataset_manifest(n_rows=1000).analysed_rows == 1000
    with pytest.raises(ValidationError, match="original_rows"):
        make_dataset_manifest(n_rows=999, sampling=sampling)


def test_a_dataset_needs_a_name_and_a_fingerprint():
    with pytest.raises(ValidationError):
        make_dataset_manifest(name="")
    with pytest.raises(ValidationError):
        make_dataset_manifest(fingerprint="")
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate({**make_dataset_manifest().model_dump(), "x": 1})


def test_a_plan_round_trips_with_dependencies():
    plan = PlanManifest(
        dsbf_version="0.1.0",
        depth="standard",
        checks=(
            PlannedCheck(check_id="skewness", check_version="1", pack="core"),
            PlannedCheck(
                check_id="ml_readiness",
                check_version="2",
                pack="core",
                params={"threshold": 0.5, "methods": ("iqr", "zscore")},
                depends_on=("skewness",),
            ),
        ),
    )

    restored = round_trip(plan)

    assert restored.check_ids == ("skewness", "ml_readiness")
    assert restored.checks[1].params["methods"] == ["iqr", "zscore"]


def test_planned_parameters_must_be_finite():
    with pytest.raises(ValidationError, match="not a finite number"):
        PlannedCheck(check_id="x", check_version="1", params={"alpha": math.nan})


def test_planned_parameters_serialise_as_strict_json():
    check = PlannedCheck(check_id="x", check_version="1", params={"alpha": 0.05})

    assert strict_loads(check.model_dump_json())["params"] == {"alpha": 0.05}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"checks": ("a", "a")}, "each check once"),
        ({"checks": ("a",), "depends": {"a": ("b",)}}, "depends on 'b'"),
        ({"checks": ("a",), "depends": {"a": ("a",)}}, "depends on itself"),
    ],
)
def test_a_plan_refers_only_to_what_it_contains(kwargs, message):
    depends = kwargs.get("depends", {})
    checks = tuple(
        PlannedCheck(
            check_id=c,
            check_version="1",
            depends_on=depends.get(c, ()),
        )
        for c in kwargs["checks"]
    )

    with pytest.raises(ValidationError, match=message):
        PlanManifest(dsbf_version="0.1.0", checks=checks)
