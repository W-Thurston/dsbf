"""A plan says before the run whether each check will run, and why not."""

import polars as pl
import pytest

from dsbf.data import load
from dsbf.engine import (
    Expected,
    Registry,
    Requires,
    check,
    make_plan,
)
from dsbf.engine import extras as extras_module
from dsbf.engine.check import SemanticTypes
from dsbf.model import Dimension
from dsbf.profiling import profile_dataset
from tests.test_engine.toys import make_dataset

pytestmark = pytest.mark.unit

NUMERIC = SemanticTypes("numeric")


def declare(check_id: str, *, applies_to=SemanticTypes(), **spec):
    @check(
        id=check_id,
        version="1.0.0",
        kind=spec.pop("kind", "rule"),
        applies_to=applies_to,
        dimensions=[Dimension.VALIDITY],
        **spec,
    )
    def declared(ctx, params):
        """A check."""

    return declared


@pytest.fixture(scope="module")
def profile():
    return profile_dataset(make_dataset())


def expectation(checks, profile=None, **plan):
    plan = make_plan(Registry(checks), profile=profile, **plan)
    return {entry.id: entry.expected for entry in plan}


class TestColumns:
    def test_a_check_will_run_on_the_columns_its_type_selects(self, profile):
        found = expectation([declare("spread", applies_to=NUMERIC)], profile)

        assert found["spread"].status is Expected.WILL_RUN
        assert found["spread"].columns == ("amount", "single")
        assert found["spread"].reason is None

    def test_a_check_about_the_whole_dataset_reads_no_columns(self, profile):
        found = expectation([declare("shape", applies_to=None)], profile)

        assert found["shape"].status is Expected.WILL_RUN
        assert found["shape"].columns is None

    def test_no_column_of_the_type_is_not_applicable(self):
        profile = profile_dataset(load(pl.DataFrame({"a": ["x", "y", "x"] * 20})))
        found = expectation([declare("spread", applies_to=NUMERIC)], profile)

        assert found["spread"].status is Expected.NOT_APPLICABLE
        assert found["spread"].reason == (
            "No column is of a type spread applies to (numeric)."
        )
        assert found["spread"].columns == ()

    def test_a_missing_profile_feature_narrows_the_columns(self, profile):
        # `single` has one value, so its histogram is empty.
        found = expectation(
            [
                declare(
                    "shape_of",
                    applies_to=NUMERIC,
                    requires=Requires(profile=["histogram"]),
                )
            ],
            profile,
        )

        assert found["shape_of"].status is Expected.WILL_RUN
        assert found["shape_of"].columns == ("amount",)

    def test_no_column_with_the_feature_is_not_applicable(self):
        constant = profile_dataset(load(pl.DataFrame({"a": [1.5] + [None] * 49})))
        found = expectation(
            [
                declare(
                    "shape_of",
                    applies_to=NUMERIC,
                    requires=Requires(profile=["histogram"]),
                )
            ],
            constant,
        )

        assert found["shape_of"].status is Expected.NOT_APPLICABLE
        assert found["shape_of"].reason == (
            "No numeric column has histogram in the profile, which shape_of requires."
        )

    def test_a_feature_a_type_never_carries_rules_the_column_out(self, profile):
        found = expectation(
            [declare("tops", requires=Requires(profile=["top_values"]))], profile
        )

        assert found["tops"].columns == ("colour", "comment")

    def test_a_dataset_feature_the_profile_lacks_is_not_applicable(self):
        bare = profile_dataset(load(pl.DataFrame({"a": [1, 2, 3]})))
        bare = bare.model_copy(update={"duplicate_rows": None})
        found = expectation(
            [
                declare(
                    "dupes",
                    applies_to=None,
                    requires=Requires(profile=["duplicate_rows"]),
                )
            ],
            bare,
        )

        assert found["dupes"].status is Expected.NOT_APPLICABLE
        assert found["dupes"].reason == (
            "The profile has no duplicate_rows, which dupes requires."
        )

    def test_without_a_profile_columns_are_not_judged(self):
        found = expectation([declare("spread", applies_to=NUMERIC)])

        assert found["spread"].status is Expected.WILL_RUN
        assert found["spread"].columns is None

    def test_the_plan_records_whether_it_saw_a_profile(self, profile):
        registry = Registry([declare("one")])

        assert make_plan(registry).profiled is False
        assert make_plan(registry, profile=profile).profiled is True


class TestExtras:
    @pytest.fixture
    def without_statsmodels(self, monkeypatch):
        real = extras_module.find_spec
        monkeypatch.setattr(
            extras_module,
            "find_spec",
            lambda m: None if m == "statsmodels" else real(m),
        )

    def test_a_missing_extra_is_unavailable_with_an_install_hint(
        self, without_statsmodels, profile
    ):
        found = expectation([declare("acf", extras=["timeseries"])], profile)

        assert found["acf"].status is Expected.UNAVAILABLE
        assert found["acf"].reason == (
            "Needs statsmodels, which is not installed ('timeseries' extra)."
        )
        assert found["acf"].hint == "Install it with `pip install 'dsbf[timeseries]'`."

    def test_unavailable_is_judged_without_a_profile(self, without_statsmodels):
        found = expectation([declare("acf", extras=["timeseries"])])

        assert found["acf"].status is Expected.UNAVAILABLE

    def test_an_installed_extra_does_not_matter(self, profile):
        found = expectation([declare("fine", extras=["pandas"])], profile)

        assert found["fine"].status is Expected.WILL_RUN

    def test_a_dependent_of_an_unavailable_check_is_skipped(
        self, without_statsmodels, profile
    ):
        found = expectation(
            [
                declare("acf", extras=["timeseries"]),
                declare(
                    "summary",
                    applies_to=None,
                    kind="synthesis",
                    requires=Requires(checks=["acf"]),
                ),
            ],
            profile,
        )

        assert found["summary"].status is Expected.SKIPPED
        assert (
            found["summary"].reason == "Requires 'acf' (unavailable), which cannot run."
        )
        assert found["summary"].blocked_by == ("acf",)

    def test_skipping_spreads_down_the_chain(self, without_statsmodels, profile):
        found = expectation(
            [
                declare("acf", extras=["timeseries"]),
                declare("middle", requires=Requires(checks=["acf"])),
                declare("last", requires=Requires(checks=["middle"])),
            ],
            profile,
        )

        assert found["last"].status is Expected.SKIPPED
        assert found["last"].blocked_by == ("middle",)

    def test_a_soft_dependency_never_blocks(self, without_statsmodels, profile):
        found = expectation(
            [
                declare("acf", extras=["timeseries"]),
                declare("later", after=["acf"]),
            ],
            profile,
        )

        assert found["later"].status is Expected.WILL_RUN

    def test_a_not_applicable_requirement_does_not_block(self):
        profile = profile_dataset(load(pl.DataFrame({"a": ["x", "y"] * 20})))
        found = expectation(
            [
                declare("spread", applies_to=NUMERIC),
                declare(
                    "summary", applies_to=None, requires=Requires(checks=["spread"])
                ),
            ],
            profile,
        )

        assert found["spread"].status is Expected.NOT_APPLICABLE
        assert found["summary"].status is Expected.WILL_RUN

    def test_a_missing_extra_outranks_a_blocked_requirement(
        self, without_statsmodels, profile
    ):
        found = expectation(
            [
                declare("acf", extras=["timeseries"]),
                declare(
                    "both", extras=["timeseries"], requires=Requires(checks=["acf"])
                ),
            ],
            profile,
        )

        assert found["both"].status is Expected.UNAVAILABLE
