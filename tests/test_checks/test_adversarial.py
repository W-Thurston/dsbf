"""Every core check survives hostile frames and still ends in an honest status.

The frames are the fixed adversarial one and Hypothesis-generated ones: odd column
names (``a|b``, ``[/x]``, empty, unicode), all-null columns, one row, mixed types and
timezone-aware datetimes.
"""

import datetime as dt
import zoneinfo

import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dsbf.checks import core_pack
from dsbf.data import load
from dsbf.engine import Reference, Registry, Selection, execute, make_plan
from dsbf.model import CheckStatus, Run
from dsbf.profiling import profile_dataset
from tests.fixtures.adversarial import build_adversarial_frame
from tests.golden.datasets import DATA_DIR
from tests.test_model.builders import strict_loads

pytestmark = pytest.mark.integration

CHECKS = core_pack().checks
TITANIC = DATA_DIR / "titanic.csv"
ACCEPTABLE = {CheckStatus.OK, CheckStatus.NOT_APPLICABLE}


def run_core(
    frame: pd.DataFrame | pl.DataFrame,
    name: str = "adversarial",
    time_index: str | None = None,
    reference: pd.DataFrame | pl.DataFrame | None = None,
) -> Run:
    """Run every core check; ``time_index`` goes to each check that takes one.

    ``reference`` is the dataset the checks that compare with one are given.
    """
    dataset = load(frame, name=name)
    profile = profile_dataset(dataset)
    compared = None
    if reference is not None:
        loaded = load(reference, name="reference")
        compared = Reference.of_data(loaded, profile_dataset(loaded))
    params = (
        {c.id: {"time_index": time_index} for c in CHECKS if _takes_time_index(c)}
        if time_index is not None
        else None
    )
    plan = make_plan(
        Registry(CHECKS),
        Selection(depth="deep", entry_points=False),
        params=params,
        profile=profile,
        has_reference=compared is not None,
    )
    return execute(
        plan, dataset, profile, dsbf_version="test", workers=2, reference=compared
    )


def _takes_time_index(check) -> bool:
    return "time_index" in check.spec.params.model_fields


def first_datetime_column(frame: pd.DataFrame) -> str | None:
    """The first column the profile types datetime, if any."""
    profile = profile_dataset(load(frame, name="probe"))
    return next(
        (c.name for c in profile.columns if c.semantic_type == "datetime"), None
    )


def trouble(run: Run) -> dict[str, object]:
    """Each check that ended badly or raised, with what it said."""
    return {
        check_id: (result.status.value, [n.message for n in result.notes])
        for check_id, result in run.results.items()
        if result.status not in ACCEPTABLE
        or any(n.code == "check_failed" for n in result.notes)
    }


@pytest.fixture(scope="module")
def adversarial_run() -> Run:
    return run_core(pl.from_pandas(build_adversarial_frame()))


def test_the_core_pack_holds_checks() -> None:
    assert CHECKS


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.id)
def test_a_core_check_runs_on_the_adversarial_frame(
    check, adversarial_run: Run
) -> None:
    result = adversarial_run.results[check.id]

    assert result.status in ACCEPTABLE, [n.message for n in result.notes]
    assert not [n for n in result.notes if n.code == "check_failed"]


TIME_SERIES = ("stationarity", "autocorrelation", "seasonal_decomposition")


def test_the_time_series_checks_run_on_the_adversarial_frame_with_a_time_index() -> (
    None
):
    run = run_core(pl.from_pandas(build_adversarial_frame()), time_index="ts_tz")

    assert trouble(run) == {}
    for check_id in TIME_SERIES:
        result = run.results[check_id]
        assert result.status is CheckStatus.OK, [n.message for n in result.notes]
        assert result.output is not None
        # The hourly zone-aware index is read in UTC; the column with infinities is
        # a series of the finite values.
        assert result.output["frequency"] == "h"
        assert "inf_col" in result.output["columns"]
        assert "all_nan_float" not in result.output["columns"]
    strict_loads(run.model_dump_json())


def test_reference_drift_compares_the_adversarial_frame_with_itself() -> None:
    frame = pl.from_pandas(build_adversarial_frame())

    run = run_core(frame, reference=frame)

    assert trouble(run) == {}
    result = run.results["reference_drift"]
    assert result.status is CheckStatus.OK, [n.message for n in result.notes]
    assert result.findings == ()
    assert result.output is not None
    assert result.output["columns_added"] == result.output["columns_dropped"] == []
    strict_loads(run.model_dump_json())


def test_an_integer_coded_boolean_column_never_appears_in_zeros(
    adversarial_run: Run,
) -> None:
    output = adversarial_run.results["zeros"].output
    assert output is not None

    # Half of each is zero, so the legacy task flagged them; the profile types them
    # categorical, so the check never reads them.
    assert {"flag_as_int", "flag_copy", "[/x]"}.isdisjoint(output["columns"])
    assert adversarial_run.results["zeros"].findings == ()
    assert "price|usd" in output["columns"]


def test_zeros_runs_on_titanic() -> None:
    run = run_core(pd.read_csv(TITANIC), name="titanic")

    result = run.results["zeros"]
    assert result.status is CheckStatus.OK
    assert result.output is not None
    # Only the continuous columns: survived, pclass, sibsp and parch are small codes.
    assert set(result.output["columns"]) == {"age", "fare"}
    assert result.output["columns"]["fare"]["zero_count"] == 15


def test_the_adversarial_run_is_strict_json(adversarial_run: Run) -> None:
    strict_loads(adversarial_run.model_dump_json())
    assert set(adversarial_run.results) == {c.id for c in CHECKS}


# --- Hypothesis frames -----------------------------------------------------------

ODD_NAMES = ["a|b", "[/x]", "", " ", "naïve résumé 名前", "with,comma", "\x00", "💥"]
ZONES = [
    dt.UTC,
    zoneinfo.ZoneInfo("US/Eastern"),
    dt.timezone(dt.timedelta(hours=5, minutes=30)),
]
MOMENTS = st.datetimes(dt.datetime(1800, 1, 1), dt.datetime(2200, 1, 1))
ANY_VALUE = st.one_of(
    st.none(),
    st.integers(-(10**6), 10**6),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=8),
    st.booleans(),
)


def column(kind: str, n: int) -> st.SearchStrategy[pd.Series]:
    def many(values: st.SearchStrategy, dtype: str | None = None):
        return st.lists(values, min_size=n, max_size=n).map(
            lambda items: pd.Series(items, dtype=dtype)
        )

    if kind == "int":
        return many(st.integers(-(10**6), 10**6), "int64")
    if kind == "small_int":
        return many(st.integers(0, 2), "int64")
    if kind == "float":
        return many(st.floats(allow_nan=True, allow_infinity=True), "float64")
    if kind == "string":
        return many(st.one_of(st.none(), st.text(max_size=12)), "object")
    if kind == "bool":
        return many(st.booleans(), "bool")
    if kind == "mixed":
        return many(ANY_VALUE, "object")
    if kind == "all_null":
        return st.just(pd.Series([None] * n, dtype="object"))
    if kind == "datetime_tz":
        zone = st.sampled_from(ZONES)
        return st.tuples(zone, st.lists(MOMENTS, min_size=n, max_size=n)).map(
            lambda pair: pd.Series(
                pd.to_datetime(pair[1]).tz_localize(pair[0], ambiguous="NaT")
            )
        )
    raise AssertionError(kind)


KINDS = [
    "int",
    "small_int",
    "float",
    "string",
    "bool",
    "mixed",
    "all_null",
    "datetime_tz",
]


def _refused(name: str) -> bool:
    # `dsbf.data.load` refuses these names (tests/test_data/test_errors.py).
    return name == "*" or (name.startswith("^") and name.endswith("$"))


@st.composite
def frames(draw) -> pd.DataFrame:
    n = draw(st.integers(1, 30))
    names = draw(
        st.lists(
            st.one_of(st.sampled_from(ODD_NAMES), st.text(max_size=6)).filter(
                lambda name: not _refused(name)
            ),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    series = {name: draw(column(draw(st.sampled_from(KINDS)), n)) for name in names}
    return pd.DataFrame(series)


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(frame=frames())
def test_every_core_check_survives_a_generated_frame(frame: pd.DataFrame) -> None:
    run = run_core(frame, name="generated")

    assert trouble(run) == {}
    assert set(run.results) == {c.id for c in CHECKS}
    strict_loads(run.model_dump_json())


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(frame=frames())
def test_the_time_series_checks_survive_a_generated_index(frame: pd.DataFrame) -> None:
    index = first_datetime_column(frame)
    run = run_core(frame, name="generated", time_index=index)

    assert trouble(run) == {}
    strict_loads(run.model_dump_json())


@settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(frame=frames())
def test_reference_drift_survives_a_generated_frame_and_reference(
    frame: pd.DataFrame,
) -> None:
    reference = frame.iloc[: max(1, len(frame) // 2)]
    run = run_core(frame, name="generated", reference=reference)

    assert trouble(run) == {}
    assert run.results["reference_drift"].status is CheckStatus.OK
    strict_loads(run.model_dump_json())
