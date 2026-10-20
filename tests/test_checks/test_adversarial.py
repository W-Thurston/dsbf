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
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import CheckStatus, Run
from dsbf.profiling import profile_dataset
from tests.fixtures.adversarial import build_adversarial_frame
from tests.golden.datasets import DATA_DIR
from tests.test_model.builders import strict_loads

pytestmark = pytest.mark.integration

CHECKS = core_pack().checks
TITANIC = DATA_DIR / "titanic.csv"
ACCEPTABLE = {CheckStatus.OK, CheckStatus.NOT_APPLICABLE}


def run_core(frame: pd.DataFrame | pl.DataFrame, name: str = "adversarial") -> Run:
    dataset = load(frame, name=name)
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(CHECKS),
        Selection(depth="deep", entry_points=False),
        profile=profile,
    )
    return execute(plan, dataset, profile, dsbf_version="test", workers=2)


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


def test_an_integer_coded_boolean_column_never_appears_in_zeros(
    adversarial_run: Run,
) -> None:
    output = adversarial_run.results["zeros"].output
    assert output is not None

    # Half of each is zero, so the legacy task flagged them; the profile types them
    # categorical, so the check never reads them.
    assert {"flag_as_int", "flag_copy", "[/x]"}.isdisjoint(output["columns"])
    assert {f.subject.columns[0] for f in adversarial_run.findings} == set()
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


@st.composite
def frames(draw) -> pd.DataFrame:
    n = draw(st.integers(1, 30))
    names = draw(
        st.lists(
            st.one_of(st.sampled_from(ODD_NAMES), st.text(max_size=6)),
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
