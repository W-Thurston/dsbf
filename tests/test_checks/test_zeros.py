"""The ``zeros`` check on small frames."""

import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks import core_pack
from dsbf.checks.zeros import ZerosParams, zeros
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import CheckStatus, Run, Severity
from dsbf.profiling import profile_dataset

pytestmark = pytest.mark.unit


def run_zeros(frame: pl.DataFrame, **params) -> Run:
    dataset = load(frame, name="zeros")
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include="zeros"),
        params={"zeros": params} if params else None,
        profile=profile,
    )
    return execute(plan, dataset, profile, dsbf_version="test", workers=1)


def columns_of(run: Run) -> dict[str, dict]:
    output = run.results["zeros"].output
    assert output is not None
    return output["columns"]


def frame() -> pl.DataFrame:
    """Numeric columns of 20 rows with enough distinct values to profile numeric."""
    return pl.DataFrame(
        {
            "mostly": [0.0] * 17 + [1.5, 2.5, 5.5],
            "half": [0.0] * 10 + [float(i) + 0.5 for i in range(10)],
            "some": [0] * 7 + list(range(1, 14)),
            "none": list(range(1, 21)),
            "with_nulls": [0] * 5 + [None] * 10 + list(range(1, 6)),
            "word": ["a", "b"] * 10,
        }
    )


def test_a_column_at_or_above_the_notice_share_is_reported_with_a_severity() -> None:
    result = run_zeros(frame()).results["zeros"]

    by_column = {f.subject.columns[0]: f for f in result.findings}
    assert set(by_column) == {"mostly", "half", "some", "with_nulls"}
    assert by_column["mostly"].severity is Severity.WARN
    assert by_column["half"].severity is Severity.WARN
    assert by_column["some"].severity is Severity.INFO
    assert {f.code for f in result.findings} == {"high_zero_rate"}


def test_the_share_is_of_the_non_null_values() -> None:
    run = run_zeros(frame())
    result = run.results["zeros"]

    assert result.status is CheckStatus.OK
    nulls = columns_of(run)["with_nulls"]
    assert nulls == {
        "zero_count": 5,
        "non_null": 10,
        "zero_share": 0.5,
        "flagged": False,
    }
    finding = next(f for f in result.findings if f.subject.columns == ("with_nulls",))
    assert finding.evidence.n_used == 10
    assert finding.evidence.n_total == 20
    assert finding.evidence.method == "zero_share"
    assert finding.evidence.metrics == {"zero_count": 5, "zero_share": 0.5}


def test_a_column_above_the_flag_share_is_flagged_in_the_output() -> None:
    columns = columns_of(run_zeros(frame(), flag_share=0.8))

    assert {name for name, c in columns.items() if c["flagged"]} == {"mostly"}
    assert columns["none"]["zero_count"] == 0
    default = columns_of(run_zeros(frame()))
    assert not any(c["flagged"] for c in default.values())


def test_only_numeric_columns_are_read() -> None:
    data = frame().with_columns(flag=pl.Series([0, 1] * 10))

    columns = columns_of(run_zeros(data))

    assert "word" not in columns
    assert "flag" not in columns, "an integer-coded boolean is categorical"


def test_thresholds_are_parameters_and_move_the_findings() -> None:
    result = run_zeros(frame(), notice_share=0.8, warn_share=0.9).results["zeros"]

    assert [f.subject.columns for f in result.findings] == [("mostly",)]
    assert result.findings[0].severity is Severity.INFO
    assert result.params == {
        "notice_share": 0.8,
        "warn_share": 0.9,
        "flag_share": 0.95,
    }


def test_a_column_with_no_value_is_not_read() -> None:
    data = pl.DataFrame(
        {"x": [0, 0, 1, 2], "empty": pl.Series([None] * 4, dtype=pl.Int64)}
    )

    columns = columns_of(run_zeros(data))

    assert set(columns) == {"x"}


def test_a_frame_with_no_numeric_column_is_not_applicable() -> None:
    result = run_zeros(pl.DataFrame({"a": ["x", "y"]})).results["zeros"]

    assert result.status is CheckStatus.NOT_APPLICABLE
    assert result.findings == ()


@pytest.mark.parametrize(
    "params",
    [
        {"notice_share": -0.1},
        {"warn_share": 1.5},
        {"notice_share": 0.6, "warn_share": 0.4},
        {"threshold": 0.5},
    ],
)
def test_parameters_outside_their_bounds_are_rejected(params) -> None:
    with pytest.raises(ValidationError):
        zeros.resolve_params(params)


def test_a_sampled_run_carries_the_sampling_on_each_finding() -> None:
    dataset = load(pl.DataFrame({"x": [0] * 80 + list(range(1, 21))})).sample(
        50, strategy="random", seed=3
    )
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(core_pack().checks), Selection(include="zeros"), profile=profile
    )

    result = execute(plan, dataset, profile, dsbf_version="test").results["zeros"]

    assert result.findings
    assert all(f.evidence.sampled is not None for f in result.findings)
    assert ZerosParams().notice_share == 0.30
