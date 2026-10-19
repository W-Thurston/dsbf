"""Adapted tasks give the results they gave when the legacy engine ran them."""

import warnings

import pytest

from dsbf.model import CheckStatus
from tests.test_legacy.support import (
    comparable,
    make_session,
    output_of,
    run_adapted,
    run_legacy,
)

pytestmark = pytest.mark.integration

# A rule-like task, a pandas-heavy one, a plotting one, a synthesis scorer and a
# time-series one.
REPRESENTATIVE = (
    "detect_constant_columns",
    "detect_outliers",
    "generate_univariate_plots",
    "data_quality_scorer",
    "detect_stationarity",
)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    warnings.simplefilter("ignore")
    # One directory for both, so the plot paths a result names are the same.
    session = make_session(tmp_path_factory.mktemp("figures"))
    return run_adapted(session, *REPRESENTATIVE), run_legacy(session, *REPRESENTATIVE)


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_the_task_ends_ok(runs, name):
    run, _ = runs

    assert run.results[name].status is CheckStatus.OK, run.results[name].notes


@pytest.mark.parametrize("name", REPRESENTATIVE)
def test_the_task_result_is_unchanged(runs, name):
    run, legacy = runs

    adapted = output_of(run, name)["result"]

    assert adapted == comparable(legacy[name])
