"""The optional-dependency groups a check can need, and whether they are installed."""

from typing import get_args

import pytest

from dsbf.engine import Extra
from dsbf.engine import extras as extras_module
from dsbf.engine.extras import EXTRA_MODULES, install_hint, missing_extras

pytestmark = pytest.mark.unit


def test_every_extra_names_the_modules_that_show_it_is_installed():
    assert set(EXTRA_MODULES) == set(get_args(Extra))
    assert all(EXTRA_MODULES.values())


@pytest.fixture
def without(monkeypatch):
    """Make the named modules look uninstalled and every other one installed."""

    def hide(*modules: str) -> None:
        monkeypatch.setattr(
            extras_module,
            "find_spec",
            lambda m: None if m in modules else object(),
        )

    return hide


def test_an_installed_extra_is_not_missing(without):
    without()

    assert missing_extras(["pandas"]) == {}


def test_a_missing_module_names_its_extra(without):
    without("statsmodels")

    assert missing_extras(["ml", "timeseries"]) == {"timeseries": ("statsmodels",)}


def test_an_extra_is_missing_when_any_of_its_modules_is(without):
    without("uvicorn", "alembic")

    assert missing_extras(["server"]) == {"server": ("uvicorn", "alembic")}


def test_the_hint_lists_each_extra_once():
    assert install_hint(["ml", "text", "ml"]) == (
        "Install it with `pip install 'dsbf[ml,text]'`."
    )
