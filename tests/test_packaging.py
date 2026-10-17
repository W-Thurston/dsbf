# tests/test_packaging.py
"""The extras in pyproject.toml stay consistent with the base install."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

EXTRAS = {"ml", "timeseries", "text", "pandas", "sql", "server"}


@pytest.fixture(scope="module")
def project():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["project"]


def test_extras_are_the_documented_set_plus_all(project):
    assert set(project["optional-dependencies"]) == EXTRAS | {"all"}


def test_all_is_the_union_of_the_other_extras(project):
    (requirement,) = (Requirement(r) for r in project["optional-dependencies"]["all"])

    assert requirement.name == project["name"]
    assert requirement.extras == EXTRAS


def test_an_extra_repeating_a_base_package_keeps_its_specifier(project):
    base = {Requirement(r).name: Requirement(r) for r in project["dependencies"]}
    repeated = [
        Requirement(r)
        for extra, requirements in project["optional-dependencies"].items()
        if extra != "all"
        for r in requirements
        if Requirement(r).name in base
    ]

    assert repeated, "no extra repeats a base package; drop this test with the base"
    for requirement in repeated:
        assert requirement.specifier == base[requirement.name].specifier, (
            f"{requirement.name} differs between the base install and an extra"
        )
