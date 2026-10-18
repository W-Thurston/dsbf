"""The registry holds checks by id; nothing registers itself."""

import pytest

from dsbf.engine import (
    Check,
    CheckDefinitionError,
    Pack,
    Registry,
    RegistryError,
    SemanticTypes,
    check,
)
from dsbf.model import Dimension

pytestmark = pytest.mark.unit


def make_check(check_id: str, packs: tuple[str, ...] = ()) -> Check:
    @check(
        id=check_id,
        version="1.0.0",
        kind="rule",
        applies_to=SemanticTypes("numeric"),
        dimensions=[Dimension.VALIDITY],
        packs=packs,
    )
    def declared(ctx, params):
        """A check for registry tests."""

    return declared


def test_declaring_a_check_registers_nothing():
    make_check("null_rate")

    assert len(Registry()) == 0


def test_registered_checks_are_found_by_id_and_listed_in_id_order():
    registry = Registry([make_check("zeros"), make_check("null_rate")])

    assert [c.id for c in registry] == ["null_rate", "zeros"]
    assert registry.get("zeros").id == "zeros"
    assert "null_rate" in registry
    assert "other" not in registry
    assert len(registry) == 2


def test_registering_the_same_check_again_is_a_no_op():
    first = make_check("null_rate")
    registry = Registry([first])

    registry.register(first, first)

    assert registry.checks == (first,)


def test_a_different_check_under_a_taken_id_is_an_error_and_adds_nothing():
    registry = Registry([make_check("null_rate")])

    with pytest.raises(RegistryError, match="'null_rate' is already registered"):
        registry.register(make_check("zeros"), make_check("null_rate"))

    assert [c.id for c in registry] == ["null_rate"]


def test_two_different_checks_with_one_id_in_one_call_are_an_error():
    with pytest.raises(RegistryError, match="'zeros'"):
        Registry().register(make_check("zeros"), make_check("zeros"))


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ("null_rat", "did you mean 'null_rate'"),
        ("nonsense_check", "unknown check 'nonsense_check'"),
    ],
)
def test_an_unknown_id_names_the_id_and_suggests_a_close_one(asked, expected):
    registry = Registry([make_check("null_rate")])

    with pytest.raises(RegistryError, match=expected):
        registry.get(asked)


def test_a_directly_registered_check_reports_that_source():
    registry = Registry([make_check("null_rate")])

    assert registry.source_of("null_rate") == "direct"


def test_packs_merge_a_checks_own_pack_names_with_pack_objects():
    declared = make_check("null_rate", packs=("core",))
    plain = make_check("zeros")
    registry = Registry([declared])

    registry.register_pack(Pack("core", [plain]))
    registry.register_pack(Pack("healthcare", [plain, make_check("icd_codes")]))

    assert registry.packs == {
        "core": ("null_rate", "zeros"),
        "healthcare": ("icd_codes", "zeros"),
    }
    assert "icd_codes" in registry


def test_a_pack_with_a_conflicting_check_registers_nothing():
    registry = Registry([make_check("zeros")])

    with pytest.raises(RegistryError):
        registry.register_pack(Pack("extras", [make_check("zeros")]))

    assert registry.packs == {}


@pytest.mark.parametrize(
    ("name", "members", "message"),
    [
        ("Bad Name", [], "lower snake case"),
        ("ok", ["null_rate"], "not a Check"),
    ],
)
def test_a_malformed_pack_is_refused_when_declared(name, members, message):
    with pytest.raises(CheckDefinitionError, match=message):
        Pack(name, members)
