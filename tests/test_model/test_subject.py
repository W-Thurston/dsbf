# tests/test_model/test_subject.py
"""A Subject names columns without ever joining them into one string key."""

import pytest
from pydantic import ValidationError

from dsbf.model import Subject

pytestmark = pytest.mark.unit

AWKWARD_NAMES = [
    "a|b",
    "|",
    "[/x]",
    "[a]",
    "{braces}",
    "ünïcödé",
    "日本語",
    "🙂",
    "",
    " ",
    "  padded  ",
    "line\nbreak",
    "quote\"s and 's",
    "back\\slash",
    "null\x00byte",
]


@pytest.mark.parametrize("name", AWKWARD_NAMES)
def test_a_column_subject_accepts_any_name_and_round_trips(name):
    subject = Subject.column(name)

    restored = Subject.model_validate_json(subject.model_dump_json())

    assert restored == subject
    assert restored.columns == (name,)


@pytest.mark.parametrize("first", AWKWARD_NAMES)
def test_a_pair_keeps_both_names_whatever_they_contain(first):
    subject = Subject.column_pair(first, "other")

    restored = Subject.model_validate_json(subject.model_dump_json())

    assert restored.columns == (first, "other")


def test_pairs_whose_encoded_keys_would_collide_stay_distinct():
    left = Subject.column_pair("a|b", "c")
    right = Subject.column_pair("a", "b|c")

    assert left != right
    assert len({left, right}) == 2
    assert left.columns == ("a|b", "c")
    assert right.columns == ("a", "b|c")


def test_the_order_of_a_pair_is_kept():
    assert Subject.column_pair("a", "b") != Subject.column_pair("b", "a")


def test_constructors_set_the_kind():
    assert Subject.dataset() == Subject(kind="dataset")
    assert Subject.column("x").kind == "column"
    assert Subject.column_pair("x", "y").kind == "column_pair"
    assert Subject.column_group("x", "y", "z").columns == ("x", "y", "z")
    assert Subject.row_set("x", "y").kind == "row_set"
    assert Subject.row_set().columns == ()


@pytest.mark.parametrize(
    ("kind", "columns"),
    [
        ("dataset", ("a",)),
        ("column", ()),
        ("column", ("a", "b")),
        ("column_pair", ("a",)),
        ("column_pair", ("a", "b", "c")),
        ("column_group", ("a",)),
        ("column_group", ()),
        ("column_pair", ("a", "a")),
        ("column_group", ("a", "b", "a")),
        ("row_set", ("a", "a")),
    ],
)
def test_the_number_and_distinctness_of_columns_is_checked(kind, columns):
    with pytest.raises(ValidationError):
        Subject(kind=kind, columns=columns)


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValidationError):
        Subject.model_validate({"kind": "table", "columns": ()})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Subject.model_validate({"kind": "dataset", "colums": []})


def test_a_subject_is_immutable_and_hashable():
    subject = Subject.column("x")

    with pytest.raises(ValidationError):
        subject.kind = "dataset"  # type: ignore[misc]
    assert {subject: 1}[Subject.column("x")] == 1


def test_a_json_array_is_read_as_the_column_tuple():
    subject = Subject.model_validate_json(
        '{"kind": "column_pair", "columns": ["a", "b"]}'
    )

    assert subject.columns == ("a", "b")
