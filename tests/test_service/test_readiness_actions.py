"""The dashboard's action dictionary covers the guidance the engine can emit."""

import json
import re
from pathlib import Path
from typing import get_args

import pytest

from dsbf.checks.categorical_encoding import ENCODE_METHODS
from dsbf.checks.imputation import IMPUTE_METHODS
from dsbf.checks.numeric_binning import Strategy as BinStrategy
from dsbf.checks.transformations import Method as TransformMethod
from dsbf.engine.tips import load_library

DICTIONARY = (
    Path(__file__).parents[2] / "frontend/src/components/readiness/actionMeta.json"
)

pytestmark = pytest.mark.unit


def _normal(method: str) -> str:
    return re.sub(r"[-. ]", "_", method.lower())


def _emitted() -> set[tuple[str, str | None]]:
    """Every (action, method) pair a finding's guidance can carry."""
    pairs: set[tuple[str, str | None]] = {
        (tip.action, _normal(tip.method) if tip.method else None)
        for tip in load_library().values()
    }
    pairs |= {("impute", m) for m in IMPUTE_METHODS}
    pairs |= {("encode", m) for m in ENCODE_METHODS}
    pairs |= {("bin", m) for m in get_args(BinStrategy)}
    pairs |= {("transform", m) for m in get_args(TransformMethod)}
    pairs |= {("drop", None), ("add_indicator", None), ("keep", None)}
    return pairs


def _key(action: str, method: str | None) -> str:
    return f"{action}:{method}" if method else action


def test_every_emitted_pair_has_an_entry_or_is_listed_as_uncovered() -> None:
    data = json.loads(DICTIONARY.read_text(encoding="utf-8"))
    entries, uncovered = data["entries"], set(data["uncovered"])
    missing = {
        _key(action, method)
        for action, method in _emitted()
        if not (
            (f"{action}:{method}" in entries or method in entries)
            if method
            else action in entries
        )
    }
    assert missing == uncovered, (
        "uncovered guidance changed; add dictionary entries or update `uncovered`"
    )
