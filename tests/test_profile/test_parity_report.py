"""``docs/parity/profile.md`` explains every accepted difference and no other."""

import re
from pathlib import Path

import pytest

from tests.test_profile.accepted import ACCEPTED_DIFFERENCES, BY_SLUG

pytestmark = pytest.mark.unit

REPORT = Path(__file__).parents[2] / "docs" / "parity" / "profile.md"
SLUG_HEADING = re.compile(r"^#{3,4} `([a-z0-9-]+)`\s*$", re.MULTILINE)
AREA_HEADING = re.compile(r"^## .*\(`(\w+)`\)\s*$", re.MULTILINE)


def _documented() -> list[str]:
    return SLUG_HEADING.findall(REPORT.read_text())


def test_every_accepted_difference_is_documented_once():
    documented = _documented()

    assert sorted(documented) == sorted(a.slug for a in ACCEPTED_DIFFERENCES)


def test_each_difference_is_under_its_area():
    sections = re.split(r"^## ", REPORT.read_text(), flags=re.MULTILINE)[1:]
    placed: dict[str, str] = {}
    for section in sections:
        area = AREA_HEADING.match("## " + section.splitlines()[0])
        slugs = SLUG_HEADING.findall(section)
        # A section with no differences, such as "Not compared", needs no area.
        assert area or not slugs, f"no area in: {section.splitlines()[0]!r}"
        for slug in slugs:
            assert area
            placed[slug] = area.group(1)

    assert placed == {a.slug: a.area for a in ACCEPTED_DIFFERENCES}


def test_every_slug_is_listed_with_a_reason():
    for accepted in ACCEPTED_DIFFERENCES:
        assert accepted.reason.strip() and accepted.legacy and accepted.new
    assert set(BY_SLUG) == {a.slug for a in ACCEPTED_DIFFERENCES}
