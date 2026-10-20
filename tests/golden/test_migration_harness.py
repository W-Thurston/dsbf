"""Unit tests for the migration harness and the ``zeros`` comparator."""

from __future__ import annotations

import pytest

from dsbf.legacy import SUPERSEDED_BY
from tests.golden.datasets import GOLDEN_DATASETS
from tests.golden.migration.harness import Accepted, Difference, compare, judge
from tests.golden.migration.registry import ACCEPTED, COMPARATORS
from tests.golden.migration.snapshot import core_snapshot
from tests.golden.migration.zeros import compare_zeros
from tests.golden.normalise import render_snapshot

pytestmark = pytest.mark.unit


def legacy_zeros(counts, percentages=None, flags=None, guidance=None, status="ok"):
    percentages = percentages or {c: n / 100 for c, n in counts.items()}
    return {
        "status": status,
        "result": {
            "data": {
                "zero_counts": counts,
                "zero_percentages": percentages,
                "zero_flags": flags or {c: False for c in counts},
            },
            "guidance": guidance or {},
        },
    }


def core_zeros(columns, severities=None, status="ok"):
    return {
        "status": status,
        "output": {
            "columns": {
                name: {
                    "zero_count": count,
                    "non_null": 100,
                    "zero_share": count / 100,
                    "flagged": False,
                }
                for name, count in columns.items()
            }
        },
        "findings": [
            {"subject": {"columns": [name]}, "severity": level}
            for name, level in (severities or {}).items()
        ],
    }


def blurb(level):
    return {"eda": [{"level": level}], "ml": [{"level": level}]}


class TestTheZerosComparator:
    def test_equal_results_have_no_difference(self):
        legacy = legacy_zeros({"a": 40}, guidance={"a": blurb("info")})
        core = core_zeros({"a": 40}, {"a": "info"})

        assert compare("zeros", compare_zeros, legacy, core) == []

    def test_a_column_read_by_one_side_only_is_listed(self):
        found = compare_zeros(
            legacy_zeros({"a": 0, "b": 0}), core_zeros({"a": 0, "c": 0})
        )

        assert found == [
            Difference("b", "column_read", "legacy only"),
            Difference("c", "column_read", "core only"),
        ]

    def test_a_count_a_share_and_a_severity_are_compared(self):
        legacy = legacy_zeros({"a": 40}, {"a": 0.2}, guidance={"a": blurb("info")})
        core = core_zeros({"a": 41}, {"a": "warn"})

        assert compare_zeros(legacy, core) == [
            Difference("a", "zero_count", "legacy 40, core 41"),
            Difference("a", "zero_share", "legacy 0.2000, core 0.4100"),
            Difference("a", "severity", "legacy info, core warn"),
        ]

    def test_a_finding_one_side_lacks_is_listed(self):
        legacy = legacy_zeros({"a": 0, "b": 40}, guidance={"b": blurb("info")})
        core = core_zeros({"a": 0, "b": 40}, {"a": "warn"})

        assert compare_zeros(legacy, core) == [
            Difference("a", "finding", "core only (warn)"),
            Difference("b", "finding", "legacy only (info)"),
        ]

    def test_a_different_status_stops_the_comparison(self):
        legacy = legacy_zeros({"a": 0}, status="ok")
        core = core_zeros({}, status="not_applicable")

        assert compare("zeros", compare_zeros, legacy, core) == [
            Difference("", "status", "legacy ok, core not_applicable")
        ]

    def test_a_legacy_task_the_engine_did_not_run_is_a_difference(self):
        found = compare("zeros", compare_zeros, None, core_zeros({"a": 0}))

        assert [d.kind for d in found] == ["legacy_not_run"]


class TestTheVerdict:
    ACCEPT = Accepted("zeros", "titanic", "a", "column_read", "legacy only", "why")

    def test_an_accepted_difference_is_clean(self):
        found = [Difference("a", "column_read", "legacy only")]

        assert judge("zeros", "titanic", found, (self.ACCEPT,)).clean

    def test_a_difference_nobody_accepted_is_unexpected(self):
        found = [Difference("b", "column_read", "legacy only")]

        verdict = judge("zeros", "titanic", found, (self.ACCEPT,))

        assert verdict.unexpected == found
        assert verdict.stale == [self.ACCEPT]

    def test_the_detail_is_part_of_what_is_accepted(self):
        found = [Difference("a", "column_read", "core only")]

        verdict = judge("zeros", "titanic", found, (self.ACCEPT,))

        assert not verdict.clean

    def test_an_entry_for_another_dataset_or_check_does_not_apply(self):
        verdict = judge("zeros", "wide", [], (self.ACCEPT,))

        assert verdict.clean


class TestTheRegistry:
    def test_every_replaced_task_has_a_comparator(self):
        assert set(SUPERSEDED_BY.values()) <= set(COMPARATORS)

    def test_every_accepted_difference_has_a_reason_and_a_known_home(self):
        datasets = {d.name for d in GOLDEN_DATASETS}
        for entry in ACCEPTED:
            assert entry.reason.strip(), entry
            assert entry.dataset in datasets, entry
            assert entry.check in COMPARATORS, entry

    def test_an_entry_is_listed_once(self):
        keys = [(a.check, a.dataset, a.difference) for a in ACCEPTED]

        assert len(keys) == len(set(keys))


class TestTheCoreSnapshot:
    RUN = {
        "run_id": "volatile",
        "dataset": {"n_rows": 10, "n_columns": 2},
        "results": {
            "zeros": {
                "check_version": "1.0.0",
                "status": "ok",
                "timing": {"wall_seconds": 0.123},
                "findings": [
                    {
                        "id": "b",
                        "check_id": "zeros",
                        "code": "high_zero_rate",
                        "subject": {"kind": "column", "columns": ["y"]},
                        "severity": "warn",
                        "message": "prose that may change",
                        "evidence": {"metrics": {"zero_share": 0.123456789}},
                    },
                    {
                        "id": "a",
                        "check_id": "zeros",
                        "code": "high_zero_rate",
                        "subject": {"kind": "column", "columns": ["x"]},
                        "severity": "info",
                        "message": "prose",
                        "evidence": {"metrics": {"zero_share": 0.5}},
                    },
                ],
            }
        },
    }

    def test_a_finding_keeps_what_a_reviewer_judges_it_on(self):
        snapshot = core_snapshot("demo", self.RUN)

        assert snapshot["shape"] == [10, 2]
        findings = snapshot["checks"]["zeros"]["findings"]
        assert [f["subject"]["columns"] for f in findings] == [["x"], ["y"]]
        assert findings[1] == {
            "id": "b",
            "check": "zeros",
            "code": "high_zero_rate",
            "subject": {"kind": "column", "columns": ["y"]},
            "severity": "warn",
            "metrics": {"zero_share": 0.123457},
        }

    def test_volatile_fields_and_prose_are_dropped(self):
        text = render_snapshot(core_snapshot("demo", self.RUN))

        for dropped in ("volatile", "prose", "wall_seconds", "timing"):
            assert dropped not in text

    def test_two_renderings_are_identical(self):
        assert render_snapshot(core_snapshot("demo", self.RUN)) == render_snapshot(
            core_snapshot("demo", self.RUN)
        )
