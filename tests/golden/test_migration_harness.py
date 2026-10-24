"""Unit tests for the migration harness and the ``zeros`` comparator."""

from __future__ import annotations

import json

import pytest

from dsbf.checks import core_pack
from dsbf.legacy import SUPERSEDED_BY
from tests.golden.datasets import GOLDEN_DATASETS
from tests.golden.migration.column_health import (
    compare_constant_columns,
    compare_duplicate_columns,
    compare_null_rate,
)
from tests.golden.migration.findings import (
    LEGACY_FINDING_CODES,
    compare_levels,
    core_levels,
    legacy_levels,
    unmapped_titles,
)
from tests.golden.migration.harness import (
    Accepted,
    AcceptedKind,
    Difference,
    compare,
    judge,
)
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


class TestAcceptedKinds:
    KIND = AcceptedKind("zeros", ("column_read",), "calibrated")

    def test_a_kind_is_accepted_for_every_dataset_and_subject(self):
        found = [
            Difference("a", "column_read", "legacy only"),
            Difference("b", "column_read", "core only"),
        ]

        for dataset in ("titanic", "wide"):
            assert judge("zeros", dataset, found, (), (self.KIND,)).clean

    def test_another_kind_or_check_is_still_unexpected(self):
        other_kind = [Difference("a", "status", "legacy ok, core failed")]

        assert judge("zeros", "titanic", other_kind, (), (self.KIND,)).unexpected == (
            other_kind
        )
        assert judge(
            "nulls", "titanic", [Difference("a", "column_read", "x")], (), (self.KIND,)
        ).unexpected

    def test_a_listed_difference_of_an_accepted_kind_is_not_stale(self):
        listed = Accepted("zeros", "titanic", "a", "column_read", "legacy only", "why")

        assert judge("zeros", "titanic", [], (listed,), (self.KIND,)).clean
        assert not judge("zeros", "titanic", [], (listed,)).clean


class TestTheRegistry:
    def test_every_replaced_task_has_a_comparator(self):
        assert set(SUPERSEDED_BY) == set(COMPARATORS)

    def test_every_accepted_difference_has_a_reason_and_a_known_home(self):
        datasets = {d.name for d in GOLDEN_DATASETS}
        for entry in ACCEPTED:
            assert entry.reason.strip(), entry
            # `<dataset>+reference` is the run against a perturbed copy.
            assert entry.dataset.removesuffix("+reference") in datasets, entry
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

    def test_a_report_only_check_keeps_its_levels_and_flagged_columns(self):
        dimension = {
            "assessed": True,
            "level": "amber",
            "coverage": 0.5,
            "affected_columns": ["y", "x"],
            "findings": [{"id": "f"}],
            "inputs": [{"check_id": "volatile prose"}],
        }
        run = {
            **self.RUN,
            "results": {
                "quality_dimensions": {
                    "check_version": "1.0.0",
                    "status": "ok",
                    "findings": [],
                    "output": {"dimensions": {"validity": dimension}},
                }
            },
        }

        kept = core_snapshot("demo", run)["checks"]["quality_dimensions"]["output"]

        assert kept == {
            "validity": {
                "assessed": True,
                "level": "amber",
                "coverage": 0.5,
                "affected_columns": ["x", "y"],
                "findings": 1,
            }
        }

    def test_every_committed_core_snapshot_covers_every_core_check(self):
        wanted = {check.id for check in core_pack().checks}

        for dataset in GOLDEN_DATASETS:
            snapshot = json.loads(dataset.core_snapshot_path.read_text())
            assert set(snapshot["checks"]) == wanted, dataset.name


class TestTheColumnHealthComparators:
    def legacy(self, data, guidance=None):
        return {"status": "ok", "result": {"data": data, "guidance": guidance or {}}}

    def core(self, output, findings=()) -> dict:
        return {
            "status": "ok",
            "output": output,
            "findings": [
                {"subject": {"columns": columns}, "severity": level}
                for columns, level in findings
            ],
        }

    def test_the_strongest_level_of_a_columns_blurbs_is_its_level(self):
        guidance = {"a": {"eda": [{"level": "info"}], "ml": [{"level": "warn"}]}}

        assert legacy_levels(guidance) == {"a": "warn"}

    def test_a_finding_about_a_group_counts_for_each_column(self):
        findings: list[dict] = [
            {"subject": {"columns": ["a", "b"]}, "severity": "warn"},
            {"subject": {"columns": []}, "severity": "info"},
        ]

        assert core_levels(findings) == {
            "a": "warn",
            "b": "warn",
            "__dataset__": "info",
        }

    def test_levels_are_compared_and_the_dataset_has_an_empty_subject(self):
        legacy = self.legacy({}, {"__dataset__": {"ml": [{"level": "warn"}]}})
        core = self.core({}, [([], "info")])

        assert compare_levels(legacy, core) == [
            Difference("", "severity", "legacy warn, core info")
        ]

    def test_levels_can_be_limited_to_the_columns_a_task_examined(self):
        legacy = self.legacy({}, {})
        core = self.core({}, [(["a"], "warn"), (["b"], "warn")])

        assert compare_levels(legacy, core, columns={"a"}) == [
            Difference("a", "finding", "core only (warn)")
        ]

    def test_equal_constant_columns_have_no_difference(self):
        legacy = self.legacy(
            {"constant_columns": ["x"]}, {"x": {"ml": [{"level": "error"}]}}
        )
        core = self.core({"constant_columns": ["x"]}, [(["x"], "error")])

        assert compare_constant_columns(legacy, core) == []

    def test_duplicate_groups_are_compared_as_pairs(self):
        legacy = self.legacy({"duplicate_column_pairs": [["a", "b"], ["a", "c"]]})
        core = self.core({"groups": [["a", "b", "c"]]})

        assert compare_duplicate_columns(legacy, core) == [
            Difference("b, c", "pair", "core only")
        ]

    def test_a_null_share_that_moved_is_reported(self):
        legacy = self.legacy(
            {
                "null_counts": {"a": 5},
                "null_percentages": {"a": 0.05},
                "high_null_columns": [],
            }
        )
        core = self.core(
            {
                "columns": {
                    "a": {"null_count": 5, "null_share": 0.5, "mostly_missing": True}
                }
            }
        )

        kinds = {d.kind for d in compare_null_rate(legacy, core)}

        assert kinds == {"null_share", "mostly_missing"}

    def test_every_legacy_title_of_the_replaced_tasks_maps_to_a_code(self):
        guidance = {
            "a": {
                "eda": [{"title": "Constant Column - Zero Information"}],
                "ml": [{"title": "Something New"}],
            }
        }

        assert unmapped_titles("detect_constant_columns", guidance) == ["Something New"]

    def test_every_replaced_task_has_title_patterns(self):
        assert set(LEGACY_FINDING_CODES) == set(SUPERSEDED_BY)
