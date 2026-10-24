"""The profile-only column health checks on small frames."""

import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks.dominant_value import DominantValueParams
from dsbf.checks.null_rate import NullRateParams
from dsbf.model import CheckStatus, Confidence, Severity
from tests.test_checks.support import (
    actions_of,
    by_column,
    output_of,
    run_core_check,
)

pytestmark = pytest.mark.unit

N = 100


def codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


class TestNullRate:
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "none": list(range(N)),
                "four": [None] * 4 + list(range(N - 4)),
                "five": [None] * 5 + list(range(N - 5)),
                "twenty": [None] * 20 + list(range(N - 20)),
                "half": [None] * 50 + list(range(N - 50)),
                "all": [None] * N,
            }
        )

    def test_severity_rises_with_the_share(self):
        found = by_column(run_core_check(self.frame(), "null_rate"))

        assert set(found) == {"five", "twenty", "half", "all"}
        assert {c: f[0].severity for c, f in found.items()} == {
            "five": Severity.INFO,
            "twenty": Severity.WARN,
            "half": Severity.ERROR,
            "all": Severity.ERROR,
        }

    def test_every_column_is_in_the_output(self):
        output = output_of(run_core_check(self.frame(), "null_rate"))

        assert output["columns"]["four"] == {
            "null_count": 4,
            "rows": N,
            "null_share": 0.04,
            "mostly_missing": False,
        }
        assert output["columns"]["all"]["mostly_missing"] is True
        assert set(output["columns"]) == set(self.frame().columns)

    def test_thresholds_are_parameters_and_must_be_ordered(self):
        result = run_core_check(self.frame(), "null_rate", notice_share=0.04)

        assert "four" in by_column(result)
        with pytest.raises(ValidationError):
            NullRateParams(notice_share=0.5, warn_share=0.2)


class TestConstantColumns:
    def test_a_column_of_one_value_or_no_value_is_reported(self):
        frame = pl.DataFrame(
            {
                "one": [7] * N,
                "one_and_nulls": [7] * 90 + [None] * 10,
                "none": [None] * N,
                "two": [0, 1] * (N // 2),
            }
        )

        result = run_core_check(frame, "constant_columns")

        assert output_of(result)["constant_columns"] == ["one", "one_and_nulls", "none"]
        found = by_column(result)
        assert set(found) == {"one", "one_and_nulls", "none"}
        assert all(f[0].severity is Severity.ERROR for f in found.values())
        assert found["one_and_nulls"][0].evidence.metrics == {
            "n_unique": 1,
            "null_share": 0.1,
        }
        assert "no values" in found["none"][0].message

    def test_a_tip_says_to_drop_it(self):
        result = run_core_check(pl.DataFrame({"one": [7] * N}), "constant_columns")

        (action,) = actions_of(result.findings[0])
        assert action.action == "drop"
        assert action.columns == ("one",)


class TestHighCardinality:
    def frame(self, distinct: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "label": [f"v{i % distinct}" for i in range(2000)],
                "few": [f"w{i % 4}" for i in range(2000)],
            }
        )

    def test_more_distinct_values_than_the_limit_is_reported(self):
        result = run_core_check(self.frame(51), "high_cardinality")

        assert set(by_column(result)) == {"label"}
        finding = result.findings[0]
        assert finding.evidence.metrics["n_unique"] == 51
        assert finding.severity is Severity.WARN
        assert finding.confidence is Confidence.HIGH

    def test_the_limit_itself_is_not_reported(self):
        assert not run_core_check(self.frame(50), "high_cardinality").findings

    def test_the_limit_is_a_parameter(self):
        result = run_core_check(self.frame(50), "high_cardinality", max_distinct=10)

        assert set(by_column(result)) == {"label"}

    def test_tips_suggest_encodings(self):
        finding = run_core_check(self.frame(60), "high_cardinality").findings[0]

        assert [(a.action, a.method) for a in actions_of(finding)] == [
            ("encode", "frequency_encoding"),
            ("encode", "target_encoding"),
            ("group", "bucket_rare_values"),
        ]


class TestIdColumns:
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "row": list(range(N)),
                "shifted": list(range(1000, 1000 + N)),
                "shuffled": [(i * 37) % N for i in range(N)],
                "gap": [*range(N - 1), N + 5],
                "repeat": [*range(N - 1), 0],
                "floats": [i + 0.5 for i in range(N)],
                "code": [f"c{i}" for i in range(N)],
                "value": [i % 7 for i in range(N)],
            }
        )

    def test_a_dense_run_of_whole_numbers_is_a_row_number(self):
        result = run_core_check(self.frame(), "id_columns")

        flagged = output_of(result)["id_columns"]
        assert flagged["row"] == "dense_integer_sequence"
        assert flagged["shifted"] == "dense_integer_sequence"
        assert flagged["shuffled"] == "dense_integer_sequence"
        assert not {"gap", "repeat", "floats", "value"} & set(flagged)

    def test_unique_strings_are_identifiers_by_their_share(self):
        result = run_core_check(self.frame(), "id_columns")

        assert output_of(result)["id_columns"]["code"] == "unique_share"
        finding = by_column(result)["code"][0]
        assert finding.severity is Severity.WARN
        assert finding.evidence.method == "unique_share"

    def test_too_few_rows_say_nothing(self):
        frame = pl.DataFrame({"row": list(range(10))})

        assert not run_core_check(frame, "id_columns").findings

    def test_the_row_floor_is_a_parameter(self):
        frame = pl.DataFrame({"row": list(range(10))})

        result = run_core_check(frame, "id_columns", min_rows=5)

        assert set(by_column(result)) == {"row"}


class TestDominantValue:
    def frame(self) -> pl.DataFrame:
        rows = 200
        return pl.DataFrame(
            {
                "even": ["a", "b"] * (rows // 2),
                "seventy": ["a"] * 140 + ["b"] * 30 + ["c"] * 30,
                "ninety": ["a"] * 180 + ["b"] * 10 + ["c"] * 10,
                "flag": [True] * 196 + [False] * 4,
                "with_nulls": ["a"] * 100 + ["b"] * 20 + [None] * 80,
                "constant": ["a"] * rows,
            }
        )

    def test_severity_follows_the_share(self):
        found = by_column(run_core_check(self.frame(), "dominant_value"))

        assert {c: f[0].severity for c, f in found.items()} == {
            "seventy": Severity.INFO,
            "ninety": Severity.WARN,
            "flag": Severity.WARN,
            "with_nulls": Severity.INFO,
        }

    def test_the_share_is_of_the_non_null_values(self):
        output = output_of(run_core_check(self.frame(), "dominant_value"))

        assert output["columns"]["with_nulls"] == {
            "top_value": "a",
            "top_count": 100,
            "non_null": 120,
            "top_share": 100 / 120,
            "distinct": 2,
            "flagged": False,
        }
        assert output["columns"]["flag"]["top_value"] == "True"
        assert output["columns"]["ninety"]["flagged"] is False

    def test_a_constant_column_is_the_constant_checks(self):
        result = run_core_check(self.frame(), "dominant_value")

        assert "constant" not in by_column(result)
        assert output_of(result)["columns"]["constant"]["flagged"] is True

    def test_shares_must_be_ordered(self):
        with pytest.raises(ValidationError):
            DominantValueParams(notice_share=0.9, warn_share=0.8)

    def test_a_column_with_no_listed_top_value_is_skipped_not_read_as_zero(self):
        long = ["x" * 300 + str(i % 3) for i in range(60)]
        frame = pl.DataFrame({"long": long, "other": ["a", "b"] * 30})

        result = run_core_check(frame, "dominant_value")

        assert "long" not in (result.output or {}).get("columns", {})

    def test_a_mode_too_long_to_list_is_still_the_mode(self):
        # A string over 256 characters is never in `top_values`, but it can be the
        # most frequent value; the profile counts it apart.
        frame = pl.DataFrame({"c": ["x" * 300] * 8 + [f"s{i % 40}" for i in range(92)]})

        result = run_core_check(frame, "dominant_value")

        column = output_of(result)["columns"]["c"]
        assert column["top_count"] == 8
        assert column["top_value"] == "(a value of over 256 characters)"
        assert column["top_share"] == pytest.approx(0.08)


class TestNearZeroVariance:
    def frame(self) -> pl.DataFrame:
        values = [i / 1000 for i in range(N)]
        return pl.DataFrame(
            {
                "tiny": [1.0 + v * 0.01 for v in values],
                "small": [v * 0.01 + 0.5 for v in values],
                "wide": [float(i) for i in range(N)],
            }
        )

    def test_a_column_at_or_below_the_variance_is_reported(self):
        result = run_core_check(self.frame(), "near_zero_variance")

        found = by_column(result)
        assert set(found) == {"tiny", "small"}
        finding = found["tiny"][0]
        assert finding.code == "near_zero_variance"
        assert finding.severity is Severity.WARN
        assert finding.evidence.metrics["variance"] < 1e-4
        assert finding.evidence.metrics["relative_spread"] is not None

    def test_the_bound_is_a_parameter(self):
        result = run_core_check(self.frame(), "near_zero_variance", max_variance=0.0)

        assert not result.findings
        assert output_of(result)["columns"]["wide"]["low"] is False

    def test_a_tip_says_to_drop_it(self):
        finding = run_core_check(self.frame(), "near_zero_variance").findings[0]

        assert [a.action for a in actions_of(finding)] == ["drop"]


class TestDuplicateRows:
    def test_repeated_rows_are_counted_over_every_column(self):
        frame = pl.DataFrame(
            {"a": [1, 2, 3, 1, 2] * 4, "b": ["x", "y", "z", "x", "y"] * 4}
        )

        result = run_core_check(frame, "duplicate_rows")

        assert result.output == {
            "duplicate_rows": 17,
            "rows": 20,
            "duplicate_share": 0.85,
        }
        (finding,) = result.findings
        assert finding.severity is Severity.WARN
        assert finding.subject.kind == "dataset"

    def test_a_share_at_or_below_the_bound_is_only_information(self):
        frame = pl.DataFrame({"a": [*range(199), 0]})

        (finding,) = run_core_check(frame, "duplicate_rows", warn_share=0.01).findings

        assert finding.severity is Severity.INFO

    def test_no_repeats_no_finding(self):
        result = run_core_check(pl.DataFrame({"a": list(range(30))}), "duplicate_rows")

        assert result.status is CheckStatus.OK
        assert not result.findings
        assert output_of(result)["duplicate_rows"] == 0


class TestDuplicateColumns:
    def frame(self) -> pl.DataFrame:
        base = [i % 17 for i in range(N)]
        return pl.DataFrame(
            {
                "a": base,
                "b": base,
                "c": base,
                "as_float": [float(v) for v in base],
                "reversed": base[::-1],
                "other": [i % 11 for i in range(N)],
                "n1": [None if i % 5 == 0 else i for i in range(N)],
                "n2": [None if i % 5 == 0 else i for i in range(N)],
            }
        )

    def test_equal_columns_are_one_group_each(self):
        result = run_core_check(self.frame(), "duplicate_columns")

        assert result.output == {"groups": [["a", "b", "c"], ["n1", "n2"]], "pairs": 4}
        first, second = result.findings
        assert first.subject.columns == ("a", "b", "c")
        assert first.evidence.metrics == {"n_columns": 3, "n_pairs": 3}
        assert first.evidence.method == "content_hash"
        assert second.subject.columns == ("n1", "n2")

    def test_the_same_values_in_another_type_or_order_are_not_equal(self):
        result = run_core_check(self.frame(), "duplicate_columns")

        grouped = {name for group in output_of(result)["groups"] for name in group}
        assert not {"as_float", "reversed", "other"} & grouped

    def test_a_tip_says_to_drop_one(self):
        finding = run_core_check(self.frame(), "duplicate_columns").findings[0]

        (action,) = actions_of(finding)
        assert action.action == "drop"
        assert action.columns == ("a", "b", "c")

    def test_no_duplicates_no_finding(self):
        frame = pl.DataFrame({"a": list(range(30)), "b": list(range(1, 31))})

        result = run_core_check(frame, "duplicate_columns")

        assert result.output == {"groups": [], "pairs": 0}
        assert not result.findings


class TestTextEncoding:
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "clean": ["ok"] * N,
                "replaced": ["caf�"] * 40 + ["ok"] * 60,
                "few_replaced": ["x�"] + ["ok"] * (N - 1),
                "nuls": ["a\x00b"] * 10 + ["ok"] * 90,
                "bad�name": ["ok"] * N,
                "numbers": list(range(N)),
            }
        )

    def test_each_kind_of_damage_has_its_own_code(self):
        found = by_column(run_core_check(self.frame(), "text_encoding"))

        assert set(found) == {"replaced", "few_replaced", "nuls", "bad�name"}
        assert [f.code for f in found["replaced"]] == ["replacement_characters"]
        assert [f.code for f in found["nuls"]] == ["nul_characters"]
        assert [f.code for f in found["bad�name"]] == ["damaged_column_name"]

    def test_severity_follows_the_share_of_values(self):
        found = by_column(run_core_check(self.frame(), "text_encoding"))

        assert found["replaced"][0].severity is Severity.WARN
        assert found["few_replaced"][0].severity is Severity.INFO
        assert found["bad�name"][0].severity is Severity.WARN

    def test_the_output_lists_only_damaged_columns(self):
        output = output_of(run_core_check(self.frame(), "text_encoding"))

        assert output["columns"]["nuls"] == {
            "name_damaged": False,
            "values_with_replacement_char": 0,
            "values_with_nul": 10,
        }
        assert "clean" not in output["columns"]

    def test_a_clean_frame_has_no_findings(self):
        result = run_core_check(pl.DataFrame({"a": ["x", "y"] * 10}), "text_encoding")

        assert result.status is CheckStatus.OK
        assert not result.findings


class TestTipsOnTheFindings:
    """The advice the legacy tasks gave comes back as actions on the findings."""

    def actions(self, result, column):
        (finding,) = by_column(result)[column]
        return [(a.action, a.method) for a in actions_of(finding)]

    def test_missing_values_get_advice_by_how_much_is_missing(self):
        frame = pl.DataFrame(
            {
                "some": [None] * 10 + list(range(90)),
                "many": [None] * 30 + list(range(70)),
                "most": [None] * 60 + list(range(40)),
            }
        )

        result = run_core_check(frame, "null_rate")

        assert self.actions(result, "some") == [("impute", "median")]
        assert self.actions(result, "many") == [
            ("add_indicator", "is_missing"),
            ("impute", "median"),
            ("impute", "model_based"),
        ]
        assert self.actions(result, "most") == [
            ("drop", None),
            ("add_indicator", "is_missing"),
        ]

    def test_an_identifier_is_dropped_from_the_features(self):
        result = run_core_check(pl.DataFrame({"row": list(range(N))}), "id_columns")

        assert self.actions(result, "row") == [("drop", None)]

    def test_a_dominant_value_gets_advice_by_its_share(self):
        frame = pl.DataFrame(
            {"mild": ["a"] * 75 + ["b"] * 25, "severe": ["a"] * 95 + ["b"] * 5}
        )

        result = run_core_check(frame, "dominant_value")

        assert self.actions(result, "mild") == [
            ("stratified_split", None),
            ("class_weight", None),
        ]
        assert self.actions(result, "severe") == [
            ("stratified_split", None),
            ("class_weight", None),
            ("drop", None),
        ]
