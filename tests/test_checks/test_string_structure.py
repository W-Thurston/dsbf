"""The string structure checks on small frames: anomalies, encodings, mixes, formats."""

import base64
import hashlib
import json
import string
import uuid

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks.encoded_columns import EncodedColumnsParams
from dsbf.checks.format_violations import FormatViolationsParams
from dsbf.checks.mixed_types import MixedTypesParams
from dsbf.checks.string_anomalies import StringAnomaliesParams
from dsbf.model import CheckStatus, Confidence, Severity
from tests.test_checks.support import actions_of, by_column, output_of, run_core_check

pytestmark = pytest.mark.unit


def generator(seed: int = 5) -> np.random.Generator:
    return np.random.default_rng(seed)


def codes(result) -> dict[str, set[str]]:
    return {c: {f.code for f in found} for c, found in by_column(result).items()}


class TestStringAnomalies:
    def test_values_that_differ_only_by_case_are_grouped_with_the_minority_rows(self):
        frame = pl.DataFrame(
            {
                "city": ["New York"] * 40
                + ["new york"] * 5
                + ["NEW YORK"] * 2
                + ["Paris"] * 30
            }
        )

        result = run_core_check(frame, "string_anomalies")

        case = output_of(result)["columns"]["city"]["mixed_case"]
        assert case["groups"] == 1 and case["values"] == 3 and case["rows"] == 47
        assert case["minority_rows"] == 7
        assert case["examples"][0]["variants"] == ["New York", "new york", "NEW YORK"]
        finding = by_column(result)["city"][0]
        assert (finding.code, finding.severity) == ("mixed_case", Severity.WARN)
        assert finding.confidence is Confidence.HIGH
        assert {a.action for a in actions_of(finding)} == {"transform"}

    def test_padding_and_hidden_characters_are_counted_in_rows(self):
        values = ["ok"] * 50 + [" padded", "padded ", "\tpadded"] + ["zero​width"] * 4
        values += ["no break", "bell\x07"]
        result = run_core_check(pl.DataFrame({"tag": values}), "string_anomalies")

        column = output_of(result)["columns"]["tag"]
        assert column["edge_whitespace"]["rows"] == 3
        assert column["invisible_characters"]["rows"] == 6
        assert column["invisible_characters"]["values"] == 3
        # Hidden characters are shown escaped so the example can be read.
        assert "zero\\u200bwidth" in column["invisible_characters"]["examples"]
        assert codes(result)["tag"] == {"edge_whitespace", "invisible_characters"}
        assert column["n_used"] == 59

    def test_nul_characters_are_left_to_the_text_encoding_check(self):
        frame = pl.DataFrame({"tag": ["fine"] * 30 + ["a\x00b"] * 3})

        assert run_core_check(frame, "string_anomalies").findings == ()

    def test_a_length_far_outside_the_range_is_an_outlier_in_a_label_column(self):
        g = generator()
        short = [
            "".join(g.choice(list("abcdef"), size=int(g.integers(4, 9))))
            for _ in range(200)
        ]
        long = "x" * 400
        result = run_core_check(
            pl.DataFrame({"code": short + [long]}), "string_anomalies"
        )

        outliers = output_of(result)["columns"]["code"]["length_outliers"]
        assert outliers["rows"] == 1 and outliers["upper_fence"] < 400
        assert outliers["examples"][0].endswith("...")
        finding = by_column(result)["code"][0]
        assert finding.code == "length_outliers" and finding.severity is Severity.INFO

    def test_a_column_of_one_length_has_no_length_fence(self):
        frame = pl.DataFrame({"code": ["abcde"] * 100 + ["abcdefghijklmnop"]})

        assert run_core_check(frame, "string_anomalies").findings == ()

    def test_free_text_gets_neither_the_case_nor_the_length_test(self):
        g = generator()
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        text = [
            " ".join(g.choice(words, size=int(g.integers(6, 12)))) for _ in range(200)
        ]
        text += ["Alpha beta gamma delta epsilon zeta one two"] * 3
        text += ["alpha beta gamma delta epsilon zeta one two"] * 3
        text += [" ".join(["word"] * 400)]
        frame = pl.DataFrame({"note": text})

        result = run_core_check(frame, "string_anomalies")

        assert result.findings == ()

    def test_a_column_not_stored_as_strings_is_skipped_with_the_reason(self):
        frame = pl.DataFrame(
            {"grade": [1, 2, 3] * 21, "tag": ["a", "b", "c"] * 20 + [" a"] * 3}
        )

        output = output_of(run_core_check(frame, "string_anomalies"))

        assert output["skipped_columns"] == {"grade": "stored as Int64, not as strings"}
        assert output["columns_read"] == 1

    def test_no_string_column_is_not_applicable(self):
        frame = pl.DataFrame({"x": [1.0, 2.0, 3.0] * 20})

        result = run_core_check(frame, "string_anomalies")

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_a_column_with_few_values_is_not_read(self):
        frame = pl.DataFrame(
            {"tag": [None] * 35 + [" a", "b", "c", "d", "e"], "ok": ["x", " y"] * 20}
        )

        output = output_of(run_core_check(frame, "string_anomalies"))

        assert output["skipped_columns"] == {"tag": "fewer than 10 values"}
        assert output["columns_read"] == 1

    def test_the_parameters_are_bounded(self):
        for bad in ({"min_values": 0}, {"fence": 0}, {"max_examples": 99}):
            with pytest.raises(ValidationError):
                StringAnomaliesParams(**bad)


def random_tokens(n: int, length: int, alphabet: str, seed: int = 1) -> list[str]:
    g = generator(seed)
    return ["".join(g.choice(list(alphabet), size=length)) for _ in range(n)]


class TestEncodedColumns:
    def test_uuids_are_found_whatever_their_version(self):
        frame = pl.DataFrame(
            {
                "id": [str(uuid.uuid4()) for _ in range(60)]
                + [str(uuid.UUID(int=i)) for i in range(40)]
            }
        )

        result = run_core_check(frame, "encoded_columns")

        column = output_of(result)["columns"]["id"]
        assert column["kind"] == "uuid" and column["share"] == 1.0
        finding = result.findings[0]
        assert finding.code == "encoded_values"
        assert finding.confidence is Confidence.HIGH
        assert finding.evidence.metrics["kind"] == "uuid"
        assert {a.action for a in actions_of(finding)} == {"drop"}

    def test_hex_digests_have_a_digest_length_and_some_letters(self):
        digests = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(80)]
        counters = [f"{i:032d}" for i in range(80)]  # 32 digits: a number, not a digest

        found = output_of(
            run_core_check(
                pl.DataFrame({"d": digests, "n": counters}), "encoded_columns"
            )
        )

        assert found["columns"]["d"]["kind"] == "hex"
        assert "n" not in found["columns"]

    def test_base64_that_decodes_to_text_or_looks_random_is_found(self):
        text = [
            base64.b64encode(f"record number {i}".encode()).decode() for i in range(100)
        ]
        rand = [
            base64.b64encode(hashlib.sha1(str(i).encode()).digest()).decode()
            for i in range(100)
        ]

        found = output_of(
            run_core_check(pl.DataFrame({"a": text, "b": rand}), "encoded_columns")
        )

        assert found["columns"]["a"]["kind"] == "base64"
        assert found["columns"]["b"]["kind"] == "base64"

    def test_plain_words_are_never_base64(self):
        # Every word is made of base64 characters; none has the length, case mix and
        # digit or padding a base64 string has.
        words = [
            "female",
            "male",
            "Internationalization",
            "Southampton",
            "Cherbourg",
            "Queenstown",
        ]
        frame = pl.DataFrame(
            {"w": words * 20, "mixed": ["Passw0rd"] * 60 + ["Welcome1"] * 60}
        )

        assert output_of(run_core_check(frame, "encoded_columns"))["columns"] == {}

    def test_random_tokens_of_one_length_are_opaque_tokens(self):
        tokens = random_tokens(300, 23, string.ascii_letters + string.digits)

        found = output_of(
            run_core_check(pl.DataFrame({"t": tokens}), "encoded_columns")
        )

        assert found["columns"]["t"]["kind"] == "opaque_token"

    def test_a_label_with_a_counter_is_not_an_opaque_token(self):
        # Long, with a high entropy per value, but made of the same words each time.
        frame = pl.DataFrame({"e": [f"invalid_email_{i}" for i in range(300)]})

        assert output_of(run_core_check(frame, "encoded_columns"))["columns"] == {}

    def test_a_kind_can_be_switched_off(self):
        frame = pl.DataFrame({"id": [str(uuid.uuid4()) for _ in range(50)]})

        result = run_core_check(frame, "encoded_columns", detect_uuid=False)

        assert output_of(result)["columns"] == {}

    def test_a_column_is_flagged_only_above_the_share(self):
        ids = [str(uuid.uuid4()) for _ in range(80)] + ["n/a"] * 20
        frame = pl.DataFrame({"id": ids})

        strict = run_core_check(frame, "encoded_columns", min_share=0.9)
        loose = run_core_check(frame, "encoded_columns", min_share=0.7)

        assert output_of(strict)["columns"] == {}
        assert output_of(loose)["columns"]["id"]["share"] == pytest.approx(0.8)

    def test_integers_are_not_read_and_the_reason_is_stated(self):
        frame = pl.DataFrame(
            {"n": [1, 2, 3, 4] * 25, "id": [str(uuid.uuid4()) for _ in range(100)]}
        )

        output = output_of(run_core_check(frame, "encoded_columns"))

        assert output["skipped_columns"] == {"n": "stored as Int64, not as strings"}

    def test_the_parameters_are_bounded(self):
        for bad in ({"min_share": 0}, {"min_values": 0}, {"min_entropy": -1}):
            with pytest.raises(ValidationError):
                EncodedColumnsParams(**bad)


class TestMixedTypes:
    def test_numbers_beside_words_are_a_mix_with_shares_and_examples(self):
        values = [str(i) + ".5" for i in range(80)] + [f"ERR_{i}" for i in range(20)]
        result = run_core_check(pl.DataFrame({"v": values}), "mixed_types")

        column = output_of(result)["columns"]["v"]
        assert set(column["kinds"]) == {"number", "text"}
        assert column["kinds"]["text"]["share"] == pytest.approx(0.2)
        assert column["source"] == "content"
        finding = result.findings[0]
        assert (finding.code, finding.severity) == ("mixed_value_types", Severity.WARN)
        assert finding.evidence.method == "value_kinds"
        assert {a.action for a in actions_of(finding)} == {"convert"}

    def test_integers_and_decimals_are_one_kind_of_number(self):
        frame = pl.DataFrame({"v": ["1", "2.5", "3", "4.25", "-7", "1e3"] * 20})

        assert run_core_check(frame, "mixed_types").findings == ()

    def test_codes_with_leading_zeros_are_text_not_numbers(self):
        frame = pl.DataFrame(
            {"zip": ["02134", "10001", "94105", "00501", "60601", "30301"] * 20}
        )

        assert run_core_check(frame, "mixed_types").findings == ()

    def test_words_that_are_booleans_beside_other_words_are_not_a_mix(self):
        frame = pl.DataFrame({"answer": ["yes", "no", "maybe", "true", "false"] * 20})

        assert run_core_check(frame, "mixed_types").findings == ()

    def test_booleans_beside_numbers_are_a_mix(self):
        frame = pl.DataFrame({"flag": ["True", "False", "1", "0"] * 25})

        column = output_of(run_core_check(frame, "mixed_types"))["columns"]["flag"]

        assert set(column["kinds"]) == {"boolean", "number"}

    def test_null_tokens_are_ignored_unless_asked_for(self):
        values = [str(i) for i in range(60)] + ["N/A"] * 20 + ["null"] * 10 + [""] * 10
        frame = pl.DataFrame({"v": values})

        ignored = run_core_check(frame, "mixed_types")
        counted = run_core_check(frame, "mixed_types", ignore_null_tokens=False)

        assert ignored.findings == ()
        column = output_of(counted)["columns"]["v"]
        assert set(column["kinds"]) == {"number", "null_token"}
        assert column["n_used"] == 100

    def test_dates_in_several_formats_are_one_kind_not_a_mix(self):
        frame = pl.DataFrame(
            {"d": ["2023-01-15", "15/01/2023", "Jan 15 2023", "20230115"] * 25}
        )

        assert run_core_check(frame, "mixed_types").findings == ()

    def test_eight_digit_numbers_alone_are_numbers_not_dates(self):
        frame = pl.DataFrame({"d": [str(20230101 + i) for i in range(60)] + ["x"] * 3})

        assert run_core_check(frame, "mixed_types").findings == ()

    def test_dates_beside_numbers_and_words_are_a_mix(self):
        frame = pl.DataFrame({"d": ["2023-01-15"] * 50 + ["12"] * 30 + ["soon"] * 20})

        column = output_of(run_core_check(frame, "mixed_types"))["columns"]["d"]

        assert set(column["kinds"]) == {"date", "number", "text"}

    def test_a_pandas_column_of_mixed_python_types_is_reported_from_the_load(self):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(
            {
                "m": ["a", 1.5, "b", 2.5] * 25 + ["c"] * 5,
                "ok": ["x", "y"] * 52 + ["z"] * 1,
            }
        )

        result = run_core_check(frame, "mixed_types")

        column = output_of(result)["columns"]["m"]
        assert column["source"] in ("load", "both")
        assert "ok" not in output_of(result)["columns"]
        assert "mixed Python types" in result.findings[0].message

    def test_a_column_of_one_kind_is_not_a_mix_whatever_the_load_said(self):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({"m": ["a", "b", "c"] * 40})

        assert run_core_check(frame, "mixed_types").findings == ()

    def test_the_parameters_are_bounded(self):
        for bad in ({"min_share": 0}, {"min_share": 0.9}, {"min_values": 0}):
            with pytest.raises(ValidationError):
                MixedTypesParams(**bad)


class TestFormatViolations:
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "email": ["a@b.com", "c@d.org", "bad", "also bad", "e@f.net"] * 10,
                "state": ["NY", "CA", "ny", "TX", "T"] * 10,
                "n": list(range(50)),
            }
        )

    def test_values_that_break_a_rule_are_counted_with_examples(self):
        rules = [{"column": "state", "pattern": "[A-Z]{2}"}]

        result = run_core_check(self.frame(), "format_violations", rules=rules)

        rule = output_of(result)["rules"][0]
        assert (rule["violations"], rule["n_used"]) == (20, 50)
        assert rule["share"] == pytest.approx(0.4)
        assert set(rule["examples"]) == {"ny", "T"}
        finding = result.findings[0]
        assert (finding.code, finding.severity) == ("format_violations", Severity.WARN)
        assert finding.evidence.method == "regex_fullmatch"
        assert finding.evidence.metrics["distinct_violations"] == 2

    def test_search_mode_needs_the_pattern_anywhere_in_the_value(self):
        rules = [{"column": "email", "pattern": "@", "mode": "search"}]

        result = run_core_check(self.frame(), "format_violations", rules=rules)

        assert output_of(result)["rules"][0]["violations"] == 20
        assert result.findings[0].evidence.method == "regex_search"

    def test_a_rule_without_violations_is_in_the_output_but_not_a_finding(self):
        rules = [{"column": "email", "pattern": r".*", "mode": "fullmatch"}]

        result = run_core_check(self.frame(), "format_violations", rules=rules)

        assert result.findings == ()
        assert output_of(result)["rules"][0]["violations"] == 0

    def test_a_rule_that_is_not_a_regular_expression_is_refused_when_planned(self):
        with pytest.raises(ValidationError, match="not a regular expression"):
            FormatViolationsParams(rules=[{"column": "a", "pattern": "("}])

    def test_two_rules_for_one_column_are_refused(self):
        with pytest.raises(ValidationError, match="two rules"):
            FormatViolationsParams(
                rules=[{"column": "a", "pattern": "x"}, {"column": "a", "pattern": "y"}]
            )

    def test_a_rule_for_a_missing_or_numeric_column_is_reported_not_run(self):
        rules = [
            {"column": "nope", "pattern": "x"},
            {"column": "n", "pattern": r"\d+"},
        ]

        result = run_core_check(self.frame(), "format_violations", rules=rules)

        output = output_of(result)
        assert output["unchecked_rules"] == {
            "nope": "no such column",
            "n": "stored as Int64, not strings",
        }
        assert output["rules"] == []
        assert {n.code for n in result.notes} == {"rule_not_run"}

    def mixed_dates(self) -> pl.DataFrame:
        formats = ["2023-01-15", "15/01/2023", "Jan 15 2023"]
        return pl.DataFrame({"d": [formats[i % 3] for i in range(90)] + [None] * 10})

    def test_dates_written_in_several_formats_are_reported_with_their_signatures(self):
        result = run_core_check(self.mixed_dates(), "format_violations")

        found = output_of(result)["mixed_dates"]["d"]
        assert found["date_share"] == 1.0
        assert {f["signature"] for f in found["formats"]} == {
            "9-9-9",
            "9/9/9",
            "Aa 9 9",
        }
        assert all(f["share"] == pytest.approx(1 / 3) for f in found["formats"])
        finding = result.findings[0]
        assert (finding.code, finding.severity) == ("mixed_date_formats", Severity.WARN)
        assert {a.action for a in actions_of(finding)} == {"convert"}

    def test_a_compact_date_counts_as_a_format_when_the_column_holds_other_dates(self):
        frame = pl.DataFrame({"d": ["2023-01-15", "20230115"] * 30})

        found = output_of(run_core_check(frame, "format_violations"))["mixed_dates"][
            "d"
        ]

        assert {f["format"] for f in found["formats"]} == {"%Y-%m-%d", "%Y%m%d"}

    def test_a_column_of_one_format_is_not_mixed(self):
        frame = pl.DataFrame({"d": ["2023-01-15", "2023-02-20"] * 30})

        assert (
            output_of(run_core_check(frame, "format_violations"))["mixed_dates"] == {}
        )

    def test_a_column_that_is_mostly_not_dates_is_not_a_column_of_dates(self):
        frame = pl.DataFrame({"d": ["2023-01-15", "15/01/2023"] * 5 + ["text"] * 90})

        assert (
            output_of(run_core_check(frame, "format_violations"))["mixed_dates"] == {}
        )

    def test_a_rare_format_below_the_share_is_not_counted(self):
        frame = pl.DataFrame({"d": ["2023-01-15"] * 97 + ["15/01/2023"] * 3})

        assert (
            output_of(run_core_check(frame, "format_violations"))["mixed_dates"] == {}
        )

    def test_nothing_to_check_is_not_applicable(self):
        frame = pl.DataFrame({"x": [1.0, 2.0, 3.0] * 10})

        result = run_core_check(frame, "format_violations")

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_no_rules_and_the_date_scan_off_reports_nothing(self):
        result = run_core_check(
            self.mixed_dates(), "format_violations", check_date_formats=False
        )

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_the_output_is_strict_json(self):
        result = run_core_check(self.mixed_dates(), "format_violations")

        json.loads(result.model_dump_json(), parse_constant=_refuse)


def _refuse(constant: str):
    raise AssertionError(f"non-finite number {constant} in the output")
