"""The fuzzy duplicate check on small frames: pairs, budgets, determinism, size."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks.fuzzy_duplicates import FuzzyDuplicatesParams
from dsbf.model import CheckStatus, Confidence, Severity
from tests.test_checks.support import actions_of, output_of, run_core_check
from tests.test_model.builders import strict_loads

pytestmark = pytest.mark.unit

CHECK = "fuzzy_duplicates"
LETTERS = list("abcdefghijklmnopqrstuvwxyz")


def words(rng: np.random.Generator, n: int, length: int = 14) -> list[str]:
    """``n`` distinct random lower-case strings, each a word or two."""
    found: set[str] = set()
    while len(found) < n:
        found.add("".join(rng.choice(LETTERS, size=length)))
    return sorted(found)


def people(n: int = 80, seed: int = 3) -> pl.DataFrame:
    """Distinct records of two text columns, in no relation to each other."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "name": [f"{a} {b}" for a, b in zip(words(rng, n), words(rng, n, 9))],
            "address": [
                f"{a} street {b}" for a, b in zip(words(rng, n, 10), words(rng, n, 6))
            ],
        }
    )


def with_rows(frame: pl.DataFrame, *rows: dict) -> pl.DataFrame:
    return pl.concat([frame, pl.DataFrame(rows, schema=frame.schema)])


def pairs_of(result) -> set[tuple[int, int]]:
    return {
        (p["first"]["row"], p["second"]["row"]) for p in output_of(result)["top_pairs"]
    }


class TestFindingPairs:
    def test_a_typo_in_one_column_is_a_pair_with_the_column_and_values_named(self):
        frame = people()
        source = frame.row(5, named=True)
        copy = {**source, "name": source["name"].replace("e", "3", 1)}

        result = run_core_check(with_rows(frame, copy), CHECK)

        assert result.status is CheckStatus.OK
        output = output_of(result)
        assert output["blocking"] == "exhaustive" and output["coverage"] == 1.0
        assert output["pairs_flagged"] == 1 and output["rows_involved"] == 2
        pair = output["top_pairs"][0]
        assert (pair["first"]["row"], pair["second"]["row"]) == (5, 80)
        assert pair["differing_columns"] == ["name"]
        assert pair["values"]["name"] == [source["name"], copy["name"]]
        assert 0.92 < pair["similarity"] < 1.0
        finding = result.findings[0]
        assert finding.code == "near_duplicate_rows"
        assert finding.subject.kind == "row_set"
        assert finding.subject.columns == ("name", "address")
        assert finding.confidence is Confidence.MODERATE
        assert finding.evidence.method == "indel_ratio_exhaustive"
        assert finding.evidence.n_used == 81
        assert {a.action for a in actions_of(finding)} == {"investigate", "transform"}

    def test_case_and_padding_make_a_pair_at_similarity_one(self):
        frame = people()
        source = frame.row(2, named=True)
        shouted = {**source, "name": source["name"].upper()}
        padded = {**frame.row(9, named=True), "address": "  " + frame["address"][9]}

        result = run_core_check(with_rows(frame, shouted, padded), CHECK)

        pairs = {
            (p["first"]["row"], p["second"]["row"]): p
            for p in output_of(result)["top_pairs"]
        }
        assert set(pairs) == {(2, 80), (9, 81)}
        assert all(p["similarity"] == 1.0 for p in pairs.values())
        assert pairs[(2, 80)]["differing_columns"] == ["name"]
        assert pairs[(9, 81)]["differing_columns"] == ["address"]

    def test_rows_equal_in_every_column_are_exact_repeats_not_pairs(self):
        frame = people()
        repeated = pl.concat([frame, frame.head(10), frame.head(10)])

        result = run_core_check(repeated, CHECK)

        output = output_of(result)
        assert output["pairs_flagged"] == 0 and result.findings == ()
        assert output["exact_repeats"] == 20
        assert output["rows"] == 100 and output["records"] == 80

    def test_a_repeated_record_counts_its_rows_in_a_pair(self):
        frame = people()
        source = frame.row(4, named=True)
        copy = {**source, "name": source["name"] + "x"}

        result = run_core_check(with_rows(frame, copy, copy, copy), CHECK)

        output = output_of(result)
        assert output["pairs_flagged"] == 1 and output["rows_involved"] == 4
        assert output["top_pairs"][0]["second"] == {"row": 80, "rows": 3}

    def test_unlike_rows_are_not_a_pair_when_one_long_field_is_shared(self):
        frame = people().with_columns(
            boilerplate=pl.Series(
                ["see the terms and conditions of the licence agreement"] * 40
                + ["no terms apply to this particular record at all, "] * 40
            )
        )

        result = run_core_check(frame, CHECK, max_label_levels=1)

        assert output_of(result)["pairs_flagged"] == 0

    def test_a_label_column_is_left_out_of_the_comparison(self):
        frame = people().with_columns(
            kind=pl.Series((["alpha", "beta", "gamma"] * 27)[:80])
        )
        source = frame.row(7, named=True)
        # Differs in the label only: the rest is the same record.
        copy = {**source, "kind": "delta"}

        result = run_core_check(with_rows(frame, copy), CHECK)

        output = output_of(result)
        assert output["columns"] == ["name", "address"]
        assert "kind" in output["skipped_columns"]
        assert "label" in output["skipped_columns"]["kind"]
        assert output["pairs_flagged"] == 0 and output["exact_repeats"] == 1

    def test_the_threshold_and_a_lower_bound_decide_what_is_flagged(self):
        frame = people()
        source = frame.row(5, named=True)
        copy = {**source, "name": source["name"][:-6]}  # six characters shorter

        strict = run_core_check(with_rows(frame, copy), CHECK, threshold=0.99)
        loose = run_core_check(with_rows(frame, copy), CHECK, threshold=0.8)

        assert output_of(strict)["pairs_flagged"] == 0
        assert output_of(loose)["pairs_flagged"] == 1
        with pytest.raises(ValidationError):
            FuzzyDuplicatesParams(threshold=0.3)

    def test_a_pair_is_a_warning_when_enough_rows_are_involved(self):
        frame = people(30)
        rows = [
            {**frame.row(i, named=True), "name": frame["name"][i] + "!"}
            for i in range(10)
        ]

        result = run_core_check(with_rows(frame, *rows), CHECK)

        assert result.findings[0].severity is Severity.WARN
        assert output_of(result)["rows_involved"] == 20


class TestColumnNames:
    @pytest.mark.parametrize("name", ["", "first", "rows", "c0", "a|b", "名前"])
    def test_any_column_name_works_and_comes_back_unchanged(self, name):
        frame = people().rename({"name": name})
        source = frame.row(5, named=True)
        copy = {**source, name: source[name] + "x"}

        result = run_core_check(with_rows(frame, copy), CHECK)

        assert result.status is CheckStatus.OK
        pair = output_of(result)["top_pairs"][0]
        assert pair["differing_columns"] == [name]
        assert pair["values"][name][0] == source[name]
        assert output_of(result)["columns"] == [name, "address"]
        assert result.findings[0].subject.columns == (name, "address")

    def test_columns_named_in_exclude_columns_are_left_out(self):
        frame = people()
        source = frame.row(5, named=True)
        copy = {**source, "address": "somewhere else entirely, 99"}

        both = run_core_check(with_rows(frame, copy), CHECK)
        name_only = run_core_check(
            with_rows(frame, copy), CHECK, exclude_columns=["address"]
        )

        assert output_of(both)["pairs_flagged"] == 0
        assert output_of(name_only)["pairs_flagged"] == 0
        assert output_of(name_only)["columns"] == ["name"]
        assert output_of(name_only)["exact_repeats"] == 1
        assert output_of(name_only)["skipped_columns"] == {
            "address": "left out by exclude_columns"
        }


class TestBlockingAndBudgets:
    def planted(self) -> tuple[pl.DataFrame, set[tuple[int, int]]]:
        frame = people(120)
        rows = [
            {**frame.row(i, named=True), "address": frame["address"][i] + "s"}
            for i in (3, 40, 77)
        ]
        return with_rows(frame, *rows), {(3, 120), (40, 121), (77, 122)}

    def test_blocking_on_rare_tokens_finds_the_pairs_exhaustive_scoring_does(self):
        frame, planted = self.planted()

        exhaustive = run_core_check(frame, CHECK)
        blocked = run_core_check(frame, CHECK, max_pairs=1000)

        assert output_of(exhaustive)["blocking"] == "exhaustive"
        assert output_of(blocked)["blocking"] == "rare_tokens"
        assert pairs_of(exhaustive) == pairs_of(blocked) == planted
        scored = output_of(blocked)
        assert scored["pairs_compared"] < 300 < scored["pairs_possible"]
        assert scored["coverage"] == scored["pairs_compared"] / scored["pairs_possible"]
        assert scored["stopped_by"] is None
        finding = blocked.findings[0]
        assert finding.evidence.method == "indel_ratio_rare_tokens"
        assert finding.evidence.metrics["coverage"] == scored["coverage"]

    def test_a_pair_budget_that_holds_every_block_leaves_no_note(self):
        frame, _ = self.planted()

        result = run_core_check(frame, CHECK, max_pairs=100, max_block_size=2)

        output = output_of(result)
        assert output["blocking"] == "rare_tokens" and output["pairs_compared"] <= 100
        assert output["stopped_by"] is None
        assert [n.code for n in result.notes] == []

    def test_candidate_pairs_beyond_the_budget_end_in_a_note(self):
        rng = np.random.default_rng(0)
        shared = words(rng, 4, 8)
        frame = pl.DataFrame(
            {
                "name": [f"{shared[i % 4]} {w}" for i, w in enumerate(words(rng, 400))],
                "city": words(rng, 400, 12),
            }
        )

        result = run_core_check(frame, CHECK, max_pairs=500, max_block_size=100)

        output = output_of(result)
        assert output["stopped_by"] == "pairs" and output["pairs_compared"] <= 500
        assert [n.code for n in result.notes] == ["partial_comparison"]
        assert "max_pairs" in result.notes[0].message

    def test_more_records_than_max_rows_are_sampled_the_same_way_each_time(self):
        frame, _ = self.planted()

        first = run_core_check(frame, CHECK, max_rows=60)
        again = run_core_check(frame, CHECK, max_rows=60)

        output = output_of(first)
        assert output["records"] == 123 and output["records_compared"] == 60
        assert output["stopped_by"] == "rows" and output["coverage"] < 0.25
        assert [n.code for n in first.notes] == ["partial_comparison"]
        assert output == output_of(again)

    def test_a_clock_that_runs_out_stops_scoring_and_says_so(self):
        frame, _ = self.planted()

        result = run_core_check(frame, CHECK, max_seconds=1e-9)

        output = output_of(result)
        assert output["stopped_by"] == "seconds" and output["pairs_compared"] == 0
        assert output["pairs_flagged"] == 0 and result.findings == ()
        assert [n.code for n in result.notes] == ["partial_comparison"]


class TestWhenThereIsNothingToCompare:
    def test_columns_of_a_few_levels_leave_no_text(self):
        frame = pl.DataFrame({"kind": ["a", "b", "c"] * 30, "size": ["s", "m"] * 45})

        result = run_core_check(frame, CHECK)

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.output is None
        assert [n.code for n in result.notes] == ["no_text_columns"]

    def test_one_distinct_row_is_too_few(self):
        rng = np.random.default_rng(1)
        frame = pl.DataFrame({"name": [words(rng, 1)[0]] * 30, "note": [None] * 30})

        result = run_core_check(frame, CHECK, max_label_levels=1)

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_numbers_and_missing_text_are_skipped_with_the_reason(self):
        frame = people().with_columns(
            note=pl.lit(None, dtype=pl.String), amount=pl.Series(range(80))
        )

        output = output_of(run_core_check(frame, CHECK))

        assert output["columns"] == ["name", "address"]
        assert "amount" not in output["columns"]


class TestTheOutputStaysSmall:
    def test_thousands_of_pairs_leave_a_short_list_and_counts(self):
        rng = np.random.default_rng(2)
        base = words(rng, 300, 20)
        # Every base string with three variants: 300 groups, 6 pairs each.
        names = [f"{w}{suffix}" for w in base for suffix in ("", "a", "b", "c")]
        cities = [c for c in words(rng, 300, 12) for _ in range(4)]
        frame = pl.DataFrame({"name": names, "city": cities})

        result = run_core_check(frame, CHECK, threshold=0.9)

        output = output_of(result)
        assert output["pairs_flagged"] >= 1800
        assert len(output["top_pairs"]) == 20
        assert len(json.dumps(output)) < 20_000
        strict_loads(json.dumps(output))


class TestAvailability:
    def test_without_the_text_extra_the_check_is_unavailable_with_a_hint(
        self, monkeypatch
    ):
        monkeypatch.setattr("dsbf.engine.extras.find_spec", lambda name: None)

        result = run_core_check(people(), CHECK)

        assert result.status is CheckStatus.UNAVAILABLE
        assert result.output is None
        assert "rapidfuzz" in result.notes[0].message

    def test_the_check_runs_in_a_worker_process_not_on_a_thread(self):
        result = run_core_check(people(), CHECK)

        assert result.status is CheckStatus.OK
        assert "ran_in_thread" not in {n.code for n in result.notes}


SCRIPT = textwrap.dedent(
    """
    import json

    import numpy as np
    import polars as pl

    from tests.test_checks.support import run_core_check


    def main():
        rng = np.random.default_rng(11)
        words = ["".join(rng.choice(list("abcdef"), size=5)) for _ in range(400)]
        frame = pl.DataFrame(
            {
                "name": [f"{words[i]} {words[(i * 7) % 400]}" for i in range(400)],
                "note": [f"{words[(i * 3) % 400]} {words[i % 50]}" for i in range(400)],
            }
        )
        result = run_core_check(
            frame, "fuzzy_duplicates", max_pairs=2000, threshold=0.6, max_label_levels=1
        )
        print(json.dumps(result.output, sort_keys=True))


    if __name__ == "__main__":
        main()
    """
)


def test_the_output_does_not_depend_on_the_string_hash_seed(tmp_path):
    script = tmp_path / "run_check.py"
    script.write_text(SCRIPT)
    root = Path(__file__).parents[2]
    outputs = []
    for seed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, str(script)],
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(root)},
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0])["blocking"] == "rare_tokens"
    assert json.loads(outputs[0])["pairs_flagged"] > 0
