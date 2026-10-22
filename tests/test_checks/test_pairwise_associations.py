"""The association and contingency checks on small frames."""

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from scipy import stats

from dsbf.checks.associations import AssociationsParams
from dsbf.checks.contingency import ContingencyParams
from dsbf.checks.pairs import (
    contingency_table,
    cramers_v,
    fisher_interval,
    level_codes,
    strength_band,
)
from dsbf.model import CheckStatus, Dimension, Severity
from tests.test_checks.support import actions_of, output_of, run_core_check

pytestmark = pytest.mark.unit


def generator(seed: int = 5) -> np.random.Generator:
    return np.random.default_rng(seed)


def codes(result) -> list[str]:
    return sorted(finding.code for finding in result.findings)


def pair(result, first: str, second: str) -> dict:
    (found,) = (
        p
        for p in output_of(result)["pairs"]
        if (p["first"], p["second"]) == (first, second)
    )
    return found


class TestSharedArithmetic:
    def test_bands_use_the_cut_points_of_each_metric(self):
        assert strength_band(-0.69, "pearson_r") == "moderate"
        assert strength_band(-0.7, "pearson_r") == "strong"
        assert strength_band(0.09, "cramers_v") == "negligible"
        assert strength_band(0.5, "cramers_v") == "strong"
        # η at the η of the conventional η² cut points 0.01, 0.06 and 0.14
        assert strength_band(0.099, "eta") == "negligible"
        assert strength_band(0.3, "eta") == "moderate"
        assert strength_band(0.38, "eta") == "strong"

    def test_levels_are_coded_with_minus_one_where_the_value_is_missing(self):
        coded, levels = level_codes(np.array(["b", None, "a", "b"], dtype=object))

        assert coded.tolist() == [1, -1, 0, 1]
        assert levels.tolist() == ["a", "b"]

    def test_float_codes_with_nan_and_booleans_with_nulls_are_coded(self):
        coded, levels = level_codes(np.array([1.0, np.nan, 2.0, 1.0]))
        assert coded.tolist() == [0, -1, 1, 0]
        assert levels.tolist() == [1.0, 2.0]

        coded, _ = level_codes(np.array([True, None, False], dtype=object))
        assert coded.tolist() == [1, -1, 0]

    def test_a_column_of_only_missing_values_has_no_levels(self):
        coded, levels = level_codes(np.array([None, None], dtype=object))

        assert coded.tolist() == [-1, -1]
        assert levels.size == 0

    def test_the_table_leaves_out_rows_missing_either_column(self):
        first = np.array([0, 1, -1, 1, 0])
        second = np.array([0, 0, 1, -1, 1])

        table, rows, cols = contingency_table(first, second)

        assert table.tolist() == [[1, 1], [1, 0]]
        assert rows.tolist() == [0, 1]
        assert cols.tolist() == [0, 1]

    def test_corrected_v_follows_bergsma_by_hand(self):
        table = np.array([[10, 20], [30, 40]])
        n = table.sum()
        chi2 = stats.chi2_contingency(table, correction=False).statistic
        phi2 = chi2 / n
        phi2_tilde = max(0.0, phi2 - 1 / (n - 1))
        r_tilde = 2 - 1 / (n - 1)
        expected = np.sqrt(phi2_tilde / (r_tilde - 1))

        got_chi2, plain, corrected = cramers_v(table)

        assert got_chi2 == pytest.approx(chi2)
        assert plain == pytest.approx(np.sqrt(phi2))
        assert corrected == pytest.approx(expected)
        assert corrected < plain

    def test_the_fisher_interval_is_the_textbook_one_and_needs_degrees_of_freedom(self):
        interval = fisher_interval(0.5, 53, "pearson_r", 0.95)
        assert interval is not None
        low, high = interval

        half = 1.959964 / np.sqrt(50)
        assert low == pytest.approx(np.tanh(np.arctanh(0.5) - half))
        assert high == pytest.approx(np.tanh(np.arctanh(0.5) + half))
        assert fisher_interval(0.5, 4, "kendalls_tau", 0.95) is None


class TestAssociations:
    def numeric_frame(self, n: int) -> pl.DataFrame:
        g = generator()
        x = g.normal(size=n)
        return pl.DataFrame(
            {"x": x, "y": 0.6 * x + g.normal(size=n), "noise": g.normal(size=n)}
        )

    def test_a_20_row_frame_gets_kendalls_tau(self):
        frame = self.numeric_frame(20)

        result = run_core_check(frame, "associations")

        assert result.status is CheckStatus.OK
        found = pair(result, "x", "y")
        assert found["metric"] == "kendalls_tau"
        assert found["n_used"] == 20
        assert found["value"] == pytest.approx(
            stats.kendalltau(frame["x"], frame["y"]).statistic
        )
        assert output_of(result)["pairs_tested"] == 3

    def test_a_large_frame_gets_pearson_with_a_fisher_interval(self):
        frame = self.numeric_frame(200)

        found = pair(run_core_check(frame, "associations"), "x", "y")

        r = float(np.corrcoef(frame["x"], frame["y"])[0, 1])
        assert found["metric"] == "pearson_r"
        assert found["value"] == pytest.approx(r)
        assert found["strength"] == "moderate"
        half = 1.959964 / np.sqrt(197)
        assert found["interval"]["low"] == pytest.approx(np.tanh(np.arctanh(r) - half))
        assert found["interval"]["high"] == pytest.approx(np.tanh(np.arctanh(r) + half))
        assert found["interval"]["level"] == 0.95

    @pytest.mark.parametrize(
        ("method", "metric", "reference"),
        [
            ("pearson", "pearson_r", lambda a, b: stats.pearsonr(a, b).statistic),
            ("spearman", "spearman_r", lambda a, b: stats.spearmanr(a, b).statistic),
            ("kendall", "kendalls_tau", lambda a, b: stats.kendalltau(a, b).statistic),
        ],
    )
    def test_each_method_matches_scipy(self, method, metric, reference):
        frame = self.numeric_frame(120).with_columns(pl.col("y").exp())

        found = pair(run_core_check(frame, "associations", method=method), "x", "y")

        assert found["metric"] == metric
        assert found["value"] == pytest.approx(reference(frame["x"], frame["y"]))

    def test_a_monotone_curve_is_stronger_by_rank_than_by_pearson(self):
        x = np.linspace(0, 6, 80)
        frame = pl.DataFrame({"x": x, "y": np.exp(x)})

        pearson = pair(
            run_core_check(frame, "associations", method="pearson"), "x", "y"
        )
        spearman = pair(
            run_core_check(frame, "associations", method="spearman"), "x", "y"
        )

        assert spearman["value"] == pytest.approx(1.0)
        assert pearson["value"] < 0.9

    def test_columns_named_with_a_pipe_keep_their_own_subjects(self):
        g = generator()
        x = g.normal(size=60)
        frame = pl.DataFrame(
            {"a|b": x, "c|d": 2 * x + 0.05 * g.normal(size=60), "e": g.normal(size=60)}
        )

        result = run_core_check(frame, "associations")

        subjects = {f.subject.columns for f in result.findings}
        assert subjects == {("a|b", "c|d")}
        (finding,) = result.findings
        assert finding.subject.kind == "column_pair"
        assert {(p["first"], p["second"]) for p in output_of(result)["pairs"]} == {
            ("a|b", "c|d"),
            ("a|b", "e"),
            ("c|d", "e"),
        }

    def test_a_near_copy_is_redundant_and_not_also_strong(self):
        g = generator()
        x = g.normal(size=100)
        frame = pl.DataFrame(
            {"celsius": x, "kelvin": x + 273.15, "other": g.normal(size=100)}
        )

        result = run_core_check(frame, "associations")

        assert codes(result) == ["redundant_columns"]
        (finding,) = result.findings
        assert finding.dimension is Dimension.REDUNDANCY
        assert finding.severity is Severity.WARN
        assert finding.subject.columns == ("celsius", "kelvin")
        assert float(finding.evidence.metrics["abs_pearson_r"]) >= 0.99  # type: ignore[arg-type]
        assert {a.action for a in actions_of(finding)} == {"drop", "investigate"}
        assert finding.evidence.n_used == 100

    def test_a_strong_pair_is_informational_and_states_the_pairs_tested(self):
        g = generator()
        x = g.normal(size=300)
        frame = pl.DataFrame(
            {
                "x": x,
                "y": x + 0.7 * g.normal(size=300),
                "n1": g.normal(size=300),
                "n2": g.normal(size=300),
            }
        )

        result = run_core_check(frame, "associations")

        (finding,) = result.findings
        assert finding.code == "strong_association"
        assert finding.dimension is Dimension.RELATIONSHIP
        assert finding.severity is Severity.INFO
        assert finding.evidence.metrics["pairs_tested"] == 6
        assert finding.evidence.metrics["pairs_flagged"] == 1
        assert finding.evidence.interval is not None
        assert output_of(result)["pairs_flagged"] == 1

    def test_pairs_use_the_rows_where_both_columns_have_a_value(self):
        g = generator()
        x = g.normal(size=100)
        y = x + g.normal(size=100)
        x_nulls = x.copy()
        x_nulls[:30] = np.nan
        y[90:] = np.inf
        frame = pl.DataFrame({"x": x_nulls, "y": y, "z": g.normal(size=100)})

        found = pair(run_core_check(frame, "associations"), "x", "y")

        assert found["n_used"] == 60
        keep = ~np.isnan(x_nulls) & np.isfinite(y)
        assert found["value"] == pytest.approx(
            np.corrcoef(x_nulls[keep], y[keep])[0, 1]
        )

    def test_a_pair_with_too_few_rows_or_no_variation_is_counted_not_reported(self):
        g = generator()
        # `y` is constant on the 25 rows where `x` has a value; `thin` has 4 values.
        frame = pl.DataFrame(
            {
                "x": [*g.normal(size=25), *[None] * 15],
                "y": [*[7.0] * 25, *g.normal(size=15)],
                "thin": [*[None] * 36, 1.0, 2.0, 3.0, 4.0],
            }
        )

        result = run_core_check(frame, "associations")

        out = output_of(result)
        assert out["untested"] == {"no_variation": 1, "too_few_rows": 2}
        assert out["pairs_tested"] == 0
        assert all(np.isfinite(p["value"]) for p in out["pairs"])

    def test_values_near_the_float_limit_give_a_number_not_a_nan(self):
        g = generator()
        x = g.normal(size=50) * 1e307
        frame = pl.DataFrame({"x": x, "y": 2 * x, "z": g.normal(size=50) * 1e307})

        result = run_core_check(frame, "associations")

        assert result.status is CheckStatus.OK
        assert pair(result, "x", "y")["value"] == pytest.approx(1.0)
        assert all(np.isfinite(p["value"]) for p in output_of(result)["pairs"])

    def test_eta_matches_the_variance_ratio_and_point_biserial_for_two_groups(self):
        g = generator()
        group = np.array(["u", "v", "w"] * 40)
        x = g.normal(size=120) + np.where(group == "w", 1.5, 0.0)
        two = np.where(group == "w", "w", "other")
        frame = pl.DataFrame({"x": x, "three": group, "two": two})

        result = run_core_check(frame, "associations", method="pearson")

        grand = x.mean()
        between = sum(
            (group == level).sum() * (x[group == level].mean() - grand) ** 2
            for level in ("u", "v", "w")
        )
        found = pair(result, "x", "three")
        assert found["metric"] == "eta"
        assert found["value"] == pytest.approx(
            np.sqrt(between / ((x - grand) ** 2).sum())
        )
        assert found["interval"] is None
        point_biserial = stats.pointbiserialr(two == "w", x).statistic
        assert pair(result, "x", "two")["value"] == pytest.approx(abs(point_biserial))

    def test_groups_of_one_row_are_left_out_of_eta(self):
        g = generator()
        level = np.array(["a", "b"] * 20 + ["lonely"])
        frame = pl.DataFrame({"x": g.normal(size=41), "grp": level})

        found = pair(run_core_check(frame, "associations"), "x", "grp")

        assert found["n_used"] == 40

    def test_categorical_columns_with_too_many_or_too_few_levels_are_skipped(self):
        g = generator()
        frame = pl.DataFrame(
            {
                "x": g.normal(size=100),
                "y": g.normal(size=100),
                "many": [f"v{i % 30}" for i in range(100)],
                "one": ["a"] * 100,
            }
        )

        out = output_of(run_core_check(frame, "associations", max_levels=10))

        assert out["skipped_columns"] == {
            "many": "more than 10 levels",
            "one": "fewer than two levels",
        }
        assert {(p["first"], p["second"]) for p in out["pairs"]} == {("x", "y")}

    def test_one_numeric_column_and_nothing_to_pair_it_with_is_not_applicable(self):
        frame = pl.DataFrame({"x": generator().normal(size=30)})

        result = run_core_check(frame, "associations")

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_too_many_pairs_skips_the_check_with_a_hint(self):
        g = generator()
        frame = pl.DataFrame({f"c{i}": g.normal(size=30) for i in range(6)})

        result = run_core_check(frame, "associations", max_pairs=10)

        assert result.status is CheckStatus.SKIPPED
        (note,) = [n for n in result.notes if n.code == "too_many_pairs"]
        assert "15" in note.message
        assert note.hint is not None
        assert "max_pairs" in note.hint

    def test_a_large_frame_is_sampled_with_a_seed(self):
        frame = self.numeric_frame(3000)

        first = run_core_check(frame, "associations", max_rows=1000)
        again = run_core_check(frame, "associations", max_rows=1000)

        assert output_of(first)["rows_used"] == 1000
        assert output_of(first)["rows_total"] == 3000
        assert pair(first, "x", "y")["n_used"] == 1000
        assert output_of(first) == output_of(again)
        assert any(n.code == "rows_sampled" for n in first.notes)

    def test_findings_are_capped_and_the_rest_stay_in_the_output(self):
        g = generator()
        base = g.normal(size=100)
        frame = pl.DataFrame(
            {f"c{i}": base + 0.2 * g.normal(size=100) for i in range(5)}
        )

        result = run_core_check(frame, "associations", max_findings=2)

        assert len(result.findings) == 2
        assert output_of(result)["pairs_flagged"] == 10
        assert any(n.code == "findings_capped" for n in result.notes)

    @pytest.mark.parametrize(
        "bad", [{"min_n": 3}, {"interval_level": 1.0}, {"method": "cramer"}]
    )
    def test_each_parameter_has_bounds(self, bad):
        with pytest.raises(ValidationError):
            AssociationsParams(**bad)


class TestContingency:
    def frame(self, n: int, link: float = 0.0) -> pl.DataFrame:
        g = generator()
        a = g.choice(list("xyz"), size=n)
        follow = g.random(n) < link
        b = np.where(follow, np.char.upper(a), g.choice(list("XYZ"), size=n))
        return pl.DataFrame({"a": a, "b": b, "c": g.choice(["p", "q"], size=n)})

    def test_the_corrected_v_is_below_the_plain_v_on_a_small_sample(self):
        frame = self.frame(40)

        found = pair(run_core_check(frame, "contingency"), "a", "b")

        assert found["cramers_v_corrected"] < found["cramers_v"]

    def test_independent_columns_have_a_corrected_v_near_zero_and_no_finding(self):
        result = run_core_check(self.frame(3000), "contingency")

        assert pair(result, "a", "b")["cramers_v_corrected"] < 0.1
        assert result.findings == ()
        assert output_of(result)["pairs_flagged"] == 0

    def test_a_strong_link_is_reported_with_a_bootstrap_interval(self):
        result = run_core_check(self.frame(600, link=0.9), "contingency")

        found = pair(result, "a", "b")
        assert found["strength"] == "strong"
        assert found["interval"]["method"] == "percentile_bootstrap"
        assert found["interval"]["low"] <= found["cramers_v_corrected"]
        assert found["cramers_v_corrected"] <= found["interval"]["high"]
        (finding,) = result.findings
        assert finding.code == "categorical_association"
        assert finding.severity is Severity.WARN
        assert finding.subject.columns == ("a", "b")
        assert finding.evidence.n_used == 600
        assert finding.evidence.metrics["pairs_tested"] == 3
        assert {a.action for a in actions_of(finding)} == {"combine", "investigate"}

    def test_the_bootstrap_is_reproducible_and_skipped_for_weak_pairs(self):
        frame = self.frame(600, link=0.9)

        first = output_of(run_core_check(frame, "contingency"))
        again = output_of(run_core_check(frame, "contingency"))
        other = output_of(run_core_check(frame, "contingency", seed=7))

        assert first == again
        assert first["pairs"][0]["interval"] != other["pairs"][0]["interval"]
        weak = {(p["first"], p["second"]): p for p in first["pairs"]}[("a", "c")]
        assert weak["interval"] is None

    def test_the_p_value_is_the_uncorrected_chi_squared_test(self):
        frame = self.frame(500, link=0.3)

        found = pair(run_core_check(frame, "contingency"), "a", "b")

        table = np.array(
            [
                [
                    sum(
                        1
                        for u, v in zip(frame["a"], frame["b"], strict=True)
                        if (u, v) == (i, j)
                    )
                    for j in "XYZ"
                ]
                for i in "xyz"
            ]
        )
        reference = stats.chi2_contingency(table, correction=False)
        assert found["chi2"] == pytest.approx(reference.statistic)
        assert found["p_value"] == pytest.approx(reference.pvalue)
        assert found["dof"] == 4
        assert found["levels"] == [3, 3]

    def test_the_printed_table_is_cut_but_the_statistics_use_every_level(self):
        g = generator()
        n = 2000
        frame = pl.DataFrame(
            {
                "wide": g.choice([f"w{i:02d}" for i in range(20)], size=n),
                "narrow": g.choice(["p", "q"], size=n),
            }
        )

        found = pair(run_core_check(frame, "contingency", top_n=5), "wide", "narrow")

        assert found["levels"] == [20, 2]
        assert len(found["table"]) == 5
        assert found["truncated"] is True
        assert found["dof"] == 19

    def test_a_sparse_table_lowers_confidence_and_records_the_broken_assumption(self):
        a = ["x", "x", "y", "y", "z", "z"] * 5
        b = ["p", "p", "q", "q", "r", "r"] * 5
        frame = pl.DataFrame({"a": a, "b": b})

        result = run_core_check(frame, "contingency", min_v=0.1)

        (finding,) = result.findings
        (assumption,) = finding.evidence.assumptions
        assert assumption.name == "expected_counts_at_least_5"
        assert pair(result, "a", "b")["sparse_share"] == pytest.approx(1.0)
        # n = 30 and 9 cells expect 3.3 rows: more than a fifth are under 5
        assert assumption.satisfied is False
        assert finding.confidence.value == "low"

    def test_columns_named_with_a_pipe_and_booleans_are_compared(self):
        g = generator()
        a = g.choice([True, False], size=400)
        frame = pl.DataFrame(
            {
                "a|b": a,
                "c|d": a ^ (g.random(400) < 0.05),
                "e": g.choice(list("pq"), size=400),
            }
        )

        result = run_core_check(frame, "contingency")

        (finding,) = result.findings
        assert finding.subject.columns == ("a|b", "c|d")
        assert len(output_of(result)["pairs"]) == 3

    def test_rows_missing_either_column_are_left_out(self):
        a = ["x", "y"] * 30
        b = ["p", "q"] * 30
        frame = pl.DataFrame({"a": a[:50] + [None] * 10, "b": [None] * 5 + b[5:]})

        found = pair(run_core_check(frame, "contingency"), "a", "b")

        assert found["n_used"] == 45

    def test_columns_with_too_many_levels_and_tiny_pairs_are_accounted_for(self):
        frame = pl.DataFrame(
            {
                "a": ["x", "y"] * 4 + [None] * 4,
                "b": ["p", "q"] * 4 + [None] * 4,
                "many": [f"v{i % 6}" for i in range(12)],
            }
        )

        out = output_of(run_core_check(frame, "contingency", max_levels=4))

        assert out["skipped_columns"] == {"many": "more than 4 levels"}
        assert out["untested"] == {"too_few_rows": 1}
        assert out["pairs"] == []

    def test_a_single_categorical_column_is_not_applicable(self):
        frame = pl.DataFrame({"a": ["x", "y"] * 20, "n": generator().normal(size=40)})

        assert run_core_check(frame, "contingency").status is CheckStatus.NOT_APPLICABLE

    def test_too_many_pairs_skips_the_check(self):
        g = generator()
        frame = pl.DataFrame({f"c{i}": g.choice(["a", "b"], size=40) for i in range(6)})

        result = run_core_check(frame, "contingency", max_pairs=5)

        assert result.status is CheckStatus.SKIPPED

    @pytest.mark.parametrize("bad", [{"n_boot": 10}, {"alpha": 1.0}])
    def test_each_parameter_has_bounds(self, bad):
        with pytest.raises(ValidationError):
            ContingencyParams(**bad)
