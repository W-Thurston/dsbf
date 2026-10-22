"""The missingness mechanism check on small frames."""

import json

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from scipy.stats import chi2_contingency, mannwhitneyu

from dsbf.checks.missingness_mechanism import MissingnessMechanismParams
from dsbf.model import CheckStatus, Confidence, Dimension, Severity
from tests.test_checks.support import actions_of, output_of, run_core_check

pytestmark = pytest.mark.unit


def generator(seed: int = 11) -> np.random.Generator:
    return np.random.default_rng(seed)


def columns_of(result) -> dict:
    return output_of(result)["columns"]


def predictor(result, subject: str, column: str) -> dict:
    found = [
        p
        for p in columns_of(result)[subject]["dependences"]
        + [columns_of(result)[subject]["strongest"]]
        if p and p["column"] == column
    ]
    assert found, f"{column} is no predictor of {subject}"
    return found[0]


def dependent_frame(n: int = 600) -> pl.DataFrame:
    """`income` is null more often where `age` is high; `noise` is null at random."""
    g = generator()
    age = g.normal(50, 10, size=n)
    income = g.normal(size=n)
    income[g.random(n) < 1 / (1 + np.exp(-(age - 55) / 3))] = np.nan
    noise = g.normal(size=n)
    noise[g.random(n) < 0.25] = np.nan
    return pl.DataFrame(
        {
            "age": age,
            "income": income,
            "noise": noise,
            "other": g.normal(size=n),
        }
    ).with_columns(pl.col("income", "noise").fill_nan(None))


class TestDependence:
    def test_a_column_whose_nulls_follow_another_is_flagged_with_the_direction(self):
        result = run_core_check(dependent_frame(), "missingness_mechanism")

        assert result.status is CheckStatus.OK
        income = columns_of(result)["income"]
        assert income["assessment"] == "depends_on_observed"
        age = predictor(result, "income", "age")
        assert age["flagged"] and age["test"] == "mann_whitney"
        # Higher age where the income is null: the rank-biserial r is positive.
        assert age["effect"] > 0.3
        assert age["p_adjusted"] < 0.001

    def test_a_column_with_random_nulls_depends_on_nothing(self):
        result = run_core_check(dependent_frame(), "missingness_mechanism")

        noise = columns_of(result)["noise"]
        assert noise["assessment"] == "no_dependence_found"
        assert noise["tests_flagged"] == 0
        finding = next(f for f in result.findings if f.subject.columns == ("noise",))
        assert finding.code == "missingness_unexplained"
        assert finding.confidence is Confidence.LOW
        assert finding.severity is Severity.INFO

    def test_the_finding_names_the_predictors_and_how_many_tests_were_run(self):
        result = run_core_check(dependent_frame(), "missingness_mechanism")

        finding = next(f for f in result.findings if f.subject.columns == ("income",))
        assert finding.code == "missingness_depends_on_observed"
        assert finding.dimension is Dimension.RELATIONSHIP
        assert "'age'" in finding.message
        metrics = finding.evidence.metrics
        assert metrics["tests_run"] == columns_of(result)["income"]["tests_run"]
        assert float(metrics["tests_flagged"]) >= 1  # type: ignore[arg-type]
        assert finding.evidence.method == "indicator_tests"
        assert finding.evidence.n_used == 600

    def test_a_categorical_predictor_gets_a_bias_corrected_cramers_v(self):
        g = generator()
        n = 800
        region = g.choice(["north", "south", "east"], size=n)
        score = g.normal(size=n)
        score[(region == "south") & (g.random(n) < 0.8)] = np.nan
        frame = pl.DataFrame({"region": region, "score": score}).with_columns(
            pl.col("score").fill_nan(None)
        )

        result = run_core_check(frame, "missingness_mechanism")

        region_test = predictor(result, "score", "region")
        assert region_test["test"] == "chi_squared" and region_test["flagged"]
        assert region_test["effect_name"] == "cramers_v"
        table = np.array(
            [
                [
                    int(((region == r) & ~np.isnan(score)).sum()),
                    int(((region == r) & np.isnan(score)).sum()),
                ]
                for r in ("east", "north", "south")
            ]
        )
        chi2, p, _, _ = chi2_contingency(table.T, correction=False)
        assert region_test["statistic"] == pytest.approx(chi2)
        assert region_test["p_value"] == pytest.approx(p)
        assert 0.3 < region_test["effect"] <= 1

    def test_a_negligible_difference_in_a_huge_sample_is_tested_but_not_flagged(self):
        g = generator()
        n = 200_000
        x = g.normal(size=n)
        y = g.normal(size=n)
        y[g.random(n) < 0.02 + 0.002 * (x > 0)] = np.nan
        frame = pl.DataFrame({"x": x, "y": y}).with_columns(pl.col("y").fill_nan(None))

        result = run_core_check(frame, "missingness_mechanism")

        test = predictor(result, "y", "x")
        assert test["p_adjusted"] < 0.05
        assert abs(test["effect"]) < 0.1
        assert not test["flagged"]
        assert columns_of(result)["y"]["assessment"] == "no_dependence_found"


class TestTests:
    def test_the_mann_whitney_test_matches_scipy_with_ties(self):
        g = generator()
        n = 300
        x = np.round(g.normal(size=n) * 3)  # many ties
        y = g.normal(size=n)
        y[(x > 1) & (g.random(n) < 0.5)] = np.nan
        frame = pl.DataFrame({"x": x, "y": y}).with_columns(pl.col("y").fill_nan(None))

        result = run_core_check(frame, "missingness_mechanism")

        missing = np.isnan(y)
        expected = mannwhitneyu(x[missing], x[~missing], method="asymptotic")
        test = predictor(result, "y", "x")
        assert test["p_value"] == pytest.approx(expected.pvalue, rel=1e-9)
        n1, n2 = int(missing.sum()), int((~missing).sum())
        assert (test["n_missing"], test["n_observed"]) == (n1, n2)
        assert test["effect"] == pytest.approx(2 * expected.statistic / (n1 * n2) - 1)

    def test_each_test_counts_only_rows_where_the_predictor_has_a_value(self):
        g = generator()
        n = 500
        x = g.normal(size=n)
        x[:100] = np.nan
        y = g.normal(size=n)
        y[100:250] = np.nan
        frame = pl.DataFrame({"x": x, "y": y}).with_columns(
            pl.col("x", "y").fill_nan(None)
        )

        result = run_core_check(frame, "missingness_mechanism")

        test = predictor(result, "y", "x")
        assert test["n_missing"] == 150  # y null, x observed
        assert test["n_observed"] == 250  # y observed, x observed (rows 250 to 499)
        assert test["n_missing"] + test["n_observed"] == 400

    def test_p_values_are_adjusted_over_every_test_of_every_column(self):
        frame = dependent_frame()

        none = run_core_check(frame, "missingness_mechanism", correction="none")
        holm = run_core_check(frame, "missingness_mechanism", correction="bonferroni")

        total = output_of(none)["tests_run"]
        assert total == sum(c["tests_run"] for c in columns_of(none).values())
        raw = predictor(none, "income", "age")
        adjusted = predictor(holm, "income", "age")
        assert raw["p_adjusted"] == raw["p_value"]
        assert adjusted["p_adjusted"] == pytest.approx(min(1.0, raw["p_value"] * total))
        assert output_of(holm)["correction"] == "bonferroni"

    def test_a_predictor_that_is_null_with_the_subject_is_not_compared(self):
        g = generator()
        n = 400
        both = g.random(n) < 0.3
        a = np.where(both, np.nan, g.normal(size=n))
        b = np.where(both, np.nan, g.normal(size=n))
        c = g.normal(size=n)
        frame = pl.DataFrame({"a": a, "b": b, "c": c}).with_columns(
            pl.col("a", "b").fill_nan(None)
        )

        result = run_core_check(frame, "missingness_mechanism")

        a_column = columns_of(result)["a"]
        assert a_column["missing_with"] == {"b": 1.0}
        # No row has `a` null and `b` observed, so there is no test of `a` on `b`.
        assert a_column["tests_run"] == 1
        assert a_column["strongest"]["column"] == "c"


class TestScope:
    def test_a_column_with_too_few_nulls_is_skipped_with_the_reason(self):
        g = generator()
        values = g.normal(size=500)
        values[:3] = np.nan
        frame = pl.DataFrame({"a": values, "b": g.normal(size=500)}).with_columns(
            pl.col("a").fill_nan(None)
        )

        result = run_core_check(frame, "missingness_mechanism")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert result.notes[0].code == "no_missing_columns"

    def test_only_the_columns_with_most_nulls_are_analysed_beyond_the_limit(self):
        g = generator()
        n = 300
        data = {}
        for i, share in enumerate((0.5, 0.4, 0.3)):
            column = g.normal(size=n)
            column[g.random(n) < share] = np.nan
            data[f"c{i}"] = column
        frame = pl.DataFrame(data).with_columns(pl.all().fill_nan(None))

        result = run_core_check(frame, "missingness_mechanism", max_subjects=2)

        output = output_of(result)
        assert set(output["columns"]) == {"c0", "c1"}
        assert "c2" in output["skipped_columns"]

    def test_a_big_frame_is_sampled_and_the_rows_used_are_stated(self):
        frame = dependent_frame(3000)

        result = run_core_check(frame, "missingness_mechanism", max_rows=1000)

        assert output_of(result)["rows_used"] == 1000
        finding = result.findings[0]
        assert finding.evidence.n_used == 1000
        assert finding.evidence.n_total == 3000

    def test_columns_that_are_not_numbers_or_levels_are_not_predictors(self):
        g = generator()
        n = 300
        values = g.normal(size=n)
        values[g.random(n) < 0.3] = np.nan
        frame = pl.DataFrame(
            {
                "y": values,
                "x": g.normal(size=n),
                "key": [f"id-{i}" for i in range(n)],
                "constant": ["same"] * n,
            }
        ).with_columns(pl.col("y").fill_nan(None))

        result = run_core_check(frame, "missingness_mechanism")

        unusable = output_of(result)["unusable_predictors"]
        assert "key" in unusable and "constant" in unusable
        assert columns_of(result)["y"]["tests_run"] == 1

    def test_a_datetime_column_is_compared_as_time(self):
        g = generator()
        n = 400
        days = np.arange(n)
        value = g.normal(size=n)
        value[days < 120] = np.nan  # the early rows are the missing ones
        frame = pl.DataFrame(
            {
                "when": pl.date_range(
                    pl.date(2024, 1, 1),
                    pl.date(2024, 1, 1) + pl.duration(days=n - 1),
                    "1d",
                    eager=True,
                ),
                "value": value,
                "x": g.normal(size=n),
            }
        ).with_columns(pl.col("value").fill_nan(None))

        result = run_core_check(frame, "missingness_mechanism")

        when = predictor(result, "value", "when")
        assert when["flagged"] and when["effect"] < -0.8  # earlier where it is null


class TestOutput:
    def test_the_output_has_no_non_finite_number(self):
        result = run_core_check(dependent_frame(), "missingness_mechanism")

        json.loads(result.model_dump_json(), parse_constant=_refuse)

    def test_a_strong_dependence_of_a_mostly_null_column_is_a_warning_with_tips(self):
        g = generator()
        n = 600
        group = g.choice(["a", "b"], size=n)
        value = g.normal(size=n)
        value[(group == "a") & (g.random(n) < 0.9)] = np.nan
        frame = pl.DataFrame({"group": group, "value": value}).with_columns(
            pl.col("value").fill_nan(None)
        )

        result = run_core_check(frame, "missingness_mechanism")

        finding = result.findings[0]
        assert finding.severity is Severity.WARN
        assert finding.confidence is Confidence.HIGH
        assert {a.action for a in actions_of(finding)} >= {"investigate"}

    def test_the_parameters_are_bounded(self):
        for bad in ({"alpha": 0}, {"min_group_n": 1}, {"correction": "magic"}):
            with pytest.raises(ValidationError):
                MissingnessMechanismParams.model_validate(bad)


def _refuse(constant: str):
    raise AssertionError(f"non-finite number {constant} in the output")
