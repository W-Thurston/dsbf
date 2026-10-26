"""The distribution analysis checks on small frames."""

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks.bimodality import BimodalityParams
from dsbf.checks.normality import NormalityParams
from dsbf.checks.out_of_bounds import Bounds, OutOfBoundsParams
from dsbf.checks.outliers import OutliersParams
from dsbf.engine import extras as extras_module
from dsbf.model import CheckStatus, Severity
from tests.test_checks.support import actions_of, by_column, output_of, run_core_check

pytestmark = pytest.mark.unit

N = 600


def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


@pytest.fixture
def without_sklearn(monkeypatch):
    real = extras_module.find_spec
    monkeypatch.setattr(
        extras_module, "find_spec", lambda m: None if m == "sklearn" else real(m)
    )


class TestOutliers:
    def frame(self) -> pl.DataFrame:
        generator = rng()
        planted = generator.normal(size=N)
        planted[:3] = [40.0, -35.0, 55.0]
        return pl.DataFrame(
            {
                "planted": planted,
                "clean": generator.uniform(size=N),
                "mostly_zero": np.r_[
                    np.zeros(N * 7 // 10), generator.exponential(size=N * 3 // 10)
                ],
                "constant": [2.0] * N,
            }
        )

    def test_planted_extremes_are_flagged_by_agreeing_rules(self):
        found = by_column(run_core_check(self.frame(), "outliers"))

        (finding,) = found["planted"]
        assert finding.code == "outliers_detected"
        assert finding.severity is Severity.WARN
        assert finding.evidence.metrics["consensus"] is True
        assert set(finding.evidence.metrics["methods_flagging"]) == {
            "iqr",
            "zscore",
            "mad",
        }
        assert finding.evidence.n_used == N
        assert "clean" not in found
        assert "constant" not in found

    def test_the_most_distant_flagged_values_come_first(self):
        result = run_core_check(self.frame(), "outliers")

        top = output_of(result)["columns"]["planted"]["top_values"]
        assert top[:3] == [55.0, 40.0, -35.0]
        assert len(top) <= 10

    def test_a_column_mostly_of_one_value_has_no_mad_rule(self):
        column = output_of(run_core_check(self.frame(), "outliers"))["columns"][
            "mostly_zero"
        ]

        assert column["mad"] is None
        assert column["zscore"] is not None

    def test_a_column_of_one_value_is_not_read_at_all(self):
        # The profile types it categorical, and `constant_columns` reports it.
        output = output_of(run_core_check(self.frame(), "outliers"))

        assert "constant" not in output["columns"]
        assert "constant" not in output["skipped"]

    def test_one_rule_alone_is_an_information_when_it_flags_few_values(self):
        result = run_core_check(self.frame(), "outliers", methods=["iqr"])

        finding = by_column(result)["planted"][0]
        assert finding.severity is Severity.INFO
        assert finding.evidence.metrics["consensus"] is False
        assert finding.evidence.method == "iqr"

    def test_a_large_share_from_one_rule_is_a_warning(self):
        generator = rng()
        frame = pl.DataFrame({"x": np.r_[generator.normal(size=90), [100.0] * 10]})

        result = run_core_check(frame, "outliers", methods=["iqr"], warn_share=0.05)

        assert result.findings[0].severity is Severity.WARN

    def test_nulls_and_infinities_are_not_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 500.0]
        frame = pl.DataFrame({"x": [*values, None, float("inf"), float("nan")]})

        result = run_core_check(frame, "outliers")

        column = output_of(result)["columns"]["x"]
        assert column["n_used"] == len(values)
        assert result.findings[0].evidence.n_total == len(values) + 3

    def test_an_integer_column_is_examined(self):
        frame = pl.DataFrame({"n": [*range(100), 100_000]})

        assert codes(run_core_check(frame, "outliers")) == {"outliers_detected"}

    def test_a_column_with_too_few_values_is_skipped_and_named(self):
        frame = pl.DataFrame(
            {"short": [*map(float, range(1, 10)), *[None] * 34], "x": range(43)}
        )

        output = output_of(run_core_check(frame, "outliers"))

        assert output["skipped"] == {"short": "fewer than 10 finite values"}
        assert "x" in output["columns"]

    def test_with_no_column_to_examine_it_is_not_applicable(self):
        result = run_core_check(pl.DataFrame({"x": [1.0, 2.0, 3.0]}), "outliers")

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_a_column_named_with_a_bar_keeps_its_subject(self):
        frame = self.frame().rename({"planted": "a|b"})

        assert by_column(run_core_check(frame, "outliers"))["a|b"]

    def test_the_cutoffs_rise_with_the_number_of_values(self):
        generator = rng()
        values = generator.normal(size=30_000)
        frame = pl.DataFrame(
            {"long": values, "short": [*values[:30], *[None] * 29_970]}
        )

        columns = output_of(run_core_check(frame, "outliers"))["columns"]
        fixed = output_of(run_core_check(frame, "outliers", false_flag_rate=1))[
            "columns"
        ]

        short, long = columns["short"], columns["long"]
        assert 3.0 < short["zscore"]["threshold"] < long["zscore"]["threshold"]
        assert long["zscore"]["threshold"] > 4.5
        assert long["iqr"]["multiplier"] > 2.5
        # Never below the fixed cutoffs, which false_flag_rate=1 uses alone.
        assert fixed["long"]["zscore"]["threshold"] == 3.0
        assert fixed["long"]["iqr"]["multiplier"] == 1.5

    def test_clean_normal_columns_of_any_length_are_rarely_flagged(self):
        generator = rng()
        frame = pl.DataFrame({f"x{i}": generator.normal(size=5000) for i in range(100)})

        def warned(**params):
            result = run_core_check(frame, "outliers", **params)
            return [f for f in result.findings if f.severity is Severity.WARN]

        # About 1 column in 100 is flagged by chance; the fixed cutoffs flag most.
        assert len(warned()) <= 6
        assert len(warned(false_flag_rate=1)) > 50

    def test_a_few_values_far_from_a_long_clean_column_are_still_flagged(self):
        generator = rng()
        values = generator.normal(size=3000)
        values[:3] = [7.0, -7.5, 8.0]

        (finding,) = run_core_check(pl.DataFrame({"x": values}), "outliers").findings

        assert finding.severity is Severity.WARN
        assert finding.evidence.metrics["consensus"] is True

    @pytest.mark.parametrize(
        "bad",
        [
            {"iqr_multiplier": 0},
            {"zscore_threshold": -1},
            {"methods": []},
            {"methods": ["dbscan"]},
            {"min_n": 2},
            {"warn_share": 0},
        ],
    )
    def test_each_parameter_has_bounds(self, bad):
        with pytest.raises(ValidationError):
            OutliersParams(**bad)


class TestNormality:
    def frame(self) -> pl.DataFrame:
        generator = rng()
        return pl.DataFrame(
            {
                "normal": generator.normal(10, 2, size=N),
                "skewed": generator.exponential(size=N) ** 2,
                "constant": [3.0] * N,
                "tiny": [1.0, 2.0, 3.0] + [None] * (N - 3),
            }
        )

    def test_a_normal_column_passes_and_a_skewed_one_is_reported(self):
        found = by_column(run_core_check(self.frame(), "normality"))

        assert "normal" not in found
        (finding,) = found["skewed"]
        assert finding.code == "non_normal"
        assert finding.severity is Severity.INFO
        assert finding.evidence.metrics["primary_test"] == "shapiro_wilk"
        assert finding.evidence.metrics["qq_pattern"] in {"heavy_tails", "right_tail"}

    def test_the_qq_data_is_kept_for_a_failing_column_only_by_default(self):
        columns = output_of(run_core_check(self.frame(), "normality"))["columns"]

        assert columns["normal"]["qq"] is None
        qq = columns["skewed"]["qq"]
        assert len(qq["theoretical"]) == len(qq["empirical"]) == 100
        assert qq["empirical"] == sorted(qq["empirical"])
        assert len(qq["reference_x"]) == len(qq["reference_y"]) == 2

    def test_every_column_gets_qq_data_when_asked_and_the_points_are_bounded(self):
        result = run_core_check(
            self.frame(), "normality", qq_all_columns=True, qq_points=25
        )

        assert len(output_of(result)["columns"]["normal"]["qq"]["empirical"]) == 25

    def test_a_large_column_gets_the_kolmogorov_smirnov_test(self):
        result = run_core_check(self.frame(), "normality", shapiro_max_n=20)

        column = output_of(result)["columns"]["skewed"]
        assert column["primary"]["test"] == "ks_fitted_normal"
        assert column["jarque_bera"]["test"] == "jarque_bera"
        assert by_column(result)["skewed"][0].evidence.method == (
            "ks_fitted_normal+jarque_bera"
        )

    def test_a_nearly_empty_column_is_skipped_with_the_reason(self):
        output = output_of(run_core_check(self.frame(), "normality"))

        assert output["skipped"] == {"tiny": "fewer than 8 finite values"}
        assert "constant" not in output["columns"]  # typed categorical, not read

    def test_with_nothing_to_test_it_is_not_applicable(self):
        frame = pl.DataFrame({"constant": [1.0] * 30})

        assert run_core_check(frame, "normality").status is CheckStatus.NOT_APPLICABLE

    def test_the_alpha_decides_the_verdict(self):
        lenient = run_core_check(
            self.frame(), "normality", alpha=0.9999, min_deviation=0
        )

        assert "normal" in by_column(lenient)

    def test_a_significant_gap_too_small_to_matter_is_not_reported(self):
        generator = rng()
        frame = pl.DataFrame({"almost": generator.standard_t(30, size=50_000)})

        strict = run_core_check(frame, "normality", min_deviation=0)
        default = run_core_check(frame, "normality")

        column = output_of(default)["columns"]["almost"]
        assert column["rejected"] is True and column["non_normal"] is False
        assert column["mean_abs_deviation_in_std"] < 0.05
        assert default.findings == ()
        assert [f.code for f in strict.findings] == ["non_normal"]

    def test_a_normal_column_is_rejected_about_alpha_of_the_time(self):
        generator = rng()
        frame = pl.DataFrame({f"x{i}": generator.normal(size=400) for i in range(200)})

        rejected = output_of(run_core_check(frame, "normality", min_deviation=0))[
            "columns"
        ]

        share = sum(c["rejected"] for c in rejected.values()) / 200
        # Two tests, each held to alpha / 2, so the pair is not above alpha.
        assert share <= 0.08

    @pytest.mark.parametrize(
        "bad",
        [
            {"alpha": 0},
            {"alpha": 1},
            {"qq_points": 5},
            {"shapiro_max_n": 5},
            {"min_deviation": -1},
        ],
    )
    def test_each_parameter_has_bounds(self, bad):
        with pytest.raises(ValidationError):
            NormalityParams(**bad)


class TestBimodality:
    def frame(self) -> pl.DataFrame:
        generator = rng()
        n = 2000
        return pl.DataFrame(
            {
                "two_peaks": np.r_[
                    generator.normal(0, 1, n // 2), generator.normal(8, 1, n // 2)
                ],
                "one_peak": generator.normal(size=n),
                "heavy": generator.standard_t(3, size=n),
                "flat": generator.uniform(0, 100, size=n),
                "skewed": generator.exponential(size=n) ** 2,
                "constant": [1.0] * n,
            }
        )

    def test_two_separate_peaks_are_reported_with_their_centres(self):
        result = run_core_check(self.frame(), "bimodality")

        found = by_column(result)
        assert set(found) == {"two_peaks"}
        (finding,) = found["two_peaks"]
        assert finding.code == "bimodal"
        assert finding.severity is Severity.WARN
        assert finding.evidence.metrics["mean_low"] == pytest.approx(0, abs=0.2)
        assert finding.evidence.metrics["mean_high"] == pytest.approx(8, abs=0.2)
        assert finding.evidence.metrics["weight_low"] == pytest.approx(0.5, abs=0.05)

    def test_a_heavy_tailed_column_is_not_two_peaks_however_much_it_improves_bic(
        self,
    ):
        column = output_of(run_core_check(self.frame(), "bimodality"))["columns"][
            "heavy"
        ]

        assert column["bic_delta"] > 10
        assert column["separation"] < 2
        assert column["bimodal"] is False

    def test_a_flat_column_is_not_two_peaks_though_two_gaussians_fit_it_better(self):
        column = output_of(run_core_check(self.frame(), "bimodality"))["columns"][
            "flat"
        ]

        assert column["bic_delta"] > 10
        assert column["separation"] >= 2
        assert column["valley_ratio"] > 0.75  # the density has no dip
        assert column["bimodal"] is False

    def test_peaks_four_spreads_apart_are_found_in_a_few_hundred_values(self):
        generator = rng()
        found = 0
        for _ in range(10):
            n = 200
            values = np.where(
                generator.random(n) < 0.5,
                generator.normal(0, 1, n),
                generator.normal(4, 1, n),
            )
            result = run_core_check(pl.DataFrame({"x": values}), "bimodality")
            found += output_of(result)["columns"]["x"]["bimodal"]

        assert found >= 9

    def test_two_peaks_have_a_deep_valley_between_them(self):
        column = output_of(run_core_check(self.frame(), "bimodality"))["columns"][
            "two_peaks"
        ]

        assert column["valley_ratio"] < 0.1

    def test_a_skewed_column_is_not_fitted_and_says_why(self):
        skipped = output_of(run_core_check(self.frame(), "bimodality"))["skipped"]

        assert list(skipped) == ["skewed"]
        assert "skewness" in skipped["skewed"]

    def test_the_decision_does_not_depend_on_the_units(self):
        frame = self.frame().select("two_peaks")
        scaled = frame.with_columns((pl.col("two_peaks") * 1000.0).alias("two_peaks"))

        small = output_of(run_core_check(frame, "bimodality"))["columns"]["two_peaks"]
        large = output_of(run_core_check(scaled, "bimodality"))["columns"]["two_peaks"]

        assert small["bimodal"] is large["bimodal"] is True
        assert small["bic_delta"] == pytest.approx(large["bic_delta"], rel=1e-6)
        assert small["separation"] == pytest.approx(large["separation"], rel=1e-6)

    def test_an_integer_column_gets_the_same_bic_as_the_same_values_as_floats(self):
        # scikit-learn mis-scores integer input (and differently in 1.7 and 1.8), so
        # the check always fits floats; the BIC of one component is then the textbook
        # -2 log-likelihood plus 2 ln n.
        generator = rng()
        ints = generator.normal(45, 6, size=2000).round().astype(int)
        as_int = pl.DataFrame({"x": ints})
        as_float = pl.DataFrame({"x": ints.astype(float)})

        by_int = output_of(run_core_check(as_int, "bimodality"))["columns"]["x"]
        by_float = output_of(run_core_check(as_float, "bimodality"))["columns"]["x"]

        n, variance = ints.size, ints.var()
        expected = n * (np.log(2 * np.pi * variance) + 1) + 2 * np.log(n)
        assert by_int["bic_1_component"] == by_float["bic_1_component"]
        assert by_int["bic_1_component"] == pytest.approx(expected, rel=1e-4)

    def test_the_fit_is_repeatable_with_a_seed(self):
        first = output_of(run_core_check(self.frame(), "bimodality", seed=3))
        second = output_of(run_core_check(self.frame(), "bimodality", seed=3))

        assert first == second

    def test_a_large_column_is_sampled_and_the_evidence_says_so(self):
        result = run_core_check(self.frame(), "bimodality", max_rows=1000)

        column = output_of(result)["columns"]["two_peaks"]
        assert (column["n_used"], column["n_finite"]) == (1000, 2000)
        assert by_column(result)["two_peaks"][0].evidence.n_used == 1000
        assert by_column(result)["two_peaks"][0].evidence.metrics["n_finite"] == 2000

    def test_without_the_ml_extra_it_is_unavailable_with_an_install_hint(
        self, without_sklearn
    ):
        result = run_core_check(self.frame(), "bimodality")

        assert result.status is CheckStatus.UNAVAILABLE
        assert "dsbf[ml]" in "".join(n.message + (n.hint or "") for n in result.notes)

    @pytest.mark.parametrize(
        "bad",
        [
            {"min_weight": 0.5},
            {"min_separation": 0},
            {"max_rows": 10},
            {"max_valley_ratio": 0},
        ],
    )
    def test_each_parameter_has_bounds(self, bad):
        with pytest.raises(ValidationError):
            BimodalityParams(**bad)


class TestMultivariateOutliers:
    def frame(self, rows: int = 2000) -> pl.DataFrame:
        generator = rng()
        x = generator.uniform(-1, 1, size=rows)
        y = x + generator.normal(scale=0.05, size=rows)
        x[7], y[7] = 0.9, -0.9  # inside each column's range, off the diagonal
        return pl.DataFrame({"x": x, "y": y, "label": ["a"] * rows})

    def test_a_row_off_the_joint_pattern_ranks_first(self):
        result = run_core_check(self.frame(), "multivariate_outliers")

        output = output_of(result)
        assert 7 in output["example_rows"][:3]
        assert output["feature_columns"] == ["x", "y"]
        assert output["n_flagged"] == pytest.approx(0.05 * 2000, abs=2)
        (finding,) = result.findings
        assert finding.code == "multivariate_anomalies"
        assert finding.severity is Severity.INFO
        assert finding.subject.kind == "row_set"
        assert finding.subject.columns == ("x", "y")

    def test_a_larger_frame_is_sampled_and_the_rows_stay_positions_in_the_data(self):
        result = run_core_check(
            self.frame(3000), "multivariate_outliers", max_rows=1000
        )

        output = output_of(result)
        assert (output["rows_used"], output["rows_total"]) == (1000, 3000)
        assert all(0 <= row < 3000 for row in output["example_rows"])

    def test_clean_normal_columns_raise_no_finding(self):
        generator = rng()
        frame = pl.DataFrame({f"x{i}": generator.normal(size=3000) for i in range(4)})

        result = run_core_check(frame, "multivariate_outliers")

        output = output_of(result)
        assert result.findings == ()
        # The forest still ranks the share it was asked to; that is not a finding.
        assert output["n_flagged"] > 100
        assert output["far_rows"] == 0
        assert output["excess"] < 0.03

    def test_rows_that_break_a_correlation_are_far_though_ordinary_in_each_column(
        self,
    ):
        generator = rng()
        n = 3000
        x = generator.normal(size=n)
        y = 0.9 * x + np.sqrt(0.19) * generator.normal(size=n)
        broken = generator.choice(n, 30, replace=False)
        x[broken], y[broken] = 1.8, -1.8
        frame = pl.DataFrame({"x": x, "y": y, "z": generator.normal(size=n)})

        result = run_core_check(frame, "multivariate_outliers")

        output = output_of(result)
        assert output["far_rows"] == 30
        assert set(output["example_rows"]) <= set(broken.tolist())
        (finding,) = result.findings
        assert finding.confidence.value == "moderate"
        assert finding.evidence.method == "robust_mahalanobis+isolation_forest"
        assert not run_core_check(frame, "outliers").findings

    def test_far_rows_do_not_stretch_the_ruler_that_measures_them(self):
        generator = rng()
        n = 3000
        x = generator.normal(size=n)
        y = 0.9 * x + np.sqrt(0.19) * generator.normal(size=n)
        broken = generator.choice(n, 90, replace=False)
        x[broken], y[broken] = 2.2, -2.2
        frame = pl.DataFrame({"x": x, "y": y})

        output = output_of(run_core_check(frame, "multivariate_outliers"))

        assert output["far_rows"] == 90

    def test_skewed_and_heavy_tailed_columns_do_not_make_rows_far(self):
        generator = rng()
        n = 2000
        shared = (
            generator.normal(size=(n, 4))
            @ np.linalg.cholesky(np.full((4, 4), 0.5) + 0.5 * np.eye(4)).T
        )
        frame = pl.DataFrame(
            {
                "lognormal_a": np.exp(shared[:, 0]),
                "lognormal_b": np.exp(shared[:, 1]),
                "cubed_a": shared[:, 2] ** 3,
                "cubed_b": shared[:, 3] ** 3,
            }
        )

        result = run_core_check(frame, "multivariate_outliers")

        assert output_of(result)["far_rows"] == 0

    def test_with_too_few_rows_per_column_no_row_is_called_far(self):
        generator = rng()
        frame = pl.DataFrame({f"x{i}": generator.normal(size=60) for i in range(6)})
        frame = frame.with_columns(
            pl.when(pl.int_range(60) == 0)
            .then(30.0)
            .otherwise(pl.col("x0"))
            .alias("x0")
        )

        output = output_of(run_core_check(frame, "multivariate_outliers"))

        assert output["far_rows"] == 0 and output["far_limit"] is None

    def test_a_column_that_repeats_another_costs_no_degree_of_freedom(self):
        generator = rng()
        x = generator.normal(size=2000)
        frame = pl.DataFrame(
            {"x": x, "copy": x * 3 + 1, "y": generator.normal(size=2000)}
        )

        result = run_core_check(frame, "multivariate_outliers")

        assert result.status is CheckStatus.OK
        assert output_of(result)["far_rows"] == 0

    def test_a_column_that_is_mostly_missing_is_not_a_feature(self):
        generator = rng()
        n = 2000
        mostly = generator.normal(5, 1, n)
        mostly[generator.random(n) < 0.9] = np.nan
        frame = pl.DataFrame(
            {
                "a": generator.normal(size=n),
                "b": generator.normal(size=n),
                "mostly_missing": mostly,
            }
        )

        result = run_core_check(frame, "multivariate_outliers")

        assert output_of(result)["feature_columns"] == ["a", "b"]
        assert [n.code for n in result.notes] == ["features_mostly_missing"]
        # Standing the median in for 90% of a column made the other rows "far".
        assert result.findings == ()
        loose = run_core_check(frame, "multivariate_outliers", max_missing_share=1.0)
        assert output_of(loose)["feature_columns"] == ["a", "b", "mostly_missing"]

    def test_the_forest_is_repeatable_with_a_seed(self):
        first = output_of(run_core_check(self.frame(), "multivariate_outliers"))
        second = output_of(run_core_check(self.frame(), "multivariate_outliers"))

        assert first == second

    def test_nulls_are_filled_with_the_median_and_the_rows_kept(self):
        frame = self.frame().with_columns(
            pl.when(pl.int_range(pl.len()) % 50 == 0)
            .then(None)
            .otherwise(pl.col("y"))
            .alias("y")
        )

        output = output_of(run_core_check(frame, "multivariate_outliers"))

        assert output["rows_used"] == 2000

    def test_one_numeric_column_is_not_enough(self):
        frame = self.frame().select("x", "label")

        result = run_core_check(frame, "multivariate_outliers")

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_without_the_ml_extra_it_is_unavailable_and_the_univariate_check_runs(
        self, without_sklearn
    ):
        assert (
            run_core_check(self.frame(), "multivariate_outliers").status
            is CheckStatus.UNAVAILABLE
        )
        assert run_core_check(self.frame(), "outliers").status is CheckStatus.OK


class TestOutOfBounds:
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "age": [25, 40, -3, 130, 61, 200] * 10,
                "score": [0.1, 0.5, 0.9, 1.4, None, 0.3] * 10,
                "name": ["a", "b", "c", "d", "e", "f"] * 10,
            }
        )

    def test_without_bounds_it_is_not_applicable(self):
        result = run_core_check(self.frame(), "out_of_bounds")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert [n.code for n in result.notes] == ["no_bounds"]

    def test_values_on_both_sides_are_counted_with_the_extremes(self):
        result = run_core_check(
            self.frame(), "out_of_bounds", bounds={"age": {"min": 0, "max": 120}}
        )

        (finding,) = result.findings
        assert finding.code == "out_of_bounds"
        assert finding.severity is Severity.WARN
        assert finding.evidence.metrics["below_count"] == 10
        assert finding.evidence.metrics["above_count"] == 20
        assert finding.evidence.metrics["lowest_below"] == -3
        assert finding.evidence.metrics["highest_above"] == 200
        assert finding.evidence.n_used == 60

    def test_nulls_are_not_values_and_a_bound_may_be_open(self):
        result = run_core_check(
            self.frame(), "out_of_bounds", bounds={"score": {"max": 1}}
        )

        assert output_of(result)["columns"]["score"] == {
            "below": 0,
            "above": 10,
            "lowest": None,
            "highest": 1.4,
        }
        assert result.findings[0].evidence.n_used == 50

    def test_a_column_inside_its_bounds_has_no_finding(self):
        result = run_core_check(
            self.frame(), "out_of_bounds", bounds={"age": {"min": -10, "max": 500}}
        )

        assert not result.findings
        assert output_of(result)["columns"]["age"]["above"] == 0

    def test_infinity_is_above_any_maximum(self):
        frame = pl.DataFrame({"x": [*np.linspace(0, 5, 40), float("inf")]})

        result = run_core_check(frame, "out_of_bounds", bounds={"x": {"max": 10}})

        assert result.findings[0].evidence.metrics["above_count"] == 1

    def test_bounds_on_a_missing_or_non_numeric_column_are_noted_not_checked(self):
        result = run_core_check(
            self.frame(),
            "out_of_bounds",
            bounds={"name": {"min": 0}, "height": {"max": 3}},
        )

        assert output_of(result)["unchecked"] == {
            "name": "not a numeric column of the data",
            "height": "not a numeric column of the data",
        }
        assert [n.code for n in result.notes] == ["bounds_not_checked"] * 2

    @pytest.mark.parametrize("bad", [{}, {"min": 5, "max": 1}])
    def test_a_bound_needs_an_end_and_they_must_be_ordered(self, bad):
        with pytest.raises(ValidationError):
            Bounds(**bad)

    def test_the_bounds_are_validated_as_parameters(self):
        with pytest.raises(ValidationError):
            OutOfBoundsParams(bounds={"x": {"min": 3, "max": 1}})


class TestTips:
    """The advice attached to each finding follows the numbers in it."""

    def methods(self, finding) -> set[str]:
        return {a.method or a.action for a in actions_of(finding)}

    def test_agreeing_outlier_rules_get_capping_and_robust_scaling(self):
        generator = rng()
        planted = generator.normal(size=N)
        planted[:3] = [40.0, -35.0, 55.0]

        (finding,) = run_core_check(pl.DataFrame({"x": planted}), "outliers").findings

        assert self.methods(finding) == {"iqr_fences", "robust_scaler"}

    def test_a_lone_rule_gets_advice_to_verify_not_to_cap(self):
        generator = rng()
        planted = generator.normal(size=N)
        planted[:3] = [40.0, -35.0, 55.0]

        result = run_core_check(
            pl.DataFrame({"x": planted}), "outliers", methods=["iqr"]
        )

        assert self.methods(result.findings[0]) == {"robust_scaler", "investigate"}

    def test_a_non_normal_column_with_negative_values_gets_yeo_johnson(self):
        generator = rng()
        frame = pl.DataFrame({"x": generator.exponential(size=N) ** 2 - 1.0})

        (finding,) = run_core_check(frame, "normality").findings

        assert self.methods(finding) == {"yeo_johnson", "monitor"}

    def test_a_positive_non_normal_column_gets_box_cox(self):
        generator = rng()
        frame = pl.DataFrame({"x": generator.exponential(size=N) ** 2 + 1.0})

        (finding,) = run_core_check(frame, "normality").findings

        assert self.methods(finding) == {"box_cox", "monitor"}

    def test_two_cleanly_separate_peaks_get_a_cluster_feature(self):
        generator = rng()
        values = np.r_[generator.normal(0, 1, 1000), generator.normal(9, 1, 1000)]

        (finding,) = run_core_check(pl.DataFrame({"x": values}), "bimodality").findings

        assert self.methods(finding) == {
            "segment",
            "gmm_cluster_indicator",
            "use_tree_model",
        }

    def test_out_of_bounds_values_get_the_three_ways_to_deal_with_them(self):
        frame = pl.DataFrame({"x": [*np.linspace(0, 5, 40), 99.0]})

        result = run_core_check(frame, "out_of_bounds", bounds={"x": {"max": 10}})

        assert self.methods(result.findings[0]) == {
            "investigate",
            "clip_to_bounds",
            "remove_rows",
        }


class TestOverflowingValues:
    """Values near the limit of a float leave a column unexamined, never a NaN."""

    def test_normality_skips_a_column_whose_tests_overflow(self):
        frame = pl.DataFrame({"x": [0.0] * 6 + [1.0, 3.5e77]})

        result = run_core_check(frame, "normality")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert not result.findings

    def test_outliers_skips_a_column_whose_fences_overflow(self):
        huge = np.finfo(np.float64).max
        frame = pl.DataFrame(
            {
                "x": [-huge, huge, huge / 2, -huge / 2] * 10,
                "fine": np.linspace(0, 1, 40),
            }
        )

        result = run_core_check(frame, "outliers")

        output = output_of(result)
        assert output["skipped"] == {"x": "values too large to compute with"}
        assert list(output["columns"]) == ["fine"]

    def test_outliers_skips_a_column_whose_z_scores_overflow(self):
        huge = np.finfo(np.float64).max
        frame = pl.DataFrame({"x": [0.0] * 8 + [-huge / 2e5, -huge]})

        result = run_core_check(frame, "outliers")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert not [n for n in result.notes if n.code == "check_failed"]

    def test_the_forest_rescales_columns_so_huge_values_do_not_stop_it(self):
        generator = rng()
        frame = pl.DataFrame(
            {
                "huge": np.r_[generator.normal(size=N) * 1e100, [1e300]],
                "y": np.arange(N + 1.0),
            }
        )

        result = run_core_check(frame, "multivariate_outliers")

        assert result.status is CheckStatus.OK
        assert output_of(result)["feature_columns"] == ["huge", "y"]

    def test_a_column_whose_range_overflows_is_left_out_and_named(self):
        generator = rng()
        huge = np.finfo(np.float64).max
        frame = pl.DataFrame(
            {
                "wide": np.r_[[-huge, huge], generator.normal(size=N)],
                "a": generator.normal(size=N + 2),
                "b": generator.normal(size=N + 2),
            }
        )

        result = run_core_check(frame, "multivariate_outliers")

        assert output_of(result)["feature_columns"] == ["a", "b"]
        assert [n.code for n in result.notes] == ["feature_dropped"]

    def test_only_infinite_violations_leave_the_extremes_empty(self):
        frame = pl.DataFrame({"x": [*np.linspace(0, 5, 40), float("inf")]})

        result = run_core_check(frame, "out_of_bounds", bounds={"x": {"max": 10}})

        assert output_of(result)["columns"]["x"]["highest"] is None
        assert result.findings[0].evidence.metrics["above_count"] == 1
