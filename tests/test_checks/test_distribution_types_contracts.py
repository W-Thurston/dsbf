"""The shape, type and contract rule checks on small frames."""

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks import core_pack
from dsbf.checks.skewness import SkewnessParams
from dsbf.data import load
from dsbf.engine import PlanError, Registry, Selection, make_plan
from dsbf.model import CheckStatus, Confidence, Severity
from dsbf.profiling import profile_dataset
from tests.test_checks.support import (
    actions_of,
    by_column,
    output_of,
    run_core_check,
)

pytestmark = pytest.mark.unit

N = 400


def codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def rng() -> np.random.Generator:
    return np.random.default_rng(7)


class TestSkewness:
    def frame(self) -> pl.DataFrame:
        generator = rng()
        return pl.DataFrame(
            {
                "symmetric": generator.normal(size=N),
                "right": generator.exponential(size=N) ** 2 + 1.0,
                "mild": generator.gamma(shape=12.0, size=N),
                "left": -(generator.exponential(size=N) ** 2) - 1.0,
                "constant": [4.0] * N,
            }
        )

    def test_the_level_follows_the_size_of_the_skew(self):
        found = by_column(run_core_check(self.frame(), "skewness"))

        assert found["right"][0].severity is Severity.WARN
        assert found["right"][0].evidence.metrics["skewness"] > 2
        assert "symmetric" not in found
        assert "constant" not in found

    def test_a_mild_skew_is_info_and_a_moderate_or_heavy_one_a_warning(self):
        generator = rng()
        frame = pl.DataFrame(
            {
                "mild": generator.gamma(shape=8.0, size=4000),  # skew 0.7
                "moderate": generator.gamma(shape=3.0, size=4000),  # skew 1.2
                "heavy": generator.gamma(shape=0.4, size=4000),  # skew 3.2
            }
        )

        found = by_column(run_core_check(frame, "skewness"))

        assert {
            c: (f[0].severity, f[0].title.split()[0]) for c, f in found.items()
        } == {
            "mild": (Severity.INFO, "Mild"),
            "moderate": (Severity.WARN, "Moderate"),
            "heavy": (Severity.WARN, "Heavy"),
        }

    def test_the_thresholds_are_parameters(self):
        found = by_column(
            run_core_check(
                self.frame(), "skewness", mild=50.0, moderate=60.0, heavy=70.0
            )
        )

        assert "skewed" not in {f.code for fs in found.values() for f in fs}

    def test_a_constant_column_is_not_read(self):
        output = output_of(run_core_check(self.frame(), "skewness"))

        assert "constant" not in output["columns"]

    def test_the_mean_median_gap_is_reported_in_standard_deviations(self):
        generator = rng()
        values = np.concatenate(
            [generator.normal(1, 0.1, 60), generator.normal(50, 1, 40)]
        )
        found = by_column(run_core_check(pl.DataFrame({"a": values}), "skewness"))

        gap = next(f for f in found["a"] if f.code == "mean_median_gap")
        metrics = gap.evidence.metrics
        assert metrics["gap_in_std"] == pytest.approx(
            abs(metrics["mean"] - metrics["median"]) / metrics["std"]
        )
        assert gap.severity is Severity.INFO

    def test_the_gap_levels_are_parameters(self):
        generator = rng()
        values = np.concatenate(
            [generator.normal(1, 0.1, 60), generator.normal(50, 1, 40)]
        )
        frame = pl.DataFrame({"a": values})

        found = by_column(
            run_core_check(frame, "skewness", gap_notice=0.1, gap_warn=0.3)
        )

        gap = next(f for f in found["a"] if f.code == "mean_median_gap")
        assert gap.severity is Severity.WARN

    def test_a_column_that_barely_varies_has_no_gap_finding(self):
        values = [0.0] * 90 + [0.001 * i for i in range(10)]
        found = by_column(run_core_check(pl.DataFrame({"a": values}), "skewness"))

        assert "mean_median_gap" not in {f.code for f in found.get("a", [])}

    def test_a_right_skew_with_no_negative_values_is_advised_a_log_transform(self):
        found = by_column(run_core_check(self.frame(), "skewness"))

        methods = {a.method for a in actions_of(found["right"][0])}
        assert {"log1p", "box_cox", "yeo_johnson"} <= methods
        assert "reflect_then_log" not in methods

    def test_a_left_skew_is_advised_to_reflect_it(self):
        found = by_column(run_core_check(self.frame(), "skewness"))

        methods = {a.method for a in actions_of(found["left"][0])}
        assert "reflect_then_log" in methods
        assert "log1p" not in methods

    def test_a_finding_carries_n_used_and_the_method(self):
        finding = run_core_check(self.frame(), "skewness").findings[0]

        assert finding.evidence.n_used == N
        assert finding.evidence.method in {"sample_skewness", "mean_median_gap"}

    def test_the_thresholds_must_be_ordered(self):
        with pytest.raises(ValidationError):
            SkewnessParams(mild=2.0, moderate=1.0)
        with pytest.raises(ValidationError):
            SkewnessParams(gap_notice=2.0, gap_warn=1.0)

    def test_it_is_not_applicable_without_a_numeric_column(self):
        result = run_core_check(pl.DataFrame({"a": ["x", "y"] * 20}), "skewness")

        assert result.status is CheckStatus.NOT_APPLICABLE


class TestKurtosis:
    def frame(self) -> pl.DataFrame:
        generator = rng()
        return pl.DataFrame(
            {
                "normal": generator.normal(size=N),
                "heavy": generator.standard_t(df=3, size=N),
                "uniform": generator.uniform(size=N),
                "few": [1.0, 2.0, 3.0] * 100 + [4.0] * 100,
            }
        )

    def test_a_heavy_tailed_column_is_a_warning(self):
        found = by_column(run_core_check(self.frame(), "kurtosis"))

        assert found["heavy"][0].code == "heavy_tails"
        assert found["heavy"][0].severity is Severity.WARN
        assert "normal" not in found

    def test_a_uniform_column_has_thin_tails(self):
        found = by_column(run_core_check(self.frame(), "kurtosis"))

        assert found["uniform"][0].code == "thin_tails"
        assert found["uniform"][0].severity is Severity.INFO

    def test_every_numeric_column_is_in_the_output_with_its_tails(self):
        output = output_of(run_core_check(self.frame(), "kurtosis"))

        assert output["columns"]["normal"]["tails"] == "normal"
        assert output["columns"]["heavy"]["tails"] == "heavy"
        assert output["columns"]["uniform"]["tails"] == "thin"

    def test_fewer_than_four_values_have_no_kurtosis(self):
        result = run_core_check(pl.DataFrame({"a": [1.0, 2.0, 5.0]}), "kurtosis")

        assert output_of(result)["columns"]["a"] == {"kurtosis": None, "tails": None}
        assert not result.findings

    def test_heavy_tails_are_advised_to_be_capped(self):
        found = by_column(run_core_check(self.frame(), "kurtosis"))

        assert "winsorize" in {a.action for a in actions_of(found["heavy"][0])}

    def test_the_thresholds_must_be_ordered(self):
        with pytest.raises(PlanError):
            run_core_check(self.frame(), "kurtosis", mild=4.0, strong=3.0)


class TestEntropy:
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "even": ["a", "b", "c", "d"] * 25,
                "skewed": ["a"] * 85 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5,
                "one": ["a"] * 100,
                "flag": [True, False] * 50,
                "number": list(range(100)),
            }
        )

    def test_an_even_column_has_the_maximum_entropy(self):
        output = output_of(run_core_check(self.frame(), "entropy"))

        even = output["columns"]["even"]
        assert even["entropy_bits"] == pytest.approx(2.0)
        assert even["max_bits"] == pytest.approx(2.0)
        assert even["evenness"] == pytest.approx(1.0)

    def test_a_skewed_column_has_less_than_the_maximum(self):
        output = output_of(run_core_check(self.frame(), "entropy"))

        assert 0 < output["columns"]["skewed"]["evenness"] < 0.5

    def test_a_column_of_one_value_has_none_and_is_not_uneven(self):
        output = output_of(run_core_check(self.frame(), "entropy"))

        assert output["columns"]["one"]["entropy_bits"] == 0
        assert output["columns"]["one"]["evenness"] == 1.0

    def test_categorical_and_boolean_columns_are_read_and_numbers_are_not(self):
        output = output_of(run_core_check(self.frame(), "entropy"))

        assert set(output["columns"]) == {"even", "skewed", "one", "flag"}
        assert output["columns"]["flag"]["entropy_bits"] == pytest.approx(1.0)

    def test_it_raises_no_finding(self):
        assert not run_core_check(self.frame(), "entropy").findings

    def test_a_small_sample_is_noted(self):
        result = run_core_check(pl.DataFrame({"a": ["x", "y", "x"]}), "entropy")

        assert [n.code for n in result.notes] == ["small_sample"]

    def test_a_sample_of_enough_rows_is_not(self):
        assert not run_core_check(self.frame(), "entropy").notes


class TestDtypeOptimizations:
    def frame(self) -> pl.DataFrame:
        generator = rng()
        n = 5000
        return pl.DataFrame(
            {
                "small_int": generator.integers(0, 100_000, n) % 120 + 1000 * 0,
                "wide_int": generator.integers(0, 2_000_000_000, n),
                "mid_int": generator.integers(-30_000, 30_000, n),
                "flag_int": generator.integers(0, 2, n),
                "yes_no": generator.choice(["yes", "no"], n),
                "city": generator.choice([f"city_{i:03d}" for i in range(40)], n),
                "measure": generator.normal(size=n),
            },
            schema_overrides={"small_int": pl.Int64, "wide_int": pl.Int64},
        )

    def suggestions(self, **params) -> dict:
        result = run_core_check(self.frame(), "dtype_optimizations", **params)
        return output_of(result)["suggestions"]

    def test_an_integer_is_narrowed_to_the_smallest_type_that_holds_its_range(self):
        found = self.suggestions()

        assert found["small_int"]["suggested_dtype"] == "Int8"
        assert found["mid_int"]["suggested_dtype"] == "Int16"
        assert found["wide_int"]["suggested_dtype"] == "Int32"

    def test_an_integer_of_zero_and_one_is_a_boolean(self):
        found = self.suggestions()

        assert found["flag_int"]["suggested_dtype"] == "Boolean"
        assert found["flag_int"]["reason"] == "fits_bool"

    def test_a_string_of_two_truthy_words_is_a_boolean(self):
        assert self.suggestions()["yes_no"]["suggested_dtype"] == "Boolean"

    def test_a_repeating_string_is_a_dictionary(self):
        found = self.suggestions()["city"]

        assert found["suggested_dtype"] == "Categorical"
        assert found["estimated_saving_bytes"] > 0

    def test_a_float_is_never_suggested(self):
        assert "measure" not in self.suggestions()

    def test_a_saving_below_the_minimum_is_not_reported(self):
        found = self.suggestions(min_saving_bytes=10**9)

        assert found == {}

    def test_a_saving_is_the_width_difference_times_the_rows(self):
        assert self.suggestions()["small_int"]["estimated_saving_bytes"] == 7 * 5000

    def test_the_output_totals_the_savings_and_each_has_a_finding(self):
        result = run_core_check(self.frame(), "dtype_optimizations")

        output = output_of(result)
        assert output["total_saving_bytes"] == sum(
            s["estimated_saving_bytes"] for s in output["suggestions"].values()
        )
        assert {f.subject.columns[0] for f in result.findings} == set(
            output["suggestions"]
        )
        assert {f.confidence for f in result.findings} == {Confidence.MODERATE}

    def test_codes_the_profile_lists_in_full_are_narrowed_too(self):
        # `pclass`-style codes are typed categorical, which has no min and max.
        frame = pl.DataFrame({"pclass": [1, 2, 3] * 2000}, schema={"pclass": pl.Int64})

        found = output_of(run_core_check(frame, "dtype_optimizations"))["suggestions"]

        assert found["pclass"]["suggested_dtype"] == "Int8"

    def test_codes_the_profile_does_not_list_in_full_are_left_alone(self):
        values = [i % 15 for i in range(30_000)]
        frame = pl.DataFrame({"code": values}, schema={"code": pl.Int64})

        found = output_of(run_core_check(frame, "dtype_optimizations"))["suggestions"]

        assert "code" not in found

    def test_a_downcast_is_advised_with_its_caution(self):
        result = run_core_check(self.frame(), "dtype_optimizations")

        finding = by_column(result)["small_int"][0]
        assert "overflow" in finding.message
        assert {a.action for a in actions_of(finding)} == {"convert"}


class TestSampleSize:
    def frame(self, rows: int, features: int = 5) -> pl.DataFrame:
        generator = rng()
        return pl.DataFrame(
            {f"c{i}": generator.normal(size=rows) for i in range(features)}
        )

    def test_enough_rows_is_adequate_with_no_finding(self):
        result = run_core_check(self.frame(2000), "sample_size")

        assert output_of(result)["verdict"] == "adequate"
        assert not result.findings

    def test_a_shortfall_is_a_finding_per_failing_rule_about_the_dataset(self):
        result = run_core_check(self.frame(120), "sample_size")

        assert codes(result) == {"too_few_rows_general_ml"}
        assert {f.subject.kind for f in result.findings} == {"dataset"}
        assert output_of(result)["verdict"] == "limited"

    def test_only_the_linear_rule_is_a_warning(self):
        result = run_core_check(self.frame(60), "sample_size")

        severity = {f.code: f.severity for f in result.findings}
        assert severity["too_few_rows_linear_model"] is Severity.WARN
        assert severity["too_few_rows_tree_model"] is Severity.INFO

    def test_neither_linear_nor_tree_passing_is_insufficient(self):
        result = run_core_check(self.frame(20), "sample_size")

        assert output_of(result)["verdict"] == "insufficient"

    def test_the_required_rows_are_the_larger_of_the_ratio_and_the_floor(self):
        rules = output_of(run_core_check(self.frame(60, features=10), "sample_size"))[
            "rules"
        ]

        assert rules["linear_model"]["required_rows"] == 200
        assert rules["tree_model"]["required_rows"] == 100
        assert rules["general_ml"]["required_rows"] == 1000
        assert rules["linear_model"]["shortfall"] == 140

    def test_identifier_and_datetime_columns_are_not_features(self):
        frame = self.frame(500, features=3).with_columns(
            pl.Series("id", [f"id-{i:06d}-{'x' * 30}" for i in range(500)]),
            pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2021, 5, 15), "1d", eager=True
            )[:500].alias("when"),
        )

        output = output_of(run_core_check(frame, "sample_size"))

        assert output["n_features"] <= 4

    def test_the_feature_count_can_be_given(self):
        result = run_core_check(
            self.frame(300, features=2), "sample_size", n_features=50
        )

        assert output_of(result)["n_features"] == 50
        assert "too_few_rows_linear_model" in codes(result)

    def test_a_finding_states_its_numbers(self):
        finding = run_core_check(self.frame(60), "sample_size").findings[0]

        assert finding.evidence.metrics["rows"] == 60
        assert finding.evidence.metrics["n_features"] == 5
        assert finding.evidence.n_used == 60
        assert {a.action for a in actions_of(finding)} >= {
            "collect_more_data",
            "reduce_features",
        }


class TestClassImbalance:
    def frame(self, share: float = 0.95, rows: int = 1000) -> pl.DataFrame:
        majority = int(rows * share)
        return pl.DataFrame(
            {
                "y": ["no"] * majority + ["yes"] * (rows - majority),
                "x": np.linspace(0, 1, rows),
                "f": [True] * (rows // 2) + [False] * (rows - rows // 2),
            }
        )

    def test_without_a_target_it_is_not_applicable(self):
        result = run_core_check(self.frame(), "class_imbalance")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert [n.code for n in result.notes] == ["no_target"]

    def test_an_imbalanced_target_is_a_warning_with_its_numbers(self):
        result = run_core_check(self.frame(0.95), "class_imbalance", target="y")

        (finding,) = result.findings
        assert finding.severity is Severity.WARN
        assert finding.subject.columns == ("y",)
        assert finding.evidence.metrics["majority_share"] == pytest.approx(0.95)
        assert finding.evidence.metrics["majority_class"] == "no"
        assert finding.evidence.metrics["imbalance_ratio"] == pytest.approx(19.0)
        assert finding.evidence.n_used == 1000

    def test_a_balanced_target_has_an_output_and_no_finding(self):
        result = run_core_check(self.frame(0.6), "class_imbalance", target="y")

        assert not result.findings
        output = output_of(result)
        assert output["imbalanced"] is False
        assert output["n_classes"] == 2
        assert output["minority_class"] == "yes"

    def test_the_threshold_is_a_parameter(self):
        result = run_core_check(
            self.frame(0.8), "class_imbalance", target="y", majority_share=0.75
        )

        assert codes(result) == {"class_imbalance"}

    def test_a_boolean_target_is_read(self):
        frame = pl.DataFrame({"y": [True] * 97 + [False] * 3})

        result = run_core_check(frame, "class_imbalance", target="y")

        assert codes(result) == {"class_imbalance"}
        assert output_of(result)["majority_class"] == "True"

    def test_the_minority_is_unknown_when_the_profile_lists_too_few_values(self):
        classes = [f"c{i}" for i in range(30)]
        frame = pl.DataFrame({"y": ["c0"] * 3000 + classes * 5})

        result = run_core_check(frame, "class_imbalance", target="y")

        output = output_of(result)
        assert output["n_classes"] == 30
        assert output["minority_class"] is None
        assert output["imbalance_ratio"] is None

    def test_a_continuous_target_is_not_applicable_with_a_note(self):
        result = run_core_check(self.frame(), "class_imbalance", target="x")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert [n.code for n in result.notes] == ["target_not_a_class"]

    def test_a_target_that_is_not_a_column_is_not_applicable(self):
        result = run_core_check(self.frame(), "class_imbalance", target="nope")

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_the_advice_is_weights_resampling_and_a_better_metric(self):
        result = run_core_check(self.frame(0.97), "class_imbalance", target="y")

        assert {a.action for a in actions_of(result.findings[0])} == {
            "class_weight",
            "resample",
            "change_metric",
        }

    def test_the_share_must_be_a_share(self):
        with pytest.raises(PlanError):
            run_core_check(self.frame(), "class_imbalance", majority_share=1.5)


class TestSchema:
    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "age": [21, 35, None, 51, 140, -3] * 10,
                "sex": ["m", "f", "f", "x", "m", "f"] * 10,
                "score": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5] * 10,
            }
        )

    def run(self, columns: dict | None = None, **extra):
        params = {"columns": columns} if columns is not None else {}
        return run_core_check(self.frame(), "schema", **params, **extra)

    def test_without_a_schema_it_is_not_applicable(self):
        result = self.run()

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert [n.code for n in result.notes] == ["no_schema"]

    def test_a_schema_the_data_meets_is_valid_with_no_finding(self):
        result = self.run(
            {"age": {"type": "numeric"}, "sex": {"allowed": ["m", "f", "x"]}}
        )

        assert not result.findings
        assert output_of(result)["valid"] is True
        assert output_of(result)["checked"] == 2

    def test_a_required_column_that_is_missing_is_one_error_about_the_dataset(self):
        result = self.run({"age": {}, "height": {}, "weight": {}})

        (finding,) = result.findings
        assert finding.code == "missing_column"
        assert finding.severity is Severity.ERROR
        assert finding.subject.kind == "dataset"
        assert finding.evidence.metrics["missing"] == ("height", "weight")
        assert output_of(result)["missing_columns"] == ["height", "weight"]
        assert output_of(result)["valid"] is False

    def test_an_optional_column_may_be_missing(self):
        result = self.run({"height": {"required": False}})

        assert not result.findings
        assert output_of(result)["valid"] is True

    def test_a_wrong_type_is_reported_with_both_names(self):
        found = by_column(self.run({"sex": {"type": "numeric", "dtype": "Int64"}}))

        assert {f.code for f in found["sex"]} == {"type_mismatch", "dtype_mismatch"}
        mismatch = next(f for f in found["sex"] if f.code == "dtype_mismatch")
        assert mismatch.evidence.metrics["actual"] == "String"

    def test_the_dtype_is_compared_without_case(self):
        assert not self.run({"sex": {"dtype": "string"}}).findings

    def test_nulls_in_a_column_that_allows_none_are_counted(self):
        found = by_column(self.run({"age": {"nullable": False}}))

        assert found["age"][0].code == "unexpected_nulls"
        assert found["age"][0].evidence.metrics["null_count"] == 10

    def test_values_outside_a_range_are_counted_on_each_side(self):
        found = by_column(self.run({"age": {"min": 0, "max": 120}}))

        by_code = {f.code: f for f in found["age"]}
        assert by_code["below_min"].evidence.metrics["count"] == 10
        assert by_code["above_max"].evidence.metrics["count"] == 10
        assert by_code["above_max"].evidence.metrics["observed"] == 140

    def test_a_range_the_data_stays_inside_is_not_read(self):
        assert not self.run({"score": {"min": 0, "max": 6}}).findings

    def test_values_outside_the_allowed_set_are_listed(self):
        found = by_column(self.run({"sex": {"allowed": ["m", "f"]}}))

        finding = found["sex"][0]
        assert finding.code == "unexpected_values"
        assert finding.evidence.metrics["examples"] == ("x",)
        assert finding.evidence.metrics["rows"] == 10

    def test_a_range_on_a_text_column_is_noted_not_checked(self):
        result = self.run({"sex": {"min": 0}})

        assert not result.findings
        assert [n.code for n in result.notes] == ["range_on_non_numeric"]

    def test_the_violations_are_listed_by_column(self):
        result = self.run({"age": {"nullable": False, "max": 120}, "sex": {}})

        assert output_of(result)["violations"] == {
            "age": ["unexpected_nulls", "above_max"]
        }

    def test_an_unknown_key_is_refused_at_planning(self):
        for params in (
            {"columns": {"age": {"minimum": 0}}},
            {"required_columns": ["age"]},
        ):
            with pytest.raises(PlanError):
                run_core_check(self.frame(), "schema", **params)

    def test_a_minimum_above_the_maximum_is_refused(self):
        with pytest.raises(PlanError):
            self.run({"age": {"min": 5, "max": 1}})

    def test_the_check_reads_values_so_it_is_an_analysis_check(self):
        check = {c.id: c for c in core_pack().checks}["schema"]

        assert check.spec.kind == "analysis"
        assert check.spec.cost.value == "cheap"


class TestTheTargetSetting:
    def test_the_plan_gives_the_target_to_a_check_that_declares_one(self):
        frame = pl.DataFrame({"y": ["a"] * 95 + ["b"] * 5, "x": range(100)})
        dataset = load(frame, name="frame")
        plan = make_plan(
            Registry(core_pack().checks),
            Selection(include="class_imbalance"),
            params={"class_imbalance": {"target": "y"}},
            profile=profile_dataset(dataset),
        )

        entry = next(e for e in plan if e.id == "class_imbalance")
        assert entry.params.model_dump()["target"] == "y"
