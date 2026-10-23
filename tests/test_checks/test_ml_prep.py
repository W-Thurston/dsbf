"""The ML-prep checks on small frames: encoding, binning, transforms, imputation."""

import datetime as dt
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks import core_pack
from dsbf.checks.categorical_encoding import (
    ENCODE_METHODS,
    CategoricalEncodingParams,
)
from dsbf.checks.imputation import IMPUTE_METHODS
from dsbf.checks.numeric_binning import BIN_METHODS, NumericBinningParams
from dsbf.checks.transformations import TRANSFORM_METHODS, TransformationsParams
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import Action, CheckStatus, Confidence, Severity
from dsbf.profiling import profile_dataset
from tests.test_checks.support import (
    actions_of,
    by_column,
    output_of,
    run_core_check,
)

pytestmark = pytest.mark.unit

N = 400
RNG = np.random.default_rng(3)


IDS = ("categorical_encoding", "numeric_binning", "transformations", "imputation")


def metric(finding, name: str) -> float:
    value = finding.evidence.metrics[name]
    assert isinstance(value, (int, float))
    return float(value)


def levels(count: int, rows: int = N) -> list[str]:
    return [f"v{i % count}" for i in range(rows)]


class TestCategoricalEncoding:
    def strategy(self, count: int, rows: int = N, **params) -> str:
        frame = pl.DataFrame({"c": levels(count, rows)})
        output = output_of(run_core_check(frame, "categorical_encoding", **params))
        return output["columns"]["c"]["strategy"]

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (2, "binary"),
            (3, "one_hot"),
            (10, "one_hot"),
            (11, "frequency"),
            (60, "frequency"),
        ],
    )
    def test_the_strategy_follows_the_cardinality(self, count, expected):
        assert self.strategy(count) == expected

    def test_too_few_rows_per_value_make_one_hot_columns_too_sparse(self):
        assert self.strategy(10, rows=190) == "frequency"  # 19 rows a value
        assert self.strategy(10, rows=200) == "one_hot"  # 20 rows a value

    def test_the_limits_are_parameters(self):
        assert self.strategy(11, max_one_hot=12) == "one_hot"
        with pytest.raises(ValidationError):
            CategoricalEncodingParams.model_validate({"max_one_hot": 1})

    def test_the_finding_carries_a_typed_encode_action(self):
        frame = pl.DataFrame({"c": levels(3)})
        finding = run_core_check(frame, "categorical_encoding").findings[0]
        (action,) = actions_of(finding)

        assert finding.guidance is not None and finding.guidance.phase == "ml"
        assert finding.severity is Severity.INFO
        assert (action.action, action.method, action.columns) == (
            "encode",
            "one_hot",
            ("c",),
        )
        assert action.params == {"n_levels": 3}
        assert finding.evidence.metrics["rows_per_level"] == pytest.approx(N / 3)

    def test_the_rarest_value_is_known_only_when_the_profile_lists_them_all(self):
        few = pl.DataFrame({"c": ["a"] * 30 + ["b"] * 9 + ["c"] * 3})
        many = pl.DataFrame({"c": levels(30)})

        listed = output_of(run_core_check(few, "categorical_encoding"))["columns"]["c"]
        unlisted = output_of(run_core_check(many, "categorical_encoding"))["columns"][
            "c"
        ]

        assert listed["rarest_level_count"] == 3
        assert unlisted["rarest_level_count"] is None

    def test_constant_and_non_categorical_columns_are_left_out(self):
        frame = pl.DataFrame(
            {"one": ["a"] * N, "none": [None] * N, "flag": [True, False] * (N // 2)}
        )

        result = run_core_check(frame, "categorical_encoding")

        assert result.findings == ()
        assert result.status in (CheckStatus.OK, CheckStatus.NOT_APPLICABLE)

    def test_a_numeric_code_with_few_values_is_encoded(self):
        frame = pl.DataFrame({"code": [1, 2, 3, 4] * (N // 4)})

        output = output_of(run_core_check(frame, "categorical_encoding"))

        assert output["columns"]["code"]["strategy"] == "one_hot"


class TestNumericBinning:
    def binning(self, values, **params) -> dict:
        frame = pl.DataFrame({"x": values})
        return output_of(run_core_check(frame, "numeric_binning", **params))

    def test_a_uniform_column_gets_equal_width_bins(self):
        output = self.binning(RNG.uniform(0, 100, 2000))

        assert output["columns"]["x"]["strategy"] == "equal_width"
        assert output["columns"]["x"]["core_span"] > 0.9

    def test_a_skewed_column_gets_quantile_bins(self):
        output = self.binning(RNG.lognormal(0, 1.2, 2000))

        assert output["columns"]["x"]["strategy"] == "quantile"
        assert output["columns"]["x"]["skewness"] > 1

    def test_a_long_tail_on_a_symmetric_column_also_gets_quantile_bins(self):
        values = np.r_[RNG.normal(0, 1, 1990), np.full(5, 1e4), np.full(5, -1e4)]
        output = self.binning(values)

        assert output["columns"]["x"]["strategy"] == "quantile"
        assert output["columns"]["x"]["core_span"] < 0.01

    def test_the_bin_count_follows_sturges_and_the_rows_per_bin(self):
        assert self.binning(RNG.uniform(0, 1, 2000))["columns"]["x"]["n_bins"] == 10
        assert self.binning(RNG.uniform(0, 1, 100))["columns"]["x"]["n_bins"] == 3
        assert (
            self.binning(RNG.uniform(0, 1, 2000), max_bins=5)["columns"]["x"]["n_bins"]
            == 5
        )

    def test_a_spike_at_zero_is_named_in_the_action(self):
        values = np.r_[np.zeros(1000), RNG.lognormal(0, 1, 1000)]
        frame = pl.DataFrame({"x": values})
        finding = run_core_check(frame, "numeric_binning").findings[0]
        (action,) = actions_of(finding)

        assert action.method == "quantile"
        assert action.params["spike_value"] == 0.0
        assert finding.evidence.metrics["spike_share"] == pytest.approx(0.5)

    def test_discrete_constant_and_short_columns_are_skipped_with_a_reason(self):
        output = self.binning(RNG.uniform(0, 1, 500), min_distinct=500)
        assert output["columns"] == {}
        assert "already discrete" in output["skipped_columns"]["x"]
        constant = pl.DataFrame({"x": [3.0] * 500, "y": RNG.uniform(0, 1, 500)})
        found = output_of(run_core_check(constant, "numeric_binning"))
        assert set(found["columns"]) == {"y"}
        short = self.binning(RNG.uniform(0, 1, 50))
        assert "fewer than two bins" in short["skipped_columns"]["x"]

    def test_the_action_is_a_bin_action_with_the_count(self):
        frame = pl.DataFrame({"x": RNG.uniform(0, 1, 2000)})
        (action,) = actions_of(run_core_check(frame, "numeric_binning").findings[0])

        assert action.action == "bin" and action.method in BIN_METHODS
        assert action.params == {"n_bins": 10}

    def test_a_profile_without_the_outer_quantiles_falls_back_to_the_skewness(self):
        dataset = load(pl.DataFrame({"x": RNG.uniform(0, 100, 2000)}), name="frame")
        profile = profile_dataset(dataset, quantiles=(0.25, 0.5, 0.75))
        plan = make_plan(
            Registry(core_pack().checks),
            Selection(include="numeric_binning"),
            profile=profile,
        )

        run = execute(plan, dataset, profile, dsbf_version="test", workers=1)
        column = output_of(run.results["numeric_binning"])["columns"]["x"]

        assert column["core_span"] is None
        assert column["strategy"] == "equal_width"

    def test_parameters_are_bounded(self):
        with pytest.raises(ValidationError):
            NumericBinningParams.model_validate({"max_bins": 2})
        with pytest.raises(ValidationError):
            NumericBinningParams.model_validate({"min_core_span": 1.5})


class TestTransformations:
    def run(self, values, **params):
        frame = pl.DataFrame({"x": values})
        return run_core_check(frame, "transformations", **params)

    def test_a_lognormal_column_gets_log1p(self):
        result = self.run(RNG.lognormal(2, 1, 5000))
        (finding,) = result.findings
        (action,) = actions_of(finding)

        assert finding.code == "transform_suggested"
        assert (action.action, action.method) == ("transform", "log1p")
        assert action.params == {}
        assert metric(finding, "skewness_before") > 3
        assert abs(metric(finding, "skewness_after")) <= 0.5
        assert metric(finding, "reduction") > 0.8

    def test_negative_values_rule_out_the_roots_and_box_cox(self):
        values = RNG.exponential(3, 5000) - 1.0
        result = self.run(values)
        column = output_of(result)["columns"]["x"]

        assert column["recommended"] == "yeo_johnson"
        assert "0 or more" in column["candidates"]["log1p"]["skipped"]
        assert column["candidates"]["box_cox"]["skipped"] == "needs values above 0"
        (action,) = actions_of(result.findings[0])
        assert action.params["lambda"] == pytest.approx(
            column["candidates"]["yeo_johnson"]["lam"]
        )

    def test_yeo_johnson_fits_the_same_power_as_scikit_learn(self):
        pytest.importorskip("sklearn")
        from sklearn.preprocessing import PowerTransformer

        values = RNG.exponential(3, 4000) - 1.0
        column = output_of(self.run(values))["columns"]["x"]
        reference = PowerTransformer(method="yeo-johnson", standardize=False).fit(
            values.reshape(-1, 1)
        )

        assert column["candidates"]["yeo_johnson"]["lam"] == pytest.approx(
            reference.lambdas_[0], abs=1e-6
        )

    def test_a_left_skewed_non_negative_column_gets_square(self):
        values = 100 - RNG.exponential(8, 5000)
        values = values - values.min()
        column = output_of(self.run(values))["columns"]["x"]

        assert column["recommended"] in ("square", "box_cox", "yeo_johnson")
        assert column["candidates"]["log1p"]["skipped"].endswith("a right skew")

    def test_x_squared_is_not_tried_on_values_below_zero(self):
        values = -RNG.exponential(3, 3000)
        column = output_of(self.run(values))["columns"]["x"]

        assert "folds the distribution" in column["candidates"]["square"]["skipped"]

    def test_the_simplest_transform_that_reaches_the_target_is_chosen(self):
        column = output_of(self.run(RNG.lognormal(3, 1, 5000)))["columns"]["x"]

        assert column["recommended"] == "log1p"
        for method in ("sqrt", "box_cox", "yeo_johnson"):
            assert column["candidates"][method]["skipped"].startswith("not needed")

    def test_a_column_no_transform_fixes_gets_a_finding_without_actions(self):
        values = np.r_[np.zeros(4900), RNG.exponential(30, 100)]
        result = self.run(values, min_reduction=0.9)
        (finding,) = result.findings

        assert finding.code == "no_effective_transform"
        assert finding.guidance is None

    def test_symmetric_constant_and_short_columns_are_not_previewed(self):
        frame = pl.DataFrame(
            {
                "normal": RNG.normal(0, 1, 500),
                "constant": [4.0] * 500,
                "short": [1.0, 2.0, 50.0] + [None] * 497,
            }
        )
        result = run_core_check(frame, "transformations")

        assert result.findings == ()
        assert output_of(result)["columns"] == {}
        assert "fewer than" in output_of(result)["skipped_columns"]["short"]

    def test_infinite_values_are_left_out_and_nothing_is_not_finite(self):
        values = np.r_[RNG.lognormal(0, 1, 2000), [np.inf, -np.inf]]
        result = self.run(values)

        assert output_of(result)["columns"]["x"]["n_used"] == 2000
        assert result.findings[0].evidence.non_finite_metrics == ()

    def test_a_large_column_is_sampled_and_says_so(self):
        values = RNG.lognormal(0, 1, 30_000)
        result = self.run(values, max_rows=5000)
        column = output_of(result)["columns"]["x"]

        assert column["sampled"] is True and column["n_used"] == 5000
        assert result.findings[0].confidence is Confidence.MODERATE
        again = output_of(self.run(values, max_rows=5000))["columns"]["x"]
        assert again == column

    def test_parameters_are_bounded(self):
        with pytest.raises(ValidationError):
            TransformationsParams.model_validate({"min_reduction": 0})
        with pytest.raises(ValidationError):
            TransformationsParams.model_validate({"max_rows": 10})


def run_together(frame: pl.DataFrame, ids: tuple[str, ...], params=None, exclude=()):
    """The results of ``ids`` and whatever else the core pack selects."""
    dataset = load(frame, name="frame")
    profile = profile_dataset(dataset)
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include=ids, exclude=exclude),
        params=params,
        profile=profile,
    )
    return execute(plan, dataset, profile, dsbf_version="test", workers=1).results


def missing_frame() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    n = 1500
    driver = rng.normal(0, 1, n)
    skewed = rng.lognormal(0, 1.5, n)

    def hole(share: float, depends: bool = False) -> list:
        if depends:
            chance = 1 / (1 + np.exp(-3 * driver)) * share * 2
            gone = rng.random(n) < chance
        else:
            gone = rng.random(n) < share
        return gone.tolist()

    def where(values, gone):
        return [None if g else v for v, g in zip(values, gone, strict=True)]

    day = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "driver": driver,
            "full": rng.normal(0, 1, n),
            "mean_fill": where(rng.normal(0, 1, n), hole(0.1)),
            "median_fill": where(skewed, hole(0.1)),
            "model_fill": where(rng.normal(0, 1, n), hole(0.2, depends=True)),
            "significant": where(rng.normal(0, 1, n), hole(0.3)),
            "severe": where(rng.normal(0, 1, n), hole(0.7)),
            "category": where(rng.choice(["a", "b", "c"], n), hole(0.1)),
            "category_big_gap": where(rng.choice(["a", "b", "c"], n), hole(0.3)),
            "flag": where(rng.integers(0, 2, n).astype(bool), hole(0.1)),
            "day": where(day, hole(0.1)),
            "irregular": where(
                [
                    dt.date(2020, 1, 1) + dt.timedelta(days=int(d))
                    for d in rng.integers(0, 3000, n)
                ],
                hole(0.1),
            ),
            "text": where([f"free text {i} " * 3 for i in range(n)], hole(0.1)),
        }
    )


class TestImputation:
    def advice(self, **params):
        results = run_together(
            missing_frame(),
            ("imputation", "missingness_mechanism"),
            params={"imputation": params} if params else None,
        )
        return results["imputation"], output_of(results["imputation"])["columns"]

    def test_each_tier_type_and_mechanism_gets_its_strategy(self):
        _, columns = self.advice()

        assert columns["mean_fill"]["strategy"] == "mean"
        assert columns["median_fill"]["strategy"] == "median"
        assert columns["model_fill"]["strategy"] == "iterative"
        assert columns["model_fill"]["mechanism"] == "depends_on_observed"
        assert columns["model_fill"]["add_indicator"] is True
        assert columns["significant"]["strategy"] == "mean"
        assert columns["significant"]["add_indicator"] is True
        assert columns["severe"]["strategy"] == "indicator_and_drop"
        assert columns["category"]["strategy"] == "mode"
        assert columns["category_big_gap"]["strategy"] == "constant_missing"
        assert columns["flag"]["strategy"] == "mode"
        assert columns["day"]["strategy"] == "forward_fill"
        assert columns["irregular"]["strategy"] == "indicator_only"

    def test_the_tier_is_the_one_null_rate_reported(self):
        _, columns = self.advice()

        assert columns["mean_fill"]["tier"] == "moderate"
        assert columns["significant"]["tier"] == "significant"
        assert columns["severe"]["tier"] == "severe"
        assert columns["model_fill"]["tier"] == "significant"

    def test_text_columns_and_complete_columns_are_left_out(self):
        _, columns = self.advice()

        assert "text" not in columns and "full" not in columns
        assert "driver" not in columns

    def test_without_the_mechanism_check_the_advice_says_it_was_not_analysed(self):
        results = run_together(
            missing_frame(), ("imputation",), exclude=("missingness_mechanism",)
        )
        result = results["imputation"]
        columns = output_of(result)["columns"]

        assert columns["model_fill"]["mechanism"] is None
        assert columns["model_fill"]["strategy"] == "mean"
        message = by_column(result)["model_fill"][0].message
        assert "was not analysed" in message

    def test_actions_are_typed_and_in_the_vocabulary(self):
        result, _ = self.advice()
        seen = set()
        for finding in result.findings:
            assert finding.guidance is not None and finding.guidance.phase == "ml"
            for action in finding.guidance.actions:
                Action.model_validate(action.model_dump())
                seen.add(action.action)
                if action.action == "impute":
                    assert action.method in IMPUTE_METHODS
                assert action.columns == finding.subject.columns
        assert seen == {"impute", "add_indicator", "drop"}

    def test_a_severe_column_is_a_warning_and_the_rest_are_not(self):
        result, _ = self.advice()
        severities = {
            column: findings[0].severity
            for column, findings in by_column(result).items()
        }

        assert severities.pop("severe") is Severity.WARN
        assert set(severities.values()) == {Severity.INFO}

    def test_the_minimum_share_is_a_parameter(self):
        _, columns = self.advice(min_null_share=0.5)

        assert set(columns) == {"severe"}

    def test_a_synthesis_check_cannot_read_values(self):
        results = run_together(missing_frame(), ("imputation",))

        assert results["imputation"].status is CheckStatus.OK
        assert results["imputation"].notes == ()


class TestActionVocabulary:
    def test_every_method_the_checks_emit_is_declared(self):
        assert set(ENCODE_METHODS) == {"binary", "one_hot", "frequency", "target"}
        assert set(BIN_METHODS) == {"quantile", "equal_width"}
        assert set(TRANSFORM_METHODS) == {
            "log1p",
            "sqrt",
            "square",
            "box_cox",
            "yeo_johnson",
        }

    def test_every_action_on_a_mixed_frame_validates(self):
        frame = missing_frame().with_columns(
            skew=pl.Series(RNG.lognormal(2, 1, 1500)),
            wide=pl.Series(RNG.uniform(0, 100, 1500)),
            many=pl.Series(levels(30, 1500)),
        )
        results = run_together(frame, IDS)
        seen = {}
        for check_id in IDS:
            for finding in results[check_id].findings:
                for action in finding.guidance.actions if finding.guidance else ():
                    assert Action.model_validate(action.model_dump()) == action
                    seen.setdefault(action.action, set()).add(action.method)

        assert seen["encode"] <= set(ENCODE_METHODS)
        assert seen["bin"] <= set(BIN_METHODS)
        assert seen["transform"] <= set(TRANSFORM_METHODS)
        assert seen["impute"] <= set(IMPUTE_METHODS)


class TestWithoutTheMlExtra:
    def test_the_checks_run_with_scikit_learn_blocked(self):
        script = textwrap.dedent(
            """
            import sys
            sys.modules["sklearn"] = None
            import numpy as np
            import polars as pl
            from dsbf.checks import core_pack
            from dsbf.data import load
            from dsbf.engine import Registry, Selection, execute, make_plan
            from dsbf.profiling import profile_dataset

            rng = np.random.default_rng(0)
            n = 600
            ids = (
                "categorical_encoding",
                "numeric_binning",
                "transformations",
                "imputation",
            )
            y = rng.normal(0, 1, n)
            frame = pl.DataFrame({
                "c": [f"v{i % 4}" for i in range(n)],
                "x": rng.lognormal(0, 1, n),
                "y": [None if i % 5 == 0 else float(v) for i, v in enumerate(y)],
            })
            dataset = load(frame, name="frame")
            profile = profile_dataset(dataset)
            plan = make_plan(
                Registry(core_pack().checks),
                Selection(include=ids),
                profile=profile,
            )
            run = execute(plan, dataset, profile, dsbf_version="test", workers=1)
            print({i: run.results[i].status.value for i in ids})
            """
        )

        done = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True
        )

        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == str(
            {
                "categorical_encoding": "ok",
                "numeric_binning": "ok",
                "transformations": "ok",
                "imputation": "ok",
            }
        )
