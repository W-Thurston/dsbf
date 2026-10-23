"""The target-aware checks on small frames: leakage and mutual information."""

import datetime as dt
import subprocess
import sys
import textwrap

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks import core_pack
from dsbf.checks.mutual_information import MutualInformationParams
from dsbf.checks.target_leakage import TargetLeakageParams
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.model import CheckStatus, Severity
from dsbf.profiling import profile_dataset
from tests.test_checks.support import (
    actions_of,
    by_column,
    output_of,
    run_core_check,
)

pytestmark = pytest.mark.unit

N = 800


def labels(seed: int = 1, n: int = N) -> np.ndarray:
    return np.random.default_rng(seed).choice(["no", "yes"], n, p=[0.6, 0.4])


def flipped(y: np.ndarray, share: float, seed: int = 2) -> list[str]:
    """``y`` with ``share`` of the labels swapped."""
    swap = np.random.default_rng(seed).random(y.size) < share
    return [("yes" if v == "no" else "no") if s else v for v, s in zip(y, swap)]


def codes(values: list[str]) -> list[str]:
    return [{"yes": "Y", "no": "N"}[v] for v in values]


def leakage(frame: pl.DataFrame, **params):
    return run_core_check(frame, "target_leakage", target="y", **params)


def flagged(result) -> dict[str, str]:
    return {f.subject.columns[0]: f.code for f in result.findings}


class TestTargetLeakage:
    def test_a_recoded_copy_of_a_class_target_is_an_error(self):
        y = labels()
        frame = pl.DataFrame({"y": y, "copy": codes(list(y))})

        result = leakage(frame)

        finding = by_column(result)["copy"][0]
        assert finding.code == "target_leakage"
        assert finding.severity is Severity.ERROR
        assert finding.evidence.method == "cramers_v_bias_corrected"
        assert finding.evidence.metrics["association"] == pytest.approx(1.0)

    def test_a_copy_with_a_few_flipped_labels_is_a_warning(self):
        y = labels()
        frame = pl.DataFrame({"y": y, "almost": flipped(y, 0.012)})

        finding = by_column(leakage(frame))["almost"][0]

        assert finding.severity is Severity.WARN

    def test_a_numeric_feature_that_separates_the_classes_is_flagged(self):
        y = labels()
        noise = np.random.default_rng(3).normal(0, 5, N)
        score = np.where(y == "yes", 100.0, 0.0) + noise
        frame = pl.DataFrame({"y": y, "score": score})

        finding = by_column(leakage(frame))["score"][0]

        assert finding.evidence.method == "correlation_ratio"
        assert finding.evidence.metrics["association"] > 0.99

    def test_a_class_feature_of_a_numeric_target_is_flagged(self):
        t = np.random.default_rng(4).normal(0, 1, N)
        band = np.digitize(t, np.quantile(t, [0.25, 0.5, 0.75]))
        frame = pl.DataFrame({"t": t, "band": [f"b{i}" for i in band]})

        result = run_core_check(frame, "target_leakage", target="t")

        # Four bands of a normal variable keep about 0.92 of its variance.
        assert output_of(result)["features"][0]["metric"] == "eta"
        assert output_of(result)["features"][0]["value"] > 0.9
        assert not result.findings

    def test_a_monotone_function_of_a_numeric_target_is_a_rank_leak(self):
        t = np.random.default_rng(5).normal(0, 1, N)
        frame = pl.DataFrame({"t": t, "exp_t": np.exp(3 * t)})

        result = run_core_check(frame, "target_leakage", target="t")

        finding = by_column(result)["exp_t"][0]
        assert finding.evidence.metrics["metric"] == "spearman_r"
        assert finding.severity is Severity.ERROR

    def test_independent_features_are_not_flagged(self):
        rng = np.random.default_rng(6)
        frame = pl.DataFrame(
            {
                "y": labels(),
                **{f"x{i}": rng.normal(0, 1, N) for i in range(10)},
                **{f"c{i}": rng.choice(list("abcd"), N) for i in range(10)},
            }
        )

        result = leakage(frame)

        assert not result.findings
        assert output_of(result)["features_tested"] == 20

    def test_a_part_of_an_identifier_that_spells_the_outcome(self):
        y = labels()
        frame = pl.DataFrame(
            {
                "y": y,
                "order_id": [f"{v}-{i:05d}" for i, v in enumerate(y)],
                "row_id": [f"row-{i:05d}" for i in range(N)],
            }
        )

        result = leakage(frame)

        finding = by_column(result)["order_id"][0]
        assert finding.code == "derived_from_target"
        assert output_of(result)["features"][0]["part"] == 1
        assert "row_id" not in flagged(result)

    def test_missing_values_that_follow_the_outcome(self):
        y = labels()
        rng = np.random.default_rng(7)
        frame = pl.DataFrame(
            {
                "y": y,
                "closed_amount": [
                    float(i) if v == "yes" else None for i, v in enumerate(y)
                ],
                "random_holes": [None if r < 0.3 else 1.0 for r in rng.random(N)],
            }
        )

        result = leakage(frame)

        assert flagged(result) == {"closed_amount": "missingness_tracks_target"}

    def test_a_datetime_after_the_time_index(self):
        y = labels()
        start = dt.datetime(2024, 1, 1)
        rng = np.random.default_rng(8)
        event = [start + dt.timedelta(days=i) for i in range(N)]
        frame = pl.DataFrame(
            {
                "y": y,
                "event": event,
                "closed": [
                    e + dt.timedelta(days=int(d))
                    for e, d in zip(event, rng.integers(1, 6, N))
                ],
                "opened": [
                    e - dt.timedelta(days=int(d))
                    for e, d in zip(event, rng.integers(1, 6, N))
                ],
            }
        )

        result = leakage(frame, time_index="event")

        assert flagged(result) == {"closed": "timestamp_after_event"}
        finding = by_column(result)["closed"][0]
        assert finding.severity is Severity.WARN
        assert finding.evidence.metrics["association"] == pytest.approx(1.0)

    def test_without_a_time_index_the_timestamps_are_left_alone(self):
        y = labels()
        start = dt.datetime(2024, 1, 1)
        frame = pl.DataFrame(
            {"y": y, "event": [start + dt.timedelta(days=i) for i in range(N)]}
        )

        result = leakage(frame)

        assert result.status is CheckStatus.OK
        assert "no_time_index" in {n.code for n in result.notes}

    def test_a_time_index_that_is_not_a_datetime_is_noted(self):
        frame = pl.DataFrame({"y": labels(), "x": np.arange(N, dtype=float)})

        result = leakage(frame, time_index="x")

        assert "time_index_not_a_datetime" in {n.code for n in result.notes}

    @pytest.mark.parametrize(
        ("params", "code"),
        [
            ({"target": None}, "no_target"),
            ({"target": "text"}, "target_not_supported"),
            ({"target": "flat"}, "target_constant"),
            ({"target": "many", "max_levels": 5}, "target_not_supported"),
        ],
    )
    def test_a_target_that_cannot_be_used_ends_the_check_not_applicable(
        self, params, code
    ):
        rng = np.random.default_rng(9)
        frame = pl.DataFrame(
            {
                "x": rng.normal(0, 1, N),
                "text": [f"some words about row number {i} and more" for i in range(N)],
                "flat": ["a"] * N,
                "many": [f"k{i % 12}" for i in range(N)],
            }
        )

        result = run_core_check(frame, "target_leakage", **params)

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert code in {n.code for n in result.notes}

    def test_a_feature_with_too_few_rows_is_counted_untested(self):
        y = labels(n=20)
        frame = pl.DataFrame({"y": y, "copy": list(y)})

        result = leakage(frame)

        assert output_of(result)["untested"] == {"too_few_rows": 1}
        assert not result.findings

    def test_a_sample_still_finds_the_leak_and_says_so(self):
        y = labels(n=3000)
        frame = pl.DataFrame({"y": y, "copy": codes(list(y))})

        result = leakage(frame, max_rows=1000)

        assert flagged(result) == {"copy": "target_leakage"}
        assert output_of(result)["rows_used"] == 1000
        assert output_of(result)["rows_total"] == 3000
        assert "rows_sampled" in {n.code for n in result.notes}

    def test_the_advice_is_a_typed_drop_action(self):
        y = labels()
        frame = pl.DataFrame({"y": y, "copy": codes(list(y))})

        finding = by_column(leakage(frame))["copy"][0]

        (action,) = actions_of(finding)
        assert (action.action, action.columns) == ("drop", ("copy",))
        assert action.condition is not None

    def test_the_parameters_are_bounded(self):
        for bad in ({"warn_at": 0}, {"min_n": 2}, {"max_levels": 1}):
            with pytest.raises(ValidationError):
                TargetLeakageParams.model_validate(bad)


def informative_frame(n: int = 3000) -> pl.DataFrame:
    """A binary target, a code that predicts it, a code that does not, and noise."""
    rng = np.random.default_rng(10)
    code = rng.integers(0, 12, n)
    rate = 0.2 + 0.6 * code / 11
    return pl.DataFrame(
        {
            "y": np.where(rng.random(n) < rate, "yes", "no"),
            "predictive": [f"p{c}" for c in code],
            "unrelated": [f"u{c}" for c in rng.integers(0, 12, n)],
            "noise": rng.normal(0, 1, n),
        }
    )


class TestMutualInformation:
    def test_it_ranks_a_non_linear_feature_a_correlation_misses(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, 2000)
        frame = pl.DataFrame(
            {
                "t": x**2 + rng.normal(0, 0.1, 2000),
                "x": x,
                "noise": rng.normal(0, 1, 2000),
            }
        )

        result = run_core_check(frame, "mutual_information", target="t")

        ranked = output_of(result)["features"]
        assert [f["feature"] for f in ranked] == ["x", "noise"]
        assert ranked[0]["notable"] and not ranked[1]["notable"]
        assert abs(float(np.corrcoef(x, frame["t"])[0, 1])) < 0.1
        assert output_of(result)["normalisation"] == "information_r2"

    def test_independent_features_are_not_notable(self):
        rng = np.random.default_rng(12)
        frame = pl.DataFrame(
            {
                "y": labels(n=2000),
                **{f"x{i}": rng.normal(0, 1, 2000) for i in range(8)},
                **{f"c{i}": rng.choice(list("abcde"), 2000) for i in range(8)},
            }
        )

        result = run_core_check(frame, "mutual_information", target="y")

        assert output_of(result)["features_notable"] == 0
        assert not result.findings

    def test_repeated_rows_do_not_change_the_estimate(self):
        rng = np.random.default_rng(13)
        base = pl.DataFrame({"t": rng.normal(0, 1, 2000), "x": rng.normal(0, 1, 2000)})
        repeated = pl.concat([base, base.head(200)])

        with_repeats = run_core_check(repeated, "mutual_information", target="t")
        without = run_core_check(base, "mutual_information", target="t")

        # 250 copies among 5 250 rows lifted an independent column from 0 to 0.057
        # nats, because the neighbour estimator reads a repeated point as dependence.
        (feature,) = output_of(with_repeats)["features"]
        (expected,) = output_of(without)["features"]
        assert feature["mutual_information"] == expected["mutual_information"]
        assert feature["n_used"] == 2000

    def test_a_class_target_reports_the_share_of_its_entropy(self):
        frame = informative_frame()

        result = run_core_check(frame, "mutual_information", target="y")

        out = output_of(result)
        assert out["normalisation"] == "target_entropy"
        assert out["target_classes"] == 2
        features = {f["feature"]: f for f in out["features"]}
        assert features["predictive"]["notable"]
        assert 0.05 < features["predictive"]["normalised"] < 0.3
        assert not features["unrelated"]["notable"]
        assert not features["noise"]["notable"]
        (finding,) = result.findings
        assert finding.subject.columns == ("predictive",)
        assert finding.evidence.method == "plug_in_mutual_information"
        assert actions_of(finding)[0].action == "keep"

    def test_a_copy_of_the_target_points_at_target_leakage(self):
        y = labels()
        frame = pl.DataFrame({"y": y, "copy": codes(list(y))})

        result = run_core_check(frame, "mutual_information", target="y")

        assert "target_leakage" in result.findings[0].message
        assert output_of(result)["features"][0]["normalised"] == pytest.approx(1.0)

    def test_missing_values_are_left_out_rather_than_filled(self):
        rng = np.random.default_rng(14)
        x = rng.normal(0, 1, 1000)
        t = 2 * x + rng.normal(0, 0.5, 1000)
        holes = np.where(rng.random(1000) < 0.3, np.nan, x)
        frame = pl.DataFrame({"t": t, "x": holes})

        result = run_core_check(frame, "mutual_information", target="t")

        (feature,) = output_of(result)["features"]
        assert feature["n_used"] == int(np.count_nonzero(~np.isnan(holes)))
        assert feature["notable"]

    def test_a_noise_floor_is_reported_for_each_kind(self):
        result = run_core_check(informative_frame(), "mutual_information", target="y")

        floor = output_of(result)["noise_floor"]
        assert set(floor) == {"discrete", "continuous"}
        assert all(0 <= v < 0.05 for v in floor.values())

    @pytest.mark.parametrize(
        ("params", "code"),
        [
            ({"target": None}, "no_target"),
            ({"target": "text"}, "target_not_supported"),
        ],
    )
    def test_without_a_usable_target_it_is_not_applicable(self, params, code):
        frame = pl.DataFrame(
            {
                "x": np.random.default_rng(15).normal(0, 1, N),
                "text": [f"some words about row number {i} and more" for i in range(N)],
            }
        )

        result = run_core_check(frame, "mutual_information", **params)

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert code in {n.code for n in result.notes}

    def test_the_parameters_are_bounded(self):
        for bad in (
            {"n_neighbors": 0},
            {"min_normalised": 1},
            {"max_rows": 10},
            {"null_shuffles": 0},
        ):
            with pytest.raises(ValidationError):
                MutualInformationParams.model_validate(bad)


def run_three(frame: pl.DataFrame, **targets):
    dataset = load(frame, name="frame")
    profile = profile_dataset(dataset)
    ids = ("categorical_encoding", "mutual_information", "target_leakage")
    plan = make_plan(
        Registry(core_pack().checks),
        Selection(include=ids),
        params={
            "mutual_information": {"target": targets.get("mi")},
            "target_leakage": {"target": targets.get("leak")},
        },
        profile=profile,
    )
    return execute(plan, dataset, profile, dsbf_version="test", workers=1).results


def encodings(result, column: str):
    (finding,) = by_column(result)[column]
    return [a for a in actions_of(finding) if a.action == "encode"]


class TestTargetEncodingAdvice:
    def test_a_predictive_column_also_gets_a_target_encoding(self):
        results = run_three(informative_frame(), mi="y", leak="y")

        methods = [
            a.method for a in encodings(results["categorical_encoding"], "predictive")
        ]
        assert methods == ["frequency", "target"]
        target = encodings(results["categorical_encoding"], "predictive")[1]
        assert target.params == {"folds": 5, "smoothing": 20}
        assert "out of fold" in (target.detail or "")
        assert output_of(results["categorical_encoding"])["columns"]["predictive"][
            "target_encoding"
        ]

    def test_a_column_that_does_not_predict_it_does_not(self):
        results = run_three(informative_frame(), mi="y", leak="y")

        assert [
            a.method for a in encodings(results["categorical_encoding"], "unrelated")
        ] == ["frequency"]

    def test_a_column_flagged_as_a_leak_is_never_target_encoded(self):
        y = np.random.default_rng(16).integers(0, 2, 1200)
        frame = pl.DataFrame(
            {"y": y, "leaky": [f"k{2 * (i % 6) + int(v)}" for i, v in enumerate(y)]}
        )

        results = run_three(frame, mi="y", leak="y")

        assert results["target_leakage"].findings
        assert [
            a.method for a in encodings(results["categorical_encoding"], "leaky")
        ] == ["frequency"]

    def test_without_the_two_checks_there_is_no_target_encoding(self):
        results = run_three(informative_frame())

        assert results["mutual_information"].status is CheckStatus.NOT_APPLICABLE
        assert [
            a.method for a in encodings(results["categorical_encoding"], "predictive")
        ] == ["frequency"]

    def test_a_target_of_many_classes_gets_none(self):
        rng = np.random.default_rng(17)
        code = rng.integers(0, 12, 3000)
        frame = pl.DataFrame(
            {
                "y": [f"class{(c + rng.integers(0, 2)) % 4}" for c in code],
                "predictive": [f"p{c}" for c in code],
            }
        )

        results = run_three(frame, mi="y", leak="y")

        assert [
            a.method for a in encodings(results["categorical_encoding"], "predictive")
        ] == ["frequency"]

    def test_a_numeric_target_gets_one(self):
        rng = np.random.default_rng(18)
        code = rng.integers(0, 12, 3000)
        frame = pl.DataFrame(
            {
                "y": code * 1.0 + rng.normal(0, 4, 3000),
                "predictive": [f"p{c}" for c in code],
            }
        )

        results = run_three(frame, mi="y", leak="y")

        assert [
            a.method for a in encodings(results["categorical_encoding"], "predictive")
        ] == [
            "frequency",
            "target",
        ]


class TestWithoutTheMlExtra:
    def test_leakage_runs_and_mutual_information_is_unavailable(self):
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
            y = rng.choice(["a", "b"], n)
            frame = pl.DataFrame({"y": y, "copy": y, "x": rng.normal(0, 1, n)})
            dataset = load(frame, name="frame")
            profile = profile_dataset(dataset)
            ids = ("target_leakage", "mutual_information")
            plan = make_plan(
                Registry(core_pack().checks),
                Selection(include=ids),
                params={i: {"target": "y"} for i in ids},
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
            {"target_leakage": "ok", "mutual_information": "unavailable"}
        )
