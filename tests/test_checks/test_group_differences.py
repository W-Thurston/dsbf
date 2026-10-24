"""The group differences check on small frames."""

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from scipy import stats

from dsbf.checks.group_differences import GroupDifferencesParams
from dsbf.checks.pairs import adjust_p_values
from dsbf.model import CheckStatus, Dimension, Severity
from tests.test_checks.support import actions_of, output_of, run_core_check

pytestmark = pytest.mark.unit


def generator(seed: int = 11) -> np.random.Generator:
    return np.random.default_rng(seed)


def pair(result, numeric: str, categorical: str) -> dict:
    (found,) = (
        p
        for p in output_of(result)["pairs"]
        if (p["numeric"], p["categorical"]) == (numeric, categorical)
    )
    return found


def grouped_frame(
    shifts: dict[str, float], n_each: int = 60, seed: int = 11
) -> pl.DataFrame:
    g = generator(seed)
    levels, values = [], []
    for level, shift in shifts.items():
        levels += [level] * n_each
        values += list(g.normal(shift, 1.0, n_each))
    return pl.DataFrame(
        {"y": values, "group": levels, "noise": g.normal(size=len(values))}
    )


class TestAdjustment:
    P = np.array([0.01, 0.04, 0.03, 0.2])

    def test_benjamini_hochberg_follows_the_step_up_procedure_by_hand(self):
        got = adjust_p_values(self.P, "fdr_bh")

        # sorted 0.01, 0.03, 0.04, 0.2 times 4/1, 4/2, 4/3, 4/4: 0.04, 0.06, 0.0533, 0.2
        # then the running minimum from the right: 0.04, 0.0533, 0.0533, 0.2
        assert got == pytest.approx([0.04, 0.053333, 0.053333, 0.2], abs=1e-6)

    def test_bonferroni_and_holm_control_any_false_positive(self):
        assert adjust_p_values(self.P, "bonferroni") == pytest.approx(
            [0.04, 0.16, 0.12, 0.8]
        )
        # sorted 0.01x4, 0.03x3, 0.04x2, 0.2x1 -> 0.04, 0.09, 0.08, 0.2 -> running max
        assert adjust_p_values(self.P, "holm") == pytest.approx([0.04, 0.09, 0.09, 0.2])

    def test_none_and_a_single_test_are_unchanged_and_nothing_exceeds_one(self):
        assert adjust_p_values(self.P, "none").tolist() == self.P.tolist()
        assert adjust_p_values(np.array([0.3]), "fdr_bh").tolist() == [0.3]
        assert adjust_p_values(np.array([0.9, 0.8]), "bonferroni").max() == 1.0


class TestOmnibusTests:
    def test_f_and_h_match_scipy_and_the_effect_sizes_follow_by_hand(self):
        frame = grouped_frame({"a": 0.0, "b": 0.5, "c": 1.0}, n_each=40)
        groups = [frame.filter(pl.col("group") == v)["y"].to_numpy() for v in "abc"]
        f_stat, f_p = stats.f_oneway(*groups)
        h_stat, h_p = stats.kruskal(*groups)
        n = sum(g.size for g in groups)
        ssb = sum(g.size * (g.mean() - frame["y"].mean()) ** 2 for g in groups)
        ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)

        found = pair(
            run_core_check(frame, "group_differences", equal_variances=True),
            "y",
            "group",
        )

        assert found["f_statistic"] == pytest.approx(f_stat)
        assert found["anova_p"] == pytest.approx(f_p)
        assert found["h_statistic"] == pytest.approx(h_stat)
        assert found["kruskal_p"] == pytest.approx(h_p)
        assert found["eta_squared"] == pytest.approx(ssb / (ssb + ssw))
        msw = ssw / (n - 3)
        assert found["omega_squared"] == pytest.approx(
            (ssb - 2 * msw) / (ssb + ssw + msw)
        )
        assert found["epsilon_squared"] == pytest.approx(h_stat / (n - 1))
        assert found["omega_squared"] < found["eta_squared"]

    def test_the_default_p_values_are_welchs_by_hand(self):
        g = generator()
        groups = [g.normal(0, sd, n) for n, sd in ((50, 1.0), (30, 2.0), (20, 5.0))]
        frame = pl.DataFrame(
            {
                "y": np.concatenate(groups),
                "group": ["a"] * 50 + ["b"] * 30 + ["c"] * 20,
            }
        )
        n = np.array([50, 30, 20])
        mean = np.array([x.mean() for x in groups])
        weight = n / np.array([x.var(ddof=1) for x in groups])
        centre = (weight * mean).sum() / weight.sum()
        tail = (((1 - weight / weight.sum()) ** 2) / (n - 1)).sum()
        f_star = ((weight * (mean - centre) ** 2).sum() / 2) / (1 + 2 * 1 / 8 * tail)
        df2 = 8 / (3 * tail)

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        assert found["f_statistic"] == pytest.approx(f_star)
        assert found["anova_p"] == pytest.approx(stats.f.sf(f_star, 2, df2))
        # The Kruskal-Wallis H stays in the output; the p-value is Welch's on ranks.
        assert found["h_statistic"] == pytest.approx(stats.kruskal(*groups).statistic)

    def test_equal_groups_of_unequal_size_and_spread_are_rarely_called_different(self):
        g = generator()
        sizes = ((600, 1.0), (300, 1.5), (100, 3.0), (30, 6.0))
        called = {True: 0, False: 0}
        for _ in range(60):
            values = np.concatenate([g.normal(0, sd, n) for n, sd in sizes])
            frame = pl.DataFrame(
                {
                    "y": values,
                    "group": np.repeat(list("abcd"), [n for n, _ in sizes]),
                }
            )
            for classic in called:
                found = pair(
                    run_core_check(frame, "group_differences", equal_variances=classic),
                    "y",
                    "group",
                )
                called[classic] += found["anova_p"] < 0.05

        # The classic F calls equal means different about half the time here.
        assert called[True] > 20
        assert called[False] <= 8

    def test_ties_are_corrected_in_h(self):
        g = generator()
        values = np.round(g.normal(size=90) * 2)
        frame = pl.DataFrame({"y": values, "group": ["a", "b", "c"] * 30})
        groups = [values[i::3] for i in range(3)]

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        assert found["h_statistic"] == pytest.approx(stats.kruskal(*groups).statistic)

    def test_columns_named_with_a_pipe_get_the_right_subjects(self):
        frame = grouped_frame({"a": 0.0, "b": 2.0}).rename({"y": "a|b", "group": "c|d"})

        result = run_core_check(frame, "group_differences")

        (finding,) = result.findings
        assert finding.subject.columns == ("a|b", "c|d")
        assert finding.subject.kind == "column_pair"

    def test_a_group_under_the_minimum_is_left_out_and_counted(self):
        frame = grouped_frame({"a": 0.0, "b": 1.0}, n_each=30)
        extra = pl.DataFrame(
            {"y": [5.0, 6.0], "group": ["tiny", "tiny"], "noise": [0.0, 1.0]}
        )

        found = pair(
            run_core_check(pl.concat([frame, extra]), "group_differences"),
            "y",
            "group",
        )

        assert found["n_groups"] == 2
        assert found["groups_dropped"] == 1
        assert found["n_used"] == 60

    def test_a_pair_with_one_usable_group_is_untested_with_its_reason(self):
        frame = pl.DataFrame(
            {"y": list(generator().normal(size=12)), "group": ["a"] * 10 + ["b"] * 2}
        )

        result = run_core_check(frame, "group_differences")

        assert output_of(result)["pairs"] == []
        assert output_of(result)["untested"] == {"too_few_groups": 1}

    def test_missing_values_in_either_column_are_left_out(self):
        frame = grouped_frame({"a": 0.0, "b": 1.0}, n_each=40).with_columns(
            pl.when(pl.int_range(pl.len()) % 8 == 0)
            .then(None)
            .otherwise(pl.col("y"))
            .alias("y")
        )

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        assert found["n_used"] == 70

    def test_columns_with_large_values_do_not_overflow(self):
        frame = grouped_frame({"a": 0.0, "b": 3.0}).with_columns(pl.col("y") * 1e300)

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        assert np.isfinite(found["anova_p"]) and found["omega_squared"] > 0.3

    def test_groups_without_spread_give_no_f_but_a_certain_difference(self):
        frame = pl.DataFrame(
            {
                "y": [1.5 * (i % 4) for i in range(40)],
                "group": [f"g{i % 4}" for i in range(40)],
            }
        )

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        assert found["f_statistic"] is None
        assert found["anova_p"] == 0.0
        assert found["omega_squared"] == pytest.approx(1.0, abs=0.01)


class TestFlagging:
    def test_a_real_difference_is_flagged_by_both_methods_with_intervals(self):
        frame = grouped_frame({"a": 0.0, "b": 1.0, "c": 2.0})

        result = run_core_check(frame, "group_differences")

        found = pair(result, "y", "group")
        assert found["flagged"] and found["flagged_by"] == ["anova", "kruskal"]
        for name in ("omega_interval", "epsilon_interval"):
            interval = found[name]
            assert interval["method"] == "percentile_bootstrap"
            assert 0 <= interval["low"] <= interval["high"] <= 1
        assert (
            found["omega_interval"]["low"]
            < found["omega_squared"]
            < found["omega_interval"]["high"]
        )
        assert (found["top_group"], found["bottom_group"]) == ("c", "a")
        (finding,) = result.findings
        assert finding.code == "group_difference"
        assert finding.dimension is Dimension.RELATIONSHIP
        assert finding.severity is Severity.INFO
        assert finding.evidence.n_used == 180
        assert finding.evidence.method == "anova+kruskal"
        assert finding.evidence.interval is not None
        assert finding.evidence.metrics["pairs_tested"] == 2
        assert finding.evidence.metrics["pairs_flagged"] == 1
        assert {a.action for a in actions_of(finding)} == {"investigate", "encode"}

    def test_the_output_states_the_pairs_tested_and_tests_run(self):
        result = run_core_check(
            grouped_frame({"a": 0.0, "b": 1.0}), "group_differences"
        )

        output = output_of(result)
        assert (output["pairs_tested"], output["pairs_flagged"]) == (2, 1)
        # two omnibus tests for each pair, plus the one contrast of the flagged pair
        assert output["tests_run"] == 2 * 2 + 1

    def test_a_negligible_effect_in_a_huge_sample_is_significant_but_not_flagged(self):
        g = generator()
        n = 200_000
        frame = pl.DataFrame(
            {
                "y": g.normal(size=n) + 0.02 * (np.arange(n) % 2),
                "group": np.where(np.arange(n) % 2, "a", "b"),
            }
        )

        found = pair(
            run_core_check(frame, "group_differences", max_rows=n), "y", "group"
        )

        assert found["anova_p_adjusted"] < 0.05
        assert found["omega_squared"] < 0.001
        assert not found["flagged"]

    def test_the_methods_can_disagree_and_the_finding_says_so(self):
        g = generator()
        a = g.normal(size=200)
        b = g.normal(size=200)
        b[:8] += 60.0  # a few extreme values move the mean, not the ranks
        frame = pl.DataFrame({"y": np.r_[a, b], "group": ["a"] * 200 + ["b"] * 200})

        result = run_core_check(frame, "group_differences")

        found = pair(result, "y", "group")
        assert found["flagged_by"] == ["anova"]
        (finding,) = result.findings
        assert "often a few extreme values" in finding.message

    def test_the_correction_is_made_per_method_over_every_pair(self):
        frame = grouped_frame({"a": 0.0, "b": 0.35}, n_each=70)

        none = pair(
            run_core_check(frame, "group_differences", correction="none"), "y", "group"
        )
        bonferroni = pair(
            run_core_check(frame, "group_differences", correction="bonferroni"),
            "y",
            "group",
        )

        assert bonferroni["anova_p_adjusted"] == pytest.approx(
            min(1.0, none["anova_p"] * 2)
        )

    def test_findings_are_capped_with_a_note_and_the_output_keeps_every_pair(self):
        g = generator()
        n = 120
        shift = np.arange(n) % 2
        frame = pl.DataFrame(
            {
                **{f"y{i}": g.normal(size=n) + 3 * shift for i in range(4)},
                "group": np.where(shift, "a", "b"),
            }
        )

        result = run_core_check(frame, "group_differences", max_findings=2)

        assert len(result.findings) == 2
        assert output_of(result)["pairs_flagged"] == 4
        assert "findings_capped" in {n.code for n in result.notes}


class TestContrasts:
    def test_a_flagged_pair_gets_every_pairwise_contrast_with_a_signed_effect(self):
        frame = grouped_frame({"a": 0.0, "b": 0.0, "c": 2.0})

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        contrasts = {
            (c["first_level"], c["second_level"]): c for c in found["contrasts"]
        }
        assert set(contrasts) == {("a", "b"), ("a", "c"), ("b", "c")}
        # positive when the first level is higher: c is above a
        assert contrasts[("a", "c")]["rank_biserial"] < -0.5
        assert contrasts[("a", "c")]["p_adjusted"] < 0.001
        assert contrasts[("a", "b")]["p_adjusted"] > 0.05
        assert contrasts[("a", "c")]["interval"] is not None
        assert contrasts[("a", "b")]["interval"] is None

    def test_the_rank_biserial_of_a_higher_first_group_is_positive(self):
        frame = grouped_frame({"low": 0.0, "high": 2.0}).with_columns(
            pl.col("group").replace({"low": "z_low", "high": "a_high"})
        )

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        (contrast,) = found["contrasts"]
        assert contrast["first_level"] == "a_high"
        assert contrast["rank_biserial"] == pytest.approx(
            2
            * stats.mannwhitneyu(
                frame.filter(pl.col("group") == "a_high")["y"].to_numpy(),
                frame.filter(pl.col("group") == "z_low")["y"].to_numpy(),
            ).statistic
            / (60 * 60)
            - 1,
            abs=1e-9,
        )
        assert contrast["rank_biserial"] > 0.5

    def test_contrasts_use_the_holm_adjustment_over_the_pair(self):
        frame = grouped_frame({"a": 0.0, "b": 0.6, "c": 1.2}, n_each=50)

        found = pair(run_core_check(frame, "group_differences"), "y", "group")

        raw = [c["p_value"] for c in found["contrasts"]]
        assert [c["p_adjusted"] for c in found["contrasts"]] == pytest.approx(
            adjust_p_values(np.array(raw), "holm").tolist()
        )

    def test_too_many_groups_skip_the_contrasts_and_say_so(self):
        frame = grouped_frame({f"g{i}": float(i) for i in range(6)}, n_each=30)

        found = pair(
            run_core_check(frame, "group_differences", max_contrast_levels=4),
            "y",
            "group",
        )

        assert found["contrasts"] == [] and found["contrasts_skipped"]

    def test_the_intervals_repeat_with_the_seed_and_change_without_it(self):
        frame = grouped_frame({"a": 0.0, "b": 1.0})

        again = pair(run_core_check(frame, "group_differences"), "y", "group")
        first = pair(run_core_check(frame, "group_differences"), "y", "group")
        other = pair(run_core_check(frame, "group_differences", seed=3), "y", "group")

        assert first["omega_interval"] == again["omega_interval"]
        assert first["omega_interval"] != other["omega_interval"]


class TestApplicability:
    def test_no_numeric_or_no_categorical_column_is_not_applicable(self):
        only_numeric = pl.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 1, 5, 3]})
        only_categorical = pl.DataFrame({"a": list("xyxy"), "b": list("pqpq")})

        for frame in (only_numeric, only_categorical):
            assert (
                run_core_check(frame, "group_differences").status
                is CheckStatus.NOT_APPLICABLE
            )

    def test_a_categorical_column_over_the_level_limit_is_skipped_with_a_reason(self):
        frame = grouped_frame({"a": 0.0, "b": 1.0}).with_columns(
            (pl.int_range(pl.len()) % 12).cast(pl.String).alias("many")
        )

        output = output_of(run_core_check(frame, "group_differences", max_levels=10))

        assert output["skipped_columns"] == {"many": "more than 10 levels"}

    def test_too_many_pairs_skips_the_check_with_a_hint(self):
        result = run_core_check(
            grouped_frame({"a": 0.0, "b": 1.0}), "group_differences", max_pairs=1
        )

        assert result.status is CheckStatus.SKIPPED
        assert "max_pairs" in (result.notes[0].hint or "")

    def test_a_large_frame_is_sampled_and_says_so(self):
        result = run_core_check(
            grouped_frame({"a": 0.0, "b": 1.0}, n_each=1500),
            "group_differences",
            max_rows=1000,
        )

        assert output_of(result)["rows_used"] == 1000
        assert "rows_sampled" in {n.code for n in result.notes}

    def test_boolean_groups_work(self):
        g = generator()
        flag = g.random(200) < 0.5
        frame = pl.DataFrame({"y": g.normal(size=200) + 2 * flag, "flag": flag})

        found = pair(run_core_check(frame, "group_differences"), "y", "flag")

        assert found["flagged"] and found["n_groups"] == 2

    @pytest.mark.parametrize(
        "bad", [{"alpha": 1.0}, {"n_boot": 10}, {"correction": "sidak"}]
    )
    def test_each_parameter_has_bounds(self, bad):
        with pytest.raises(ValidationError):
            GroupDifferencesParams(**bad)
