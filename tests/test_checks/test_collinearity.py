"""The collinearity check on small frames."""

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from dsbf.checks.collinearity import CollinearityParams
from dsbf.model import CheckStatus, Dimension, Severity
from tests.test_checks.support import actions_of, output_of, run_core_check

pytestmark = pytest.mark.unit


def generator(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def vifs(result) -> dict[str, float]:
    return {c["column"]: c["vif"] for c in output_of(result)["columns"]}


def independent_frame(n: int = 400) -> pl.DataFrame:
    g = generator()
    return pl.DataFrame(
        {
            "a": g.normal(loc=1000, scale=5, size=n),
            "b": g.normal(loc=-300, scale=2, size=n),
            "c": g.normal(size=n),
        }
    )


class TestVif:
    def test_independent_columns_with_large_means_have_a_vif_of_one(self):
        result = run_core_check(independent_frame(), "collinearity")

        assert result.status is CheckStatus.OK
        assert all(1.0 <= v < 1.05 for v in vifs(result).values())
        assert result.findings == ()

    def test_the_vif_is_the_diagonal_of_the_inverse_correlation_matrix(self):
        g = generator()
        x = g.normal(size=300)
        frame = pl.DataFrame(
            {"x": x, "y": x + g.normal(size=300), "z": g.normal(size=300) + 0.5 * x}
        )

        got = vifs(run_core_check(frame, "collinearity"))

        expected = np.diag(np.linalg.inv(np.corrcoef(frame.to_numpy(), rowvar=False)))
        assert [got[c] for c in "xyz"] == pytest.approx(expected)

    def test_the_vif_equals_one_over_one_minus_r_squared_with_an_intercept(self):
        g = generator()
        x, noise = g.normal(size=500), g.normal(size=500)
        frame = pl.DataFrame({"x": x + 50.0, "y": 2 * x + noise})
        r2 = np.corrcoef(frame["x"], frame["y"])[0, 1] ** 2

        got = vifs(run_core_check(frame, "collinearity"))

        assert got["x"] == pytest.approx(1 / (1 - r2))
        assert got["y"] == pytest.approx(1 / (1 - r2))

    def test_a_column_the_others_nearly_reproduce_is_flagged_with_its_partners(self):
        g = generator()
        base = g.normal(size=300)
        frame = pl.DataFrame(
            {
                "base": base,
                "copy": base + 0.05 * g.normal(size=300),
                "other": g.normal(size=300),
            }
        )

        result = run_core_check(frame, "collinearity")

        by_column = {f.subject.columns[0]: f for f in result.findings}
        assert set(by_column) == {"base", "copy"}
        finding = by_column["base"]
        assert finding.severity is Severity.WARN
        assert finding.dimension is Dimension.REDUNDANCY
        assert finding.code == "multicollinearity"
        assert finding.evidence.method == "vif_centred"
        assert finding.evidence.n_used == 300
        assert finding.evidence.metrics["partner_1"] == "copy"
        assert float(finding.evidence.metrics["partner_1_r"]) > 0.99  # type: ignore[arg-type]
        assert {a.action for a in actions_of(finding)} == {
            "drop",
            "regularise",
            "transform",
        }

    def test_an_exact_copy_has_a_capped_vif_and_no_infinity(self):
        g = generator()
        x = g.normal(size=100)
        frame = pl.DataFrame({"x": x, "copy": x, "scaled": 3 * x + 1})

        result = run_core_check(frame, "collinearity")

        output = output_of(result)
        assert all(c["vif"] == 1e6 and c["capped"] for c in output["columns"])
        assert np.isfinite(output["condition_number"])
        assert {f.title for f in result.findings} == {
            "Exact linear combination of other columns"
        }

    def test_the_vif_is_rescaled_so_columns_near_the_float_limit_do_not_overflow(self):
        g = generator()
        frame = pl.DataFrame(
            {"big": g.normal(size=100) * 1e300, "small": g.normal(size=100)}
        )

        got = vifs(run_core_check(frame, "collinearity"))

        assert got["big"] == pytest.approx(1.0, abs=0.1)


class TestMissingData:
    def frame_with_gaps(self, n: int = 400) -> pl.DataFrame:
        g = generator()
        x = g.normal(size=n)
        sparse = g.normal(size=n)
        sparse[: int(n * 0.7)] = np.nan
        return pl.DataFrame(
            {
                "x": x,
                "y": x + g.normal(size=n),
                "z": g.normal(size=n),
                "sparse": sparse,
            }
        )

    def test_a_sparse_column_is_left_out_before_any_row_is(self):
        result = run_core_check(self.frame_with_gaps(), "collinearity")

        output = output_of(result)
        assert output["n_used"] == 400
        assert output["skipped_columns"] == {"sparse": "70% of values are missing"}
        assert output["n_columns"] == 3

    def test_rows_are_dropped_only_for_the_columns_used(self):
        g = generator()
        frame = self.frame_with_gaps().with_columns(
            pl.Series(
                "z",
                [None if i % 10 == 0 else v for i, v in enumerate(g.normal(size=400))],
            )
        )

        output = output_of(run_core_check(frame, "collinearity"))

        assert output["n_used"] == 360
        assert output["rows_total"] == 400

    def test_a_column_is_dropped_when_too_few_complete_rows_remain(self):
        g = generator()
        n = 60
        # three columns each missing on a different 30% of rows: no row is complete
        columns = {}
        for i, name in enumerate("abc"):
            values = g.normal(size=n)
            values[i * 20 : i * 20 + 20] = np.nan
            columns[name] = values
        columns["d"] = g.normal(size=n)

        result = run_core_check(pl.DataFrame(columns), "collinearity", min_rows=30)

        output = output_of(result)
        assert result.status is CheckStatus.OK
        assert output["dropped_for_rows"]
        assert output["n_used"] >= 30
        assert output["n_used"] >= output["n_columns"] + 1

    def test_too_few_rows_for_the_columns_leaves_the_check_not_applicable(self):
        g = generator()
        frame = pl.DataFrame({c: g.normal(size=5) for c in "abcdefg"})

        result = run_core_check(frame, "collinearity")

        assert result.status is CheckStatus.NOT_APPLICABLE
        assert "fewer than the 10 needed" in result.notes[0].message

    def test_kitchen_like_frame_with_no_complete_row_still_gets_vifs(self):
        g = generator()
        n = 500
        columns = {}
        for i in range(8):
            values = g.normal(size=n)
            values[g.random(n) < 0.15] = np.nan
            columns[f"c{i}"] = values
        frame = pl.DataFrame(columns).fill_nan(None)
        assert frame.drop_nulls().height < 0.3 * n

        output = output_of(run_core_check(frame, "collinearity"))

        assert output["n_columns"] >= 2
        assert output["n_used"] >= 30

    def test_a_column_constant_over_the_complete_rows_is_left_out_with_its_reason(self):
        g = generator()
        n = 200
        varies_only_where_other_is_missing = np.ones(n)
        varies_only_where_other_is_missing[:20] = g.normal(size=20)
        gappy = g.normal(size=n)
        gappy[:20] = np.nan
        frame = pl.DataFrame(
            {
                "a": g.normal(size=n),
                "gappy": gappy,
                "narrow": varies_only_where_other_is_missing,
            }
        ).fill_nan(None)

        output = output_of(run_core_check(frame, "collinearity"))

        assert output["skipped_columns"] == {
            "narrow": "no variation among the complete rows"
        }
        assert output["n_used"] == 180

    def test_infinite_values_count_as_missing(self):
        frame = independent_frame(100).with_columns(
            pl.Series("c", [float("inf"), *generator().normal(size=99)])
        )

        output = output_of(run_core_check(frame, "collinearity"))

        assert output["n_used"] == 99


class TestApplicability:
    def test_one_numeric_column_is_not_applicable(self):
        result = run_core_check(independent_frame().select("a"), "collinearity")

        assert result.status is CheckStatus.NOT_APPLICABLE

    def test_too_many_columns_skips_the_check_with_a_hint(self):
        result = run_core_check(independent_frame(), "collinearity", max_columns=2)

        assert result.status is CheckStatus.SKIPPED
        assert "max_columns" in (result.notes[0].hint or "")

    def test_a_large_frame_is_sampled_and_says_so(self):
        frame = independent_frame(2000)

        result = run_core_check(frame, "collinearity", max_rows=1000)

        assert output_of(result)["rows_read"] == 1000
        assert "rows_sampled" in {n.code for n in result.notes}

    def test_few_rows_lower_the_confidence_of_a_finding(self):
        g = generator()
        base = g.normal(size=20)
        frame = pl.DataFrame({"a": base, "b": base + 0.01 * g.normal(size=20)})

        result = run_core_check(frame, "collinearity")

        assert {f.confidence.value for f in result.findings} == {"low"}

    def test_parameter_bounds_are_enforced(self):
        with pytest.raises(ValidationError):
            CollinearityParams(vif_threshold=1.0)
        with pytest.raises(ValidationError):
            CollinearityParams(max_missing_share=1.0)
