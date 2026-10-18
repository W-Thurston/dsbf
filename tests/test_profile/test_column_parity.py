"""Column profiles agree with the legacy ``summarize_*`` and length tasks.

The legacy tasks run live on each golden dataset as pandas reads it, and the profile
runs on the same CSV as ``load`` reads it. Counts and numeric statistics have no
accepted differences except the integer codes in ``CODES_WITHOUT_STATISTICS``; top
values and lengths list theirs in ``ACCEPTED_VALUES``.
"""

import math

import pandas as pd
import pytest

from dsbf.data import load
from dsbf.eda.tasks.categorical_length_stats import CategoricalLengthStats
from dsbf.eda.tasks.summarize_nulls import SummarizeNulls
from dsbf.eda.tasks.summarize_numeric import SummarizeNumeric
from dsbf.eda.tasks.summarize_unique import SummarizeUnique
from dsbf.eda.tasks.summarize_value_counts import SummarizeValueCounts
from dsbf.model.profile import CategoricalProfile, NumericProfile, TextProfile
from dsbf.profile import profile_columns
from dsbf.profile.values import MAX_TOP_VALUE_CHARS
from tests.golden.datasets import GOLDEN_DATASETS, GoldenDataset
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies
from tests.test_profile.accepted import cells

REL = 1e-9
# Reasons for both tables are in ``accepted``.
ACCEPTED_VALUES = cells("nul-byte-cuts-a-pandas-field")
CODES_WITHOUT_STATISTICS = cells("categorical-integer-codes-have-no-statistics")
LEGACY_TOP_K = 5
# Numeric-only datasets: nothing to compare.
NO_CATEGORICAL_COLUMNS = {"severe_multicollinearity", "single_column"}

LEGACY_PERCENTILES = {
    0.01: "1%",
    0.05: "5%",
    0.25: "25%",
    0.5: "50%",
    0.75: "75%",
    0.95: "95%",
    0.99: "99%",
}


def _legacy(task_cls, frame: pd.DataFrame, tmp_path):
    ctx, _ = make_ctx_and_task(
        task_cls, frame, global_overrides={"output_dir": str(tmp_path)}
    )
    result = run_task_with_dependencies(ctx, task_cls)
    assert result.status == "success"
    assert result.data is not None
    return result.data


def _same(old: float, new: float | None) -> bool:
    if new is None:
        return math.isnan(old)
    return old == pytest.approx(new, rel=REL, abs=1e-12) or old == new


def _compare(dataset: GoldenDataset, path, tmp_path) -> None:
    frame = pd.read_csv(path)
    numeric = _legacy(SummarizeNumeric, frame, tmp_path)
    nulls = _legacy(SummarizeNulls, frame, tmp_path)
    unique = _legacy(SummarizeUnique, frame, tmp_path)
    profiles = {p.name: p for p in profile_columns(load(path))}

    differences: list[tuple[str, str, object, object]] = []
    for name, profile in profiles.items():
        counts = profile.counts
        for what, old, new in (
            ("nulls", nulls["null_counts"][name], counts.nulls),
            ("distinct", unique[name], counts.distinct),
        ):
            if old != new:
                differences.append((name, what, old, new))
    # The legacy task summarises every numeric dtype, whatever its semantic type, so
    # it also covers integer codes the profile calls categorical; those have no
    # statistics here. Type agreement is checked in ``test_type_parity``.
    codes = set()
    for name, profile in profiles.items():
        if not isinstance(profile, NumericProfile):
            if name in numeric:
                assert isinstance(profile, CategoricalProfile), (name, profile)
                codes.add(name)
            continue
        if name not in numeric or profile.stats is None:
            differences.append((name, "missing", name in numeric, profile.stats))
            continue
        legacy = numeric[name]
        stats = profile.stats
        expected = {
            "count": profile.counts.non_null - profile.non_finite,
            "min": stats.min,
            "max": stats.max,
            "mean": stats.mean,
            "std": stats.std,
        }
        for quantile in stats.quantiles:
            expected[LEGACY_PERCENTILES[quantile.p]] = quantile.value
        differences.extend(
            (name, key, legacy[key], value)
            for key, value in expected.items()
            if not _same(legacy[key], value)
        )

    assert not differences, f"{dataset.name}: unlisted differences {differences}"
    # A listed difference that no longer happens should come off the list.
    assert codes == {c for name, c in CODES_WITHOUT_STATISTICS if name == dataset.name}


@pytest.mark.integration
@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_matches_the_legacy_summaries(dataset: GoldenDataset, tmp_path):
    _compare(dataset, dataset.write_csv(tmp_path), tmp_path)


def _same_top(old: list[tuple], new: list[tuple]) -> bool:
    """Equal values and counts; floats may differ in the last digit.

    pandas' default CSV float parser is not correctly rounded, so a value such as
    100.00020190141393 reads one unit in the last place away from Polars' reading.
    """
    return len(old) == len(new) and all(
        o_count == n_count
        and (o_value == n_value or o_value == pytest.approx(n_value, rel=1e-12))
        for (o_value, o_count), (n_value, n_count) in zip(old, new, strict=True)
    )


def _compare_values(dataset: GoldenDataset, path, tmp_path) -> None:
    frame = pd.read_csv(path)
    legacy_top = _legacy(SummarizeValueCounts, frame, tmp_path)
    legacy_lengths = _legacy(CategoricalLengthStats, frame, tmp_path)
    profiles = profile_columns(load(path), top_k=LEGACY_TOP_K)

    differences: list[tuple[str, str, object, object]] = []
    compared = 0
    for profile in profiles:
        if not isinstance(profile, CategoricalProfile | TextProfile):
            continue
        name = profile.name
        if name in legacy_top:
            # The legacy table counts null as a value (`dropna=False`); the profile's
            # top values are of the non-null values, and a text column lists only
            # values that repeat; strings over the length limit are never listed.
            minimum = 2 if isinstance(profile, TextProfile) else 1
            expected = [
                (value, count)
                for value, count in legacy_top[name].items()
                if not pd.isna(value)
                and count >= minimum
                and len(str(value)) <= MAX_TOP_VALUE_CHARS
            ]
            actual = [(v.value, v.count) for v in profile.top_values]
            compared += 1
            if not _same_top(expected, actual[: len(expected)]):
                differences.append((name, "top_values", expected, actual))
        if name in legacy_lengths and profile.lengths is not None:
            old = legacy_lengths[name]
            lengths = profile.lengths
            compared += 1
            for key, value in (
                ("min_length", lengths.min),
                ("max_length", lengths.max),
                ("mean_length", lengths.mean),
            ):
                if old[key] != pytest.approx(value, rel=REL):
                    differences.append((name, key, old[key], value))

    unexplained = [
        d for d in differences if (dataset.name, d[0]) not in ACCEPTED_VALUES
    ]
    assert not unexplained, f"{dataset.name}: unlisted differences {unexplained}"
    listed = {column for name, column in ACCEPTED_VALUES if name == dataset.name}
    assert {d[0] for d in differences} == listed
    assert compared or dataset.name in NO_CATEGORICAL_COLUMNS


@pytest.mark.integration
@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_matches_the_legacy_top_values_and_lengths(dataset: GoldenDataset, tmp_path):
    _compare_values(dataset, dataset.write_csv(tmp_path), tmp_path)
