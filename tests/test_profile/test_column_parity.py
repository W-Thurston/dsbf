"""Column counts and numeric statistics agree with the legacy ``summarize_*`` tasks.

The legacy tasks run live on each golden dataset as pandas reads it, and the profile
runs on the same CSV as ``load`` reads it. Differences are listed in ``ACCEPTED``.
"""

import math

import pandas as pd
import pytest

from dsbf.data import load
from dsbf.eda.tasks.summarize_nulls import SummarizeNulls
from dsbf.eda.tasks.summarize_numeric import SummarizeNumeric
from dsbf.eda.tasks.summarize_unique import SummarizeUnique
from dsbf.model.profile import NumericProfile
from dsbf.profile import profile_columns
from tests.golden.datasets import GOLDEN_DATASETS, GoldenDataset
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

REL = 1e-9
ACCEPTED: dict[tuple[str, str], str] = {}

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
    for name, profile in profiles.items():
        if not isinstance(profile, NumericProfile):
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

    unexplained = [d for d in differences if (dataset.name, d[0]) not in ACCEPTED]
    assert not unexplained, f"{dataset.name}: unlisted differences {unexplained}"
    listed = {column for name, column in ACCEPTED if name == dataset.name}
    assert {d[0] for d in differences} == listed


@pytest.mark.integration
@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_matches_the_legacy_summaries(dataset: GoldenDataset, tmp_path):
    _compare(dataset, dataset.write_csv(tmp_path), tmp_path)
