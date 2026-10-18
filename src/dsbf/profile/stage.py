# dsbf/profile/stage.py
"""How prepared the data is, read from its profile."""

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from dsbf.model.base import Contract
from dsbf.model.profile import DatasetProfile
from dsbf.model.vocabulary import Stage


class StageThresholds(Contract):
    """The limits stage inference applies; the names are the config's own keys."""

    null_ratio_raw_threshold: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="A mean column null ratio above this is `raw`.",
    )
    percent_numeric_for_model_ready: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="The share of numeric columns from which data is `modeling`.",
    )
    high_cardinality_threshold: int = Field(
        default=50,
        ge=1,
        description="A categorical column with more distinct values than this is "
        "high-cardinality, which keeps the data from being `cleaned`.",
    )
    max_null_ratio_cleaned: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="`cleaned` needs a mean column null ratio below this.",
    )
    max_null_ratio_modeling: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="`modeling` needs a mean column null ratio below this.",
    )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "StageThresholds":
        """The thresholds under ``engine.stage_inference``; defaults where absent."""
        section = (config.get("engine") or {}).get("stage_inference") or {}
        return cls.model_validate(section)


def infer_stage(
    profile: DatasetProfile, thresholds: StageThresholds | None = None
) -> Stage:
    """The stage the data looks to be at: ``raw``, ``cleaned`` or ``modeling``.

    Rules are tried in this order. Data with a high mean null ratio is ``raw``. Data
    that is mostly numeric and nearly complete is ``modeling``. Data that is nearly
    complete and has no high-cardinality categorical column is ``cleaned``. Anything
    else hasn't been prepared yet and is ``raw``. A profile with no rows or no columns
    is ``raw``.

    The answer depends only on the profile, so the same data gives the same stage
    whichever library it came from.
    """
    limits = thresholds or StageThresholds()
    columns = profile.columns
    if not columns or profile.n_rows == 0:
        return Stage.RAW

    mean_null_ratio = sum(c.counts.nulls for c in columns) / (
        len(columns) * profile.n_rows
    )
    if mean_null_ratio > limits.null_ratio_raw_threshold:
        return Stage.RAW

    numeric_share = sum(c.semantic_type == "numeric" for c in columns) / len(columns)
    if (
        numeric_share >= limits.percent_numeric_for_model_ready
        and mean_null_ratio < limits.max_null_ratio_modeling
    ):
        return Stage.MODELING

    high_cardinality = any(
        c.semantic_type == "categorical"
        and c.counts.distinct > limits.high_cardinality_threshold
        for c in columns
    )
    if mean_null_ratio < limits.max_null_ratio_cleaned and not high_cardinality:
        return Stage.CLEANED
    return Stage.RAW
