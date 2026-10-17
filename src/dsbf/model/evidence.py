# dsbf/model/evidence.py
"""The numbers behind a finding, and where they came from."""

import math
from datetime import date, datetime
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from dsbf.model.base import Contract, FiniteFloat

# What a metric may hold. `bool` comes first so a flag isn't read as a number.
Metric = bool | int | FiniteFloat | str | None


class Interval(Contract):
    """A confidence interval for a statistic."""

    low: FiniteFloat
    high: FiniteFloat
    level: FiniteFloat = Field(gt=0, lt=1, description="Coverage, for example 0.95.")
    method: str | None = None

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError("an interval's low bound is at most its high bound")
        return self


class AssumptionCheck(Contract):
    """Whether a statistical assumption behind a result held."""

    name: str = Field(min_length=1)
    satisfied: bool | None = Field(
        default=None, description="None when the assumption could not be assessed."
    )
    method: str | None = None
    detail: str | None = None


class SamplingInfo(Contract):
    """How a sample was drawn from the data. No sampling is the absence of this."""

    strategy: Literal["head", "random", "stratified"]
    original_rows: int = Field(ge=0)
    sampled_rows: int = Field(ge=0)
    seed: int | None = None
    stratify_by: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_strategy(self) -> Self:
        if self.sampled_rows > self.original_rows:
            raise ValueError("a sample has at most as many rows as the data")
        if self.strategy == "random" and self.seed is None:
            raise ValueError("a random sample records its seed")
        if self.strategy != "random" and self.seed is not None:
            raise ValueError("only a random sample has a seed")
        if self.strategy == "stratified" and not self.stratify_by:
            raise ValueError("a stratified sample names its stratification columns")
        if self.strategy != "stratified" and self.stratify_by:
            raise ValueError("only a stratified sample has stratification columns")
        return self


def _normalise_metric(value: Any) -> tuple[Any, bool]:
    """Return the JSON-safe form of a metric and whether it was undefined."""
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (bool, int, float, str, datetime)):
        value = item()  # NumPy scalar -> Python scalar
    if isinstance(value, datetime):
        if value != value:  # NaT
            return None, True
        return value.isoformat(), False
    if isinstance(value, date):
        return value.isoformat(), False
    if isinstance(value, float) and not math.isfinite(value):
        return None, True
    return value, False


class Evidence(Contract):
    """The numbers behind a finding: what was measured, on how much data, and how."""

    metrics: dict[str, Metric]
    non_finite_metrics: tuple[str, ...] = Field(
        default=(),
        description=(
            "Metrics whose value was NaN, infinite or NaT, so they are null in "
            "`metrics`."
        ),
    )
    n_used: int = Field(ge=0)
    n_total: int = Field(ge=0)
    method: str = Field(min_length=1, description="For example `spearman_r` or `vif`.")
    assumptions: tuple[AssumptionCheck, ...] = ()
    interval: Interval | None = None
    sampled: SamplingInfo | None = None
    details_ref: str | None = Field(
        default=None, description="Pointer to bulky evidence, fetched on demand."
    )

    @model_validator(mode="before")
    @classmethod
    def _scrub_metrics(cls, data: Any) -> Any:
        """Make metric values JSON-safe, flagging the undefined ones (never raising).

        NumPy scalars become Python scalars, dates become ISO 8601 strings, and
        NaN, infinity and NaT become null with the metric's name recorded.
        """
        if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
            return data
        metrics: dict[Any, Any] = {}
        undefined = set(data.get("non_finite_metrics") or ())
        for name, value in data["metrics"].items():
            metrics[name], is_undefined = _normalise_metric(value)
            if is_undefined:
                undefined.add(name)
        return {
            **data,
            "metrics": metrics,
            "non_finite_metrics": tuple(sorted(undefined)),
        }

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if self.n_used > self.n_total:
            raise ValueError("n_used is at most n_total")
        return self
