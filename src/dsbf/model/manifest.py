# dsbf/model/manifest.py
"""What a run was given and what it was asked to do: the dataset and the plan."""

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from dsbf.model.base import Contract
from dsbf.model.evidence import SamplingInfo
from dsbf.model.scrub import json_safe

SourceKind = Literal["dataframe", "file", "glob", "sql", "table"]


class SourceRef(Contract):
    """Where the data came from. An in-memory frame has no location or format."""

    kind: SourceKind
    location: str | None = Field(
        default=None, description="A path, glob, URL or table name."
    )
    format: str | None = Field(default=None, description="For example `csv`.")


class DatasetManifest(Contract):
    """The identity of the data a run looked at."""

    name: str = Field(min_length=1)
    modality: Literal["tabular"] = "tabular"
    source: SourceRef
    key: str = Field(
        default="",
        description="Groups the runs of one dataset into a history: the dataset the "
        "run was given, else the source's location. Empty for an in-memory frame "
        "with neither.",
    )
    fingerprint: str = Field(
        min_length=1, description="A content hash: equal data, equal fingerprint."
    )
    n_rows: int = Field(ge=0, description="Rows in the source, before any sampling.")
    n_columns: int = Field(ge=0)
    sampling: SamplingInfo | None = Field(
        default=None, description="Absent when the whole dataset was used."
    )

    @model_validator(mode="after")
    def _check_sampling(self) -> Self:
        if self.sampling and self.sampling.original_rows != self.n_rows:
            raise ValueError("a sample's original_rows is the source's n_rows")
        return self

    @property
    def analysed_rows(self) -> int:
        return self.sampling.sampled_rows if self.sampling else self.n_rows


class PlannedCheck(Contract):
    """One check the plan includes, with the parameters it will run under."""

    check_id: str = Field(min_length=1)
    check_version: str = Field(min_length=1)
    pack: str | None = None
    params: dict[str, Any] = Field(
        default_factory=dict, description="The resolved parameters."
    )
    depends_on: tuple[str, ...] = Field(
        default=(), description="The planned checks whose results this one needs."
    )

    @model_validator(mode="before")
    @classmethod
    def _make_params_json_safe(cls, data: Any) -> Any:
        """Make `params` JSON-safe; NaN or infinity there is an error."""
        if isinstance(data, dict) and isinstance(data.get("params"), dict):
            params, _ = json_safe(data["params"], null_non_finite=False)
            data = {**data, "params": params}
        return data


class PlanManifest(Contract):
    """The checks a run was asked to perform, and what produced the plan."""

    dsbf_version: str = Field(min_length=1)
    depth: str | None = Field(default=None, description="The depth preset, if any.")
    checks: tuple[PlannedCheck, ...] = ()

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        ids = [check.check_id for check in self.checks]
        if len(set(ids)) != len(ids):
            raise ValueError("a plan lists each check once")
        for check in self.checks:
            for dependency in check.depends_on:
                if dependency == check.check_id:
                    raise ValueError(f"{check.check_id} depends on itself")
                if dependency not in ids:
                    raise ValueError(
                        f"{check.check_id} depends on {dependency!r}, "
                        "which the plan lacks"
                    )
        return self

    @property
    def check_ids(self) -> tuple[str, ...]:
        return tuple(check.check_id for check in self.checks)
