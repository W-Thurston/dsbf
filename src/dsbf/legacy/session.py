# dsbf/legacy/session.py
"""What the legacy tasks of one run share: the data, the config and the profile."""

import copy
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from dsbf.data import Dataset
from dsbf.model import DatasetProfile

# The profile's semantic types in the legacy engine's words. The two inferences are
# separate rule sets (the legacy one calls a binary or low-cardinality number
# categorical), so a task can see a column typed differently from before.
TO_LEGACY_TYPE: dict[str, str] = {
    "numeric": "continuous",
    "categorical": "categorical",
    "boolean": "categorical",
    "datetime": "datetime",
    "text": "text",
    "identifier": "id",
    "unknown": "unknown",
}


class LegacySession:
    """The run-wide state behind a set of adapted legacy tasks.

    The caller builds one from the dataset and profile it passes to ``execute``,
    plus the legacy config, a directory for the plots tasks write (or ``None``) and
    the reference frame some tasks compare against. Adapted checks read the data
    and the profile from here, because a ``CheckContext`` shows a check only the
    columns it applies to and legacy tasks choose their own columns.

    Each check gets its own copy of a pandas frame, as tasks that edit their input
    would otherwise corrupt the checks running beside them.
    """

    def __init__(
        self,
        dataset: Dataset,
        profile: DatasetProfile,
        *,
        config: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        reference_data: Any | None = None,
    ) -> None:
        if [column.name for column in profile.columns] != dataset.columns:
            raise ValueError("the profile and the dataset have different columns")
        self.dataset = dataset
        self.profile = profile
        self.config: dict[str, Any] = copy.deepcopy(config or {})
        self.output_dir = None if output_dir is None else str(output_dir)
        self.reference_data = reference_data
        self._lock = threading.Lock()
        self._pandas: pd.DataFrame | None = None
        self._reliability_flags: dict[str, Any] | None = None

    @property
    def backend(self) -> str:
        return str(self.config.get("engine", {}).get("backend", "pandas"))

    @property
    def semantic_types(self) -> dict[str, str]:
        """Every column's type, in the legacy engine's words."""
        return {c.name: TO_LEGACY_TYPE[c.semantic_type] for c in self.profile.columns}

    def frame(self) -> pd.DataFrame | pl.DataFrame:
        """A private copy of the data in the backend the config names."""
        if self.backend == "polars":
            return self.dataset.frame.clone()
        with self._lock:
            if self._pandas is None:
                self._pandas = self.dataset.frame.to_pandas()
            return self._pandas.copy()

    def inferred_dtypes(self) -> dict[str, str]:
        """Every column's dtype as the legacy type inference reported it."""
        if self.backend == "polars":
            return {name: str(dtype) for name, dtype in self.dataset.schema.items()}
        with self._lock:
            if self._pandas is None:
                self._pandas = self.dataset.frame.to_pandas()
            return {str(c): str(t) for c, t in self._pandas.dtypes.items()}

    def reliability_flags(self, compute: Any) -> dict[str, Any]:
        """The data's reliability flags, worked out once for the whole run."""
        with self._lock:
            if self._reliability_flags is None:
                self._reliability_flags = compute()
            return copy.deepcopy(self._reliability_flags)

    def reference_digest(self) -> str | None:
        """A hash of the reference frame, or None without one."""
        frame = self.reference_data
        if frame is None:
            return None
        if isinstance(frame, pl.DataFrame):
            return str(int(frame.hash_rows().sum()))
        return str(int(pd.util.hash_pandas_object(frame, index=True).sum()))
