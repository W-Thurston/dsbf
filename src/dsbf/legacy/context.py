# dsbf/legacy/context.py
"""The legacy ``AnalysisContext``, built for one adapted check."""

from typing import Any, cast

import pandas as pd
import polars as pl

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.legacy.session import LegacySession
from dsbf.utils.backend import is_polars
from dsbf.utils.reliability_stats import compute_reliability_flags as compute_flags


class LegacyContext(AnalysisContext):
    """What a legacy task sees: the data, its config and the results it reads.

    ``results`` holds only the upstream tasks' results, rebuilt from the engine's
    stored output. The metadata holds the types the profile gives every column,
    the dtypes the legacy inference reported, and no plugin warnings.
    """

    def __init__(
        self,
        session: LegacySession,
        config: dict[str, Any],
        upstream: dict[str, TaskResult],
    ) -> None:
        super().__init__(
            data=session.frame(),
            config=config,
            output_dir=session.output_dir,
            reference_data=session.reference_data,
        )
        self._session = session
        self.results = dict(upstream)
        self.metadata["semantic_types"] = session.semantic_types
        self.metadata["inferred_dtypes"] = session.inferred_dtypes()
        self.metadata["plugin_warnings"] = []

    def compute_reliability_flags(self, df: pd.DataFrame | pl.DataFrame) -> None:
        if self.reliability_flags:
            return
        frame = (
            cast(pl.DataFrame, df).to_pandas()
            if is_polars(df)
            else cast(pd.DataFrame, df)
        )
        self.reliability_flags = self._session.reliability_flags(
            lambda: compute_flags(frame)
        )
