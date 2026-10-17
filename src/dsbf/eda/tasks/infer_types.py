# dsbf/eda/tasks/infer_types.py

from typing import TYPE_CHECKING, Any

import pandas as pd

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars

if TYPE_CHECKING:
    from pandas import Timestamp
    from pandas._libs import NaTType
    from polars import Series

# Patterns that identify fiscal year, quarter, or date-period labels that look
# numeric but are categorical (e.g. "FY2024", "Q1 2023", "'23", "2023-Q1").
_YEAR_LIKE_PATTERNS: tuple[str, ...] = (
    r"(?i)^FY\d{2,4}$",
    r"^\d{4}$",
    r"^'\d{2}$",
    r"^\d{4}[-/]\d{2,4}$",
    r"^'\d{2}[-/]\d{2}$",
    r"^\d{2}/\d{2}$",
    r"(?i)^Q[1-4]\s?\d{2,4}$",
)


@register_task(
    display_name="Infer Column Types",
    description="Infers both raw and analysis-intent dtypes for each column.",
    depends_on=[],
    profiling_depth="basic",
    stage="raw",
    phase="eda",
    domain="core",
    runtime_estimate="fast",
    tags=["typing", "metadata"],
)
class InferTypes(BaseTask):
    """
    Infer raw dtypes and analysis-intent semantic types for each column.

    The task records both the underlying dataframe dtype (e.g. ``int64``,
    ``object``) and a DSBF semantic type that downstream tasks use for plot
    routing, column filtering, diagnostics, and recommendations.

    Semantic types and raw dtypes are written to the shared ``AnalysisContext``
    metadata store so all downstream tasks can read them without re-computing.

    Supported semantic types:

    - ``continuous`` - numeric column with many distinct values
    - ``categorical`` - low-cardinality column or boolean
    - ``datetime`` - parseable date/time column
    - ``text`` - long free-text string column (mean length > 30 chars)
    - ``id`` - near-unique string column (identifier)
    - ``unknown`` - all-null or unclassifiable
    """

    def run(self) -> None:
        """
        Infer semantic types for every column and write results to context.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df: pd.DataFrame = self._to_pandas(self.input_data)
            results: dict[str, dict[str, str]] = {}

            for column in df.columns:
                series: Series = df[column]
                inferred_dtype = str(series.dtype)
                analysis_intent_dtype: str = self._infer_analysis_intent(series)
                results[column] = {
                    "inferred_dtype": inferred_dtype,
                    "analysis_intent_dtype": analysis_intent_dtype,
                }

            self._store_metadata(results)
            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": f"Inferred types for {len(results)} columns."},
                data=results,
            )
        except Exception as error:
            if self.context:
                raise
            self._log(
                f"Task failed outside execution context: "
                f"{type(error).__name__} - {error}",
                level="warn",
            )
            self.output = make_failure_result(self.name, error)

    def _to_pandas(self, data: Any) -> pd.DataFrame:
        """
        Convert a supported dataframe backend to pandas.

        Args:
            data: Input DataFrame (pandas or Polars).

        Returns:
            pandas DataFrame.

        Raises:
            TypeError: If data is neither pandas nor Polars.

        """
        if is_polars(data):
            return data.to_pandas()
        if isinstance(data, pd.DataFrame):
            return data
        msg: str = f"Expected pandas or polars dataframe, got {type(data).__name__}."
        raise TypeError(msg)

    def _store_metadata(self, results: dict[str, dict[str, str]]) -> None:
        """
        Persist inferred type metadata into the shared analysis context.

        Args:
            results: Dict mapping column name →
                ``{inferred_dtype, analysis_intent_dtype}``.

        """
        if not self.context:
            return

        self.context.set_metadata(
            "semantic_types",
            {col: info["analysis_intent_dtype"] for col, info in results.items()},
        )
        self.context.set_metadata(
            "inferred_dtypes",
            {col: info["inferred_dtype"] for col, info in results.items()},
        )

    def _infer_analysis_intent(self, series: pd.Series) -> str:  # noqa: PLR0911
        """
        Infer the DSBF semantic type for a single column.

        Args:
            series: Column values from the dataframe.

        Returns:
            Semantic type string: one of ``continuous``, ``categorical``,
            ``datetime``, ``text``, ``id``, or ``unknown``.

        """
        non_null: Series = series.dropna()
        if non_null.empty:
            return "unknown"

        nunique = int(non_null.nunique())
        total = int(non_null.size)
        unique_ratio: float = nunique / total if total else 0.0

        if pd.api.types.is_bool_dtype(non_null):
            return "categorical"

        if pd.api.types.is_numeric_dtype(non_null):
            return self._infer_numeric_intent(
                nunique=nunique,
                unique_ratio=unique_ratio,
                series=non_null,
            )

        if pd.api.types.is_datetime64_any_dtype(non_null):
            return "datetime"

        string_values: Series = self._as_string_series(non_null)
        if self._is_datetime_like(string_values):
            return "datetime"
        if self._is_year_like(string_values, nunique):
            return "categorical"
        if self._is_id_like(string_values, unique_ratio):
            return "id"
        if float(string_values.str.len().mean()) > 30:  # noqa: PLR2004
            return "text"
        return "categorical"

    def _infer_numeric_intent(
        self,
        *,
        nunique: int,
        unique_ratio: float,
        series: pd.Series | None = None,
    ) -> str:
        """
        Infer semantic type for a numeric column.

        Evaluates three criteria in priority order:

        1. **Binary columns** (exactly 2 distinct values) → ``categorical``
        2. **Low-cardinality columns** (unique ratio < 5% and ≤ 20 distinct
        values) → ``categorical``
        3. **Unix timestamps** (integer column whose median falls in the
        epoch-seconds range 2001-2033 or the epoch-milliseconds equivalent)
        → ``datetime``. Requires ``series`` to be passed; skipped otherwise.
        4. **Everything else** → ``continuous``

        Args:
            nunique: Number of distinct non-null values.
            unique_ratio: Ratio of distinct values to total non-null values.
            series: The non-null numeric Series, used for Unix timestamp
                detection. Optional - pass ``None`` to skip that check.

        Returns:
            One of ``"categorical"``, ``"datetime"``, or ``"continuous"``.

        """
        if nunique == 2:
            return "categorical"
        if unique_ratio < 0.05 and nunique <= 20:
            return "categorical"
        # Unix timestamp heuristic: large integers in plausible epoch range
        if series is not None and self._is_unix_timestamp(series):
            return "datetime"
        return "continuous"

    def _is_unix_timestamp(self, series: pd.Series) -> bool:
        """True if values look like Unix epoch seconds or milliseconds."""
        non_null: Series = series.dropna()
        if len(non_null) < 5:
            return False
        # Epoch seconds: ~1e9 to ~2e9 (year 2001-2033)
        # Epoch milliseconds: ~1e12 to ~2e12
        median = float(non_null.median())
        return (1_000_000_000 < median < 2_000_000_000) or (
            1_000_000_000_000 < median < 2_000_000_000_000
        )

    def _as_string_series(self, series: pd.Series) -> pd.Series:
        """
        Return a whitespace-trimmed string view of a series.

        Args:
            series: Non-null column values.

        Returns:
            String-cast, whitespace-stripped pandas Series.

        """
        return series.astype("string").str.strip()

    def _is_datetime_like(self, series: pd.Series) -> bool:
        """
        Check whether string values parse cleanly as datetimes.

        Tries ISO format first, then falls back to dateutil inference.

        Args:
            series: String-cast column values.

        Returns:
            True if the majority of values parse as datetimes.

        """
        if series.empty:
            return False
        try:
            parsed: NaTType | Timestamp = pd.to_datetime(
                series,
                format="%Y-%m-%d",
                errors="coerce",
                utc=True,
            )
            if parsed.notna().mean() > 0.9:
                return True
        except (TypeError, ValueError):
            pass
        try:
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            return bool(parsed.notna().mean() > 0.9)
        except (TypeError, ValueError):
            return False

    def _is_year_like(self, series: pd.Series, nunique: int) -> bool:
        """
        Check whether string values mostly represent fiscal or year-period labels.

        Args:
            series: String-cast column values.
            nunique: Number of distinct values.

        Returns:
            True if > 50% of values match a year/fiscal-period pattern.

        """
        if series.empty or nunique >= 100:  # noqa: PLR2004
            return False

        combined_match: Series = pd.Series(False, index=series.index, dtype="boolean")
        for pattern in _YEAR_LIKE_PATTERNS:
            combined_match = combined_match | series.str.contains(pattern, na=False)

        return float(combined_match.fillna(False).mean()) > 0.5  # noqa: PLR2004

    def _is_id_like(self, series: pd.Series, unique_ratio: float) -> bool:
        """
        Check whether string values look like identifiers rather than labels.

        Args:
            series: String-cast column values.
            unique_ratio: Ratio of distinct values to total non-null values.

        Returns:
            True if > 80% of values are hex-like strings, or if unique ratio > 0.9.

        """
        if series.empty:
            return False

        hex_like_ratio = float(
            series.str.fullmatch(r"[A-Fa-f0-9\-]{8,}").fillna(False).mean(),
        )
        return hex_like_ratio > 0.8 or unique_ratio > 0.9  # noqa: PLR2004
