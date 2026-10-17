# dsbf/data/frames.py
"""Adapters for tables that are already in memory: Polars, Arrow and pandas.

Only the library that owns the object is imported, and only after its type is
recognised from the module name, so a Polars user never pays for pandas.
"""

import polars as pl

from dsbf.data.errors import SourceReadError

# What a conversion raises for data Arrow cannot hold: ArrowInvalid is a ValueError
# and ArrowTypeError a TypeError, and Polars wraps the rest.
_CONVERSION_ERRORS = (TypeError, ValueError, pl.exceptions.PolarsError)


def adapt_frame(obj: object, label: str) -> tuple[pl.DataFrame, dict[str, str]] | None:
    """Return ``obj`` as a Polars frame plus the columns it had to coerce.

    ``None`` means ``obj`` is not an in-memory table DSBF knows.
    """
    if isinstance(obj, pl.DataFrame):
        return obj, {}
    if isinstance(obj, pl.LazyFrame):
        return obj.collect(), {}
    root = type(obj).__module__.partition(".")[0]
    if root == "pyarrow":
        return _from_arrow(obj)
    if root == "pandas" and type(obj).__name__ == "DataFrame":
        return _from_pandas(obj, label)
    return None


def _from_arrow(obj: object) -> tuple[pl.DataFrame, dict[str, str]] | None:
    import pyarrow as pa

    if isinstance(obj, pa.RecordBatch):
        obj = pa.Table.from_batches([obj])
    if not isinstance(obj, pa.Table):
        return None
    frame = pl.from_arrow(obj)
    assert isinstance(frame, pl.DataFrame)
    return frame, {}


def _from_pandas(df, label: str) -> tuple[pl.DataFrame, dict[str, str]]:
    """Convert through Arrow, sharing memory where the dtypes allow.

    A float ``NaN`` becomes null, which is what pandas means by it. A column of
    mixed Python types cannot be an Arrow array; it is converted to text and
    reported, so the load goes on and later checks can still look at it.
    """
    names = [str(name) for name in df.columns]
    if len(set(names)) != len(names):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise SourceReadError(
            f"The pandas frame has duplicate column names: {', '.join(duplicated)}",
            source=label,
            hint="Rename the columns so every name is unique.",
        )
    coerced: dict[str, str] = {}
    try:
        frame = pl.from_pandas(df)
    except _CONVERSION_ERRORS:
        columns = []
        for position, name in enumerate(names):
            series = df.iloc[:, position]
            try:
                columns.append(pl.from_pandas(series))
            except _CONVERSION_ERRORS:
                columns.append(pl.from_pandas(series.astype("string")))
                coerced[name] = "mixed types, converted to text"
        frame = pl.DataFrame(columns)
    # Polars renames an empty column name to `column_<n>`; keep the caller's names.
    frame.columns = names
    return frame, coerced
