# dsbf/utils/data_utils.py


def data_sampling(df, config, log_fn=None):
    limits = config.get("resource_limits", {})
    if not limits.get("enable_sampling", True):
        return df, None  # No sampling

    threshold = limits.get("sample_threshold_rows", 1_000_000)
    strategy = limits.get("sample_strategy", "head")

    if df.shape[0] <= threshold:
        return df, None  # No sampling needed

    if log_fn:
        log_fn(
            f"Sampling dataset (rows > {threshold}) using strategy '{strategy}'",
            level="info",
        )

    sampled_df = {
        "head": lambda d: d.head(threshold),
        "random": lambda d: (
            d.sample(n=threshold, random_state=42)
            if hasattr(d, "sample")
            else d.head(threshold)
        ),
        "stratified": lambda d: d.head(
            threshold
        ),  # TODO: implement when labels available
    }.get(strategy, lambda d: d.head(threshold))(df)

    return sampled_df, {
        "original_rows": df.shape[0],
        "sampled_rows": threshold,
        "strategy": strategy,
    }


def is_integer_polars(series):
    import polars as pl

    return hasattr(series, "dtype") and series.dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
