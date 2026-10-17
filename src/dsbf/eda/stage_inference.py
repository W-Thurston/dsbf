# dsbf/eda/stage_inference.py


def infer_stage(df, config=None):
    if config is None:
        config = {}

    thresholds = config.get("stage_inference", {})
    null_ratio_raw_threshold = thresholds.get("null_ratio_raw_threshold", 0.4)
    percent_numeric_for_model_ready = thresholds.get(
        "percent_numeric_for_model_ready", 0.7
    )
    high_cardinality_threshold = thresholds.get("high_cardinality_threshold", 50)

    n_rows = df.shape[0]
    n_cols = len(df.columns)

    if hasattr(df, "null_count"):  # Polars
        import polars as pl

        null_ratios = df.null_count() / n_rows
        mean_null_ratio = null_ratios.select(pl.all().mean())[0, 0]
    else:  # Pandas
        null_ratios = df.isnull().mean()
        mean_null_ratio = null_ratios.mean()

    if mean_null_ratio > null_ratio_raw_threshold:
        return "raw"

    numeric_cols = (
        df.select_dtypes(include="number").columns
        if hasattr(df, "select_dtypes")
        else [col for col in df.columns if df[col].dtype in ("i64", "f64")]
    )
    percent_numeric = len(numeric_cols) / n_cols if n_cols else 0

    high_card_cols = []
    for col in df.columns:
        try:
            nunique = (
                df[col].nunique() if hasattr(df, "nunique") else df[col].n_unique()
            )
            if nunique > high_cardinality_threshold:
                high_card_cols.append(col)
        except Exception:
            continue

    if percent_numeric >= percent_numeric_for_model_ready and mean_null_ratio < 0.05:
        return "model_ready"

    if mean_null_ratio < 0.1 and len(high_card_cols) == 0:
        return "cleaned"

    return "exploratory"
