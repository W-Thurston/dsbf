# dsbf/eda/tasks/sample_head.py


def sample_head(df, n=5):
    try:
        df_head = df.head(n)
        return (
            df_head.to_dict(as_series=False)
            if hasattr(df_head, "to_dict")
            else df_head.rows()
        )
    except Exception:
        return {"error": "Unable to compute sample_head"}
