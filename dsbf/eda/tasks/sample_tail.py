# dsbf/eda/tasks/sample_tail.py


def sample_tail(df, n=5):
    try:
        df_tail = df.tail(n)
        return (
            df_tail.to_dict(as_series=False)
            if hasattr(df_tail, "to_dict")
            else df_tail.rows()
        )
    except Exception:
        return {"error": "Unable to compute sample_tail"}
