# dsbf/core/eda/tasks/summarize_text_fields.py

from dsbf.utils.backend import is_polars, is_text_pandas, is_text_polars


def summarize_text_fields(df):
    results = {}
    if is_polars(df):
        for col in df.columns:
            if is_text_polars(df[col]):
                try:
                    words = df[col].drop_nulls().to_list()
                    char_counts = [len(s) for s in words]
                    word_counts = [len(s.split()) for s in words]
                    total_chars = sum(char_counts)
                    total_words = sum(word_counts)
                    avg_word_len = total_chars / total_words if total_words else 0
                    results[col] = {
                        "avg_char_length": (
                            sum(char_counts) / len(char_counts) if char_counts else 0
                        ),
                        "avg_word_count": (
                            sum(word_counts) / len(word_counts) if word_counts else 0
                        ),
                        "avg_word_length": avg_word_len,
                    }
                except Exception:
                    continue
    else:
        for col in df.columns:
            if is_text_pandas(df[col]):
                try:
                    texts = df[col].dropna().astype(str)
                    char_counts = texts.map(len)
                    word_counts = texts.map(lambda s: len(s.split()))
                    total_chars = char_counts.sum()
                    total_words = word_counts.sum()
                    avg_word_len = total_chars / total_words if total_words else 0
                    results[col] = {
                        "avg_char_length": char_counts.mean(),
                        "avg_word_count": word_counts.mean(),
                        "avg_word_length": avg_word_len,
                    }
                except Exception:
                    continue
    return results
