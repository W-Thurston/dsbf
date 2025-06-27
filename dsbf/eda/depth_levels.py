DEPTH_LEVELS = {
    "basic": [
        ("infer_types", []),
        ("summarize_nulls", ["infer_types"]),
    ],
    "standard": [
        ("infer_types", []),
        ("summarize_nulls", ["infer_types"]),
        ("detect_id_columns", ["infer_types"]),
        ("detect_constant_columns", ["infer_types"]),
        ("categorical_length_stats", ["infer_types"]),
        # Add more tasks here when ready
    ],
    "full": [
        # Flesh this out later
        ("infer_types", []),
        ("summarize_nulls", ["infer_types"]),
        ("detect_id_columns", ["infer_types"]),
        ("detect_constant_columns", ["infer_types"]),
        # ...
    ],
}
