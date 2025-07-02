from dsbf.eda.task_result import TaskResult


def validate_task_result(result: TaskResult, raise_on_error: bool = False) -> bool:
    """
    Validate the structure and content of a TaskResult object.

    Args:
        result (TaskResult): The result to validate.
        raise_on_error (bool): If True, raises ValueError on failure. Else logs warning.

    Returns:
        bool: True if valid, False if invalid and not raised.
    """
    valid = True

    def fail(msg: str):
        nonlocal valid
        valid = False
        if raise_on_error:
            raise ValueError(msg)
        else:
            print(f"[WARN] Invalid TaskResult: {msg}")

    if not result.name:
        fail("Missing 'name' field")

    if result.status not in {"success", "failed", "skipped"}:
        fail(f"Invalid 'status': {result.status}")

    if result.recommendations is not None:
        if not isinstance(result.recommendations, list):
            fail("'recommendations' must be a list of strings")
        elif not all(isinstance(r, str) for r in result.recommendations):
            fail("All items in 'recommendations' must be strings")

    if hasattr(result, "ml_impact_score"):
        score = getattr(result, "ml_impact_score")
        if score is not None and not (0 <= score <= 1):
            fail(f"'ml_impact_score' must be in [0, 1], got: {score}")

    return valid
