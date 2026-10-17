# dsbf/utils/task_and_column_mapping_utils.py

from dsbf.eda.task_result import TaskResult


def build_column_task_mappings(
    results: dict[str, TaskResult],
) -> dict[str, dict[str, set[str]]]:
    """
    Build metadata maps:
      - column_task_map: columns → set of tasks that touched them
      - task_column_map: tasks   → set of columns they analyzed

    Args:
        results (dict[str, dict]): report["results"] parsed from report.json

    Returns:
        dict[str, dict[str, set[str]]]: dict with both mappings.
    """
    column_task_map: dict[str, set[str]] = {}
    task_column_map: dict[str, set[str]] = {}

    for task_name, task_result in results.items():
        result = task_result.to_dict()
        columns = set()

        # Collect from known locations
        summary = result.get("summary", {})
        metadata = result.get("metadata", {})

        if "column" in summary:
            columns.add(summary["column"])
        if "columns" in summary:
            columns.update(summary["columns"])
        if "columns" in metadata:
            columns.update(metadata["columns"])
        if "data" in result and isinstance(result["data"], dict):
            columns.update(k for k in result["data"] if isinstance(k, str))

        task_column_map[task_name] = columns

        for col in columns:
            column_task_map.setdefault(col, set()).add(task_name)

    return {
        "column_task_map": column_task_map,
        "task_column_map": task_column_map,
    }
