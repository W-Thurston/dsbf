# dsbf/utils/validation.py

from typing import List

import networkx as nx

from dsbf.eda.task_registry import (
    TASK_REGISTRY,
    get_all_task_specs,
    get_plugin_warnings,
)


def validate_config_and_graph(config: dict) -> List[str]:
    """
    Validate config integrity and DAG safety before profiling run.

    Returns:
        List of human-readable error messages.
    """
    errors = []

    # -- Task name typos --
    defined_task_names = set(config.get("tasks", {}).keys())
    valid_task_names = set(TASK_REGISTRY.keys())
    for name in defined_task_names:
        if name not in valid_task_names:
            errors.append(f"Unknown task name in config['tasks']: '{name}'")

    # -- Dependency safety --
    all_specs = get_all_task_specs()
    for spec in all_specs:
        for dep in spec.depends_on or []:
            if dep not in TASK_REGISTRY:
                errors.append(f"Task '{spec.name}' depends on unknown task '{dep}'")

    # -- DAG cycle check --
    try:
        G = nx.DiGraph()
        for spec in all_specs:
            G.add_node(spec.name)
            for dep in spec.depends_on or []:
                G.add_edge(dep, spec.name)
        _ = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        errors.append("Cycle detected in task dependencies (DAG is not acyclic)")

    # -- Plugin file sanity --
    for warning in get_plugin_warnings():
        errors.append(f"[Plugin Warning] {warning['message']}")

    return errors
