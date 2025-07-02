# tests/eda/test_tasks/test_registry_integrity.py

import json

from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.utils.task_utils import write_task_metadata


def test_all_task_names_are_unique():
    assert len(TASK_REGISTRY) == len(
        set(TASK_REGISTRY.keys())
    ), "Duplicate task names found."


def test_all_required_metadata_fields_present():
    for name, spec in TASK_REGISTRY.items():
        assert spec.description is not None, f"{name} missing description"
        assert spec.domain is not None, f"{name} missing domain"
        assert spec.profiling_depth in {
            "basic",
            "standard",
            "full",
        }, f"{name} has invalid profiling_depth"


def test_all_dependencies_exist():
    for name, spec in TASK_REGISTRY.items():
        if not spec.depends_on:
            continue
        for dep in spec.depends_on:
            assert dep in TASK_REGISTRY, f"{name} depends on unknown task '{dep}'"


def test_all_tags_are_lists_or_none():
    for name, spec in TASK_REGISTRY.items():
        tags = spec.tags
        assert tags is None or isinstance(tags, list), f"{name} has invalid tags type"


def test_metadata_json_is_consistent_with_registry(tmp_path):
    output_file = tmp_path / "task_metadata.json"
    write_task_metadata(str(output_file))

    with open(output_file, "r") as f:
        metadata = json.load(f)

    registry_task_names = set(TASK_REGISTRY.keys())
    metadata_task_names = set(metadata.keys())

    # 1. Check all live tasks are included
    assert registry_task_names == metadata_task_names, (
        f"Mismatch in task names. "
        f"Missing from metadata: {registry_task_names - metadata_task_names}. "
        f"Unexpected extras: {metadata_task_names - registry_task_names}"
    )

    # 2. Check metadata completeness and value types
    required_fields = {
        "domain": str,
        "tags": list,
        "runtime_estimate": str,
        "summary": str,
        "profiling_depth": str,
        "experimental": bool,
    }

    for task_name, entry in metadata.items():
        assert isinstance(
            entry, dict
        ), f"Metadata for task '{task_name}' must be a dict."

        for field, expected_type in required_fields.items():
            assert (
                field in entry
            ), f"Missing '{field}' in metadata for task '{task_name}'"
            assert isinstance(entry[field], expected_type) or entry[field] is None, (
                f"Field '{field}' in task '{task_name}'"
                f" must be {expected_type.__name__} or None"
            )

        # 3. Summary should not be empty or whitespace
        summary = entry.get("summary", "")
        assert (
            summary.strip() != ""
        ), f"Summary for task '{task_name}' is empty or blank"
