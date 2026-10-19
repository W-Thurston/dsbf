# tests/test_task_registry_plugin.py

import uuid
from pathlib import Path

import pytest

import dsbf
from dsbf.eda.task_registry import TASK_REGISTRY, load_task_group

pytestmark = pytest.mark.unit

CUSTOM_PLUGINS_DIR = Path(dsbf.__file__).parent / "custom_plugins"


def test_load_real_plugin_print_shape_debug():
    plugin_dir = CUSTOM_PLUGINS_DIR / "example_plugin_domain"
    assert plugin_dir.exists(), f"Missing plugin directory: {plugin_dir}"

    # Clear first in case of repeated test runs
    TASK_REGISTRY.pop("print_shape_debug", None)

    try:
        load_task_group(str(plugin_dir))

        assert "print_shape_debug" in TASK_REGISTRY
        task_spec = TASK_REGISTRY["print_shape_debug"]

        assert task_spec.domain == "internal"
        assert "debug" in (task_spec.tags or [])
        assert task_spec.profiling_depth == "basic"
        assert task_spec.runtime_estimate == "fast"
        assert "Prints the shape" in (task_spec.description or "")
    finally:
        # Clean up to avoid registry contamination
        TASK_REGISTRY.pop("print_shape_debug", None)


def test_load_nonexistent_plugin_path_does_not_crash(caplog):
    bogus_path = str(CUSTOM_PLUGINS_DIR / f"_nonexistent_{uuid.uuid4().hex}")
    before = set(TASK_REGISTRY.keys())

    try:
        load_task_group(bogus_path)
        after = set(TASK_REGISTRY.keys())
        assert before == after

        assert "does not exist" in caplog.text or "Failed to load" in caplog.text
    finally:
        # Defensive cleanup (should be no-op)
        for key in set(TASK_REGISTRY.keys()) - before:
            TASK_REGISTRY.pop(key, None)
