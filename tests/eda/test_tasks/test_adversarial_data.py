# tests/eda/test_tasks/test_adversarial_data.py
"""Every registered task must survive hostile but legal data without raising."""

import pytest

from dsbf.config import load_default_config
from dsbf.eda.task_registry import TASK_REGISTRY, load_task_group
from dsbf.utils.task_utils import validate_task_result
from tests.fixtures.adversarial import build_adversarial_frame
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies

for _group in load_default_config().get("task_groups", ["core"]):
    load_task_group(_group)

# Tasks whose "failed" status is their documented answer to a bare run.
EXPECTED_FAILED = {
    "identify_bottleneck_tasks": "needs task durations that only a graph run records",
}


@pytest.mark.unit
@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY))
def test_task_completes_on_adversarial_frame(task_name, tmp_path):
    spec = TASK_REGISTRY[task_name]
    ctx, _ = make_ctx_and_task(
        spec.cls,
        build_adversarial_frame(),
        global_overrides={"output_dir": str(tmp_path)},
    )

    result = run_task_with_dependencies(ctx, spec.cls)

    assert validate_task_result(result)
    if task_name not in EXPECTED_FAILED:
        assert result.status != "failed", result.summary
