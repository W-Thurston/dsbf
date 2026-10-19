"""A task that reads another task's result is ordered after it."""

import re
from pathlib import Path

import pytest

from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.legacy import UNDECLARED_READS

pytestmark = pytest.mark.unit

READ = re.compile(r"""(?:results\.get\(|results\[|get_result\()\s*["'](\w+)["']""")


def known(name: str) -> set[str]:
    return set(TASK_REGISTRY[name].depends_on or ()) | set(
        UNDECLARED_READS.get(name, ())
    )


def source_of(name: str) -> str:
    import inspect

    path = inspect.getsourcefile(TASK_REGISTRY[name].cls)
    assert path is not None
    return Path(path).read_text()


def test_every_task_result_a_task_reads_is_a_dependency_or_a_listed_read():
    load_all_tasks()
    missing = {}
    for name in TASK_REGISTRY:
        reads = {r for r in READ.findall(source_of(name)) if r in TASK_REGISTRY}
        if reads - known(name):
            missing[name] = sorted(reads - known(name))

    assert not missing, f"add these to UNDECLARED_READS: {missing}"


def test_the_time_series_helper_reads_the_gap_result_for_every_task_using_it():
    load_all_tasks()
    users = [name for name in TASK_REGISTRY if "_time_series_utils" in source_of(name)]

    assert users
    for name in users:
        assert "temporal_gap_detection" in known(name) | {name}, name


def test_the_listed_reads_are_registered_tasks():
    load_all_tasks()
    for name, reads in UNDECLARED_READS.items():
        assert name in TASK_REGISTRY
        assert set(reads) <= set(TASK_REGISTRY)
        assert not set(reads) & set(TASK_REGISTRY[name].depends_on or ())
