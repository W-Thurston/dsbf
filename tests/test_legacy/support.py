"""A small frame, and the same tasks run the legacy way and through the adapter."""

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from dsbf.config import load_default_config
from dsbf.core.context import AnalysisContext
from dsbf.data import Dataset, load
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.engine import Check, Registry, Selection, execute, make_plan
from dsbf.legacy import LegacyParams, LegacySession, legacy_checks
from dsbf.legacy.result import plain
from dsbf.model import Run
from dsbf.model.scrub import json_safe
from dsbf.profiling import profile_dataset
from dsbf.utils.task_utils import instantiate_task

N_ROWS = 240


def build_frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    age = rng.normal(40, 12, N_ROWS)
    return pl.DataFrame(
        {
            "age": [None if i % 9 == 0 else float(v) for i, v in enumerate(age)],
            "fare": rng.lognormal(3, 1, N_ROWS),
            "group": rng.choice(["a", "b", "c"], N_ROWS),
            "flag": rng.integers(0, 2, N_ROWS).astype(bool),
            "when": [
                dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(N_ROWS)
            ],
            "trend": np.arange(N_ROWS) * 0.5 + rng.normal(0, 3, N_ROWS),
            "comment": [
                f"the quick brown fox jumps over lazy dog number {i % 70}"
                for i in range(N_ROWS)
            ],
            "row_id": [f"ID{i:05d}" for i in range(N_ROWS)],
        }
    )


def make_config() -> dict[str, Any]:
    config = load_default_config()
    config["time_series"] = {
        **config.get("time_series", {}),
        "enabled": True,
        "datetime_index_column": "when",
        "value_columns": ["trend"],
    }
    return config


def make_dataset() -> Dataset:
    return load(build_frame(), name="legacy_demo")


def make_session(output_dir: Path, dataset: Dataset | None = None, config=None):
    dataset = dataset or make_dataset()
    profile = profile_dataset(dataset)
    return LegacySession(
        dataset,
        profile,
        config=config if config is not None else make_config(),
        output_dir=output_dir,
    )


def run_adapted(session: LegacySession, *names: str, cache=None, workers=2) -> Run:
    """Run ``names`` and the checks they require through the engine."""
    every = {check.id: check for check in legacy_checks(session)}
    wanted: dict[str, Any] = {}
    pending = list(names)
    while pending:
        name = pending.pop()
        if name not in wanted:
            wanted[name] = every[name]
            pending.extend(every[name].spec.requires.checks)
    plan = make_plan(
        Registry(wanted.values()), Selection(depth="deep"), profile=session.profile
    )
    return execute(
        plan,
        session.dataset,
        session.profile,
        dsbf_version="test",
        workers=workers,
        cache=cache,
    )


def run_legacy(session: LegacySession, *names: str) -> dict[str, Any]:
    """The same tasks run the legacy way: one context, in dependency order."""
    load_all_tasks()
    context = AnalysisContext(
        session.frame(),
        config=copy.deepcopy(session.config),
        output_dir=session.output_dir,
    )
    context.metadata["semantic_types"] = session.semantic_types
    context.metadata["inferred_dtypes"] = session.inferred_dtypes()
    done: set[str] = set()

    def run(name: str) -> None:
        if name in done:
            return
        for dep in TASK_REGISTRY[name].depends_on or ():
            run(dep)
        task = instantiate_task(name, session.config.get("tasks", {}).get(name))
        context.run_task(task)
        # infer_types replaces the types; keep the profile's, as the adapter does
        context.metadata["semantic_types"] = session.semantic_types
        context.metadata["inferred_dtypes"] = session.inferred_dtypes()
        done.add(name)

    for name in names:
        run(name)
    return context.results


def comparable(result: Any) -> dict[str, Any]:
    """A ``TaskResult`` as the engine stores it: JSON-plain, undefined numbers null."""
    value, _ = json_safe(json.loads(json.dumps(plain(result.to_dict()), default=str)))
    return value


def params_of(check: Check) -> LegacyParams:
    params = check.resolve_params()
    assert isinstance(params, LegacyParams)
    return params


def output_of(run: Run, check_id: str) -> dict[str, Any]:
    output = run.results[check_id].output
    assert output is not None, run.results[check_id].notes
    return output
