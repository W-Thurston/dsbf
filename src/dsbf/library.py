# dsbf/library.py
"""The library entry point: profile a table and get the ``Run`` back."""

import difflib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from dsbf.config import (
    DEFAULT_PACKS,
    Backend,
    ConfigError,
    ConfigIssue,
    LimitsConfig,
    ProfileConfig,
    SampleConfig,
    load_config,
    load_default_config,
)
from dsbf.data import Dataset, load
from dsbf.data.sampling import Strategy
from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.engine import (
    Depth,
    DiskCache,
    Hook,
    Limits,
    Registry,
    ResultCache,
    Selection,
    execute,
    make_plan,
)
from dsbf.legacy import LegacySession, legacy_checks
from dsbf.model import Run
from dsbf.profiling import profile_dataset
from dsbf.utils.versioning import get_dsbf_version

# What a caller passes to ``sample=``: the settings, or just a row count.
Sample = SampleConfig

__all__ = ["DEFAULT_PACKS", "Sample", "profile"]


def profile(
    source: object,
    *,
    name: str | None = None,
    packs: Iterable[str] | str | None = None,
    checks: Iterable[str] | str = (),
    depth: Depth | str | None = None,
    target: str | None = None,
    time_index: str | None = None,
    sample: Sample | int | None = None,
    backend: Backend | str | None = None,
    limits: LimitsConfig | None = None,
    workers: int | None = None,
    plugins: Iterable[str | Path] | str | Path = (),
    tasks: Mapping[str, Mapping[str, Any]] | None = None,
    cache: bool | ResultCache | None = None,
    hooks: Iterable[Hook] = (),
    config: str | Path | None = None,
) -> Run:
    """Profile ``source`` and return the ``Run``, which stays in memory.

    ``source`` is anything ``dsbf.data.load`` accepts: a pandas, Polars or Arrow
    table, a ``Dataset``, or the path of a CSV, TSV, Parquet, Excel, JSON or NDJSON
    file. Nothing is written to disk unless you pass a ``cache``, and a failing
    check never stops the run: it ends with a status and a note, and
    ``Run.render_health()`` says which and why. The ``Run`` can be megabytes on
    text-heavy data, because a check's output is kept whole.

    ``packs`` are the packs whose checks are candidates (the ``legacy`` pack, which
    holds every existing task, when unset). ``checks`` adjusts that by id: ``"+id"``
    or ``"id"`` adds a check, ``"-id"`` removes one, and a check that another needs
    is pulled in regardless. ``depth`` (``quick``, ``standard`` or ``deep``) is the
    most expensive kind of check to run. ``target`` and ``time_index`` name the
    columns the checks that need a target or a time axis use. ``sample`` profiles a
    ``Sample`` of the rows (or the first ``n`` rows, given a number) and the ``Run``
    records how they were chosen. ``backend`` is the frame library the legacy tasks
    are handed (``pandas`` unless set). ``limits`` bounds the time of a check and of
    the run, ``workers`` is how many checks run at once, ``plugins`` are directories
    of check plugins, and ``tasks`` maps a legacy task's name to its settings.

    ``cache`` keeps results between runs: ``True`` uses the DSBF home directory and
    a ``ResultCache`` is used as given. A ``hook`` receives an event as each check
    starts, records a note and finishes. ``config`` is the path of a settings file
    (the same names as these arguments); an argument here overrides the file, and
    ``DSBF__SECTION__KEY`` environment variables sit between the two. No file is
    read unless you name it.

    Raises:
        ConfigError: a setting is invalid, such as an unknown task or a ``target``
            that is not a column; every problem is listed.
        SourceError: the data cannot be read (see ``dsbf.data.load``).
        SamplingError: the sample cannot be drawn as asked.
        PlanError: the selection cannot be planned, such as an unknown pack.
    """
    settings = _settings(
        packs=packs,
        checks=checks,
        depth=depth,
        target=target,
        time_index=time_index,
        sample=sample,
        backend=backend,
        limits=limits,
        workers=workers,
        plugins=plugins,
        tasks=tasks,
        config=config,
    )
    dataset = load(source, name=name)
    _check_columns(settings, dataset)
    if settings.sample is not None:
        dataset = _sampled(dataset, settings.sample)
    data_profile = profile_dataset(dataset)
    session = LegacySession(
        dataset, data_profile, config=_legacy_config(settings, dataset)
    )
    registry = Registry(legacy_checks(session))
    plan = make_plan(
        registry,
        Selection(
            packs=settings.packs,
            include=settings.include,
            exclude=settings.exclude,
            depth=settings.depth,
            plugins=settings.plugins,
        ),
        profile=data_profile,
    )
    return execute(
        plan,
        dataset,
        data_profile,
        dsbf_version=get_dsbf_version(),
        limits=Limits(**settings.limits.model_dump()),
        hooks=hooks,
        workers=settings.workers,
        cache=DiskCache.default() if cache is True else cache or None,
    )


def _settings(
    *,
    packs: Iterable[str] | str | None,
    checks: Iterable[str] | str,
    depth: Depth | str | None,
    target: str | None,
    time_index: str | None,
    sample: Sample | int | None,
    backend: Backend | str | None,
    limits: LimitsConfig | None,
    workers: int | None,
    plugins: Iterable[str | Path] | str | Path,
    tasks: Mapping[str, Mapping[str, Any]] | None,
    config: str | Path | None,
) -> ProfileConfig:
    include, exclude = parse_checks(checks)
    given: dict[str, Any] = {
        "packs": _names(packs),
        "include": include,
        "exclude": exclude,
        "depth": depth,
        "target": target,
        "time_index": time_index,
        "sample": {"rows": sample} if isinstance(sample, int) else sample,
        "backend": backend,
        "limits": limits,
        "workers": workers,
        "plugins": _names(plugins),
        "tasks": tasks,
    }
    overrides = {
        key: _plain(value)
        for key, value in given.items()
        if value is not None and value != ()
    }
    return load_config(ProfileConfig, project_file=config, overrides=overrides)


def parse_checks(
    checks: Iterable[str] | str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``"+id"``, ``"id"`` and ``"-id"`` into the ids to add and to remove."""
    if isinstance(checks, str):
        checks = (checks,)
    include: list[str] = []
    exclude: list[str] = []
    for item in checks:
        text = item.strip()
        (exclude if text.startswith("-") else include).append(text.lstrip("+-").strip())
    return tuple(include), tuple(exclude)


def _names(values: Iterable[str | Path] | str | Path | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str | Path):
        return (str(values),)
    return tuple(str(value) for value in values)


def _plain(value: Any) -> Any:
    if isinstance(value, SampleConfig | LimitsConfig):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _check_columns(settings: ProfileConfig, dataset: Dataset) -> None:
    issues = [
        ConfigIssue(
            path,
            f"{column!r} is not a column of the data",
            "overrides",
            allowed=tuple(dataset.columns) if dataset.n_columns <= 30 else (),
        )
        for path, column in (
            ("target", settings.target),
            ("time_index", settings.time_index),
        )
        if column is not None and column not in dataset.columns
    ]
    issues += [
        ConfigIssue(
            f"tasks.{task}",
            "is not a registered task"
            + "".join(
                f"; did you mean {close!r}?"
                for close in difflib.get_close_matches(task, TASK_REGISTRY, n=1)
            ),
            "overrides",
        )
        for task in settings.tasks
        if task not in TASK_REGISTRY
    ]
    if issues:
        raise ConfigError(tuple(issues))


def _sampled(dataset: Dataset, sample: SampleConfig) -> Dataset:
    return dataset.sample(
        sample.rows,
        strategy=cast(Strategy, sample.strategy.value),
        seed=sample.seed,
        stratify_by=sample.stratify_by,
    )


def _legacy_config(settings: ProfileConfig, dataset: Dataset) -> dict[str, Any]:
    """``default_config.yaml`` with these settings applied, for the legacy tasks."""
    config = load_default_config()
    config["engine"]["backend"] = settings.backend.value
    config["metadata"]["dataset_name"] = dataset.name
    for task, values in settings.tasks.items():
        config["tasks"].setdefault(task, {}).update(values)
    if settings.target is not None:
        for values in config["tasks"].values():
            if "target_column" in values:
                values["target_column"] = settings.target
    if settings.time_index is not None:
        config["time_series"].update(
            enabled=True, datetime_index_column=settings.time_index
        )
    return config
