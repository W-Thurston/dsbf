# dsbf/library.py
"""The library entry point: profile a table and get the ``Run`` back."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from dsbf.checks import core_pack
from dsbf.config import (
    DEFAULT_PACKS,
    ConfigError,
    ConfigIssue,
    LimitsConfig,
    ProfileConfig,
    SampleConfig,
    load_config,
)
from dsbf.data import Dataset, load
from dsbf.data.sampling import Strategy
from dsbf.engine import (
    Depth,
    DiskCache,
    Hook,
    Limits,
    Reference,
    Registry,
    ResultCache,
    Selection,
    execute,
    make_plan,
)
from dsbf.model import Policy, Run
from dsbf.policy import evaluate_policy, load_policy
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
    reference: object | None = None,
    sample: Sample | int | None = None,
    limits: LimitsConfig | None = None,
    workers: int | None = None,
    plugins: Iterable[str | Path] | str | Path = (),
    params: Mapping[str, Mapping[str, Any]] | None = None,
    cache: bool | ResultCache | None = None,
    hooks: Iterable[Hook] = (),
    config: str | Path | None = None,
    policy: Policy | Mapping[str, Any] | str | Path | None = None,
    previous: Run | None = None,
) -> Run:
    """Profile ``source`` and return the ``Run``, which stays in memory.

    ``source`` is anything ``dsbf.data.load`` accepts: a pandas, Polars or Arrow
    table, a ``Dataset``, or the path of a CSV, TSV, Parquet, Excel, JSON or NDJSON
    file. Nothing is written to disk unless you pass a ``cache``, and a failing
    check never stops the run: it ends with a status and a note, and
    ``Run.render_health()`` says which and why. The ``Run`` can be megabytes on
    text-heavy data, because a check's output is kept whole.

    A Python warning a check raises becomes a ``python_warning`` note on that
    check's result, one per category and message with a ``count``, instead of
    reaching stderr. While the call lasts, the process's warning filters show a
    warning each time it is raised rather than once (your ``ignore`` and ``error``
    filters still apply) and are restored afterwards, so another thread of yours
    may print a repeated warning more than once meanwhile.

    ``packs`` are the packs whose checks are candidates (``core`` when unset).
    ``checks`` adjusts that by id: ``"+id"`` or ``"id"`` adds a check, ``"-id"``
    removes one, and a check that another needs is pulled in regardless.
    ``depth`` (``quick``, ``standard`` or ``deep``) is the most expensive kind of
    check to run. ``target`` and ``time_index`` name the
    columns the checks that need a target or a time axis use. ``reference`` is the
    dataset the checks that compare with one (``reference_drift``) are given: any
    source ``dsbf.data.load`` accepts, profiled like the data (with the same
    ``sample``), or a ``Run`` whose stored profile is used as it is. Without one,
    those checks are not applicable. ``sample`` profiles a
    ``Sample`` of the rows (or the first ``n`` rows, given a number) and the ``Run``
    records how they were chosen. ``limits`` bounds the time of a check and of the
    run, ``workers`` is how many checks run at once (one per core, at most 8),
    ``plugins`` are directories of check plugins, and ``params`` maps a check's id to
    the parameters it runs with (they win over ``target`` and ``time_index``).

    ``cache`` keeps results between runs: ``True`` uses the DSBF home directory and
    a ``ResultCache`` is used as given. A ``hook`` receives an event as each check
    starts, records a note and finishes. ``config`` is the path of a settings file
    (the same names as these arguments); an argument here overrides the file, and
    ``DSBF__SECTION__KEY`` environment variables sit between the two. No file is
    read unless you name it.

    ``policy`` is a ``Policy``, a mapping, or the path of a ``policy.yaml``: its rules
    are evaluated over the findings and the health of the run, and ``Run.policy`` says
    which fired. A policy is report-only unless a rule says ``fail``, and then it
    only sets ``Run.policy.exit_code``; the run still completes. ``previous`` is the
    earlier ``Run`` that ``new_since: previous_run`` compares with, and is ignored
    without a policy.

    Raises:
        PolicyError: the ``policy`` cannot be read or a rule is invalid, found before
            the data is read; every problem is listed under its rule's name.
        ConfigError: a setting is invalid, such as a ``target`` that is not a
            column or a ``depth`` that does not exist; every problem is listed.
        SourceError: the data or the ``reference`` cannot be read (see
            ``dsbf.data.load``).
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
        limits=limits,
        workers=workers,
        plugins=plugins,
        params=params,
        config=config,
    )
    rules = None if policy is None else load_policy(policy)
    dataset = load(source, name=name)
    _check_columns(settings, dataset)
    if settings.sample is not None:
        dataset = _sampled(dataset, settings.sample)
    data_profile = profile_dataset(dataset)
    reference_data = _reference(reference, settings)
    registry = Registry()
    registry.register_pack(core_pack())
    plan = make_plan(
        registry,
        Selection(
            packs=settings.packs,
            include=settings.include,
            exclude=settings.exclude,
            depth=settings.depth,
            plugins=settings.plugins,
        ),
        params=_setting_params(registry, settings),
        profile=data_profile,
        has_reference=reference_data is not None,
    )
    run = execute(
        plan,
        dataset,
        data_profile,
        dsbf_version=get_dsbf_version(),
        limits=Limits(**settings.limits.model_dump()),
        hooks=hooks,
        workers=settings.workers,
        cache=DiskCache.default() if cache is True else cache or None,
        reference=reference_data,
    )
    if rules is None:
        return run
    outcome, notes = evaluate_policy(run, rules, previous)
    # Not revalidated, which would walk a megabyte-sized run again: the outcome is
    # built from this run's own ids and statuses.
    return run.model_copy(update={"policy": outcome, "notes": (*run.notes, *notes)})


def _reference(source: object | None, settings: ProfileConfig) -> Reference | None:
    """The reference to compare with: a ``Run``'s profile, or a profiled table."""
    if source is None:
        return None
    if isinstance(source, Run):
        return Reference.of_run(source)
    dataset = load(source)
    if settings.sample is not None:
        dataset = _sampled(dataset, settings.sample)
    return Reference.of_data(dataset, profile_dataset(dataset))


def _setting_params(
    registry: Registry, settings: ProfileConfig
) -> dict[str, dict[str, Any]]:
    """``target`` and ``time_index``, as the parameter of that name of each check,
    then the ``params`` the caller gave, which win.

    A check takes a setting only if it declares a parameter called the same.
    """
    given = {
        name: value
        for name, value in (
            ("target", settings.target),
            ("time_index", settings.time_index),
        )
        if value is not None
    }
    params: dict[str, dict[str, Any]] = {}
    for check in registry:
        fields = check.spec.params.model_fields
        wanted = {name: value for name, value in given.items() if name in fields}
        if wanted:
            params[check.spec.id] = wanted
    for check_id, values in settings.params.items():
        params.setdefault(check_id, {}).update(values)
    return params


def _settings(
    *,
    packs: Iterable[str] | str | None,
    checks: Iterable[str] | str,
    depth: Depth | str | None,
    target: str | None,
    time_index: str | None,
    sample: Sample | int | None,
    limits: LimitsConfig | None,
    workers: int | None,
    plugins: Iterable[str | Path] | str | Path,
    params: Mapping[str, Mapping[str, Any]] | None,
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
        "limits": limits,
        "workers": workers,
        "plugins": _names(plugins),
        "params": params,
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
    if issues:
        raise ConfigError(tuple(issues))


def _sampled(dataset: Dataset, sample: SampleConfig) -> Dataset:
    return dataset.sample(
        sample.rows,
        strategy=cast(Strategy, sample.strategy.value),
        seed=sample.seed,
        stratify_by=sample.stratify_by,
    )
