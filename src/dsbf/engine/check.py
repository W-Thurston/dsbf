# dsbf/engine/check.py
"""Declaring a check: its metadata, and the function that runs it."""

import importlib
import pickle
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from dsbf.engine.errors import CheckDefinitionError
from dsbf.model import CheckParams, Contract, Cost, Dimension, SemanticType, Stage

if TYPE_CHECKING:
    from dsbf.engine.context import CheckContext, CheckOutput

CheckKind = Literal["rule", "analysis", "synthesis"]

# Where the executor runs a check. A check that spends its time in compiled code
# (NumPy, Polars) shares the process's threads; one that loops in Python holds the
# interpreter lock, so it gets a worker process, which can also be killed.
RunsIn = Literal["thread", "process"]

# What the profile holds, so a check can declare the parts it reads.
ProfileFeature = Literal[
    "counts",
    "numeric_stats",
    "quantiles",
    "histogram",
    "top_values",
    "text_shape",
    "datetime",
    "content_hash",
    "missingness",
    "duplicate_rows",
]

# The optional-dependency groups in pyproject.toml, bar the catch-all `all`.
Extra = Literal["ml", "timeseries", "text", "excel", "pandas", "sql", "server"]

_NAME = re.compile(r"[a-z][a-z0-9_]*")
_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _lookup(module: str, qualname: str) -> Any:
    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    return found


def _require_name(what: str, value: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise CheckDefinitionError(
            f"{what} {value!r} must be lower snake case, such as 'null_rate'"
        )


class NoParams(CheckParams):
    """The parameters of a check that has none."""


class CheckOutputModel(Contract):
    """Base for a check's typed output.

    A check that returns more than findings declares a subclass, so its output has
    a schema and an empty result has a defined shape.
    """


@dataclass(frozen=True, init=False)
class SemanticTypes:
    """The columns a check applies to, chosen by semantic type.

    With no names it applies to every column.
    """

    names: tuple[str, ...]

    def __init__(self, *names: str) -> None:
        known = get_args(SemanticType)
        unknown = [name for name in names if name not in known]
        if unknown:
            raise CheckDefinitionError(
                f"unknown semantic type {unknown[0]!r}; choose from {', '.join(known)}"
            )
        object.__setattr__(self, "names", tuple(dict.fromkeys(names)))

    def selects(self, semantic_type: str) -> bool:
        return not self.names or semantic_type in self.names

    def describe(self) -> str:
        return " or ".join(self.names) if self.names else "any type"


@dataclass(frozen=True, init=False)
class Requires:
    """What a check cannot run without.

    ``profile`` names the parts of the profile it reads. ``checks`` are checks whose
    results it needs: if one does not complete, this check is skipped.
    """

    profile: tuple[ProfileFeature, ...]
    checks: tuple[str, ...]

    def __init__(
        self, profile: Iterable[ProfileFeature] = (), checks: Iterable[str] = ()
    ) -> None:
        known = get_args(ProfileFeature)
        features = _unique(profile)
        for feature in features:
            if feature not in known:
                raise CheckDefinitionError(
                    f"requires unknown profile feature {feature!r}; "
                    f"choose from {', '.join(known)}"
                )
        ids = _unique(checks)
        for check_id in ids:
            _require_name("requires.checks", check_id)
        object.__setattr__(self, "profile", features)
        object.__setattr__(self, "checks", ids)


@dataclass(frozen=True)
class CheckSpec:
    """Everything the engine knows about a check without running it."""

    id: str
    version: str
    kind: CheckKind
    applies_to: SemanticTypes | None = field(
        metadata={"doc": "None for a check about the dataset as a whole."}
    )
    dimensions: tuple[Dimension, ...]
    params: type[CheckParams] = NoParams
    output: type[CheckOutputModel] | None = None
    requires: Requires = field(default_factory=Requires)
    after: tuple[str, ...] = field(
        default=(), metadata={"doc": "Soft ordering: run later, but never skipped."}
    )
    cost: Cost = Cost.MODERATE
    stage: Stage = field(
        default=Stage.ANY,
        metadata={
            "doc": "The data stage the check is written for; a plan can skip some."
        },
    )
    runs_in: RunsIn = field(
        default="thread",
        metadata={
            "doc": "'process' for a check that loops in pure Python; see "
            "``check`` for what that asks of its module."
        },
    )
    extras: tuple[Extra, ...] = ()
    packs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        _require_name("id", self.id)
        if not _VERSION.fullmatch(self.version):
            raise CheckDefinitionError(
                f"{self.id}: version {self.version!r} must be MAJOR.MINOR.PATCH"
            )
        if self.kind not in get_args(CheckKind):
            raise CheckDefinitionError(
                f"{self.id}: kind {self.kind!r} must be one of "
                f"{', '.join(get_args(CheckKind))}"
            )
        if self.runs_in not in get_args(RunsIn):
            raise CheckDefinitionError(
                f"{self.id}: runs_in {self.runs_in!r} must be one of "
                f"{', '.join(get_args(RunsIn))}"
            )
        if not (isinstance(self.params, type) and issubclass(self.params, CheckParams)):
            raise CheckDefinitionError(f"{self.id}: params must be a CheckParams class")
        if self.output is not None and not (
            isinstance(self.output, type) and issubclass(self.output, CheckOutputModel)
        ):
            raise CheckDefinitionError(
                f"{self.id}: output must be a CheckOutputModel class"
            )
        if self.applies_to is not None and not isinstance(
            self.applies_to, SemanticTypes
        ):
            raise CheckDefinitionError(
                f"{self.id}: applies_to is SemanticTypes(...) or None"
            )
        self._normalise()
        self._check_dependencies()
        if self.kind == "synthesis":
            if self.applies_to is not None:
                raise CheckDefinitionError(
                    f"{self.id}: a synthesis check reads other checks, not columns; "
                    "set applies_to=None"
                )
            if not (self.requires.checks or self.after):
                raise CheckDefinitionError(
                    f"{self.id}: a synthesis check names the checks it reads, "
                    "in requires.checks or after"
                )

    def _normalise(self) -> None:
        extras = get_args(Extra)
        try:
            object.__setattr__(self, "cost", Cost(self.cost))
        except ValueError as error:
            raise CheckDefinitionError(f"{self.id}: {error}") from error
        try:
            object.__setattr__(self, "stage", Stage(self.stage))
        except ValueError as error:
            raise CheckDefinitionError(f"{self.id}: {error}") from error
        for name in ("extras", "packs", "tags", "after", "dimensions"):
            object.__setattr__(self, name, _unique(getattr(self, name)))
        if not self.dimensions:
            raise CheckDefinitionError(f"{self.id}: declare at least one dimension")
        try:
            dimensions = tuple(Dimension(value) for value in self.dimensions)
        except ValueError as error:
            raise CheckDefinitionError(f"{self.id}: {error}") from error
        object.__setattr__(self, "dimensions", dimensions)
        for extra in self.extras:
            if extra not in extras:
                raise CheckDefinitionError(
                    f"{self.id}: unknown extra {extra!r}; "
                    f"choose from {', '.join(extras)}"
                )
        for pack in self.packs:
            _require_name(f"{self.id}: pack", pack)

    def _check_dependencies(self) -> None:
        for other in self.after:
            _require_name(f"{self.id}: after", other)
        named = (*self.requires.checks, *self.after)
        if self.id in named:
            raise CheckDefinitionError(f"{self.id}: a check cannot depend on itself")
        both = set(self.requires.checks) & set(self.after)
        if both:
            raise CheckDefinitionError(
                f"{self.id}: {sorted(both)[0]!r} is both required and in after; "
                "a dependency is hard or soft, not both"
            )

    @property
    def upstream(self) -> tuple[str, ...]:
        """Every check whose result this one may read."""
        return (*self.requires.checks, *self.after)


CheckFunction = Callable[["CheckContext", Any], "CheckOutput | None"]


@dataclass(frozen=True)
class Check:
    """A declared check: its spec and the function that does the work."""

    spec: CheckSpec
    function: CheckFunction

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def version(self) -> str:
        return self.spec.version

    def resolve_params(self, raw: Mapping[str, Any] | None = None) -> CheckParams:
        """Validate ``raw`` against the check's parameter model.

        Raises:
            pydantic.ValidationError: a key the check does not declare, or a value
                outside the declared type or bounds.
        """
        return self.spec.params.model_validate(dict(raw or {}))

    def __call__(self, context: "CheckContext", params: Any) -> "CheckOutput | None":
        return self.function(context, params)

    def __reduce__(self) -> tuple[Callable[[str, str], "Check"], tuple[str, str]]:
        """Pickle by reference, as a function is: a worker process imports the check.

        That works for a check declared with ``@check`` at the top level of an
        importable module. The decorator replaces the function with the ``Check``,
        so pickling the function itself would fail.

        Raises:
            pickle.PicklingError: the check is not what its module names.
        """
        function = self.function
        module = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        found = None
        if module and qualname:
            try:
                found = _lookup(module, qualname)
            except (ImportError, AttributeError):
                pass
        if found is not self:
            raise pickle.PicklingError(
                f"check {self.id} cannot be sent to a worker process: it is not "
                f"the object {module}.{qualname}; declare it with @check at the "
                "top level of an importable module"
            )
        return _lookup, (module, qualname)  # type: ignore[return-value]


def check(
    *,
    id: str,
    version: str,
    kind: CheckKind,
    applies_to: SemanticTypes | None,
    dimensions: Iterable[Dimension | str],
    params: type[CheckParams] = NoParams,
    output: type[CheckOutputModel] | None = None,
    requires: Requires | None = None,
    after: Iterable[str] = (),
    cost: Cost = Cost.MODERATE,
    stage: Stage = Stage.ANY,
    runs_in: RunsIn = "thread",
    extras: Iterable[Extra] = (),
    packs: Iterable[str] = (),
    tags: Iterable[str] = (),
) -> Callable[[CheckFunction], Check]:
    """Declare a check.

    The decorated function takes ``(ctx, params)`` and returns what
    ``ctx.output(...)`` or ``ctx.stop(...)`` builds, or ``None``. Decorating
    registers nothing; a registry collects checks explicitly.

    ``runs_in="process"`` runs the check in a worker process of its own, so a long
    pure-Python loop neither holds up other checks nor outlives its timeout. The
    worker imports the check by name, so it must be declared at the top level of an
    importable module (otherwise it runs on a thread, with a note saying so), and
    a script that runs such checks needs the usual ``if __name__ == "__main__":``
    guard, as for any ``multiprocessing`` use.

    Raises:
        CheckDefinitionError: the metadata cannot work, such as a malformed id.
    """

    def declare(function: CheckFunction) -> Check:
        doc = (function.__doc__ or "").strip()
        spec = CheckSpec(
            id=id,
            version=version,
            kind=kind,
            applies_to=applies_to,
            dimensions=tuple(dimensions),  # type: ignore[arg-type]
            params=params,
            output=output,
            requires=requires or Requires(),
            after=tuple(after),
            cost=cost,
            stage=stage,
            runs_in=runs_in,
            extras=tuple(extras),
            packs=tuple(packs),
            tags=tuple(tags),
            summary=doc.splitlines()[0] if doc else "",
        )
        return Check(spec=spec, function=function)

    return declare
