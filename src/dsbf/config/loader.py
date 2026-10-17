# dsbf/config/loader.py
"""
Layered configuration.

Settings come from up to five layers, later ones overriding earlier ones:
    1. ``defaults``: a mapping of baseline values over the model's own field defaults
    2. the project file, ``dsbf.yaml``
    3. environment variables named ``DSBF__SECTION__KEY``
    4. ``cli``: options the command line already parsed into a nested mapping
    5. ``overrides``: the keyword arguments of a library call

Mappings merge key by key. Everything else (scalars, lists, ``None``) replaces the
earlier value as a whole, and no input mapping is modified. The merged result is
validated against a model, and every problem is reported at once with the layer that
supplied the offending value.

Environment variable values stay strings for pydantic to coerce (``"3"`` becomes ``3``),
except that a value starting with ``[`` or ``{``, or exactly ``null``, is read as JSON.
Names are lowercased, so the environment layer can't address a mixed-case dict key.
"""

import copy
import json
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError

PROJECT_FILE_NAME = "dsbf.yaml"
ENV_PREFIX = "DSBF__"
_ENV_DELIMITER = "__"

_KeyPath = tuple[str, ...]


@dataclass(frozen=True)
class ConfigIssue:
    """One problem with the configuration.

    ``path`` is the dotted location of the setting, ``source`` names the layer that
    supplied it (``None`` when no layer did, such as a missing required key), and
    ``allowed`` lists the keys a section accepts when the problem is an unknown one.
    """

    path: str
    message: str
    source: str | None = None
    allowed: tuple[str, ...] = ()

    def __str__(self) -> str:
        text = f"{self.path}: {self.message}" if self.path else self.message
        if self.source:
            text += f" (from {self.source})"
        if self.allowed:
            text += f"; allowed: {', '.join(self.allowed)}"
        return text


class ConfigError(ValueError):
    """The configuration is invalid; ``issues`` holds every problem found."""

    def __init__(self, issues: tuple[ConfigIssue, ...]) -> None:
        self.issues = issues
        count = len(issues)
        header = (
            f"Invalid configuration ({count} {'issue' if count == 1 else 'issues'}):"
        )
        super().__init__("\n".join([header, *(f"  {issue}" for issue in issues)]))


def find_project_file(directory: str | Path | None = None) -> Path | None:
    """Return ``dsbf.yaml`` in ``directory`` (default: the working directory).

    Parent directories are not searched.
    """
    candidate = Path(directory or Path.cwd()) / PROJECT_FILE_NAME
    return candidate if candidate.is_file() else None


@dataclass
class _Layer:
    source: str
    values: dict[str, Any]
    # Per-setting sources, such as the variable behind each environment value.
    leaf_sources: dict[_KeyPath, str]

    def source_of(self, path: _KeyPath) -> str:
        return self.leaf_sources.get(path, self.source)


def load_config[M: BaseModel](
    model: type[M],
    *,
    defaults: Mapping[str, Any] | None = None,
    project_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    cli: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> M:
    """Merge the layers and validate the result as ``model``.

    ``env`` defaults to ``os.environ``; pass ``{}`` to ignore the environment.
    Raises ``ConfigError`` listing every problem.
    """
    issues: list[ConfigIssue] = []
    layers: list[_Layer] = []

    if defaults:
        layers.append(_mapping_layer("defaults", defaults))
    if project_file is not None:
        file_layer = _file_layer(Path(project_file), issues)
        if file_layer is not None:
            layers.append(file_layer)
    layers.append(_env_layer(os.environ if env is None else env, issues))
    if cli:
        layers.append(_mapping_layer("cli", cli))
    if overrides:
        layers.append(_mapping_layer("overrides", overrides))
    if issues:
        raise ConfigError(tuple(issues))

    merged: dict[str, Any] = {}
    origins: dict[_KeyPath, str] = {}
    for layer in layers:
        _merge(merged, layer.values, layer, origins, ())

    try:
        return model.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_issues_from(exc, model, origins)) from None


def _mapping_layer(source: str, values: Mapping[str, Any]) -> _Layer:
    return _Layer(source, _plain(values), {})


def _plain(value: Any) -> Any:
    """Copy a nested mapping with string keys, so merging leaves the input alone."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _file_layer(path: Path, issues: list[ConfigIssue]) -> _Layer | None:
    source = f"file {path}"
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(ConfigIssue("", "the configuration file does not exist", source))
        return None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        reason = " ".join(str(exc).split())
        issues.append(
            ConfigIssue("", f"cannot read the configuration file: {reason}", source)
        )
        return None
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        issues.append(
            ConfigIssue(
                "",
                "the top level must be a mapping of settings, "
                f"not {type(loaded).__name__}",
                source,
            )
        )
        return None
    return _mapping_layer(source, loaded)


def _env_layer(env: Mapping[str, str], issues: list[ConfigIssue]) -> _Layer:
    layer = _Layer("env", {}, {})
    for name in sorted(env):
        if not name.startswith(ENV_PREFIX):
            continue
        source = f"env {name}"
        parts = tuple(name.removeprefix(ENV_PREFIX).lower().split(_ENV_DELIMITER))
        if not all(parts):
            issues.append(
                ConfigIssue(
                    name,
                    f"expected {ENV_PREFIX}SECTION{_ENV_DELIMITER}KEY, no empty parts",
                    source,
                )
            )
            continue
        node = layer.values
        for depth, part in enumerate(parts[:-1], start=1):
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                issues.append(
                    ConfigIssue(
                        ".".join(parts[:depth]),
                        "is set to a value and also has sub-keys",
                        source,
                    )
                )
                break
            node = child
        else:
            if isinstance(node.get(parts[-1]), dict):
                issues.append(
                    ConfigIssue(
                        ".".join(parts),
                        "is set to a value and also has sub-keys",
                        source,
                    )
                )
                continue
            node[parts[-1]] = _decode_env_value(env[name])
            layer.leaf_sources[parts] = source
    return layer


def _decode_env_value(raw: str) -> Any:
    text = raw.strip()
    if text == "null" or text.startswith(("[", "{")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw
    return raw


def _merge(
    base: dict[str, Any],
    layer_values: Mapping[str, Any],
    layer: _Layer,
    origins: dict[_KeyPath, str],
    prefix: _KeyPath,
) -> None:
    for key, value in layer_values.items():
        path = (*prefix, key)
        origins[path] = layer.source_of(path)
        if isinstance(value, Mapping):
            existing = base.get(key)
            if not isinstance(existing, dict):
                base[key] = existing = {}
                _forget_below(origins, path)
            _merge(existing, value, layer, origins, path)
        else:
            if isinstance(base.get(key), dict):
                _forget_below(origins, path)
            base[key] = value


def _forget_below(origins: dict[_KeyPath, str], path: _KeyPath) -> None:
    for known in [
        known for known in origins if known[: len(path)] == path and known != path
    ]:
        del origins[known]


def _issues_from(
    exc: ValidationError, model: type[BaseModel], origins: dict[_KeyPath, str]
) -> tuple[ConfigIssue, ...]:
    issues = []
    for error in exc.errors():
        location = tuple(error["loc"])
        allowed: tuple[str, ...] = ()
        if error["type"] == "extra_forbidden":
            message = "unknown key"
            allowed = _allowed_keys(model, location[:-1])
        else:
            message = error["msg"]
            if not isinstance(error["input"], (Mapping, list, tuple)) and location:
                message += f" (got {_short(error['input'])})"
        issues.append(
            ConfigIssue(
                path=_dotted(location),
                message=message,
                source=_origin_of(origins, location),
                allowed=allowed,
            )
        )
    return tuple(issues)


def _dotted(location: tuple[int | str, ...]) -> str:
    text = ""
    for part in location:
        text += f"[{part}]" if isinstance(part, int) else f".{part}"
    return text.removeprefix(".")


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _origin_of(
    origins: dict[_KeyPath, str], location: tuple[int | str, ...]
) -> str | None:
    """Return the layer behind the longest known prefix of an error location."""
    for length in range(len(location), 0, -1):
        key = tuple(str(part) for part in location[:length])
        if key in origins:
            return origins[key]
    return None


def _allowed_keys(
    model: type[BaseModel], parent: tuple[int | str, ...]
) -> tuple[str, ...]:
    """List the keys the section at ``parent`` accepts, if it is a single model."""
    node: Any = model
    for part in parent:
        node = _step(node, part)
        if node is None:
            return ()
    node = _unwrap(node)
    if isinstance(node, type) and issubclass(node, BaseModel):
        return tuple(node.model_fields)
    return ()


def _unwrap(annotation: Any) -> Any:
    """Strip ``Annotated`` and ``| None``; a union of several types gives nothing."""
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation = get_args(annotation)[0]
        elif origin in (Union, types.UnionType):
            members = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(members) != 1:
                return None
            annotation = members[0]
        else:
            return annotation


def _step(annotation: Any, part: int | str) -> Any:
    """Follow one location segment from a field annotation to its child annotation."""
    node = _unwrap(annotation)
    if isinstance(node, type) and issubclass(node, BaseModel):
        field = node.model_fields.get(str(part))
        return field.annotation if field is not None else None
    args = get_args(node)
    origin = get_origin(node)
    if origin is not None and isinstance(origin, type):
        if issubclass(origin, Mapping) and len(args) == 2:
            return args[1]
        if args:
            return args[0]
    return None
