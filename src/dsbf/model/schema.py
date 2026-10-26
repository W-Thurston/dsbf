# dsbf/model/schema.py
"""The published JSON Schema of a run, and the rules that decide a change is breaking.

``run_schema`` describes what a reader of a dumped ``Run`` sees: defaulted fields are
required, since a run is always written out in full, and field titles are dropped so the
TypeScript generator doesn't name a type per field. ``compare_schemas`` reads two such
schemas and lists what changed for the people who depend on them.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode

from dsbf.model.policy import Policy
from dsbf.model.run import Run

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Keywords that explain a schema without changing which documents it accepts.
_ANNOTATIONS = frozenset(
    {"title", "description", "$comment", "examples", "default", "deprecated", "$schema"}
)
_MEMBER_LISTS = ("anyOf", "oneOf", "allOf")
_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class _SchemaGenerator(GenerateJsonSchema):
    def field_title_should_be_set(self, schema: Any) -> bool:
        return False


def run_schema() -> dict[str, Any]:
    """The JSON Schema of ``Run``, as the checked-in ``run.schema.json``."""
    schema = Run.model_json_schema(
        mode="serialization", schema_generator=_SchemaGenerator
    )
    return {"$schema": JSON_SCHEMA_DIALECT, **schema}


def policy_schema() -> dict[str, Any]:
    """The JSON Schema of ``Policy``, as the checked-in ``policy.schema.json``.

    It describes the stored form of a rule. The loose spellings the loader also
    accepts (``severity: ">=error"``, ``then: fail``) are in the field descriptions.
    """
    schema = Policy.model_json_schema(
        mode="validation", schema_generator=_SchemaGenerator
    )
    return {"$schema": JSON_SCHEMA_DIALECT, **schema}


def model_schema(model: type[BaseModel], mode: JsonSchemaMode) -> dict[str, Any]:
    """The JSON Schema of any model, written the way the published ones are."""
    return model.model_json_schema(mode=mode, schema_generator=_SchemaGenerator)


def schema_text(schema: dict[str, Any]) -> str:
    """The exact text written to disk, so the same schema is the same bytes."""
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def typescript_schema(schema: Any) -> Any:
    """A copy of the schema that gives the TypeScript generator one type per definition.

    A ``$ref`` that carries a description or default of its own makes the generator
    inline the definition under a numbered name (``Confidence1`` ... ``Confidence7``).
    Naming the type with ``tsType`` keeps the description and reuses the definition.
    """
    if isinstance(schema, list):
        return [typescript_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/") and len(schema) > 1:
        named: dict[str, Any] = {"tsType": ref.removeprefix("#/$defs/")}
        if "description" in schema:
            named["description"] = schema["description"]
        return named
    return {key: typescript_schema(value) for key, value in schema.items()}


def schema_version(schema: dict[str, Any]) -> str:
    """The ``schema_version`` a schema was exported at."""
    return schema["properties"]["schema_version"]["default"]


@dataclass(frozen=True)
class Change:
    """One difference between two schemas."""

    path: str
    breaking: bool
    message: str

    def __str__(self) -> str:
        kind = "breaking" if self.breaking else "additive"
        return f"{kind}: {self.path}: {self.message}"


def compare_schemas(old: dict[str, Any], new: dict[str, Any]) -> list[Change]:
    """Every change that affects which documents the schema describes.

    A change is breaking when a document that was valid before may not be valid now
    (a stored run no longer loads) or when a reader may meet a value it was not told
    about (a new enum member). Only a new property that has a default, or a property
    that becomes required, is additive. Annotations (titles, descriptions, defaults) are
    ignored.
    """
    changes: list[Change] = []
    old_defs, new_defs = old.get("$defs", {}), new.get("$defs", {})
    _node(_without(old, "$defs"), _without(new, "$defs"), "#", changes)
    for name in old_defs.keys() - new_defs.keys():
        changes.append(Change(f"#/$defs/{name}", True, "definition removed"))
    for name in new_defs.keys() - old_defs.keys():
        changes.append(Change(f"#/$defs/{name}", False, "definition added"))
    for name in old_defs.keys() & new_defs.keys():
        _node(old_defs[name], new_defs[name], f"#/$defs/{name}", changes)
    return changes


def _without(schema: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in schema.items() if k != key}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _node(old: Any, new: Any, path: str, changes: list[Change]) -> None:
    if not isinstance(old, dict) or not isinstance(new, dict):
        if old != new:
            was, now = _canonical(old), _canonical(new)
            changes.append(Change(path, True, f"{was} became {now}"))
        return
    _properties(old, new, path, changes)
    for key in _MEMBER_LISTS:
        if key in old or key in new:
            _members(key, old.get(key, []), new.get(key, []), path, changes)
    for key in ("items", "additionalProperties"):
        if key in old and key in new:
            _node(old[key], new[key], f"{path}/{key}", changes)
        elif old.get(key) != new.get(key):
            changes.append(_keyword(path, key, old.get(key), new.get(key)))
    handled = {"properties", "required", "items", "additionalProperties"}
    handled.update(_MEMBER_LISTS)
    for key in (old.keys() | new.keys()) - handled - _ANNOTATIONS:
        if _unordered(old.get(key)) != _unordered(new.get(key)):
            changes.append(_keyword(path, key, old.get(key), new.get(key)))


def _keyword(path: str, key: str, old: Any, new: Any) -> Change:
    was = "absent" if old is None else _canonical(old)
    now = "absent" if new is None else _canonical(new)
    return Change(path, True, f"{key} was {was}, now {now}")


def _unordered(value: Any) -> Any:
    """Lists such as ``enum`` and ``type`` are sets; their order carries no meaning."""
    if isinstance(value, list):
        return sorted(_canonical(item) for item in value)
    return value


def _properties(
    old: dict[str, Any], new: dict[str, Any], path: str, changes: list[Change]
) -> None:
    old_props, new_props = old.get("properties", {}), new.get("properties", {})
    old_required = set(old.get("required", []))
    new_required = set(new.get("required", []))
    for name in old_props.keys() - new_props.keys():
        changes.append(Change(f"{path}/{name}", True, "property removed"))
    for name in new_props.keys() - old_props.keys():
        if "default" in new_props[name]:
            changes.append(Change(f"{path}/{name}", False, "property added"))
        else:
            changes.append(
                Change(
                    f"{path}/{name}",
                    True,
                    "property added without a default, so stored documents lack it",
                )
            )
    for name in old_props.keys() & new_props.keys():
        _node(old_props[name], new_props[name], f"{path}/{name}", changes)
        if name in old_required and name not in new_required:
            changes.append(Change(f"{path}/{name}", True, "no longer always present"))
        elif name in new_required and name not in old_required:
            changes.append(Change(f"{path}/{name}", False, "now always present"))


def _members(
    key: str, old: list[Any], new: list[Any], path: str, changes: list[Change]
) -> None:
    before = {_canonical(member) for member in old}
    after = {_canonical(member) for member in new}
    for member in sorted(before - after):
        changes.append(Change(path, True, f"{key} member removed: {member}"))
    for member in sorted(after - before):
        changes.append(Change(path, True, f"{key} member added: {member}"))


def _version(text: str) -> tuple[int, int, int]:
    match = _SEMVER.fullmatch(text)
    if not match:
        raise ValueError(f"{text!r} is not a major.minor.patch version")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def unannounced(old_version: str, new_version: str, changes: list[Change]) -> list[str]:
    """Why the version doesn't announce these changes; empty when it does.

    A breaking change needs a higher major version, or, while the major is 0, a higher
    minor. An additive change needs a higher minor. A change that leaves the shape alone
    needs nothing.
    """
    old, new = _version(old_version), _version(new_version)
    if new < old:
        return [f"the schema version went backwards, {old_version} to {new_version}"]
    breaking = [c for c in changes if c.breaking]
    if breaking:
        announced = new[0] > old[0] or (old[0] == 0 and new[:2] > old[:2])
        needed = "major" if old[0] > 0 else "minor"
    elif changes:
        announced = new[:2] > old[:2]
        needed = "minor"
    else:
        return []
    if announced:
        return []
    kind = "breaking" if breaking else "additive"
    return [
        f"{len(breaking or changes)} {kind} change(s) need a {needed} version bump, "
        f"but the schema version is {old_version} to {new_version}"
    ]
