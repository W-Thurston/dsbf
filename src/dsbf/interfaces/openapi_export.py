# dsbf/interfaces/openapi_export.py
"""Write and guard the OpenAPI document of API v2 (behind ``dsbf schema``)."""

import json
from pathlib import Path
from typing import Any

from dsbf.interfaces.schema_export import (
    SchemaComparison,
    SchemaToolError,
    schema_at_ref,
)
from dsbf.model.schema import Change, compare_schemas, unannounced

OPENAPI_FILE = "openapi.json"
_COMPONENTS = "#/components/schemas/"
_METHODS = frozenset({"get", "put", "post", "delete", "patch", "head", "options"})
_ADDED_WITHOUT_DEFAULT = "property added without a default"


def openapi_text() -> str:
    """The OpenAPI document as the checked-in text: sorted keys, a final newline."""
    try:
        from dsbf.service import openapi_document  # noqa: PLC0415 (needs the extra)
    except ImportError as error:
        raise SchemaToolError(str(error)) from error
    return json.dumps(openapi_document(), indent=2, sort_keys=True) + "\n"


def openapi_version(document: dict[str, Any]) -> str:
    return document["info"]["version"]


def _as_defs(value: Any) -> Any:
    """``value`` with its component references written as ``$defs`` references, which
    is what the schema comparison understands."""
    if isinstance(value, dict):
        return {
            key: (
                "#/$defs/" + item.removeprefix(_COMPONENTS)
                if key == "$ref" and isinstance(item, str)
                else _as_defs(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_as_defs(item) for item in value]
    return value


def _reader_changes(old: Any, new: Any, where: str) -> list[Change]:
    """Changes to a response shape, as a client that reads it sees them.

    A field the response gains is additive for such a client, though the run schema
    counts it as breaking for stored documents that lack it.
    """
    found = compare_schemas(_as_defs(old), _as_defs(new))
    return [
        Change(
            f"{where}{change.path[1:]}",
            change.breaking and _ADDED_WITHOUT_DEFAULT not in change.message,
            change.message,
        )
        for change in found
    ]


def _parameters(operation: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(p["in"], p["name"]): p for p in operation.get("parameters", [])}


def _operation_changes(
    old: dict[str, Any], new: dict[str, Any], where: str
) -> list[Change]:
    changes: list[Change] = []
    was, now = _parameters(old), _parameters(new)
    for key in was.keys() - now.keys():
        changes.append(Change(f"{where} {key[0]} {key[1]}", True, "parameter removed"))
    for key in now.keys() - was.keys():
        required = now[key].get("required", False)
        message = "required parameter added" if required else "parameter added"
        changes.append(Change(f"{where} {key[0]} {key[1]}", required, message))
    for key in was.keys() & now.keys():
        label = f"{where} {key[0]} {key[1]}"
        if now[key].get("required", False) and not was[key].get("required", False):
            changes.append(Change(label, True, "parameter became required"))
        before, after = was[key].get("schema", {}), now[key].get("schema", {})
        for change in compare_schemas(_as_defs(before), _as_defs(after)):
            changes.append(Change(f"{label}{change.path[1:]}", True, change.message))
    old_responses, new_responses = old.get("responses", {}), new.get("responses", {})
    for status in old_responses.keys() - new_responses.keys():
        changes.append(Change(f"{where} {status}", True, "response removed"))
    for status in new_responses.keys() - old_responses.keys():
        changes.append(Change(f"{where} {status}", False, "response added"))
    for status in old_responses.keys() & new_responses.keys():
        changes += _reader_changes(
            old_responses[status].get("content", {}),
            new_responses[status].get("content", {}),
            f"{where} {status}",
        )
    return changes


def compare_openapi(old: dict[str, Any], new: dict[str, Any]) -> list[Change]:
    """What changed for a client of the API between two OpenAPI documents.

    A removed path, operation, parameter or response, a parameter that becomes
    required, a changed parameter type, and any change to a response shape that
    could surprise a reader (a removed field, a narrowed or widened enum, a changed
    type) are breaking. Added paths, operations, optional parameters and response
    fields are additive. Descriptions and examples are ignored.
    """
    changes: list[Change] = []
    old_paths, new_paths = old.get("paths", {}), new.get("paths", {})
    for path in sorted(old_paths.keys() - new_paths.keys()):
        changes.append(Change(path, True, "path removed"))
    for path in sorted(new_paths.keys() - old_paths.keys()):
        changes.append(Change(path, False, "path added"))
    for path in sorted(old_paths.keys() & new_paths.keys()):
        before = {m: o for m, o in old_paths[path].items() if m in _METHODS}
        after = {m: o for m, o in new_paths[path].items() if m in _METHODS}
        for method in sorted(before.keys() - after.keys()):
            changes.append(
                Change(f"{method.upper()} {path}", True, "operation removed")
            )
        for method in sorted(after.keys() - before.keys()):
            changes.append(Change(f"{method.upper()} {path}", False, "operation added"))
        for method in sorted(before.keys() & after.keys()):
            changes += _operation_changes(
                before[method], after[method], f"{method.upper()} {path}"
            )
    changes += _reader_changes(
        {"$defs": old.get("components", {}).get("schemas", {})},
        {"$defs": new.get("components", {}).get("schemas", {})},
        "components",
    )
    return changes


def compare_openapi_with_base(ref: str, document: Path) -> SchemaComparison | None:
    """Compare the OpenAPI document with its committed version at ``ref``.

    None means the base has no document yet, so there is nothing to break.
    """
    base_text = schema_at_ref(ref, document)
    if base_text is None:
        return None
    if not document.is_file():
        raise SchemaToolError(f"{document} doesn't exist; run `dsbf schema export`")
    old, new = json.loads(base_text), json.loads(document.read_text("utf-8"))
    changes = compare_openapi(old, new)
    old_version, new_version = openapi_version(old), openapi_version(new)
    problems = unannounced(old_version, new_version, changes)
    return SchemaComparison(old_version, new_version, changes, problems)
