# dsbf/interfaces/reference.py
"""The command-line and settings reference, generated from the code defining them."""

import types
from typing import Annotated, Any, Union, get_args, get_origin

import typer
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typer.core import TyperGroup

from dsbf.config import ProfileConfig

REFERENCE_FILE = "CLI_REFERENCE.md"

_HEADER = """\
# DSBF command-line and settings reference

> **Generated file.** Do not edit it. `dsbf docs export` writes it from the Typer app
> and from `ProfileConfig`, and `dsbf docs export --check` fails when it is stale.

Every command exits 0 on success, 2 when the run cannot start (bad input, settings or
policy), and, for `dsbf check`, 1 when a policy rule with a `fail` action matched.
"""


def render_reference(app: typer.Typer) -> str:
    """The Markdown reference for ``app``'s commands and the settings."""
    root = typer.main.get_command(app)
    assert isinstance(root, TyperGroup)
    parts = [_HEADER, "## Commands\n", _render_group(root, "dsbf", level=3)]
    parts += [
        "## Settings\n",
        "The settings file (`--config`), `DSBF__SECTION__KEY` environment variables, "
        "`--set a.b=c` and the options of `dsbf profile` all fill these settings, "
        "later layers winning in that order. `dsbf.profile()` takes the same names as "
        "arguments.\n",
        _render_model(ProfileConfig, prefix=""),
    ]
    return "\n".join(parts).rstrip("\n") + "\n"


def _render_group(group: TyperGroup, path: str, *, level: int) -> str:
    context = typer.Context(group, info_name=path)
    lines = [_render_command(group, path, level=level)]
    for name in group.list_commands(context):
        command = group.get_command(context, name)
        assert command is not None
        child = f"{path} {name}"
        if isinstance(command, TyperGroup):
            lines.append(_render_group(command, child, level=level))
        else:
            lines.append(_render_command(command, child, level=level))
    return "\n".join(lines)


def _render_command(command: Any, path: str, *, level: int) -> str:
    context = typer.Context(command, info_name=path)
    pieces = command.collect_usage_pieces(context)
    usage = " ".join([path, *(piece.strip("{}").upper() for piece in pieces)])
    lines = [f"{'#' * level} `{path}`\n", f"```\n{usage}\n```\n"]
    if command.help:
        lines.append(command.help.strip() + "\n")
    rows = []
    for param in command.get_params(context):
        if param.param_type_name != "option" or param.name == "help":
            continue
        record = param.get_help_record(context)
        assert record is not None
        names, description = record
        rows.append(f"| `{_cell(names)}` | {_cell(description)} |")
    if rows:
        lines += ["| Option | Description |", "| --- | --- |", *rows, ""]
    return "\n".join(lines)


def _render_model(model: type[BaseModel], *, prefix: str) -> str:
    rows = ["| Setting | Type | Default | Description |", "| --- | --- | --- | --- |"]
    sections: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        section = _section_of(field.annotation)
        if section is not None:
            sections.append(
                f"### `{path}`\n\n{_render_model(section, prefix=path + '.')}"
            )
        rows.append(
            f"| `{path}` | {_cell(_type_name(field.annotation))} | "
            f"{_cell(_default(field))} | {_cell(_describe(field, section))} |"
        )
    return "\n".join([*rows, "", *sections])


def _describe(field: FieldInfo, section: type[BaseModel] | None) -> str:
    text = field.description or (section.__doc__ or "").strip().split("\n\n")[0]
    return " ".join(text.split())


def _section_of(annotation: Any) -> type[BaseModel] | None:
    node = _unwrap(annotation)
    if isinstance(node, type) and issubclass(node, BaseModel):
        return node
    return None


def _unwrap(annotation: Any) -> Any:
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation = get_args(annotation)[0]
        elif origin in (Union, types.UnionType):
            members = [a for a in get_args(annotation) if a is not type(None)]
            if len(members) != 1:
                return annotation
            annotation = members[0]
        else:
            return annotation


def _type_name(annotation: Any) -> str:
    node = _unwrap(annotation)
    if isinstance(node, type) and not get_args(node):
        return node.__name__
    return str(node).replace("typing.", "").replace("dsbf.config.profile.", "")


def _default(field: FieldInfo) -> str:
    if field.is_required():
        return "required"
    if field.default_factory is not None:
        return f"`{field.get_default(call_default_factory=True)!r}`"
    default = field.default
    if isinstance(default, BaseModel):
        return "see below"
    value = getattr(default, "value", default)
    return f"`{value!r}`" if value is not None else "unset"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")
