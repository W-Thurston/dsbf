# dsbf/interfaces/check_reference.py
"""The check reference, generated from the core pack's specs and sources."""

import inspect
import textwrap

from pydantic.fields import FieldInfo

from dsbf.checks import core_pack
from dsbf.engine import Check, Registry

CHECK_REFERENCE_FILE = "CHECK_REFERENCE.md"

_BOUNDS = {"ge": "≥", "gt": ">", "le": "≤", "lt": "<"}

_HEADER = """\
# DSBF check reference

> **Generated file.** Do not edit it. `dsbf docs export` writes it from the core
> pack's check specs, parameter models and sources, and `dsbf docs export --check`
> fails when it is stale.

Each section lists what a check applies to, what it needs, its parameters and the
finding codes it can raise. A parameter is set by check id with
`dsbf.profile(params={"<check id>": {"<parameter>": value}})` or
`dsbf profile --param <check id>.<parameter>=value`. The check's own text
carries the limits of its method.
"""


def render_check_reference() -> str:
    """The Markdown reference for every check in the core pack."""
    registry = Registry()
    registry.register_pack(core_pack())
    checks = sorted(registry.checks, key=lambda item: item.id)
    parts = [_HEADER, "## Checks\n", _index(checks)]
    parts += [_render_check(check) for check in checks]
    return "\n".join(parts).rstrip("\n") + "\n"


def _index(checks: list[Check]) -> str:
    rows = ["| Check | Kind | Cost | Summary |", "| --- | --- | --- | --- |"]
    for check in checks:
        spec = check.spec
        rows.append(
            f"| [`{spec.id}`](#{spec.id}) | {spec.kind} | {spec.cost.value} | "
            f"{_cell(spec.summary)} |"
        )
    return "\n".join([*rows, ""])


def _render_check(check: Check) -> str:
    spec = check.spec
    lines = [f"## {spec.id}\n"]
    if spec.summary:
        lines.append(spec.summary.strip() + "\n")
    doc = inspect.getdoc(check.function)
    if doc:
        body = doc.removeprefix(spec.summary.strip()).strip()
        if body:
            lines.append(body + "\n")
    facts = [
        ("Version", spec.version),
        ("Kind", str(spec.kind)),
        ("Cost", spec.cost.value),
        (
            "Applies to",
            spec.applies_to.describe() if spec.applies_to else "the dataset",
        ),
        ("Dimensions", ", ".join(d.value for d in spec.dimensions) or "none"),
        ("Needs profile", ", ".join(spec.requires.profile) or "nothing"),
        ("Needs checks", ", ".join(spec.requires.checks) or "none"),
        ("Needs parameters", ", ".join(spec.requires.params) or "none"),
        ("Needs reference", "yes" if spec.requires.reference else "no"),
        ("Extras", ", ".join(str(e) for e in spec.extras) or "none"),
        ("Packs", ", ".join(spec.packs) or "none"),
    ]
    lines += ["| | |", "| --- | --- |"]
    lines += [f"| {name} | {_cell(value)} |" for name, value in facts]
    lines.append("")
    lines.append("**Parameters**\n")
    fields = spec.params.model_fields
    if fields:
        lines += ["| Parameter | Default | Description |", "| --- | --- | --- |"]
        lines += [
            f"| `{name}` | {_cell(_default(field))} | {_cell(_describe(field))} |"
            for name, field in fields.items()
        ]
        lines.append("")
    else:
        lines.append("None.\n")
    lines.append("**Output**\n")
    if spec.output is None:
        lines.append("Findings only.\n")
    else:
        lines.append(
            f"`{spec.output.__name__}`: "
            + ", ".join(f"`{name}`" for name in spec.output.model_fields)
            + "\n"
        )
    codes = spec.codes
    lines.append("**Finding codes**\n")
    lines.append((", ".join(f"`{code}`" for code in codes) or "None.") + "\n")
    return "\n".join(lines)


def _default(field: FieldInfo) -> str:
    if field.is_required():
        return "required"
    if field.default_factory is not None:
        return f"`{field.default_factory()!r}`"  # type: ignore[call-arg]
    return f"`{field.default!r}`"


def _describe(field: FieldInfo) -> str:
    text = textwrap.dedent(field.description or "")
    bounds = [
        f"{_BOUNDS[kind]} {getattr(meta, kind)}"
        for meta in field.metadata
        if (kind := type(meta).__name__.lower()) in _BOUNDS
    ]
    return " ".join([*text.split(), *(f"({b})" for b in bounds)])


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")
