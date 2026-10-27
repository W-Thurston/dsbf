# dsbf/interfaces/check_reference.py
"""The check reference, generated from the core pack's specs and sources."""

import ast
import inspect
import textwrap
from pathlib import Path

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


def finding_codes(check: Check) -> tuple[str, ...]:
    """The finding codes the check's module sets, read from its source.

    A code is a literal passed as ``code=``, a string bound to a ``code`` name
    (``code, ... = (...)`` or ``for code, ... in (...)``), a value of a
    module-level table named ``..._CODES``, an argument named ``code`` annotated
    with a ``Literal``, or an f-string prefix, shown as ``prefix_<...>``.
    """
    module = inspect.getmodule(check.function)
    if module is None or module.__file__ is None:
        return ()
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    codes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "code":
                    codes |= _codes_in(keyword.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                codes |= _bound_to_code(target, node.value)
            if any(
                isinstance(t, ast.Name) and t.id.endswith("_CODES")
                for t in node.targets
            ) and isinstance(node.value, ast.Dict):
                for value in node.value.values:
                    codes |= _codes_in(value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id.endswith("_CODES")
                and isinstance(node.value, ast.Dict)
            ):
                for value in node.value.values:
                    codes |= _codes_in(value)
        elif isinstance(node, ast.arg) and node.arg == "code":
            codes |= _literal_strings(node.annotation)
        elif isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            for item in node.iter.elts:
                codes |= _bound_to_code(node.target, item)
    return tuple(sorted(codes))


def _literal_strings(annotation: ast.expr | None) -> set[str]:
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "Literal"
    ):
        members = annotation.slice
        items = members.elts if isinstance(members, ast.Tuple) else [members]
        return {
            item.value
            for item in items
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
    return set()


def _codes_in(node: ast.expr | None) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.JoinedStr) and node.values:
        head = node.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return {f"{head.value}<...>"}
    if isinstance(node, ast.IfExp):
        return _codes_in(node.body) | _codes_in(node.orelse)
    return set()


def _bound_to_code(target: ast.expr, value: ast.expr) -> set[str]:
    if isinstance(target, ast.Name) and target.id == "code":
        return _codes_in(value)
    if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
        found: set[str] = set()
        for name, item in zip(target.elts, value.elts):
            found |= _bound_to_code(name, item)
        return found
    return set()


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
    codes = finding_codes(check)
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
