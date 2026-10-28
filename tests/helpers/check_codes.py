"""Read the finding codes a check's module sets, from its source.

The declarations on ``@check(codes=...)`` are what the reference prints. This scan is
how a test notices a code that a declaration misses.
"""

import ast
import inspect
from pathlib import Path

from dsbf.engine import Check


def scanned_codes(check: Check) -> tuple[str, ...]:
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
