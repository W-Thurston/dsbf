# dsbf/policy/load.py
"""Read a policy from a file or a mapping, and say which rule is wrong."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dsbf.config import ConfigError, ConfigIssue
from dsbf.model import Policy

__all__ = ["PolicyError", "load_policy"]


class PolicyError(ConfigError):
    """The policy is invalid; ``issues`` names each problem and the rule it is in."""

    def __init__(self, issues: tuple[ConfigIssue, ...]) -> None:
        super().__init__(issues)
        count = len(issues)
        header = f"Invalid policy ({count} {'issue' if count == 1 else 'issues'}):"
        self.args = ("\n".join([header, *(f"  {issue}" for issue in issues)]),)


def load_policy(source: str | Path | Mapping[str, Any] | Policy) -> Policy:
    """A ``Policy`` from a YAML file, a mapping, or a ``Policy`` (returned as is).

    Raises:
        PolicyError: the file cannot be read or is not valid YAML, or the rules do
            not validate; every problem is listed, each under its rule's name.
    """
    if isinstance(source, Policy):
        return source
    origin = None
    if isinstance(source, str | Path):
        origin = f"file {source}"
        document = _read(Path(source), origin)
    else:
        document = source
    if not isinstance(document, Mapping):
        message = "a policy is a mapping with `version` and `rules`"
        raise PolicyError((ConfigIssue("", message, origin),))
    try:
        return Policy.model_validate(dict(document))
    except ValidationError as error:
        raise PolicyError(_issues(error, document, origin)) from None


def _read(path: Path, origin: str) -> Any:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PolicyError(
            (ConfigIssue("", "the policy file does not exist", origin),)
        ) from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        reason = " ".join(str(exc).split())
        raise PolicyError(
            (ConfigIssue("", f"cannot read the policy file: {reason}", origin),)
        ) from None
    return {} if loaded is None else loaded


def _issues(
    error: ValidationError, document: Mapping[str, Any], origin: str | None
) -> tuple[ConfigIssue, ...]:
    rules = document.get("rules")
    issues = []
    for item in error.errors():
        location = tuple(item["loc"])
        message = item["msg"].removeprefix("Value error, ")
        if item["type"] == "extra_forbidden":
            message = "unknown key"
        issues.append(ConfigIssue(_path(location, rules), message, origin))
    return tuple(issues)


def _path(location: tuple[int | str, ...], rules: Any) -> str:
    """Dotted location with a rule's index replaced by its name when it has one."""
    text = ""
    for position, part in enumerate(location):
        if isinstance(part, int):
            in_rules = location[:position] == ("rules",)
            name = _rule_name(rules, part) if in_rules else None
            text += f"[{name or part}]"
        else:
            text += f".{part}" if text else str(part)
    return text


def _rule_name(rules: Any, index: int) -> str | None:
    if not isinstance(rules, list | tuple) or index >= len(rules):
        return None
    rule = rules[index]
    name = rule.get("name") if isinstance(rule, Mapping) else None
    return name if isinstance(name, str) and name else None
