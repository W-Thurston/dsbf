"""Compare a run through the new engine with a legacy golden snapshot."""

from __future__ import annotations

from typing import Any

from tests.golden.normalise import normalise_task

# What the snapshot writes for an undefined number. A result holds it as null, and says
# which output keys held one (``CheckResult.non_finite_outputs``).
UNDEFINED = frozenset({"NaN", "Infinity", "-Infinity"})

# How the engine's statuses read in the legacy report's words.
LEGACY_STATUS = {"ok": "success", "skipped": "skipped", "error": "failed"}


def undefined_as_null(value: Any) -> Any:
    """``value`` with the snapshot's spelling of an undefined number made null."""
    if isinstance(value, str):
        return None if value in UNDEFINED else value
    if isinstance(value, dict):
        return {k: undefined_as_null(v) for k, v in value.items()}
    if isinstance(value, list):
        return [undefined_as_null(v) for v in value]
    return value


def differences(
    snapshot: dict[str, Any], engine: dict[str, Any], roots: list[str]
) -> dict[str, str]:
    """What differs, by task, between the committed snapshot and the engine's run.

    A check that ended ``ok`` is compared whole, on the snapshot's own terms (see
    ``normalise_task``), reading the snapshot's ``NaN`` as the null a result holds;
    anything else is compared by status. The value says what differs: a status pair,
    or the fields of the entry that moved.
    """
    found: dict[str, str] = {}
    checks = engine["checks"]
    legacy_tasks = snapshot["tasks"]
    for name in sorted(set(legacy_tasks) | set(checks)):
        if name not in checks:
            found[name] = "not run by the engine"
            continue
        if name not in legacy_tasks:
            found[name] = f"run only by the engine ({checks[name]['status']})"
            continue
        legacy = legacy_tasks[name]
        check = checks[name]
        status = LEGACY_STATUS.get(check["status"], check["status"])
        if status != legacy["status"]:
            found[name] = f"status: legacy {legacy['status']}, engine {status}"
        elif check["status"] == "ok":
            actual = normalise_task(name, check["result"], roots)
            moved = sorted(
                k for k in actual if actual[k] != undefined_as_null(legacy.get(k))
            )
            if moved:
                found[name] = "changed: " + ", ".join(moved)
    return found
