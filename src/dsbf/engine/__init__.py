# dsbf/engine/__init__.py
"""The check interface: how a check is declared and what it is given while it runs."""

from dsbf.engine.check import (
    Check,
    CheckKind,
    CheckOutputModel,
    CheckSpec,
    Extra,
    NoParams,
    ProfileFeature,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.engine.context import CheckContext, CheckOutput, ProfileView
from dsbf.engine.errors import (
    CheckContractError,
    CheckDefinitionError,
    PlanError,
    RegistryError,
)
from dsbf.engine.executor import execute
from dsbf.engine.expectation import Expectation, Expected
from dsbf.engine.pack import Pack
from dsbf.engine.plan import Plan, PlanEntry, make_plan
from dsbf.engine.registry import Registry
from dsbf.engine.render import format_plan
from dsbf.engine.runner import run_check
from dsbf.engine.selection import Depth, Selection, select

__all__ = [
    "Check",
    "CheckContext",
    "CheckContractError",
    "CheckDefinitionError",
    "CheckKind",
    "CheckOutput",
    "CheckOutputModel",
    "CheckSpec",
    "Depth",
    "Expectation",
    "Expected",
    "Extra",
    "NoParams",
    "Pack",
    "Plan",
    "PlanEntry",
    "PlanError",
    "ProfileFeature",
    "ProfileView",
    "Registry",
    "RegistryError",
    "Requires",
    "Selection",
    "SemanticTypes",
    "check",
    "execute",
    "format_plan",
    "make_plan",
    "run_check",
    "select",
]
