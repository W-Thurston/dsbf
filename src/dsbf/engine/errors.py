# dsbf/engine/errors.py
"""Errors raised by the check interface."""


class CheckDefinitionError(ValueError):
    """A check was declared with metadata that cannot work.

    Raised when the check's module is imported, so a typo never reaches a run.
    """


class CheckContractError(RuntimeError):
    """A check used its context in a way its declaration does not allow.

    For example reading a column outside ``applies_to``, or the values of a rule
    check. It is a bug in the check, so the executor reports it as an error.
    """


class RegistryError(ValueError):
    """A check could not be registered or looked up.

    For example a second, different check under an id already taken, or an unknown id.
    """
