# dsbf/model/params.py
"""Base class for the parameters a check declares."""

from typing import Any

from dsbf.model.base import Contract


class CheckParams(Contract):
    """Declared, validated parameters of one check.

    A check subclasses this with one typed field per parameter. Every field needs a
    default, so the check runs with no configuration and its defaults, bounds and
    descriptions are declared in one place. Unknown keys are errors, so a misspelt
    parameter can't be silently ignored.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        required = [
            name for name, field in cls.model_fields.items() if field.is_required()
        ]
        if required:
            raise TypeError(
                f"{cls.__name__} declares parameters without a default: "
                f"{', '.join(required)}. Every check parameter needs one."
            )
