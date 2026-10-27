# dsbf/model/base.py
"""Shared base class and scalar types for the result contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# A float that cannot be NaN or infinite, so it always serialises to valid JSON.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# A JSON scalar. `bool` comes first so a flag isn't read as a number.
Scalar = bool | int | FiniteFloat | str | None


class Contract(BaseModel):
    """Base for every contract model.

    Instances are immutable, so nothing can change a value after it has been
    validated, and unknown fields are errors, so a misspelt name is caught.
    A model is always written out in full, so its serialisation schema lists
    defaulted fields as required.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_serialization_defaults_required=True
    )


def fields_equal(left: BaseModel, right: object) -> bool:
    """Whether ``right`` is the same model with equal fields, private data aside.

    For a model that carries data outside its fields (artifact payloads), which
    ``BaseModel.__eq__`` would compare too.
    """
    if type(left) is not type(right):
        return NotImplemented  # type: ignore[return-value]
    return all(
        getattr(left, name) == getattr(right, name) for name in type(left).model_fields
    )
