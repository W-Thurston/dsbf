# dsbf/model/base.py
"""Shared base class and scalar types for the result contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# A float that cannot be NaN or infinite, so it always serialises to valid JSON.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class Contract(BaseModel):
    """Base for every contract model.

    Instances are immutable, so nothing can change a value after it has been
    validated, and unknown fields are errors, so a misspelt name is caught.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
