from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[
    str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
]
BusinessCode = Annotated[
    str, StringConstraints(strict=True, pattern=r"^[a-z][a-z0-9_]{0,63}$")
]
Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonNegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
