from typing import Literal

from .common import BusinessCode, ContractModel, Identifier, Text


class ApiError(ContractModel):
    code: BusinessCode
    message: Text
    request_id: Identifier


class HealthStatus(ContractModel):
    status: Literal["ok"]
