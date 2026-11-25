from typing import Annotated, Literal

from pydantic import Field, SecretStr

from app.contracts.common import ContractModel, Identifier, PositiveInt, Text

Provider = Literal["deepseek", "kimi", "glm", "openai", "minimax", "ollama", "custom"]
Capability = Literal["chat", "embedding", "rerank", "parser"]
TestStatus = Literal["untested", "passed", "failed"]


class ModelProvider(ContractModel):
    provider: Provider
    label: str
    base_urls: tuple[str, ...]
    requires_key: bool


class ModelConfigInput(ContractModel):
    name: Annotated[Text, Field(max_length=80)]
    capability: Capability = "chat"
    provider: Provider
    model_id: Annotated[Text, Field(max_length=200)]
    base_url: Annotated[Text, Field(max_length=300)] | None = None
    api_key: Annotated[SecretStr, Field(min_length=1, max_length=4000)] | None = Field(default=None, exclude=True)
    clear_key: bool = False


class UpdateModelConfig(ModelConfigInput):
    revision: PositiveInt


class ModelRevision(ContractModel):
    revision: PositiveInt


class ModelConfigView(ContractModel):
    config_id: Identifier
    capability: Capability = "chat"
    name: str
    provider: Provider
    model_id: str
    base_url: str
    has_key: bool
    dimensions: PositiveInt | None = None
    revision: PositiveInt
    test_status: TestStatus
    test_message: str | None
    test_at: int | None
    active: bool


class ModelConfigList(ContractModel):
    providers: tuple[ModelProvider, ...]
    configs: list[ModelConfigView]
    key_storage_ready: bool


class ActiveModelView(ContractModel):
    mode: Literal["configured", "offline", "unavailable"]
    name: str
    model_id: str | None = None
    provider: Provider | None = None
