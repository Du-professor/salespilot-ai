from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.errors import ApiProblem
from app.api.session import check_origin, require_permission, require_principal
from app.auth.sessions import Principal
from app.contracts.common import Identifier
from app.contracts.model_configs import (
    ActiveModelView, Capability, ModelConfigInput, ModelConfigList, ModelConfigView, ModelRevision, UpdateModelConfig,
)
from app.infrastructure.model_configs import ModelConfigs

router = APIRouter(prefix="/model-configs", tags=["model-configs"])
Actor = Annotated[Principal, Depends(require_principal)]
Origin = [Depends(check_origin)]


def store(request: Request) -> ModelConfigs:
    return request.app.state.model_configs


@router.get("/active", response_model=ActiveModelView)
def active_model(request: Request, actor: Actor, capability: Capability = "chat") -> ActiveModelView:
    if not request.app.state.workspace_model_routing:
        return ActiveModelView(mode="unavailable", name="模型路由未启用")
    row = store(request).active(actor.org_id, capability)
    if row is None:
        return ActiveModelView(mode="offline", name="本地演示" if capability == "chat" else "未启用远程服务")
    return ActiveModelView(mode="configured", name=row["name"], model_id=row["model_id"], provider=row["provider"])


@router.get("", response_model=ModelConfigList)
def list_model_configs(
    request: Request, actor: Actor, capability: Capability | None = None,
) -> ModelConfigList:
    require_permission(actor, "workspace:manage")
    repository = store(request)
    return ModelConfigList(
        providers=repository.providers, configs=repository.list(actor.org_id, capability),
        key_storage_ready=repository.cipher is not None,
    )


@router.post("", response_model=ModelConfigView, status_code=201, dependencies=Origin)
def create_model_config(body: ModelConfigInput, request: Request, actor: Actor) -> ModelConfigView:
    require_permission(actor, "workspace:manage")
    return store(request).save(actor, request.state.request_id, body)


@router.patch("/{config_id}", response_model=ModelConfigView, dependencies=Origin)
def update_model_config(
    config_id: Identifier, body: UpdateModelConfig, request: Request, actor: Actor,
) -> ModelConfigView:
    require_permission(actor, "workspace:manage")
    return store(request).save(actor, request.state.request_id, body, config_id)


@router.post("/{config_id}/test", response_model=ModelConfigView, dependencies=Origin)
def test_model_config(config_id: Identifier, body: ModelRevision, request: Request, actor: Actor) -> ModelConfigView:
    require_permission(actor, "workspace:manage")
    return store(request).test(actor, request.state.request_id, config_id, body.revision)


@router.post("/{config_id}/activate", response_model=ModelConfigView, dependencies=Origin)
def activate_model_config(
    config_id: Identifier, body: ModelRevision, request: Request, actor: Actor,
) -> ModelConfigView:
    require_permission(actor, "workspace:manage")
    if not request.app.state.workspace_model_routing:
        raise ApiProblem(409, "model_routing_disabled", "请在后端启用工作台 RAG 模式，再启用此模型。")
    return store(request).activate(actor, request.state.request_id, config_id, body.revision)


@router.post("/{config_id}/deactivate", response_model=ModelConfigView, dependencies=Origin)
def deactivate_model_config(config_id: Identifier, request: Request, actor: Actor) -> ModelConfigView:
    require_permission(actor, "workspace:manage")
    return store(request).deactivate(actor, request.state.request_id, config_id)


@router.delete("/{config_id}", status_code=204, dependencies=Origin)
def delete_model_config(config_id: Identifier, request: Request, actor: Actor) -> Response:
    require_permission(actor, "workspace:manage")
    store(request).delete(actor, request.state.request_id, config_id)
    return Response(status_code=204)
