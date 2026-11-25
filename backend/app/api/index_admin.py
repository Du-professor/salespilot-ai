from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field

from app.api.errors import ApiProblem
from app.api.knowledge_bases import (
    index_scope, record_index_activation, require_base,
)
from app.api.library import index_write
from app.api.session import check_origin, require_permission, require_principal
from app.auth.sessions import Principal
from app.contracts.common import ContractModel, Identifier
from app.contracts.rag_workspace import (
    IndexAvailability, IndexDescriptor, IndexGcResult, IndexSwitchPlan, RagIndexConflict,
    RagIndexGarbageCollector, RagIndexManager, RagIndexNotFound,
    RagModelInvalid, RagModelUnavailable, RebuildTask,
)
from app.infrastructure.library import LibraryRebuildSource, published_documents
from app.infrastructure.sqlite import record_audit

router = APIRouter(prefix="/library", tags=["library"])
Actor = Annotated[Principal, Depends(require_principal)]


class IndexCreateRequest(ContractModel):
    embedding_model_id: Identifier | None = None
    activate: bool = False


class IndexActivateRequest(ContractModel):
    allow_stale: bool = False
    reason: str = "manual"


class IndexGcRequest(ContractModel):
    keep_superseded: Annotated[int, Field(ge=0, le=10)] = 1


class RebuildRequest(ContractModel):
    embedding_model_id: Identifier | None = None
    task_id: Identifier | None = None
    background: bool = True


class RebuildResumeRequest(ContractModel):
    background: bool = True


def _engine(request: Request) -> RagIndexManager:
    engine = request.app.state.rag_engine
    if not isinstance(engine, RagIndexManager):
        raise ApiProblem(503, "library_index_unavailable", "当前索引服务未接入索引管理。")
    return engine


def _translate(exc: Exception) -> ApiProblem:
    if isinstance(exc, RagModelUnavailable):
        return ApiProblem(503, exc.code, str(exc))
    if isinstance(exc, RagIndexConflict):
        return ApiProblem(503 if exc.retryable else 409, exc.code, str(exc))
    if isinstance(exc, RagModelInvalid):
        return ApiProblem(409, exc.code, str(exc))
    if isinstance(exc, RagIndexNotFound):
        return ApiProblem(503 if exc.retryable else 404, exc.code, str(exc))
    return ApiProblem(503, "library_index_unavailable", str(exc))


def _base(request: Request, actor: Principal, base_id: str) -> dict:
    with request.app.state.database.connect() as connection:
        require_base(connection, actor, base_id)
        return index_scope(connection, actor.org_id, base_id)


def _require_org_model(request: Request, actor: Principal, model_id: str) -> None:
    if not model_id.startswith("cfg-"):
        return
    model_store = getattr(request.app.state, "model_configs", None)
    allowed = {item[0] for item in model_store.embedding_options_for_org(actor.org_id)} if model_store else set()
    if model_id not in allowed:
        raise ApiProblem(503, "embedding_model_unavailable", "当前组织内找不到已通过测试的远程向量模型配置。")


def _documents(request: Request, actor: Principal, base_id: str):
    with request.app.state.database.connect() as connection:
        return published_documents(connection, actor.org_id, base_id=base_id)


def _activate(request: Request, actor: Principal, base_id: str, version: str, body: IndexActivateRequest):
    database = request.app.state.database
    engine = _engine(request)
    try:
        descriptor = engine.index_descriptor(actor.org_id, base_id, version)
        ready, reason, code = engine.model_availability(descriptor.embedding_model_id)
    except (RagIndexConflict, RagIndexNotFound, RagModelInvalid, RagModelUnavailable) as exc:
        raise _translate(exc) from None
    if not ready:
        raise ApiProblem(503, code or "embedding_model_unavailable", reason or "目标索引的模型不可用。")
    with index_write(database, scope=(actor.org_id, base_id)):
        previous_scope = _base(request, actor, base_id)
        previous_version = previous_scope["active_index_version"]
        documents = _documents(request, actor, base_id)
        try:
            plan = engine.mark_active(
                actor.org_id, base_id, version, documents=documents, allow_stale=body.allow_stale,
            )
            descriptor = engine.index_descriptor(actor.org_id, base_id, version)
        except Exception as exc:
            raise _translate(exc) from None
        try:
            with database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                record_index_activation(
                    connection, org_id=actor.org_id, user_id=actor.user_id, base_id=base_id,
                    descriptor=descriptor, reason=body.reason[:64] or "activated", request_id=request.state.request_id,
                )
        except Exception:
            if previous_version is not None and previous_version != version:
                try:
                    engine.mark_active(actor.org_id, base_id, previous_version, allow_stale=True)
                except Exception:
                    pass
            raise
    return plan, descriptor


@router.get("/knowledge-bases/{knowledge_base_id}/index", response_model=IndexAvailability)
def index_availability(knowledge_base_id: Identifier, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    scope = _base(request, actor, knowledge_base_id)
    engine = _engine(request)
    available = engine.availability(actor.org_id, knowledge_base_id)
    if available.active_index_version != scope["active_index_version"] and scope["active_index_version"] is not None:
        # Business DB is the source of truth for active index pointer; idempotently reconcile index store.
        try:
            target = engine.index_descriptor(actor.org_id, knowledge_base_id, scope["active_index_version"])
            if target.state in {"ready", "active", "superseded", "legacy"}:
                engine.mark_active(actor.org_id, knowledge_base_id, scope["active_index_version"], allow_stale=True)
                available = engine.availability(actor.org_id, knowledge_base_id)
        except Exception:
            pass
    if available.active_index_version == scope["active_index_version"]:
        if available.active_index_version is None:
            return available.model_copy(update={"embedding_model_id": scope["embedding_model_id"]})
        return available
    return available.model_copy(update={
        "state": "failed",
        "active_index_version": scope["active_index_version"],
        "embedding_model_id": scope["embedding_model_id"],
        "embedding_model_revision": scope["embedding_model_revision"],
        "dimensions": None,
        "ready": False,
        "reason": "索引状态与业务库记录不一致，请核对后重新启用索引版本。",
        "retryable": True,
    })


@router.get(
    "/knowledge-bases/{knowledge_base_id}/indexes", response_model=tuple[IndexDescriptor, ...],
)
def list_index_versions(knowledge_base_id: Identifier, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    _base(request, actor, knowledge_base_id)
    return _engine(request).list_index_versions(actor.org_id, knowledge_base_id)


@router.post(
    "/knowledge-bases/{knowledge_base_id}/indexes",
    response_model=IndexDescriptor, status_code=201, dependencies=[Depends(check_origin)],
)
def create_index_version(
    knowledge_base_id: Identifier, body: IndexCreateRequest, request: Request, actor: Actor,
):
    require_permission(actor, "workspace:manage")
    scope = _base(request, actor, knowledge_base_id)
    engine = _engine(request)
    model_id = body.embedding_model_id or scope["embedding_model_id"]
    _require_org_model(request, actor, model_id)
    try:
        descriptor = engine.ensure_index_version(
            actor.org_id, knowledge_base_id, model_id, activate=False,
        )
    except Exception as exc:
        raise _translate(exc) from None
    if body.activate and descriptor.index_version != "legacy":
        _, descriptor = _activate(
            request, actor, knowledge_base_id, descriptor.index_version, IndexActivateRequest(reason="activated"),
        )
    return descriptor


@router.post(
    "/knowledge-bases/{knowledge_base_id}/indexes/{index_version}/activate",
    response_model=IndexSwitchPlan, dependencies=[Depends(check_origin)],
)
def activate_index_version(
    knowledge_base_id: Identifier, index_version: Identifier, body: IndexActivateRequest,
    request: Request, actor: Actor,
):
    require_permission(actor, "workspace:manage")
    _base(request, actor, knowledge_base_id)
    plan, _ = _activate(request, actor, knowledge_base_id, index_version, body)
    return plan


@router.post(
    "/knowledge-bases/{knowledge_base_id}/indexes/{index_version}/retire",
    status_code=204, dependencies=[Depends(check_origin)],
)
def retire_index_version(
    knowledge_base_id: Identifier, index_version: Identifier, request: Request, actor: Actor,
):
    require_permission(actor, "workspace:manage")
    _base(request, actor, knowledge_base_id)
    engine = _engine(request)
    database = request.app.state.database
    with index_write(database, scope=(actor.org_id, knowledge_base_id)):
        try:
            engine.retire(actor.org_id, knowledge_base_id, index_version)
        except Exception as exc:
            raise _translate(exc) from None
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            record_audit(
                connection, org_id=actor.org_id, user_id=actor.user_id, action="index_retired",
                request_id=request.state.request_id, object_id=index_version,
            )


@router.post(
    "/knowledge-bases/{knowledge_base_id}/indexes/gc",
    response_model=IndexGcResult, dependencies=[Depends(check_origin)],
)
def purge_retired_indexes(
    knowledge_base_id: Identifier,
    request: Request,
    actor: Actor,
    body: IndexGcRequest | None = None,
):
    require_permission(actor, "workspace:manage")
    _base(request, actor, knowledge_base_id)
    engine = request.app.state.rag_engine
    if not isinstance(engine, RagIndexGarbageCollector):
        raise ApiProblem(503, "library_index_unavailable", "当前索引服务未接入退役索引物理回收。")
    keep = body.keep_superseded if body is not None else 1
    database = request.app.state.database
    with index_write(database, scope=(actor.org_id, knowledge_base_id)):
        try:
            result = engine.purge_retired(actor.org_id, knowledge_base_id, keep_superseded=keep)
        except Exception as exc:
            raise _translate(exc) from None
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            record_audit(
                connection,
                org_id=actor.org_id,
                user_id=actor.user_id,
                action="index_garbage_collected",
                request_id=request.state.request_id,
                object_id=knowledge_base_id,
            )
    return result


@router.post(
    "/knowledge-bases/{knowledge_base_id}/rebuild",
    response_model=RebuildTask, dependencies=[Depends(check_origin)],
)
def rebuild_index(
    knowledge_base_id: Identifier, body: RebuildRequest, request: Request, actor: Actor,
):
    require_permission(actor, "workspace:manage")
    scope = _base(request, actor, knowledge_base_id)
    engine = _engine(request)
    model_id = body.embedding_model_id or scope["embedding_model_id"]
    _require_org_model(request, actor, model_id)
    source = LibraryRebuildSource(request.app.state.database, getattr(request.app.state, "object_store", None))
    try:
        task = engine.start_rebuild(
            actor.org_id, knowledge_base_id, model_id, source,
            task_id=body.task_id, background=body.background,
        )
    except Exception as exc:
        raise _translate(exc) from None
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        record_audit(
            connection, org_id=actor.org_id, user_id=actor.user_id, action="index_rebuild_started",
            request_id=request.state.request_id, object_id=task.task_id,
        )
    return task


@router.get("/knowledge-bases/{knowledge_base_id}/rebuilds", response_model=tuple[RebuildTask, ...])
def list_rebuilds(
    knowledge_base_id: Identifier,
    request: Request,
    actor: Actor,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
):
    require_permission(actor, "workspace:manage")
    _base(request, actor, knowledge_base_id)
    return _engine(request).list_rebuilds(actor.org_id, knowledge_base_id, limit=limit)


@router.get("/rebuilds/{task_id}", response_model=RebuildTask)
def rebuild_status(task_id: Identifier, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    try:
        task = _engine(request).rebuild_status(task_id)
    except Exception as exc:
        raise _translate(exc) from None
    if task.org_id != actor.org_id:
        raise ApiProblem(404, "index_task_not_found", "找不到重建任务。")
    return task


@router.post(
    "/rebuilds/{task_id}/resume", response_model=RebuildTask, dependencies=[Depends(check_origin)],
)
def resume_rebuild(
    task_id: Identifier, body: RebuildResumeRequest, request: Request, actor: Actor,
):
    require_permission(actor, "workspace:manage")
    engine = _engine(request)
    try:
        current = engine.rebuild_status(task_id)
    except Exception as exc:
        raise _translate(exc) from None
    if current.org_id != actor.org_id:
        raise ApiProblem(404, "index_task_not_found", "找不到重建任务。")
    source = LibraryRebuildSource(request.app.state.database, getattr(request.app.state, "object_store", None))
    try:
        return engine.resume_rebuild(task_id, source, background=body.background)
    except Exception as exc:
        raise _translate(exc) from None


__all__ = ["router"]
