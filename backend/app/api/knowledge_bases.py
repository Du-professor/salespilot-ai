import sqlite3
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from pydantic import Field, StringConstraints

from app.api.errors import ApiProblem
from app.api.session import check_origin, require_permission, require_principal
from app.auth.sessions import Principal
from app.contracts.common import ContractModel, Identifier, Text
from app.contracts.rag_workspace import IndexDescriptor
from app.infrastructure.sqlite import record_audit
from app.rag import model_options

router = APIRouter(prefix="/library", tags=["library"])
Actor = Annotated[Principal, Depends(require_principal)]
DEFAULT_BASE_ID = "default"
DEFAULT_MODEL_ID = "workspace-default"
DEFAULT_MODEL_REVISION = "workspace"
DEFAULT_BASE_NAME = "默认知识库"
DEFAULT_INDEX_VERSION = "legacy"


class EmbeddingModelOption(ContractModel):
    model_id: Identifier
    name: str
    selectable: bool


def embedding_model_options(request: Request, actor: Actor) -> tuple[EmbeddingModelOption, ...]:
    settings = request.app.state.settings
    from app.rag.config import RagSettings

    catalog_settings = RagSettings(
        models_dir=settings.rag_models_dir,
        embedding_device=settings.rag_embedding_device,
        embedding_model_id=settings.rag_embedding_model_id,
    )
    options = [
        EmbeddingModelOption(model_id=model_id, name=name, selectable=selectable)
        for model_id, name, selectable in model_options(catalog_settings, settings.rag_models_dir)
    ]
    model_store = getattr(request.app.state, "model_configs", None)
    if model_store is not None:
        for model_id, name, selectable in model_store.embedding_options_for_org(actor.org_id):
            options.append(EmbeddingModelOption(model_id=model_id, name=name, selectable=selectable))
    return tuple(options)


ModelOptions = Annotated[tuple[EmbeddingModelOption, ...], Depends(embedding_model_options)]


AccessPolicy = Literal["org_published", "restricted"]
GrantSubjectType = Literal["user", "role", "group"]


class KnowledgeBaseGrant(ContractModel):
    subject_type: GrantSubjectType
    subject_id: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=128)]


class KnowledgeBaseAcl(ContractModel):
    knowledge_base_id: Identifier
    access_policy: AccessPolicy = "org_published"
    grants: tuple[KnowledgeBaseGrant, ...] = ()


class KnowledgeBaseAclUpdate(ContractModel):
    access_policy: AccessPolicy
    grants: tuple[KnowledgeBaseGrant, ...] = Field(default=(), max_length=200)


class KnowledgeBaseCreate(ContractModel):
    name: Annotated[Text, Field(max_length=100)]
    description: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, max_length=500)] = ""
    embedding_model_id: Identifier = DEFAULT_MODEL_ID
    access_policy: AccessPolicy = "org_published"


class KnowledgeBaseUpdate(ContractModel):
    name: Annotated[Text, Field(max_length=100)]
    description: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, max_length=500)] | None = None


class KnowledgeBase(KnowledgeBaseCreate):
    knowledge_base_id: Identifier
    index_state: Literal["legacy", "awaiting_adapter", "building", "available", "failed"]
    active_index_version: Identifier | None = None
    embedding_model_revision: str | None = None


DEFAULT_BASE = KnowledgeBase(
    knowledge_base_id=DEFAULT_BASE_ID,
    name=DEFAULT_BASE_NAME,
    index_state="legacy",
    active_index_version=DEFAULT_INDEX_VERSION,
    embedding_model_revision=DEFAULT_MODEL_REVISION,
)
_COLUMNS = (
    "knowledge_base_id, name, description, embedding_model_id, "
    "embedding_model_revision, active_index_version, index_state, access_policy"
)


def _has_grant(connection: sqlite3.Connection, actor: Principal, base_id: str) -> bool:
    candidates: list[tuple[str, str]] = [("user", actor.user_id), ("role", actor.role)]
    for grp in actor.groups:
        candidates.append(("group", grp))
    placeholders = ", ".join("(?, ?)" for _ in candidates)
    params: list[str] = [actor.org_id, base_id]
    for stype, sid in candidates:
        params.extend((stype, sid))
    row = connection.execute(
        f"""SELECT 1 FROM library_kb_grants
        WHERE org_id=? AND knowledge_base_id=? AND (subject_type, subject_id) IN ({placeholders})
        LIMIT 1""",
        params,
    ).fetchone()
    return row is not None


def can_access_base(connection: sqlite3.Connection, actor: Principal, base_id: str) -> bool:
    if base_id == DEFAULT_BASE_ID:
        return True
    row = connection.execute(
        """SELECT base.access_policy, EXISTS (
            SELECT 1 FROM library_document_versions AS doc
            WHERE doc.org_id=base.org_id AND doc.knowledge_base_id=base.knowledge_base_id AND doc.status='published'
        ) AS has_published
        FROM library_knowledge_bases AS base
        WHERE base.org_id=? AND base.knowledge_base_id=?""",
        (actor.org_id, base_id),
    ).fetchone()
    if row is None:
        return False
    if actor.allows("workspace:manage"):
        return True
    if not row["has_published"]:
        return False
    if row["access_policy"] != "restricted":
        return True
    return _has_grant(connection, actor, base_id)


def accessible_base_ids(connection: sqlite3.Connection, actor: Principal) -> set[str] | None:
    if actor.allows("workspace:manage"):
        return None
    allowed = {DEFAULT_BASE_ID}
    rows = connection.execute(
        """SELECT knowledge_base_id, access_policy FROM library_knowledge_bases AS base
        WHERE org_id=? AND EXISTS (
            SELECT 1 FROM library_document_versions AS doc
            WHERE doc.org_id=base.org_id AND doc.knowledge_base_id=base.knowledge_base_id AND doc.status='published'
        )""",
        (actor.org_id,),
    ).fetchall()
    for row in rows:
        kb_id = row["knowledge_base_id"]
        if row["access_policy"] != "restricted" or _has_grant(connection, actor, kb_id):
            allowed.add(kb_id)
    return allowed


def require_base(connection: sqlite3.Connection, actor: Principal, base_id: str) -> None:
    if not can_access_base(connection, actor, base_id):
        raise ApiProblem(404, "knowledge_base_not_found", "找不到可访问的知识库。")


def index_scope(connection: sqlite3.Connection, org_id: str, base_id: str) -> dict:
    if base_id == DEFAULT_BASE_ID:
        override = connection.execute(
            "SELECT name, description FROM library_knowledge_bases WHERE org_id=? AND knowledge_base_id=?",
            (org_id, DEFAULT_BASE_ID),
        ).fetchone()
        extra = {"name": override["name"], "description": override["description"]} if override is not None else {}
        activated = connection.execute(
            """SELECT index_version, embedding_model_id, embedding_model_revision
            FROM library_index_activations
            WHERE org_id=? AND knowledge_base_id=? ORDER BY sequence DESC LIMIT 1""",
            (org_id, DEFAULT_BASE_ID),
        ).fetchone()
        if activated is not None:
            return {
                "knowledge_base_id": DEFAULT_BASE_ID,
                "embedding_model_id": activated["embedding_model_id"],
                "embedding_model_revision": activated["embedding_model_revision"],
                "active_index_version": activated["index_version"],
                "index_state": "available",
                **extra,
            }
        return {
            "knowledge_base_id": DEFAULT_BASE_ID,
            "embedding_model_id": DEFAULT_MODEL_ID,
            "embedding_model_revision": DEFAULT_MODEL_REVISION,
            "active_index_version": DEFAULT_INDEX_VERSION,
            "index_state": "legacy",
            **extra,
        }
    row = connection.execute(
        f"""SELECT {_COLUMNS} FROM library_knowledge_bases
        WHERE org_id=? AND knowledge_base_id=?""",
        (org_id, base_id),
    ).fetchone()
    if row is None:
        raise ApiProblem(404, "knowledge_base_not_found", "找不到可访问的知识库。")
    return dict(row)


def record_index_activation(
    connection: sqlite3.Connection,
    *,
    org_id: str,
    user_id: str,
    base_id: str,
    descriptor: IndexDescriptor,
    reason: str,
    request_id: str,
) -> None:
    if base_id != DEFAULT_BASE_ID:
        connection.execute(
            """UPDATE library_knowledge_bases
            SET embedding_model_id=?, embedding_model_revision=?, active_index_version=?, index_state='available'
            WHERE org_id=? AND knowledge_base_id=?""",
            (
                descriptor.embedding_model_id, descriptor.embedding_model_revision,
                descriptor.index_version, org_id, base_id,
            ),
        )
    connection.execute(
        """INSERT INTO library_index_activations
        (org_id, knowledge_base_id, index_version, embedding_model_id, embedding_model_revision,
         reason, activated_at, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            org_id, base_id, descriptor.index_version, descriptor.embedding_model_id,
            descriptor.embedding_model_revision, reason, datetime.now(UTC).isoformat(), user_id,
        ),
    )
    record_audit(
        connection, org_id=org_id, user_id=user_id, action=f"index_{reason}",
        request_id=request_id, object_id=descriptor.index_version,
    )


def set_base_index_state(connection: sqlite3.Connection, org_id: str, base_id: str, state: str) -> None:
    if base_id == DEFAULT_BASE_ID:
        return
    connection.execute(
        "UPDATE library_knowledge_bases SET index_state=? WHERE org_id=? AND knowledge_base_id=?",
        (state, org_id, base_id),
    )


@router.get("/knowledge-bases", response_model=list[KnowledgeBase])
def list_knowledge_bases(request: Request, actor: Actor):
    require_permission(actor, "catalog:read")
    with request.app.state.database.connect() as connection:
        active_default = DEFAULT_BASE.model_copy(update=index_scope(connection, actor.org_id, DEFAULT_BASE_ID))
        rows = connection.execute(
            f"""SELECT {_COLUMNS} FROM library_knowledge_bases AS base
            WHERE org_id=? AND knowledge_base_id != ?
            ORDER BY created_at DESC, knowledge_base_id""",
            (actor.org_id, DEFAULT_BASE_ID),
        ).fetchall()
        visible_rows = [
            row for row in rows
            if can_access_base(connection, actor, row["knowledge_base_id"])
        ]
    return [active_default, *(KnowledgeBase(**dict(row)) for row in visible_rows)]


@router.get("/embedding-models", response_model=list[EmbeddingModelOption])
def list_embedding_models(actor: Actor, models: ModelOptions):
    require_permission(actor, "workspace:manage")
    return models


@router.post("/knowledge-bases", status_code=201, response_model=KnowledgeBase, dependencies=[Depends(check_origin)])
def create_knowledge_base(body: KnowledgeBaseCreate, request: Request, actor: Actor, models: ModelOptions):
    require_permission(actor, "workspace:manage")
    if not any(model.model_id == body.embedding_model_id and model.selectable for model in models):
        raise ApiProblem(409, "embedding_model_unavailable", "请选择当前可用的索引模型。")
    base_id = uuid4().hex
    with request.app.state.database.connect() as connection:
        default_name = index_scope(connection, actor.org_id, DEFAULT_BASE_ID).get("name", DEFAULT_BASE_NAME)
        if body.name.casefold() == default_name.casefold():
            raise ApiProblem(409, "knowledge_base_exists", "已有同名知识库。")
        inserted = connection.execute(
            """INSERT INTO library_knowledge_bases
            (org_id, knowledge_base_id, name, name_key, description, embedding_model_id, access_policy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING RETURNING knowledge_base_id""",
            (actor.org_id, base_id, body.name, body.name.casefold(), body.description,
             body.embedding_model_id, body.access_policy, datetime.now(UTC).isoformat()),
        ).fetchone()
        if inserted is None:
            raise ApiProblem(409, "knowledge_base_exists", "已有同名知识库。")
        record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action="knowledge_base_created",
                     request_id=request.state.request_id, object_id=base_id)
    return KnowledgeBase(
        **body.model_dump(),
        knowledge_base_id=base_id,
        index_state="awaiting_adapter",
        active_index_version=None,
        embedding_model_revision=None,
    )


@router.patch(
    "/knowledge-bases/{knowledge_base_id}",
    response_model=KnowledgeBase,
    dependencies=[Depends(check_origin)],
)
def update_knowledge_base(
    knowledge_base_id: Identifier, body: KnowledgeBaseUpdate, request: Request, actor: Actor,
) -> KnowledgeBase:
    require_permission(actor, "workspace:manage")
    name = body.name.strip()
    name_key = name.casefold()
    now = datetime.now(UTC).isoformat()
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        default_scope = index_scope(connection, actor.org_id, DEFAULT_BASE_ID)
        default_name = default_scope.get("name", DEFAULT_BASE_NAME)
        if knowledge_base_id != DEFAULT_BASE_ID and name_key == default_name.casefold():
            raise ApiProblem(409, "knowledge_base_exists", "已有同名知识库。")
        conflict = connection.execute(
            """SELECT 1 FROM library_knowledge_bases
            WHERE org_id=? AND name_key=? AND knowledge_base_id!=?""",
            (actor.org_id, name_key, knowledge_base_id),
        ).fetchone()
        if conflict is not None:
            raise ApiProblem(409, "knowledge_base_exists", "已有同名知识库。")
        if knowledge_base_id == DEFAULT_BASE_ID:
            description = body.description if body.description is not None else default_scope.get("description", "")
            try:
                connection.execute(
                    """INSERT INTO library_knowledge_bases
                    (org_id, knowledge_base_id, name, name_key, description, embedding_model_id,
                     embedding_model_revision, active_index_version, index_state, access_policy, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'org_published', ?)
                    ON CONFLICT (org_id, knowledge_base_id) DO UPDATE SET
                        name=excluded.name,
                        name_key=excluded.name_key,
                        description=excluded.description""",
                    (
                        actor.org_id, DEFAULT_BASE_ID, name, name_key, description,
                        default_scope["embedding_model_id"], default_scope.get("embedding_model_revision"),
                        default_scope.get("active_index_version"), default_scope.get("index_state", "legacy"), now,
                    ),
                )
            except sqlite3.IntegrityError:
                raise ApiProblem(409, "knowledge_base_exists", "已有同名知识库。") from None
            record_audit(
                connection, org_id=actor.org_id, user_id=actor.user_id, action="knowledge_base_updated",
                request_id=request.state.request_id, object_id=DEFAULT_BASE_ID,
            )
            return DEFAULT_BASE.model_copy(update=index_scope(connection, actor.org_id, DEFAULT_BASE_ID))

        existing = connection.execute(
            f"SELECT {_COLUMNS} FROM library_knowledge_bases WHERE org_id=? AND knowledge_base_id=?",
            (actor.org_id, knowledge_base_id),
        ).fetchone()
        if existing is None:
            raise ApiProblem(404, "knowledge_base_not_found", "找不到可访问的知识库。")
        description = body.description if body.description is not None else existing["description"]
        try:
            connection.execute(
                """UPDATE library_knowledge_bases
                SET name=?, name_key=?, description=?
                WHERE org_id=? AND knowledge_base_id=?""",
                (name, name_key, description, actor.org_id, knowledge_base_id),
            )
        except sqlite3.IntegrityError:
            raise ApiProblem(409, "knowledge_base_exists", "已有同名知识库。") from None
        record_audit(
            connection, org_id=actor.org_id, user_id=actor.user_id, action="knowledge_base_updated",
            request_id=request.state.request_id, object_id=knowledge_base_id,
        )
        updated = connection.execute(
            f"SELECT {_COLUMNS} FROM library_knowledge_bases WHERE org_id=? AND knowledge_base_id=?",
            (actor.org_id, knowledge_base_id),
        ).fetchone()
    return KnowledgeBase(**dict(updated))


@router.get("/knowledge-bases/{knowledge_base_id}/acl", response_model=KnowledgeBaseAcl)
def get_knowledge_base_acl(knowledge_base_id: Identifier, request: Request, actor: Actor) -> KnowledgeBaseAcl:
    require_permission(actor, "workspace:manage")
    if knowledge_base_id == DEFAULT_BASE_ID:
        return KnowledgeBaseAcl(knowledge_base_id=DEFAULT_BASE_ID, access_policy="org_published", grants=())
    with request.app.state.database.connect() as connection:
        row = connection.execute(
            "SELECT access_policy FROM library_knowledge_bases WHERE org_id=? AND knowledge_base_id=?",
            (actor.org_id, knowledge_base_id),
        ).fetchone()
        if row is None:
            raise ApiProblem(404, "knowledge_base_not_found", "找不到可访问的知识库。")
        grants = connection.execute(
            """SELECT subject_type, subject_id FROM library_kb_grants
            WHERE org_id=? AND knowledge_base_id=? ORDER BY subject_type, subject_id""",
            (actor.org_id, knowledge_base_id),
        ).fetchall()
    return KnowledgeBaseAcl(
        knowledge_base_id=knowledge_base_id,
        access_policy=row["access_policy"],
        grants=tuple(KnowledgeBaseGrant(subject_type=g["subject_type"], subject_id=g["subject_id"]) for g in grants),
    )


@router.put(
    "/knowledge-bases/{knowledge_base_id}/acl",
    response_model=KnowledgeBaseAcl,
    dependencies=[Depends(check_origin)],
)
def update_knowledge_base_acl(
    knowledge_base_id: Identifier, body: KnowledgeBaseAclUpdate, request: Request, actor: Actor,
) -> KnowledgeBaseAcl:
    require_permission(actor, "workspace:manage")
    if knowledge_base_id == DEFAULT_BASE_ID:
        raise ApiProblem(409, "default_kb_acl_immutable", "默认知识库保持全组织已发布可见；请为受限资料创建独立知识库。")
    now = datetime.now(UTC).isoformat()
    deduped = list(dict.fromkeys((g.subject_type, g.subject_id) for g in body.grants))
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT 1 FROM library_knowledge_bases WHERE org_id=? AND knowledge_base_id=?",
            (actor.org_id, knowledge_base_id),
        ).fetchone()
        if row is None:
            raise ApiProblem(404, "knowledge_base_not_found", "找不到可访问的知识库。")
        connection.execute(
            "UPDATE library_knowledge_bases SET access_policy=? WHERE org_id=? AND knowledge_base_id=?",
            (body.access_policy, actor.org_id, knowledge_base_id),
        )
        connection.execute(
            "DELETE FROM library_kb_grants WHERE org_id=? AND knowledge_base_id=?",
            (actor.org_id, knowledge_base_id),
        )
        if deduped:
            connection.executemany(
                """INSERT INTO library_kb_grants (org_id, knowledge_base_id, subject_type, subject_id, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                [(actor.org_id, knowledge_base_id, stype, sid, now) for stype, sid in deduped],
            )
        record_audit(
            connection, org_id=actor.org_id, user_id=actor.user_id, action="knowledge_base_acl_updated",
            request_id=request.state.request_id, object_id=knowledge_base_id,
        )
    return KnowledgeBaseAcl(
        knowledge_base_id=knowledge_base_id,
        access_policy=body.access_policy,
        grants=tuple(KnowledgeBaseGrant(subject_type=stype, subject_id=sid) for stype, sid in deduped),
    )
