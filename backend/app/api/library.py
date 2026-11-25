import hashlib
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import Field, TypeAdapter, ValidationError, field_validator

from app.api.errors import ApiProblem
from app.api.knowledge_bases import (
    DEFAULT_BASE_ID, index_scope, record_index_activation, require_base, set_base_index_state,
)
from app.api.session import check_origin, require_permission, require_principal
from app.auth.sessions import Principal
from app.contracts import ApiError
from app.contracts.common import ContractModel, Identifier, Text
from app.contracts.rag_workspace import (
    DocumentVersionKey, LibraryDocument, RagChunkReader, RagContext, RagDiagnostics, RagDocumentError,
    RagIndexConflict, RagIndexManager, RagIndexRemover, RagModelInvalid, RagModelUnavailable, RagUnavailable,
)
from app.infrastructure.library import (
    CONTENT_TYPES, METADATA, LibraryIndexBusy, document_view, download_response, library_index_write,
    published_documents, streaming_download_response,
)
from app.infrastructure.library_inspection import ChunkPage, ChunkPreview, RetrievalTest, inspect_retrieval
from app.infrastructure.publish_queue import PublishTask
from app.infrastructure.sqlite import record_audit
from app.infrastructure.telemetry import (
    RetrievalTelemetrySummary, record_retrieval_event, summarize_retrieval_telemetry,
)
from app.infrastructure.uploads import (
    LIBRARY_DOCX_MAX_EXPANDED_BYTES, LIBRARY_MAX_BYTES, LIBRARY_PPTX_MAX_EXPANDED_BYTES, LibraryFilePayload,
    stream_multipart_upload, stream_raw_upload, validate_archive, validate_archive_file, validate_pdf_file,
    validate_text_file,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)

router = APIRouter(prefix="/library", tags=["library"])
Actor = Annotated[Principal, Depends(require_principal)]


class LibraryUpload(ContractModel):
    file: LibraryFilePayload
    version: Identifier = "v1"
    document_id: Identifier | None = None
    folder_id: Identifier | None = None
    knowledge_base_id: Identifier = DEFAULT_BASE_ID


class FolderFields(ContractModel):
    name: Annotated[Text, Field(max_length=100)]
    parent_id: Identifier | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, name: str) -> str:
        if name in {".", ".."} or any(char in "/\\" or ord(char) < 32 for char in name):
            raise ValueError("Invalid folder name")
        return name


class FolderCreate(FolderFields):
    knowledge_base_id: Identifier = DEFAULT_BASE_ID


class LibraryFolder(FolderFields):
    folder_id: Identifier


class LibraryEntry(LibraryDocument):
    folder_id: Identifier | None


class LibraryBrowser(ContractModel):
    folders: list[LibraryFolder]
    documents: list[LibraryEntry]


class DocumentLocation(ContractModel):
    folder_id: Identifier | None


class RetrievalQuery(ContractModel):
    text: Annotated[Text, Field(max_length=4000)]
    knowledge_base_id: Identifier = DEFAULT_BASE_ID
    mode: Literal["selection", "raw_retrieval"] = "selection"
    strategy: Literal["hybrid", "vector", "bm25"] | None = None
    index_version: Identifier | None = None
    top_k: Annotated[int, Field(ge=1, le=32)] | None = None


def _require_folder(connection, org_id: str, base_id: str, folder_id: str | None) -> None:
    if folder_id is not None and connection.execute(
        "SELECT 1 FROM library_folders WHERE org_id=? AND knowledge_base_id=? AND folder_id=?",
        (org_id, base_id, folder_id),
    ).fetchone() is None:
        raise ApiProblem(404, "library_folder_not_found", "找不到可访问的文件夹。")


def _location(connection, org_id: str, document_id: str) -> str | None:
    row = connection.execute(
        "SELECT folder_id FROM library_document_locations WHERE org_id=? AND document_id=?",
        (org_id, document_id),
    ).fetchone()
    return row["folder_id"] if row else None


def _set_location(connection, org_id: str, document_id: str, folder_id: str | None) -> None:
    if folder_id is None:
        connection.execute(
            "DELETE FROM library_document_locations WHERE org_id=? AND document_id=?",
            (org_id, document_id),
        )
    else:
        connection.execute(
            """INSERT INTO library_document_locations (org_id, document_id, folder_id) VALUES (?, ?, ?)
            ON CONFLICT (org_id, document_id) DO UPDATE SET folder_id=excluded.folder_id""",
            (org_id, document_id, folder_id),
        )


def _validate_filename(filename: str) -> str:
    cleaned = filename.strip()
    if not cleaned or len(cleaned) > 180:
        raise ApiProblem(422, "invalid_file", "请选择有效的文件名。")
    if any(character in "/\\" or ord(character) < 32 for character in cleaned):
        raise ApiProblem(422, "invalid_file", "请选择有效的文件名。")
    suffix = Path(cleaned).suffix.lower()
    if suffix not in CONTENT_TYPES:
        raise ApiProblem(422, "unsupported_document", "支持 TXT、Markdown、PDF、DOCX 和 PPTX。")
    return suffix


def _check_file(file: LibraryFilePayload, payload: bytes) -> None:
    suffix = _validate_filename(file.filename)
    if suffix == ".pdf" and not payload.startswith(b"%PDF-"):
        raise ApiProblem(422, "invalid_document", "文件不是有效的 PDF。")
    if suffix == ".docx":
        validate_archive(payload, "word/document.xml", max_expanded_bytes=LIBRARY_DOCX_MAX_EXPANDED_BYTES)
    elif suffix == ".pptx":
        validate_archive(payload, "ppt/presentation.xml", max_expanded_bytes=LIBRARY_PPTX_MAX_EXPANDED_BYTES)
    if suffix in {".txt", ".md"}:
        try:
            decoded = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ApiProblem(422, "invalid_text_encoding", "请上传 UTF-8 文本。") from None
        if not decoded.strip():
            raise ApiProblem(422, "invalid_document_text", "文件没有可检索的文字。")


def _check_file_on_disk(filename: str, path: Path) -> None:
    suffix = _validate_filename(filename)
    if suffix == ".pdf":
        validate_pdf_file(path)
    elif suffix == ".docx":
        validate_archive_file(path, "word/document.xml", max_expanded_bytes=LIBRARY_DOCX_MAX_EXPANDED_BYTES)
    elif suffix == ".pptx":
        validate_archive_file(path, "ppt/presentation.xml", max_expanded_bytes=LIBRARY_PPTX_MAX_EXPANDED_BYTES)
    elif suffix in {".txt", ".md"}:
        validate_text_file(path)


def _document_row(connection, key, *, include_content: bool = False):
    columns = f"{METADATA}, knowledge_base_id, digest"
    if include_content:
        columns += ", content, storage_uri"
    row = connection.execute(
        f"SELECT {columns} FROM library_document_versions WHERE org_id=? AND document_id=? AND version=?", key,
    ).fetchone()
    if row is None:
        raise ApiProblem(404, "document_not_found", "找不到可访问的资料版本。")
    return row


def _read_document_bytes(app_state, row) -> bytes:
    storage_uri = row["storage_uri"]
    object_store = getattr(app_state, "object_store", None)
    if storage_uri and object_store is not None:
        try:
            return object_store.read_bytes(storage_uri)
        except (FileNotFoundError, OSError, ValueError):
            raise ApiProblem(404, "document_not_found", "找不到可访问的资料原件。") from None
    return row["content"]


def _removed_version(connection, key) -> bool:
    return connection.execute(
        "SELECT 1 FROM library_deleted_versions WHERE org_id=? AND document_id=? AND version=?", key,
    ).fetchone() is not None


@contextmanager
def index_write(database, *, scope: tuple[str, str] | None = None):
    lock = getattr(database, "index_lock", None)
    try:
        if lock is not None:
            with lock.acquire(scope=scope):
                yield
        else:
            with library_index_write(database.path, scope=scope):
                yield
    except LibraryIndexBusy:
        raise ApiProblem(409, "library_index_busy", "资料库正在发布或删除资料，请稍后重试。") from None


def _visible_documents(
    connection, actor: Principal, document_id: str | None = None, *, base_id: str | None = None,
):
    return connection.execute(
        f"""SELECT {METADATA}, (
                SELECT folder_id FROM library_document_locations AS location
                WHERE location.org_id=doc.org_id AND location.document_id=doc.document_id
            ) AS folder_id, EXISTS (
                SELECT 1 FROM library_deleted_versions AS removed
                WHERE removed.org_id=doc.org_id AND removed.document_id=doc.document_id AND removed.version=doc.version
            ) AS deleting FROM library_document_versions AS doc
            WHERE org_id=? AND (status='published' OR ?) AND (? IS NULL OR document_id=?)
                AND (? IS NULL OR knowledge_base_id=?)
            ORDER BY uploaded_at DESC, document_id, version""",
        (actor.org_id, actor.allows("workspace:manage"), document_id, document_id, base_id, base_id),
    )


@router.get("/documents", response_model=list[LibraryDocument])
def list_library(request: Request, actor: Actor, knowledge_base_id: Identifier = DEFAULT_BASE_ID):
    require_permission(actor, "catalog:read")
    with request.app.state.database.connect() as connection:
        require_base(connection, actor, knowledge_base_id)
        return [document_view(row) for row in _visible_documents(connection, actor, base_id=knowledge_base_id)]


@router.get("/browser", response_model=LibraryBrowser)
def browse_library(request: Request, actor: Actor, knowledge_base_id: Identifier = DEFAULT_BASE_ID):
    require_permission(actor, "catalog:read")
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN")
        require_base(connection, actor, knowledge_base_id)
        documents = [
            LibraryEntry(**document_view(row).model_dump(), folder_id=row["folder_id"])
            for row in _visible_documents(connection, actor, base_id=knowledge_base_id)
        ]
        folders = {
            row["folder_id"]: LibraryFolder(**dict(row))
            for row in connection.execute(
                """SELECT folder_id, parent_id, name FROM library_folders
                WHERE org_id=? AND knowledge_base_id=? ORDER BY name_key, folder_id""",
                (actor.org_id, knowledge_base_id),
            )
        }
    if actor.allows("workspace:manage"):
        return LibraryBrowser(folders=list(folders.values()), documents=documents)
    visible: set[str] = set()
    for document in documents:
        folder_id = document.folder_id
        while folder_id is not None and folder_id in folders and folder_id not in visible:
            visible.add(folder_id)
            folder_id = folders[folder_id].parent_id
    return LibraryBrowser(
        folders=[folder for key, folder in folders.items() if key in visible], documents=documents,
    )


@router.get("/documents/{document_id}/versions", response_model=list[LibraryDocument])
def list_document_versions(document_id: Identifier, request: Request, actor: Actor):
    require_permission(actor, "catalog:read")
    with request.app.state.database.connect() as connection:
        rows = list(_visible_documents(connection, actor, document_id))
        if rows:
            require_base(connection, actor, rows[0]["knowledge_base_id"])
        versions = [document_view(row) for row in rows]
    if not versions:
        raise ApiProblem(404, "document_not_found", "找不到可访问的资料版本。")
    return versions


def _chunk_reader(request: Request) -> RagChunkReader:
    engine = request.app.state.rag_engine
    if not isinstance(engine, RagChunkReader):
        raise ApiProblem(503, "library_index_unavailable", "当前索引服务不支持查看片段，请稍后重试。")
    return engine


def _diagnostics_engine(request: Request) -> RagDiagnostics:
    engine = request.app.state.rag_engine
    if not isinstance(engine, RagDiagnostics):
        raise ApiProblem(503, "library_index_unavailable", "当前索引服务不支持检索测试，请稍后重试。")
    return engine


def _index_api_error(exc: Exception) -> ApiProblem:
    if isinstance(exc, RagModelUnavailable):
        return ApiProblem(503, exc.code, str(exc))
    if isinstance(exc, (RagModelInvalid, RagIndexConflict)):
        status = 503 if getattr(exc, "retryable", False) else 409
        return ApiProblem(status, getattr(exc, "code", "rag_unavailable"), str(exc))
    if isinstance(exc, RagDocumentError):
        return ApiProblem(422, "rag_document_invalid", "无法索引该文件，请检查资料内容。")
    return ApiProblem(503, "rag_unavailable", "资料索引服务暂不可用，请稍后重试。")


@router.get("/documents/{document_id}/{version}/chunks", response_model=ChunkPage)
def inspect_chunks(
    document_id: Identifier, version: Identifier, request: Request, actor: Actor,
    offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=50)] = 20,
):
    require_permission(actor, "catalog:read")
    key = (actor.org_id, document_id, version)
    database = request.app.state.database
    with database.connect() as connection:
        row = _document_row(connection, key)
        require_base(connection, actor, row["knowledge_base_id"])
        if row["status"] != "published":
            if not actor.allows("workspace:manage"):
                raise ApiProblem(404, "document_not_found", "找不到可访问的资料版本。")
            return ChunkPage(total=0, chunks=[])
        scope = index_scope(connection, actor.org_id, row["knowledge_base_id"])
        if row["knowledge_base_id"] != DEFAULT_BASE_ID and not scope["active_index_version"]:
            raise ApiProblem(503, "knowledge_base_index_pending", "此知识库的索引尚未接入。")

    try:
        page = _chunk_reader(request).read_chunks(
            actor.org_id,
            scope["knowledge_base_id"],
            scope["active_index_version"],
            DocumentVersionKey(document_id=document_id, version=version),
            offset=offset,
            limit=limit,
        )
    except (RagModelUnavailable, RagModelInvalid, RagIndexConflict, RagUnavailable) as exc:
        raise _index_api_error(exc) from None
    with database.connect() as connection:
        if _document_row(connection, key)["status"] != "published":
            raise ApiProblem(404, "document_not_found", "找不到可访问的资料版本。")
    return ChunkPage(
        total=page.total,
        chunks=[ChunkPreview(**chunk.model_dump()) for chunk in page.chunks],
    )


@router.post("/retrieval-test", response_model=RetrievalTest, dependencies=[Depends(check_origin)])
def test_library_retrieval(body: RetrievalQuery, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    database = request.app.state.database
    with database.connect() as connection:
        require_base(connection, actor, body.knowledge_base_id)
        scope = index_scope(connection, actor.org_id, body.knowledge_base_id)
        if body.knowledge_base_id != DEFAULT_BASE_ID and not scope["active_index_version"]:
            raise ApiProblem(503, "knowledge_base_index_pending", "此知识库的索引尚未接入。")
        documents = published_documents(connection, actor.org_id, base_id=body.knowledge_base_id)
    engine = _diagnostics_engine(request)
    target_index_version = body.index_version or scope["active_index_version"]
    target_model_id = scope["embedding_model_id"]
    if body.index_version and isinstance(engine, RagIndexManager):
        try:
            descriptor = engine.index_descriptor(actor.org_id, scope["knowledge_base_id"], body.index_version)
            target_model_id = descriptor.embedding_model_id
        except (RagModelUnavailable, RagModelInvalid, RagIndexConflict, RagUnavailable) as exc:
            raise _index_api_error(exc) from None
    t0 = time.perf_counter()
    try:
        result, used = inspect_retrieval(
            engine,
            RagContext(
                org_id=actor.org_id,
                user_id=actor.user_id,
                text=body.text,
                previous_turns=(),
                documents=documents,
                knowledge_base_id=scope["knowledge_base_id"],
                index_version=target_index_version,
                embedding_model_id=target_model_id,
            ),
            mode=body.mode,
            strategy=body.strategy,
            top_k=body.top_k,
        )
    except RagModelUnavailable as exc:
        raise ApiProblem(503, exc.code, str(exc)) from None
    except (RagIndexConflict, RagModelInvalid) as exc:
        raise ApiProblem(409, getattr(exc, "code", "index_task_conflict"), str(exc)) from None
    except RagUnavailable:
        raise ApiProblem(503, "rag_unavailable", "向量检索暂不可用，请稍后重试。") from None
    latency_ms = (time.perf_counter() - t0) * 1000.0
    with database.connect() as connection:
        current = index_scope(connection, actor.org_id, body.knowledge_base_id)
        visible = {
            (doc.document_id, doc.version)
            for doc in published_documents(connection, actor.org_id, base_id=body.knowledge_base_id)
        }
    if (body.index_version is None and current["active_index_version"] != scope["active_index_version"]) or not used <= visible:
        raise ApiProblem(409, "library_sources_changed", "查询期间资料状态有变更，请重新测试。")
    record_retrieval_event(
        database,
        org_id=actor.org_id,
        knowledge_base_id=body.knowledge_base_id,
        user_id=actor.user_id,
        source="retrieval_test",
        strategy=body.strategy or "hybrid",
        status=result.status,
        hit_count=len(result.hits),
        top_score=result.hits[0].score if result.hits else None,
        latency_ms=latency_ms,
        query_text=body.text,
    )
    return result


@router.get("/telemetry", response_model=RetrievalTelemetrySummary)
def get_library_telemetry(
    request: Request,
    actor: Actor,
    knowledge_base_id: Identifier = DEFAULT_BASE_ID,
    days: Annotated[int, Query(ge=1, le=90)] = 7,
) -> RetrievalTelemetrySummary:
    require_permission(actor, "workspace:manage")
    database = request.app.state.database
    with database.connect() as connection:
        require_base(connection, actor, knowledge_base_id)
    return summarize_retrieval_telemetry(
        database, org_id=actor.org_id, knowledge_base_id=knowledge_base_id, days=days,
    )

@router.post("/folders", response_model=LibraryFolder, status_code=201, dependencies=[Depends(check_origin)])
def create_folder(body: FolderCreate, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    knowledge_base_id = body.knowledge_base_id
    folder = LibraryFolder(folder_id=uuid4().hex, name=body.name, parent_id=body.parent_id)
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        require_base(connection, actor, knowledge_base_id)
        _require_folder(connection, actor.org_id, knowledge_base_id, folder.parent_id)
        added = connection.execute(
            """INSERT INTO library_folders (org_id, knowledge_base_id, folder_id, parent_id, name, name_key)
            VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING RETURNING folder_id""",
            (actor.org_id, knowledge_base_id, folder.folder_id, folder.parent_id, folder.name, folder.name.casefold()),
        ).fetchone()
        if added is None:
            raise ApiProblem(409, "library_folder_exists", "此目录已有同名文件夹。")
        record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action="library_folder_created",
                     request_id=request.state.request_id, object_id=folder.folder_id)
    return folder


@router.patch("/documents/{document_id}/location", status_code=204, dependencies=[Depends(check_origin)])
def move_document(document_id: Identifier, body: DocumentLocation, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    with request.app.state.database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        document = connection.execute(
            "SELECT knowledge_base_id FROM library_document_versions WHERE org_id=? AND document_id=? LIMIT 1",
            (actor.org_id, document_id),
        ).fetchone()
        if document is None:
            raise ApiProblem(404, "document_not_found", "找不到要移动的资料。")
        _require_folder(connection, actor.org_id, document["knowledge_base_id"], body.folder_id)
        if _location(connection, actor.org_id, document_id) != body.folder_id:
            _set_location(connection, actor.org_id, document_id, body.folder_id)
            record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action="library_document_moved",
                         request_id=request.state.request_id, object_id=document_id)
    return Response(status_code=204)


def _parse_identifier(value: str | None, *, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    try:
        return _IDENTIFIER_ADAPTER.validate_python(value)
    except ValidationError:
        raise ApiProblem(422, "invalid_request", f"字段 {field_name} 格式不正确。") from None


def _persist_uploaded_document(
    *,
    database,
    actor: Principal,
    request_id: str,
    filename: str,
    size_bytes: int,
    digest: str,
    storage_uri: str,
    version: str,
    requested_document_id: str | None,
    has_folder_field: bool,
    folder_id: str | None,
    base_id: str,
) -> LibraryDocument:
    document_identity = filename + "\0" + digest
    if base_id != DEFAULT_BASE_ID:
        document_identity = base_id + "\0" + document_identity
    identity = hashlib.sha256(document_identity.encode()).hexdigest()
    document_id = requested_document_id or f"manual-{identity[:32]}"
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        require_base(connection, actor, base_id)
        if requested_document_id and connection.execute(
            "SELECT 1 FROM library_document_versions WHERE org_id=? AND document_id=? AND knowledge_base_id=?",
            (actor.org_id, requested_document_id, base_id),
        ).fetchone() is None:
            raise ApiProblem(404, "document_not_found", "找不到要新增版本的资料。")
        key = (actor.org_id, document_id, version)
        if requested_document_id and _removed_version(connection, key):
            raise ApiProblem(409, "document_version_deleted", "此版本号已删除，请使用新版本号。")
        # Preserve upload deduplication while keeping previously deleted links invalid.
        copy_number = 0
        while _removed_version(connection, key):
            copy_number += 1
            key = (actor.org_id, f"{document_id}-{copy_number}", version)
        document_id = key[1]
        if has_folder_field:
            _require_folder(connection, actor.org_id, base_id, folder_id)
            existing = connection.execute(
                "SELECT 1 FROM library_document_versions WHERE org_id=? AND document_id=?",
                (actor.org_id, document_id),
            ).fetchone()
            if existing and _location(connection, actor.org_id, document_id) != folder_id:
                raise ApiProblem(409, "library_document_location_conflict", "此资料已在其他文件夹，可使用移动资料。")
            if not existing and folder_id is not None:
                _set_location(connection, actor.org_id, document_id, folder_id)
        inserted = connection.execute(
            """INSERT INTO library_document_versions
            (org_id, document_id, version, knowledge_base_id, filename, size_bytes, digest, content, storage_uri, status, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?) ON CONFLICT DO NOTHING RETURNING document_id""",
            (*key, base_id, filename, size_bytes, digest, b"", storage_uri, datetime.now(UTC).isoformat()),
        ).fetchone()
        row = _document_row(connection, key)
        if row["digest"] != digest or row["filename"] != filename:
            raise ApiProblem(409, "document_version_conflict", "此版本已有其他文件，请使用新版本号。")
        if inserted is not None:
            record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action="library_uploaded",
                         request_id=request_id, object_id=document_id)
        return document_view(row)


@router.post("/documents", response_model=LibraryDocument, status_code=201, dependencies=[Depends(check_origin)])
def upload_library(body: LibraryUpload, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    payload = body.file.bytes()
    _check_file(body.file, payload)
    digest = hashlib.sha256(payload).hexdigest()
    storage_uri = request.app.state.object_store.put_bytes(payload, digest=digest)
    return _persist_uploaded_document(
        database=request.app.state.database,
        actor=actor,
        request_id=request.state.request_id,
        filename=body.file.filename,
        size_bytes=len(payload),
        digest=digest,
        storage_uri=storage_uri,
        version=body.version,
        requested_document_id=body.document_id,
        has_folder_field="folder_id" in body.model_fields_set,
        folder_id=body.folder_id,
        base_id=body.knowledge_base_id,
    )


@router.post("/documents/upload", response_model=LibraryDocument, status_code=201, dependencies=[Depends(check_origin)])
async def upload_library_stream(
    request: Request,
    actor: Actor,
    filename: str | None = Query(default=None),
    version: str | None = Query(default=None),
    document_id: str | None = Query(default=None),
    folder_id: str | None = Query(default=None),
    knowledge_base_id: str | None = Query(default=None),
):
    """Streaming upload (up to 200 MB) supporting application/octet-stream and multipart/form-data."""
    require_permission(actor, "workspace:manage")
    content_type = request.headers.get("content-type", "").lower()
    with tempfile.NamedTemporaryFile(prefix="salespilot-upload-", delete=False) as temp_handle:
        temp_path = Path(temp_handle.name)
    try:
        fields: dict[str, str] = {}
        if content_type.startswith("multipart/form-data"):
            fields, part_filename, size_bytes, digest = await stream_multipart_upload(
                request, temp_path, max_bytes=LIBRARY_MAX_BYTES,
            )
            effective_filename = (filename or part_filename or "").strip()
        else:
            effective_filename = (filename or request.headers.get("x-filename") or "").strip()
            size_bytes, digest = await stream_raw_upload(request, temp_path, max_bytes=LIBRARY_MAX_BYTES)

        effective_version = _parse_identifier(
            version if version is not None else fields.get("version") or "v1", field_name="version",
        ) or "v1"
        effective_doc_id = _parse_identifier(
            document_id if document_id is not None else fields.get("document_id"), field_name="document_id",
        )
        raw_base = knowledge_base_id if knowledge_base_id is not None else fields.get("knowledge_base_id")
        effective_base_id = _parse_identifier(raw_base or DEFAULT_BASE_ID, field_name="knowledge_base_id") or DEFAULT_BASE_ID
        has_folder_field = ("folder_id" in request.query_params) or ("folder_id" in fields)
        raw_folder = folder_id if "folder_id" in request.query_params else fields.get("folder_id")
        effective_folder_id = _parse_identifier(raw_folder, field_name="folder_id")

        _check_file_on_disk(effective_filename, temp_path)
        storage_uri = request.app.state.object_store.put_file(temp_path, digest=digest)
    finally:
        temp_path.unlink(missing_ok=True)

    return _persist_uploaded_document(
        database=request.app.state.database,
        actor=actor,
        request_id=request.state.request_id,
        filename=effective_filename,
        size_bytes=size_bytes,
        digest=digest,
        storage_uri=storage_uri,
        version=effective_version,
        requested_document_id=effective_doc_id,
        has_folder_field=has_folder_field,
        folder_id=effective_folder_id,
        base_id=effective_base_id,
    )


def _document_base_id(database, key: tuple[str, str, str]) -> str | None:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT knowledge_base_id FROM library_document_versions WHERE org_id=? AND document_id=? AND version=?",
            key,
        ).fetchone()
    return row["knowledge_base_id"] if row else None


@router.post(
    "/documents/{document_id}/{version}/publish-async",
    response_model=PublishTask, status_code=202,
    dependencies=[Depends(check_origin)], responses={503: {"model": ApiError}},
)
def publish_library_async(
    document_id: Identifier, version: Identifier, request: Request, actor: Actor,
):
    require_permission(actor, "workspace:manage")
    app_state = request.app.state
    database = app_state.database
    key = (actor.org_id, document_id, version)
    with database.connect() as connection:
        row = _document_row(connection, key)
    base_id = row["knowledge_base_id"]
    if row["status"] == "published":
        now = datetime.now(UTC)
        return PublishTask(
            task_id=f"pub-done-{uuid4().hex[:8]}",
            org_id=actor.org_id,
            knowledge_base_id=base_id,
            document_id=document_id,
            version=version,
            stage="succeeded",
            created_at=now,
            updated_at=now,
        )
    if row["status"] != "draft":
        raise ApiProblem(409, "document_status_conflict", "此资料版本无法再发布。")
    if app_state.rag_engine is None:
        raise ApiProblem(503, "rag_unavailable", "资料索引服务尚未接入。文件已保存，可稍后发布。")
    if base_id != DEFAULT_BASE_ID and not isinstance(app_state.rag_engine, RagIndexManager):
        raise ApiProblem(503, "knowledge_base_index_pending", "此知识库的索引引擎尚未接入。")

    publish_queue = app_state.publish_queue
    task, created = publish_queue.create_or_reuse(
        org_id=actor.org_id,
        knowledge_base_id=base_id,
        document_id=document_id,
        version=version,
    )
    if created:
        request_id = request.state.request_id

        def _run_publish(task_id: str) -> None:
            try:
                publish_queue.update_stage(task_id, "parsing")
                with index_write(database, scope=(actor.org_id, base_id)):
                    _change_status(
                        app_state,
                        request_id,
                        actor,
                        key,
                        "publish",
                        on_stage=lambda stage: publish_queue.update_stage(task_id, stage),
                    )
                publish_queue.update_stage(task_id, "succeeded")
            except ApiProblem as exc:
                publish_queue.update_stage(
                    task_id, "failed", error_code=exc.code, error_message=str(exc),
                )

        publish_queue.submit(task.task_id, _run_publish)
    return task


@router.post(
    "/documents/{document_id}/{version}/{action}", response_model=LibraryDocument,
    dependencies=[Depends(check_origin)], responses={503: {"model": ApiError}},
)
def change_library_status(
    document_id: Identifier, version: Identifier, action: Literal["publish", "withdraw"],
    request: Request, actor: Actor,
):
    require_permission(actor, "workspace:manage")
    database = request.app.state.database
    key = (actor.org_id, document_id, version)
    base_id = _document_base_id(database, key)
    lock_scope = (actor.org_id, base_id) if base_id is not None else None
    with index_write(database, scope=lock_scope):
        return _change_status(request.app.state, request.state.request_id, actor, key, action)


@router.get("/publish-tasks", response_model=list[PublishTask])
def list_publish_tasks(
    request: Request,
    actor: Actor,
    knowledge_base_id: Identifier | None = Query(default=None),
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
):
    require_permission(actor, "workspace:manage")
    if knowledge_base_id is not None:
        with request.app.state.database.connect() as connection:
            require_base(connection, actor, knowledge_base_id)
    return request.app.state.publish_queue.list_tasks(
        actor.org_id, knowledge_base_id, limit=limit,
    )


@router.get("/publish-tasks/{task_id}", response_model=PublishTask)
def get_publish_task(task_id: Identifier, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    task = request.app.state.publish_queue.get(actor.org_id, task_id)
    if task is None:
        raise ApiProblem(404, "publish_task_not_found", "找不到发布任务。")
    return task


def _publish_target(app_state, actor, row):
    database = app_state.database
    engine = app_state.rag_engine
    if engine is None:
        raise ApiProblem(503, "rag_unavailable", "资料索引服务尚未接入。文件已保存，可稍后发布。")
    base_id = row["knowledge_base_id"]
    with database.connect() as connection:
        scope = index_scope(connection, actor.org_id, base_id)
    if base_id == DEFAULT_BASE_ID and scope["active_index_version"] == "legacy":
        target = document_view(row).model_copy(update={
            "knowledge_base_id": base_id,
            "index_version": "legacy",
            "embedding_model_id": scope["embedding_model_id"],
            "embedding_model_revision": scope["embedding_model_revision"],
        })
        return target, None, scope
    if not isinstance(engine, RagIndexManager):
        raise ApiProblem(503, "knowledge_base_index_pending", "此知识库的索引引擎尚未接入。")
    active = scope["active_index_version"]
    if active:
        descriptor = engine.index_descriptor(actor.org_id, base_id, active)
        if descriptor.state not in {"ready", "active"}:
            raise ApiProblem(503, "knowledge_base_index_pending", "此知识库的活动索引版本尚未就绪。")
    else:
        descriptor = engine.ensure_index_version(
            actor.org_id, base_id, scope["embedding_model_id"], activate=False,
        )
    target = document_view(row).model_copy(update={
        "knowledge_base_id": base_id,
        "index_version": descriptor.index_version,
        "embedding_model_id": descriptor.embedding_model_id,
        "embedding_model_revision": descriptor.embedding_model_revision,
        "dimensions": descriptor.dimensions,
    })
    return target, descriptor, scope


def _change_status(app_state, request_id: str, actor: Principal, key, action, *, on_stage=None):
    database = app_state.database
    expected, target = ("draft", "published") if action == "publish" else ("published", "withdrawn")
    descriptor = None
    scope = None
    if action == "publish":
        with database.connect() as connection:
            row = _document_row(connection, key, include_content=True)
        if row["status"] == target:
            return document_view(row)
        if row["status"] != expected:
            raise ApiProblem(409, "document_status_conflict", "此资料版本无法再发布。")
        payload = _read_document_bytes(app_state, row)
        scope = None
        try:
            target_document, descriptor, scope = _publish_target(app_state, actor, row)
            if on_stage is not None:
                on_stage("indexing")
            engine = app_state.rag_engine
            engine.index_document(actor.org_id, target_document, payload)
            if descriptor is not None:
                descriptor = engine.mark_ready(actor.org_id, descriptor.knowledge_base_id, descriptor.index_version)
                if scope is not None and scope["active_index_version"] != descriptor.index_version:
                    engine.mark_active(actor.org_id, descriptor.knowledge_base_id, descriptor.index_version)
        except (RagModelUnavailable, RagModelInvalid, RagIndexConflict, RagDocumentError, RagUnavailable) as exc:
            raise _index_api_error(exc) from None

    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = _document_row(connection, key)
        if row["status"] == target:
            return document_view(row)
        if row["status"] != expected:
            raise ApiProblem(409, "document_status_conflict", "当前资料状态无法执行此操作。")
        connection.execute(
            "UPDATE library_document_versions SET status=? WHERE org_id=? AND document_id=? AND version=?",
            (target, *key),
        )
        if descriptor is not None:
            if scope is not None and scope["active_index_version"] == descriptor.index_version:
                set_base_index_state(connection, actor.org_id, descriptor.knowledge_base_id, "available")
            else:
                record_index_activation(
                    connection,
                    org_id=actor.org_id,
                    user_id=actor.user_id,
                    base_id=descriptor.knowledge_base_id,
                    descriptor=descriptor,
                    reason="activated",
                    request_id=request_id,
                )
        record_audit(connection, org_id=actor.org_id, user_id=actor.user_id, action=f"library_{target}",
                     request_id=request_id, object_id=key[1])
        return document_view(_document_row(connection, key))


@router.delete(
    "/documents/{document_id}/{version}", status_code=204, dependencies=[Depends(check_origin)],
    responses={409: {"model": ApiError}, 503: {"model": ApiError}},
)
def delete_library_version(document_id: Identifier, version: Identifier, request: Request, actor: Actor):
    require_permission(actor, "workspace:manage")
    database = request.app.state.database
    scope = (actor.org_id, document_id, version)
    remover = request.app.state.rag_engine
    base_id = _document_base_id(database, scope)
    lock_scope = (actor.org_id, base_id) if base_id is not None else None
    orphan_uri: str | None = None
    with index_write(database, scope=lock_scope):
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                """SELECT knowledge_base_id, status, storage_uri FROM library_document_versions
                WHERE org_id=? AND document_id=? AND version=?""", scope,
            ).fetchone()
            if exists is None:
                if _removed_version(connection, scope):
                    return
                raise ApiProblem(404, "document_not_found", "找不到可访问的资料版本。")
            needs_index_removal = exists["knowledge_base_id"] == DEFAULT_BASE_ID or exists["status"] != "draft"
            if needs_index_removal and not isinstance(remover, RagIndexRemover):
                raise ApiProblem(503, "rag_index_removal_unavailable", "资料索引清理尚未接入，请稍后重试。")
            connection.execute(
                "INSERT INTO library_deleted_versions (org_id, document_id, version) VALUES (?, ?, ?) "
                "ON CONFLICT DO NOTHING", scope,
            )
            if not needs_index_removal:
                orphan_uri = _delete_document_record(connection, request, actor, scope)
            else:
                connection.execute(
                    "UPDATE library_document_versions SET status='withdrawn' WHERE org_id=? AND document_id=? AND version=?",
                    scope,
                )
        if not needs_index_removal:
            _gc_object(request.app.state, orphan_uri)
            return
        try:
            remover.remove_document(actor.org_id, DocumentVersionKey(document_id=document_id, version=version))
        except RagUnavailable:
            raise ApiProblem(503, "rag_index_removal_failed", "资料已停止检索，索引清理失败，请重试删除。") from None
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            orphan_uri = _delete_document_record(connection, request, actor, scope)
    _gc_object(request.app.state, orphan_uri)


def _delete_document_record(
    connection, request: Request, actor: Principal, scope: tuple[str, str, str],
) -> str | None:
    org_id, document_id, _ = scope
    row = connection.execute(
        "SELECT storage_uri FROM library_document_versions WHERE org_id=? AND document_id=? AND version=?",
        scope,
    ).fetchone()
    storage_uri = row["storage_uri"] if row else None
    connection.execute(
        "DELETE FROM library_document_versions WHERE org_id=? AND document_id=? AND version=?", scope,
    )
    connection.execute(
        """DELETE FROM library_document_locations
        WHERE org_id=? AND document_id=? AND NOT EXISTS (
            SELECT 1 FROM library_document_versions AS doc
            WHERE doc.org_id=library_document_locations.org_id
              AND doc.document_id=library_document_locations.document_id
        )""",
        (org_id, document_id),
    )
    record_audit(connection, org_id=org_id, user_id=actor.user_id, action="library_deleted",
                 request_id=request.state.request_id, object_id=document_id)
    if storage_uri:
        remaining = connection.execute(
            "SELECT 1 FROM library_document_versions WHERE storage_uri=? LIMIT 1", (storage_uri,),
        ).fetchone()
        if remaining is None:
            return storage_uri
    return None


def _gc_object(app_state, storage_uri: str | None) -> None:
    object_store = getattr(app_state, "object_store", None)
    if storage_uri and object_store is not None:
        object_store.delete(storage_uri)


@router.get("/documents/{document_id}/{version}/file")
def get_library_file(document_id: Identifier, version: Identifier, request: Request, actor: Actor):
    require_permission(actor, "catalog:read")
    with request.app.state.database.connect() as connection:
        row = _document_row(connection, (actor.org_id, document_id, version), include_content=True)
        require_base(connection, actor, row["knowledge_base_id"])
    if row["status"] != "published" and not (row["status"] == "draft" and actor.allows("workspace:manage")):
        raise ApiProblem(404, "document_not_found", "找不到可访问的资料版本。")
    storage_uri = row["storage_uri"]
    object_store = getattr(request.app.state, "object_store", None)
    if storage_uri and object_store is not None:
        if not object_store.exists(storage_uri):
            raise ApiProblem(404, "document_not_found", "找不到可访问的资料原件。")
        return streaming_download_response(
            object_store.open_stream(storage_uri),
            row["filename"],
            size_bytes=row["size_bytes"],
        )
    return download_response(row["content"], row["filename"])
