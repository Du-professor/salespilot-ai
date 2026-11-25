import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Response
from fastapi.responses import StreamingResponse

from app.contracts.rag_workspace import DocumentVersionKey, LibraryDocument, RagDocumentError

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".md": "text/markdown",
    ".txt": "text/plain",
}
METADATA = "document_id, version, filename, size_bytes, uploaded_at, status, knowledge_base_id, digest"


class LibraryIndexBusy(RuntimeError):
    """A library publish or delete is updating this workspace's index."""


def _is_lock_error(exc: sqlite3.OperationalError) -> bool:
    return exc.sqlite_errorcode in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or "locked" in str(exc).lower()


@contextmanager
def library_index_write(database_path: Path, *, scope: tuple[str, str] | None = None):
    # Keep slow indexing outside the workspace data transaction. SQLite releases
    # these separate write locks if the request's worker exits.
    # Global lock (scope=None) takes EXCLUSIVE so maintenance/tests block all KBs;
    # scoped locks take SHARED on the global lock + IMMEDIATE on the per-(org, kb) lock,
    # allowing concurrent operations across distinct knowledge bases.
    global_lock_path = database_path.with_name(database_path.stem + "-library-operations.sqlite3")
    with ExitStack() as stack:
        global_conn = stack.enter_context(closing(sqlite3.connect(global_lock_path, timeout=0.25)))
        try:
            if scope is None:
                global_conn.execute("BEGIN EXCLUSIVE")
            else:
                global_conn.execute("BEGIN DEFERRED")
                global_conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
        except sqlite3.OperationalError as exc:
            if _is_lock_error(exc):
                raise LibraryIndexBusy() from None
            raise

        if scope is not None:
            digest = hashlib.sha256(f"{scope[0]}\0{scope[1]}".encode("utf-8")).hexdigest()[:16]
            scoped_lock_path = database_path.with_name(f"{database_path.stem}-library-op-{digest}.sqlite3")
            scoped_conn = stack.enter_context(closing(sqlite3.connect(scoped_lock_path, timeout=0.25)))
            try:
                scoped_conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if _is_lock_error(exc):
                    raise LibraryIndexBusy() from None
                raise
        yield


def document_view(row: sqlite3.Row) -> LibraryDocument:
    fields = {
        field: row[field]
        for field in LibraryDocument.model_fields
        if field != "index_status" and field in row.keys()
    }
    fields.setdefault("knowledge_base_id", "default")
    if "deleting" in row.keys() and row["deleting"]:
        fields["status"] = "deleting"
    return LibraryDocument(**fields, index_status="ready" if row["status"] == "published" else "pending")


class LibraryRebuildSource:
    """Provide the published snapshot and source bytes to the RAG rebuild core."""

    def __init__(self, database, object_store=None):
        self.database = database
        self.object_store = object_store

    def snapshot(self, org_id: str, knowledge_base_id: str) -> tuple[LibraryDocument, ...]:
        with self.database.connect() as connection:
            return published_documents(connection, org_id, base_id=knowledge_base_id)

    def is_published(self, org_id: str, key: DocumentVersionKey) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM library_document_versions AS doc
                WHERE doc.org_id=? AND doc.document_id=? AND doc.version=? AND doc.status='published'
                  AND NOT EXISTS (
                      SELECT 1 FROM library_deleted_versions AS removed
                      WHERE removed.org_id=doc.org_id AND removed.document_id=doc.document_id
                        AND removed.version=doc.version
                  )""",
                (org_id, key.document_id, key.version),
            ).fetchone()
        return row is not None

    def read_payload(self, org_id: str, key: DocumentVersionKey) -> bytes:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT content, storage_uri, status, EXISTS (
                    SELECT 1 FROM library_deleted_versions AS removed
                    WHERE removed.org_id=doc.org_id AND removed.document_id=doc.document_id
                      AND removed.version=doc.version
                ) AS deleting
                FROM library_document_versions AS doc
                WHERE org_id=? AND document_id=? AND version=?""",
                (org_id, key.document_id, key.version),
            ).fetchone()
        if row is None or row["status"] != "published" or row["deleting"]:
            raise RagDocumentError("重建时找不到可使用的已发布原件。")
        storage_uri = row["storage_uri"]
        if storage_uri and self.object_store is not None:
            try:
                return self.object_store.read_bytes(storage_uri)
            except (FileNotFoundError, OSError, ValueError):
                raise RagDocumentError("重建时找不到对象存储中的原件。") from None
        return row["content"]


def published_documents(
    connection: sqlite3.Connection,
    org_id: str,
    *,
    base_id: str | None = None,
    allowed_bases: set[str] | None = None,
) -> tuple[LibraryDocument, ...]:
    rows = connection.execute(
        f"""SELECT {METADATA} FROM library_document_versions
        WHERE org_id=? AND status='published' AND (? IS NULL OR knowledge_base_id=?)
        ORDER BY document_id, version""", (org_id, base_id, base_id),
    )
    docs = tuple(map(document_view, rows))
    if allowed_bases is not None:
        docs = tuple(doc for doc in docs if doc.knowledge_base_id in allowed_bases)
    return docs


def download_response(content: bytes | str, filename: str) -> Response:
    return Response(
        content,
        media_type=CONTENT_TYPES[Path(filename).suffix.lower()],
        headers={
            "Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename, safe=""),
            "X-Content-Type-Options": "nosniff",
        },
    )


def streaming_download_response(
    stream: Iterator[bytes], filename: str, *, size_bytes: int | None = None,
) -> StreamingResponse:
    headers = {
        "Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename, safe=""),
        "X-Content-Type-Options": "nosniff",
    }
    if size_bytes is not None:
        headers["Content-Length"] = str(size_bytes)
    return StreamingResponse(
        stream,
        media_type=CONTENT_TYPES[Path(filename).suffix.lower()],
        headers=headers,
    )
