import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.contracts.rag_workspace import IndexDescriptor, LibraryDocument, PreviousTurn, RagContext
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "fixtures"
ORG = "org-a"
USER = "user-a"


def library_document(
    document_id: str,
    *,
    version: str = "v1",
    filename: str = "manual.md",
    size: int = 1,
) -> LibraryDocument:
    return LibraryDocument(
        document_id=document_id,
        version=version,
        filename=filename,
        size_bytes=size,
        uploaded_at=datetime(2026, 1, 1, tzinfo=UTC),
        status="published",
        index_status="ready",
    )


def index_bytes(
    engine: SqliteRagEngine,
    payload: bytes,
    document_id: str,
    *,
    version: str = "v1",
    filename: str = "manual.md",
    org_id: str = ORG,
) -> LibraryDocument:
    document = library_document(document_id, version=version, filename=filename, size=len(payload))
    engine.index_document(org_id, document, payload)
    return document


def index_file(
    engine: SqliteRagEngine,
    path: Path,
    document_id: str,
    *,
    version: str = "v1",
    org_id: str = ORG,
) -> LibraryDocument:
    return index_bytes(engine, path.read_bytes(), document_id, version=version, filename=path.name, org_id=org_id)


def ask(
    engine: SqliteRagEngine,
    text: str,
    documents: tuple[LibraryDocument, ...],
    *,
    previous_turns: tuple[PreviousTurn, ...] = (),
    org_id: str = ORG,
    knowledge_base_id: str = "default",
    index_version: str | None = None,
    embedding_model_id: str | None = None,
):
    return engine.recommend(RagContext(
        org_id=org_id,
        user_id=USER,
        text=text,
        previous_turns=previous_turns,
        documents=documents,
        knowledge_base_id=knowledge_base_id,
        index_version=index_version,
        embedding_model_id=embedding_model_id,
    ))


def scoped_document(
    descriptor: IndexDescriptor,
    document_id: str,
    *,
    version: str = "v1",
    filename: str = "manual.md",
    size: int = 1,
    digest: str | None = None,
) -> LibraryDocument:
    return LibraryDocument(
        document_id=document_id,
        version=version,
        filename=filename,
        size_bytes=size,
        uploaded_at=datetime(2026, 1, 1, tzinfo=UTC),
        status="published",
        index_status="ready",
        knowledge_base_id=descriptor.knowledge_base_id,
        index_version=descriptor.index_version,
        embedding_model_id=descriptor.embedding_model_id,
        embedding_model_revision=descriptor.embedding_model_revision,
        dimensions=descriptor.dimensions,
        digest=digest,
    )


def index_scoped(
    engine: SqliteRagEngine,
    descriptor: IndexDescriptor,
    payload: bytes,
    document_id: str,
    *,
    version: str = "v1",
    filename: str = "manual.md",
) -> LibraryDocument:
    document = scoped_document(
        descriptor,
        document_id,
        version=version,
        filename=filename,
        size=len(payload),
        digest=hashlib.sha256(payload).hexdigest(),
    )
    engine.index_document(descriptor.org_id, document, payload)
    return document


@pytest.fixture
def settings(tmp_path: Path) -> RagSettings:
    artifacts = os.getenv("SALESPILOT_DOCLING_ARTIFACTS")
    return RagSettings(
        index_path=tmp_path / "index.sqlite3",
        embedding_provider="hash",
        chat_provider="template",
        max_retries=0,
        docling_artifacts=Path(artifacts) if artifacts else None,
    )


@pytest.fixture
def engine(settings: RagSettings) -> SqliteRagEngine:
    return SqliteRagEngine(settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=None)


@pytest.fixture
def cases() -> dict:
    return json.loads((FIXTURES / "rag" / "cases.json").read_text(encoding="utf-8"))


@pytest.fixture
def docling_ready(settings: RagSettings) -> RagSettings:
    if settings.docling_artifacts is None or not settings.docling_artifacts.is_dir():
        pytest.skip("SALESPILOT_DOCLING_ARTIFACTS is not configured")
    pytest.importorskip("docling")
    return settings


@pytest.fixture
def vision(engine: SqliteRagEngine) -> LibraryDocument:
    return index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "vision")


@pytest.fixture
def ipc_documents(engine: SqliteRagEngine) -> tuple[LibraryDocument, ...]:
    return tuple(
        index_file(engine, FIXTURES / "specs" / name, name.removesuffix(".md"))
        for name in ("sp-ipc-620.md", "sp-ipc-670.md", "sp-ipc-680.md", "rtx-a2000-12gb.md")
    )