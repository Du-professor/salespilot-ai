import sqlite3
from pathlib import Path

import pytest

from app.api.errors import ApiProblem
from app.contracts.rag_workspace import RagDocumentError, RagUnavailable
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine

from conftest import FIXTURES, ORG, index_bytes, index_file, library_document


def test_markdown_index_is_idempotent_and_persists(engine, settings):
    document = index_file(engine, FIXTURES / "specs" / "sp-ipc-670.md", "ipc-670")
    first_ids = engine.store.chunk_ids(ORG, document.document_id, document.version)
    first_count = engine.store.count_chunks(ORG, document.document_id, document.version)
    assert first_count == len(first_ids) > 0

    index_file(engine, FIXTURES / "specs" / "sp-ipc-670.md", "ipc-670")
    assert engine.store.chunk_ids(ORG, document.document_id, document.version) == first_ids
    assert engine.store.count_chunks(ORG, document.document_id, document.version) == first_count

    reopened = SqliteRagEngine(settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=None)
    assert reopened.store.chunk_ids(ORG, document.document_id, document.version) == first_ids
    assert reopened.store.document_count(ORG, document.document_id, document.version) == 1


def test_index_store_releases_database_connections(engine, monkeypatch):
    connect = sqlite3.connect
    opened = []

    def track_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr("app.rag.store.sqlite3.connect", track_connection)
    document = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "release-connections")
    assert engine.store.count_chunks(ORG, document.document_id, document.version) > 0
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def test_markdown_keeps_sections_and_models(engine):
    document = index_file(engine, FIXTURES / "specs" / "sp-vision-cameras.md", "vision")
    chunks = [chunk for chunk, _ in engine.store.load_chunks(ORG, {(document.document_id, document.version)})]
    assert chunks
    assert all(chunk.page is None for chunk in chunks)
    assert all(chunk.section for chunk in chunks)
    camera = next(chunk for chunk in chunks if "SP-VC-120" in chunk.text)
    assert camera.model == "SP-VC-120"
    assert "SP-VC-120 标准装配" in camera.section


def test_unsupported_or_empty_document_raises_document_error(engine):
    with pytest.raises(RagDocumentError):
        engine.index_document(ORG, library_document("bad", filename="bad.exe"), b"MZ")
    with pytest.raises(RagDocumentError):
        engine.index_document(ORG, library_document("empty"), b"")


def test_parser_failures_map_to_core_errors(engine, monkeypatch):
    def unavailable(filename, payload, artifacts):
        raise ApiProblem(503, "document_parser_unavailable", "解析组件不可用。")

    monkeypatch.setattr("app.rag.parsing.parse_sections", unavailable)
    with pytest.raises(RagUnavailable):
        engine.index_document(ORG, library_document("unavailable"), b"%PDF-1.4")

    def invalid(filename, payload, artifacts):
        raise ApiProblem(422, "document_parse_failed", "资料无法解析。")

    monkeypatch.setattr("app.rag.parsing.parse_sections", invalid)
    with pytest.raises(RagDocumentError):
        engine.index_document(ORG, library_document("invalid"), b"%PDF-1.4")


def test_embedding_failure_is_retryable(engine, settings):
    class FlakyEmbeddings(HashingEmbeddings):
        def __init__(self, dimensions):
            super().__init__(dimensions)
            self.fail = True

        def embed_documents(self, texts):
            if self.fail:
                self.fail = False
                raise RagUnavailable("Embedding 暂时不可用。")
            return super().embed_documents(texts)

    flaky = FlakyEmbeddings(settings.embedding_dimensions)
    retry_engine = SqliteRagEngine(settings, embedder=flaky, chat=None)
    with pytest.raises(RagUnavailable):
        index_file(retry_engine, FIXTURES / "specs" / "sp-ipc-620.md", "retry")
    assert retry_engine.store.count_chunks(ORG, "retry", "v1") == 0

    index_file(retry_engine, FIXTURES / "specs" / "sp-ipc-620.md", "retry")
    assert retry_engine.store.count_chunks(ORG, "retry", "v1") > 0


def test_embedding_dimension_mismatch_is_unavailable(engine, settings):
    class WrongDimensions(HashingEmbeddings):
        def __init__(self):
            super().__init__(settings.embedding_dimensions)

        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

    broken = SqliteRagEngine(settings, embedder=WrongDimensions(), chat=None)
    with pytest.raises(RagUnavailable):
        index_file(broken, FIXTURES / "specs" / "sp-ipc-620.md", "wrong-dimensions")


def test_real_pdf_keeps_pages(engine, docling_ready):
    document = index_file(engine, FIXTURES / "rag" / "demo-vision-manual.pdf", "pdf-vision")
    chunks = [chunk for chunk, _ in engine.store.load_chunks(ORG, {(document.document_id, document.version)})]
    pages = sorted({chunk.page for chunk in chunks if chunk.page is not None})
    assert pages == [1, 2]
    page_one = "\n".join(chunk.text for chunk in chunks if chunk.page == 1)
    page_two = "\n".join(chunk.text for chunk in chunks if chunk.page == 2)
    assert "SP-VC-110" in page_one and "5 MP" in page_one
    assert "SP-VC-120" in page_two and "12 MP" in page_two and "120 fps" in page_two


def test_pdf_without_artifacts_is_unavailable(tmp_path):
    settings = RagSettings(index_path=tmp_path / "no-artifacts.sqlite3", docling_artifacts=None)
    engine = SqliteRagEngine(settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=None)
    path = FIXTURES / "rag" / "demo-vision-manual.pdf"
    with pytest.raises(RagUnavailable):
        index_bytes(engine, path.read_bytes(), "pdf-no-artifacts", filename=path.name)


def test_text_file_sections_are_indexed(engine):
    payload = "# SP-VC-120 标准装配\n\nSP-VC-120 使用 GigE 接口，分辨率为 12 MP，帧率为 120 fps。\n".encode()
    document = index_bytes(engine, payload, "text-camera", filename="camera.md")
    chunks = [chunk for chunk, _ in engine.store.load_chunks(ORG, {(document.document_id, document.version)})]
    assert any("SP-VC-120 标准装配" in (chunk.section or "") for chunk in chunks)
    assert any("12 MP" in chunk.text for chunk in chunks)
