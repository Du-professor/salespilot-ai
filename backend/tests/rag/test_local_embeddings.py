import hashlib
import json
import math
import socket
from pathlib import Path

import pytest

from app.contracts.rag_workspace import DocumentVersionKey, RagModelUnavailable
from app.rag import LocalTransformerEmbeddings, RagSettings, SqliteRagEngine
from app.rag.model_catalog import (
    BGE_M3,
    BGE_SMALL_ZH,
    MANIFEST_NAME,
    model_availability,
    resolve_model_spec,
)

from conftest import ORG, scoped_document

MODELS_DIR = Path(__file__).resolve().parents[3] / ".salespilot" / "models"


def test_catalog_uses_pinned_revisions_and_rejects_unknown_ids():
    settings = RagSettings(models_dir=MODELS_DIR)
    assert BGE_M3.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert BGE_SMALL_ZH.revision == "7999e1d3359715c523056ef9478215996d62a620"
    assert BGE_M3.dimensions == 1024 and BGE_SMALL_ZH.dimensions == 512
    assert BGE_SMALL_ZH.max_tokens == 512
    assert resolve_model_spec("bge-m3", settings).directory_name == "bge-m3"
    with pytest.raises(RagModelUnavailable) as raised:
        resolve_model_spec("not-registered", settings)
    assert raised.value.code == "embedding_model_unavailable"
    assert raised.value.retryable is False


def test_missing_directory_and_incomplete_manifest_are_retryable(tmp_path):
    settings = RagSettings(models_dir=tmp_path)
    ready, reason, code = model_availability(BGE_M3, settings)
    assert not ready and code == "embedding_model_directory_missing" and "缺失" in reason

    directory = tmp_path / "bge-m3"
    directory.mkdir()
    (directory / MANIFEST_NAME).write_text(json.dumps({
        "model_id": "bge-m3", "revision": BGE_M3.revision, "files": {"model.safetensors": "x"},
    }), encoding="utf-8")
    ready, _, code = model_availability(BGE_M3, settings)
    assert not ready and code == "embedding_model_incomplete"

    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"x")
    (directory / MANIFEST_NAME).write_text(json.dumps({
        "model_id": "bge-m3", "revision": "wrong", "files": {"model.safetensors": "x"},
    }), encoding="utf-8")
    ready, _, code = model_availability(BGE_M3, settings)
    assert not ready and code == "embedding_model_invalid"

    embedder = LocalTransformerEmbeddings(BGE_M3, settings)
    with pytest.raises(RagModelUnavailable) as raised:
        embedder.embed_query("检查")
    assert raised.value.code == "embedding_model_invalid"


def _require_bge_small():
    settings = RagSettings(models_dir=MODELS_DIR)
    ready, reason, code = model_availability(BGE_SMALL_ZH, settings)
    if not ready:
        pytest.skip(f"BGE Small 本地模型未就绪：{code}: {reason}")
    return settings


def test_bge_small_check_query_and_document_vectors(monkeypatch):
    settings = _require_bge_small()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    embedder = LocalTransformerEmbeddings(BGE_SMALL_ZH, settings, models_dir=MODELS_DIR)
    descriptor = embedder.check()
    assert descriptor.dimensions == 512
    assert descriptor.revision == BGE_SMALL_ZH.revision
    assert descriptor.runtime["torch"] != "missing"

    query = embedder.embed_query("需要 GigE 工业相机，至少 8 MP。")
    document = embedder.embed_documents(["SP-VC-120 使用 GigE 接口，12 MP。"])[0]
    assert len(query) == len(document) == 512
    assert all(math.isfinite(value) for value in (*query, *document))
    assert sum(value * value for value in query) == pytest.approx(1.0, abs=1e-5)
    assert sum(value * value for value in document) == pytest.approx(1.0, abs=1e-5)
    counts = embedder.token_counts(["长" * 600])
    assert counts[0] > BGE_SMALL_ZH.max_tokens


def test_bge_small_loads_with_external_network_blocked(monkeypatch):
    settings = _require_bge_small()
    original = socket.socket.connect

    def blocked(sock, address):
        host = address[0] if isinstance(address, tuple) else ""
        if host in {"127.0.0.1", "::1", "localhost"}:
            return original(sock, address)
        raise OSError("external network disabled for local model check")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    embedder = LocalTransformerEmbeddings(BGE_SMALL_ZH, settings, models_dir=MODELS_DIR)
    descriptor = embedder.check()
    assert descriptor.dimensions == 512
    assert math.isfinite(embedder.embed_query("离线查询")[0])


def test_local_index_avoids_truncation_and_marks_oversized_chunks(tmp_path, monkeypatch):
    settings = _require_bge_small()
    engine_settings = RagSettings(
        index_path=tmp_path / "local-index.sqlite3",
        embedding_provider="local",
        embedding_model_id=BGE_SMALL_ZH.model_id,
        models_dir=MODELS_DIR,
        max_retries=0,
    )
    engine = SqliteRagEngine(engine_settings, chat=None)
    descriptor = engine.ensure_index_version(ORG, "kb-local", BGE_SMALL_ZH.model_id, activate=True)
    payload = ("# 长文档\n\n" + "相机规格。" * 300).encode("utf-8")
    document = scoped_document(
        descriptor, "long", filename="long.md", size=len(payload),
        digest=hashlib.sha256(payload).hexdigest(),
    )
    engine.index_document(ORG, document, payload)
    page = engine.read_chunks(ORG, "kb-local", descriptor.index_version,
                              DocumentVersionKey(document_id="long", version="v1"), limit=50)
    # Tokenizer-aware chunking splits cleanly below BGE-Small's 512-token ceiling (0% truncation).
    assert page.total >= 3
    assert all(not chunk.truncated for chunk in page.chunks)

    # If a pre-built chunk still exceeds max_tokens, engine still records truncated=True accurately.
    from app.rag.chunking import build_chunks as real_build_chunks

    monkeypatch.setattr(
        "app.rag.engine.build_chunks",
        lambda org_id, doc_id, ver, sections, **kwargs: real_build_chunks(org_id, doc_id, ver, sections, max_tokens=0),
    )
    forced_doc = scoped_document(
        descriptor, "forced-long", filename="forced-long.md", size=len(payload),
        digest=hashlib.sha256(payload).hexdigest(),
    )
    engine.index_document(ORG, forced_doc, payload)
    forced_page = engine.read_chunks(
        ORG, "kb-local", descriptor.index_version,
        DocumentVersionKey(document_id="forced-long", version="v1"), limit=50,
    )
    assert forced_page.total > 0
    assert any(chunk.truncated for chunk in forced_page.chunks)
