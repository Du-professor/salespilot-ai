import asyncio
import hashlib
import json
import struct
from datetime import UTC, datetime
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from app.api.errors import ApiProblem
from app.contracts.rag_workspace import (
    ChunkReadPage,
    DiagnosticHit,
    DocumentVersionKey,
    EmbeddingModelDescriptor,
    IndexAvailability,
    IndexDescriptor,
    IndexSwitchPlan,
    LibraryDocument,
    RagContext,
    RagIndexConflict,
    RagModelUnavailable,
    RagRecommendation,
    ReadChunk,
    RebuildTask,
    RetrievalDiagnostics,
)
from app.infrastructure.library import LibraryIndexBusy, LibraryRebuildSource, library_index_write
from app.infrastructure.uploads import validate_archive
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine
from app.rag.embeddings import build_embeddings
from app.rag.model_catalog import BGE_SMALL_ZH, write_manifest
from app.rag.model_registry import RagModelRegistry
from app.rag import store as store_module

from test_rag_workspace import FIXTURES, LIBRARY, stub_application, upload_body
from workspace_helpers import PREFIX, browser, login

BASES = f"{PREFIX}/library/knowledge-bases"
RETRIEVAL = f"{PREFIX}/library/retrieval-test"


def _engine(tmp_path):
    settings = RagSettings(
        index_path=tmp_path / "p0-index.sqlite3",
        embedding_provider="hash",
        chat_provider="template",
        max_retries=0,
    )
    return SqliteRagEngine(settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=None)


def test_p0_1_multi_version_document_rebuild_and_activate(tmp_path):
    """P0-1: Rebuilding a KB with multiple published versions of the same document preserves both versions."""
    v1_file = FIXTURES / "rag/ax-920-v1.md"
    v2_file = FIXTURES / "rag/ax-920-v2.md"

    async def check():
        database = tmp_path / "multi-version.sqlite3"
        engine = _engine(tmp_path)
        app = stub_application(database, engine, models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            created = await client.post(BASES, json={
                "name": "多版本知识库", "description": "", "embedding_model_id": "workspace-default",
            })
            assert created.status_code == 201
            base_id = created.json()["knowledge_base_id"]

            up1 = await client.post(LIBRARY, json=upload_body(v1_file, version="v1", knowledge_base_id=base_id))
            assert up1.status_code == 201, up1.text
            doc_id = up1.json()["document_id"]
            up2 = await client.post(LIBRARY, json=upload_body(
                v2_file, document_id=doc_id, version="v2", knowledge_base_id=base_id,
            ))
            assert up2.status_code == 201, up2.text

            assert (await client.post(f"{LIBRARY}/{doc_id}/v1/publish")).status_code == 200
            assert (await client.post(f"{LIBRARY}/{doc_id}/v2/publish")).status_code == 200

            rebuild = await client.post(f"{BASES}/{base_id}/rebuild", json={"background": False})
            assert rebuild.status_code == 200, rebuild.text
            task = rebuild.json()
            assert task["state"] == "succeeded"
            assert task["total"] == 2 and task["completed"] == 2 and task["failed"] == 0

            # P0-2: business DB index_state must remain 'available' rather than stuck at 'building'.
            bases = {item["knowledge_base_id"]: item for item in (await client.get(BASES)).json()}
            assert bases[base_id]["index_state"] == "available"

            # Activate must pass strict scope validation with zero missing/changed diffs.
            activate = await client.post(
                f"{BASES}/{base_id}/indexes/{task['target_index_version']}/activate",
                json={"allow_stale": False},
            )
            assert activate.status_code == 200, activate.text
            assert activate.json()["scope_changed"] is False
            assert activate.json()["differences"] == []

            # Both versions have distinct chunks corresponding to their own source text.
            v1_chunks = (await client.get(f"{LIBRARY}/{doc_id}/v1/chunks")).json()
            v2_chunks = (await client.get(f"{LIBRARY}/{doc_id}/v2/chunks")).json()
            assert v1_chunks["total"] > 0 and v2_chunks["total"] > 0
            assert v1_chunks["chunks"][0]["text"] != v2_chunks["chunks"][0]["text"]

    asyncio.run(check())


def test_p0_2_and_p0_3_rebuild_idempotency_concurrency_and_reconciliation(tmp_path):
    """P0-2 & P0-3: Same task_id retry is idempotent, concurrent KB rebuilds conflict, split-brain reconciles."""
    sample = FIXTURES / "specs/sp-vision-cameras.md"

    async def check():
        database = tmp_path / "consistency.sqlite3"
        engine = _engine(tmp_path)
        app = stub_application(database, engine, models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            base_id = (await client.post(BASES, json={
                "name": "一致性测试库", "description": "", "embedding_model_id": "workspace-default",
            })).json()["knowledge_base_id"]
            doc_id = (await client.post(
                LIBRARY, json=upload_body(sample, knowledge_base_id=base_id),
            )).json()["document_id"]
            assert (await client.post(f"{LIBRARY}/{doc_id}/v1/publish")).status_code == 200
            initial_active = (await client.get(f"{BASES}/{base_id}/index")).json()["active_index_version"]

            # Same task_id retried returns the exact same task without 503 or orphan index versions.
            first = await client.post(f"{BASES}/{base_id}/rebuild", json={
                "task_id": "task-idem-01", "background": False,
            })
            assert first.status_code == 200, first.text
            versions_before = len((await client.get(f"{BASES}/{base_id}/indexes")).json())
            second = await client.post(f"{BASES}/{base_id}/rebuild", json={
                "task_id": "task-idem-01", "background": False,
            })
            assert second.status_code == 200, second.text
            assert second.json()["target_index_version"] == first.json()["target_index_version"]
            assert len((await client.get(f"{BASES}/{base_id}/indexes")).json()) == versions_before

            # Reusing task_id with another knowledge base conflicts with 409.
            conflict = await client.post(f"{BASES}/default/rebuild", json={
                "task_id": "task-idem-01", "background": False,
            })
            assert conflict.status_code == 409
            assert conflict.json()["code"] == "index_task_conflict"

            # Concurrent rebuild while another task is running on same KB is rejected with 409/503 conflict.
            engine.store.set_rebuild_task_state("task-idem-01", "running")
            concurrent = await client.post(f"{BASES}/{base_id}/rebuild", json={"background": False})
            assert concurrent.status_code in {409, 503}
            assert concurrent.json()["code"] == "index_task_conflict"
            engine.store.set_rebuild_task_state("task-idem-01", "succeeded")

            # Simulate split-brain: index store flipped to rebuilt version while business DB still points to initial_active.
            rebuilt_ver = first.json()["target_index_version"]
            engine.store.activate_index_version("demo", base_id, rebuilt_ver)
            assert engine.availability("demo", base_id).active_index_version == rebuilt_ver

            # GET /index reconciles index store back to business DB's authoritative pointer.
            reconciled = await client.get(f"{BASES}/{base_id}/index")
            assert reconciled.status_code == 200, reconciled.text
            assert reconciled.json()["active_index_version"] == initial_active
            assert reconciled.json()["ready"] is True
            assert engine.availability("demo", base_id).active_index_version == initial_active

    asyncio.run(check())


def test_p0_2_rebuild_prevents_ghost_resurrection_on_concurrent_delete(tmp_path):
    """P0-2: Document withdrawn/deleted mid-rebuild is removed from target index version."""
    sample = FIXTURES / "specs/sp-vision-cameras.md"
    payload = sample.read_bytes()
    doc = LibraryDocument(
        document_id="doc-ghost",
        version="v1",
        filename=sample.name,
        size_bytes=len(payload),
        uploaded_at=datetime.now(UTC),
        status="published",
        index_status="ready",
        knowledge_base_id="kb-ghost",
        digest=hashlib.sha256(payload).hexdigest(),
    )

    class VanishingSource:
        def __init__(self):
            self.deleted = False

        def snapshot(self, org_id, kb_id):
            return (doc,)

        def read_payload(self, org_id, key):
            # Simulate concurrent deletion right after payload is read.
            self.deleted = True
            return payload

        def is_published(self, org_id, key):
            return not self.deleted

    engine = _engine(tmp_path)
    task = engine.start_rebuild("org-a", "kb-ghost", "workspace-default", VanishingSource(), background=False)
    assert task.state == "failed"
    assert task.failed == 1
    assert engine.store.index_version_counts("org-a", "kb-ghost", task.target_index_version) == (0, 0)


def test_p1_1_read_chunks_and_spec_query_do_not_unpack_vectors(tmp_path, monkeypatch):
    """P1-1: Paginated chunk reading and spec lookup must not read or unpack vector BLOBs."""
    sample = FIXTURES / "specs/sp-vision-cameras.md"
    payload = sample.read_bytes()
    engine = _engine(tmp_path)
    desc = engine.ensure_index_version("org-a", "kb-fast", "workspace-default", activate=True)
    doc = LibraryDocument(
        document_id="cam-1",
        version="v1",
        filename=sample.name,
        size_bytes=len(payload),
        uploaded_at=datetime.now(UTC),
        status="published",
        index_status="ready",
        knowledge_base_id="kb-fast",
        index_version=desc.index_version,
        embedding_model_id=desc.embedding_model_id,
        embedding_model_revision=desc.embedding_model_revision,
        dimensions=desc.dimensions,
        digest=hashlib.sha256(payload).hexdigest(),
    )
    engine.index_document("org-a", doc, payload)

    def forbid_unpack(*args, **kwargs):
        raise AssertionError("Vector unpacking must not be invoked for text-only chunk reads or spec queries")

    monkeypatch.setattr(store_module, "_unpack", forbid_unpack)

    page = engine.read_chunks(
        "org-a", "kb-fast", desc.index_version, DocumentVersionKey(document_id="cam-1", version="v1"),
        offset=0, limit=1,
    )
    assert page.total >= 1 and len(page.chunks) == 1

    reply = engine.recommend(RagContext(
        org_id="org-a",
        user_id="user-a",
        text="SP-VC-120 的帧率是多少？",
        previous_turns=(),
        documents=(doc,),
        knowledge_base_id="kb-fast",
        index_version=desc.index_version,
        embedding_model_id=desc.embedding_model_id,
    ))
    assert reply.status == "recommended"
    assert "120" in reply.answer


def test_p1_5_model_registry_recovers_after_files_arrive(tmp_path):
    """P1-5: Negative cache auto-invalidates when model files/manifest appear without restarting process."""
    models_dir = tmp_path / "models"
    settings = RagSettings(index_path=tmp_path / "reg.sqlite3", models_dir=models_dir)
    calls = {"count": 0}

    class FakeLocalEmbedder:
        def __init__(self, spec):
            self.spec = spec

        def check(self):
            calls["count"] += 1
            if not (models_dir / self.spec.directory_name / "model_manifest.json").is_file():
                raise RagModelUnavailable("missing manifest", code="embedding_model_incomplete")
            return self.spec.descriptor()

    registry = RagModelRegistry(
        settings,
        HashingEmbeddings(256),
        models_dir=models_dir,
        local_factory=FakeLocalEmbedder,
        error_ttl_seconds=300.0,
    )
    with pytest.raises(RagModelUnavailable):
        registry.check(BGE_SMALL_ZH.model_id)

    # Now populate the model directory & manifest; signature changes and cache auto-recovers.
    target = models_dir / BGE_SMALL_ZH.directory_name
    target.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (target / name).write_bytes(b"stub")
    write_manifest(BGE_SMALL_ZH, target, files=["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"])

    recovered = registry.check(BGE_SMALL_ZH.model_id)
    assert recovered.model_id == BGE_SMALL_ZH.model_id
    assert calls["count"] == 2


def test_p1_6_scoped_library_locks_allow_parallel_kbs_and_respect_global_lock(tmp_path):
    """P1-6: Distinct knowledge bases can write concurrently while same KB and global locks serialize."""
    db_path = tmp_path / "locks.sqlite3"
    with library_index_write(db_path, scope=("org-a", "kb-1")):
        # Another KB in the same org succeeds concurrently.
        with library_index_write(db_path, scope=("org-a", "kb-2")):
            pass
        # Same KB is blocked immediately.
        with pytest.raises(LibraryIndexBusy):
            with library_index_write(db_path, scope=("org-a", "kb-1")):
                pass

    # Global lock (scope=None) blocks scoped locks.
    with library_index_write(db_path):
        with pytest.raises(LibraryIndexBusy):
            with library_index_write(db_path, scope=("org-a", "kb-1")):
                pass


def test_p2_1_protocol_only_engine_works_without_sqlite_rag_engine(tmp_path):
    """P2-1: API routes depend on RagIndexManager/RagChunkReader/RagDiagnostics protocols, not SqliteRagEngine."""
    sample = FIXTURES / "specs/sp-vision-cameras.md"

    class ProtocolOnlyEngine:
        def index_document(self, org_id, document, payload):
            pass

        def recommend(self, context):
            return RagRecommendation(status="insufficient_evidence", answer="none")

        def diagnose(self, context):
            doc = context.documents[0]
            return RetrievalDiagnostics(
                status="retrieved",
                message="ok",
                knowledge_base_id=context.knowledge_base_id,
                index_version=context.index_version or "legacy",
                total=1,
                hits=(DiagnosticHit(
                    chunk_id="chk-proto-1",
                    document_id=doc.document_id,
                    version=doc.version,
                    knowledge_base_id=context.knowledge_base_id,
                    index_version=context.index_version or "legacy",
                    embedding_model_id=context.embedding_model_id or "workspace-default",
                    page=1,
                    section="规格",
                    text="协议引擎命中片段",
                    score=0.95,
                    truncated=True,
                ),),
            )

        def read_chunks(self, org_id, kb_id, index_version, key, *, offset=0, limit=20):
            return ChunkReadPage(
                total=1,
                chunks=(ReadChunk(chunk_id="chk-proto-1", page=1, section="规格", text="协议引擎片段", truncated=True),),
            )

        def availability(self, org_id, knowledge_base_id):
            return IndexAvailability(
                knowledge_base_id=knowledge_base_id,
                state="legacy",
                active_index_version="legacy",
                embedding_model_id="workspace-default",
                embedding_model_revision="workspace",
                dimensions=256,
                ready=True,
            )

        def model_availability(self, model_id):
            return True, None, None

        def ensure_index_version(self, org_id, kb_id, model_id, *, index_version=None, activate=False):
            return self.index_descriptor(org_id, kb_id, index_version or "legacy")

        def index_descriptor(self, org_id, kb_id, index_version):
            return IndexDescriptor(
                org_id=org_id, knowledge_base_id=kb_id, index_version=index_version,
                embedding_model_id="workspace-default", embedding_model_revision="workspace",
                dimensions=256, state="active", document_count=1, chunk_count=1,
            )

        def list_index_versions(self, org_id, kb_id):
            return (self.index_descriptor(org_id, kb_id, "legacy"),)

        def mark_ready(self, org_id, kb_id, index_version):
            return self.index_descriptor(org_id, kb_id, index_version)

        def start_rebuild(self, org_id, kb_id, model_id, source, *, task_id=None, background=True):
            raise NotImplementedError

        def rebuild_status(self, task_id):
            raise NotImplementedError

        def list_rebuilds(self, org_id, kb_id, *, limit=10):
            return ()

        def resume_rebuild(self, task_id, source, *, background=True):
            raise NotImplementedError

        def recover_rebuilds(self):
            pass

        def validate_scope(self, org_id, kb_id, index_version, documents):
            return IndexSwitchPlan(
                org_id=org_id, knowledge_base_id=kb_id, target_index_version=index_version,
                embedding_model_id="workspace-default", embedding_model_revision="workspace",
                dimensions=256, ready=True,
            )

        def mark_active(self, org_id, kb_id, index_version, *, documents=None, allow_stale=False):
            return self.validate_scope(org_id, kb_id, index_version, documents or ())

        def retire(self, org_id, kb_id, index_version):
            pass

    async def check():
        app = stub_application(tmp_path / "proto.sqlite3", ProtocolOnlyEngine(), models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            doc_id = (await client.post(LIBRARY, json=upload_body(sample))).json()["document_id"]
            assert (await client.post(f"{LIBRARY}/{doc_id}/v1/publish")).status_code == 200

            chunks = await client.get(f"{LIBRARY}/{doc_id}/v1/chunks")
            assert chunks.status_code == 200
            assert chunks.json()["chunks"][0]["truncated"] is True

            diag = await client.post(RETRIEVAL, json={"text": "测试协议引擎"})
            assert diag.status_code == 200, diag.text
            assert diag.json()["hits"][0]["truncated"] is True

            avail = await client.get(f"{BASES}/default/index")
            assert avail.status_code == 200
            assert avail.json()["ready"] is True

    asyncio.run(check())


def test_p2_2_and_p2_4_openai_dimensions_and_forged_zip_bomb():
    """P2-2 & P2-4: OpenAI explicit dimensions propagate; forged ZIP headers cannot bypass decompression cap."""
    explicit = build_embeddings(RagSettings(
        embedding_provider="openai", embedding_dimensions=1536, embedding_api_key="k",
    ))
    assert explicit.dimensions == 1536

    implicit = build_embeddings(RagSettings(embedding_provider="openai", embedding_api_key="k"))
    assert implicit.dimensions is None

    # Create a valid ZIP whose uncompressed payload is 20 KB, then tamper file_size header down to 10 bytes.
    buf = BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "A" * 20_000)
    raw = bytearray(buf.getvalue())
    # Patch uncompressed size field in local file header (offset 22) and central directory header (offset +24 from PK\x01\x02).
    raw[22:26] = struct.pack("<I", 10)
    cd = raw.index(b"PK\x01\x02")
    raw[cd + 24:cd + 28] = struct.pack("<I", 10)

    with pytest.raises(ApiProblem) as exc:
        validate_archive(bytes(raw), "word/document.xml", max_expanded_bytes=1024)
    assert exc.value.status == 422
    assert exc.value.code == "invalid_archive"
