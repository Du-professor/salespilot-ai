import hashlib
import json
import math
import os

import time
from pathlib import Path

import pytest

from app.contracts.rag_workspace import RagContext
from app.rag import HashingEmbeddings, LocalTransformerEmbeddings, RagSettings, SqliteRagEngine
from app.rag.model_catalog import LOCAL_MODELS, model_availability

from conftest import FIXTURES, ORG, USER, scoped_document

if os.getenv("SALESPILOT_RUN_EMBEDDING_BENCHMARK") != "1":
    pytest.skip(
        "set SALESPILOT_RUN_EMBEDDING_BENCHMARK=1 to run the 20-query benchmark",
        allow_module_level=True,
    )

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / ".salespilot" / "models"
ARTIFACTS = Path(os.getenv("SALESPILOT_DOCLING_ARTIFACTS", str(ROOT / ".salespilot" / "docling-artifacts")))
BENCHMARK = json.loads((FIXTURES / "rag" / "embedding-benchmark.json").read_text(encoding="utf-8"))
CAMERA = BENCHMARK["sets"][0]
IPC = BENCHMARK["sets"][1]
BENCHMARK_MODEL_ID = os.getenv("SALESPILOT_BENCHMARK_MODEL", "bge-m3")
MODEL_SPEC = LOCAL_MODELS[BENCHMARK_MODEL_ID]


class TimedEmbedder:
    def __init__(self, base):
        self.base = base
        self.document_seconds: list[float] = []
        self.query_seconds: list[float] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def embed_documents(self, texts):
        started = time.perf_counter()
        result = self.base.embed_documents(texts)
        self.document_seconds.append(time.perf_counter() - started)
        return result

    def embed_query(self, text):
        started = time.perf_counter()
        result = self.base.embed_query(text)
        self.query_seconds.append(time.perf_counter() - started)
        return result


def _settings(name: str, provider: str, model_id: str) -> RagSettings:
    return RagSettings(
        index_path=ROOT / ".salespilot" / f"benchmark-{name}.sqlite3",
        embedding_provider=provider,
        embedding_model_id=model_id,
        models_dir=MODELS_DIR,
        docling_artifacts=ARTIFACTS,
        chat_provider="template",
        max_retries=0,
        top_k=8,
    )


def _build(provider: str, model_id: str, model_spec=None):
    settings = _settings(provider, provider, model_id)
    if provider == "local":
        timed = TimedEmbedder(LocalTransformerEmbeddings(model_spec, settings, models_dir=MODELS_DIR))
        engine = SqliteRagEngine(settings, embedder=timed, chat=None)
        return engine, timed, settings
    engine = SqliteRagEngine(
        settings, embedder=HashingEmbeddings(settings.embedding_dimensions), chat=None,
    )
    return engine, None, settings


def _index(engine: SqliteRagEngine, knowledge_base_id: str, model_id: str, files: list[tuple[str, str]]):
    descriptor = engine.ensure_index_version(ORG, knowledge_base_id, model_id, activate=True)
    documents = []
    for filename, document_id in files:
        path = FIXTURES / filename
        payload = path.read_bytes()
        document = scoped_document(
            descriptor,
            document_id,
            filename=path.name,
            size=len(payload),
            digest=hashlib.sha256(payload).hexdigest(),
        )
        engine.index_document(ORG, document, payload)
        documents.append(document)
    return descriptor, tuple(documents)


def _hit_matches(hit, expected_file: str, *, page: int | None = None, section: str | None = None):
    if hit.document_id != expected_file:
        return False
    if page is not None:
        return hit.page == page
    if section is not None:
        return (hit.section or "").startswith(section)
    return True


def _run_query(engine: SqliteRagEngine, timed: TimedEmbedder | None, descriptor, documents, text, matcher):
    if timed is not None:
        before = len(timed.query_seconds)
    started = time.perf_counter()
    diagnostics, _ = engine.diagnose_detailed(RagContext(
        org_id=ORG,
        user_id=USER,
        text=text,
        previous_turns=(),
        documents=documents,
        knowledge_base_id=descriptor.knowledge_base_id,
        index_version=descriptor.index_version,
        embedding_model_id=descriptor.embedding_model_id,
    ))
    elapsed = time.perf_counter() - started
    embed_seconds = timed.query_seconds[-1] if timed is not None and len(timed.query_seconds) > before else 0.0
    top5 = diagnostics.hits[:5]
    matched = any(matcher(hit) for hit in top5)
    return {
        "matched": matched,
        "top5": top5,
        "embed_seconds": embed_seconds,
        "retrieval_seconds": max(elapsed - embed_seconds, 0.0),
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(0.95 * len(ordered)) - 1, 0)
    return ordered[index]


def _run_suite(provider: str, model_id: str, model_spec=None):
    engine, timed, _ = _build(provider, model_id, model_spec)
    camera_descriptor, camera_documents = _index(
        engine, f"bench-camera-{provider}", model_id,
        [(CAMERA["files"][0], "camera")],
    )
    ipc_descriptor, ipc_documents = _index(
        engine, f"bench-ipc-{provider}", model_id,
        [(name, Path(name).stem) for name in IPC["files"]],
    )

    records = []
    for text in CAMERA["queries"]:
        result = _run_query(
            engine, timed, camera_descriptor, camera_documents, text,
            lambda hit: _hit_matches(hit, "camera", page=CAMERA["expected_page"]),
        )
        records.append(("工业相机", camera_descriptor.knowledge_base_id, text, result))
    prefix = IPC["query_prefix"]
    ipc_expected_id = Path(IPC["expected_file"]).stem
    for item in IPC["queries"]:
        text = f"{prefix} {item['text']}"
        result = _run_query(
            engine, timed, ipc_descriptor, ipc_documents, text,
            lambda hit, section=item["expected_section"]: _hit_matches(hit, ipc_expected_id, section=section),
        )
        records.append(("工业计算机", ipc_descriptor.knowledge_base_id, text, result))

    for name, knowledge_base_id, text, result in records:
        top5 = [
            (
                hit.document_id,
                f"p{hit.page}" if hit.page else hit.section,
                round(hit.score, 4),
            )
            for hit in result["top5"]
        ]
        print(
            f"[{provider}] {name} kb={knowledge_base_id} hit={result['matched']} "
            f"embed={result['embed_seconds'] * 1000:.1f}ms "
            f"retrieval={result['retrieval_seconds'] * 1000:.1f}ms top5={top5}"
        )
    return records


def test_bge_m3_and_hash_use_same_documents_and_queries():
    settings = RagSettings(models_dir=MODELS_DIR)
    ready, reason, code = model_availability(MODEL_SPEC, settings)
    if not ready:
        pytest.skip(f"{MODEL_SPEC.model_id} 未就绪：{code}: {reason}")
    if not ARTIFACTS.is_dir():
        pytest.skip(f"Docling 工件目录不存在：{ARTIFACTS}")

    hash_records = _run_suite("hash", "workspace-default")
    bge_records = _run_suite("local", MODEL_SPEC.model_id, MODEL_SPEC)

    hash_camera = sum(1 for name, _, _, result in hash_records if name == "工业相机" and result["matched"])
    hash_ipc = sum(1 for name, _, _, result in hash_records if name == "工业计算机" and result["matched"])
    bge_camera = sum(1 for name, _, _, result in bge_records if name == "工业相机" and result["matched"])
    bge_ipc = sum(1 for name, _, _, result in bge_records if name == "工业计算机" and result["matched"])
    bge_total = bge_camera + bge_ipc

    query_times = [result["embed_seconds"] for _, _, _, result in bge_records]
    retrieval_times = [result["retrieval_seconds"] for _, _, _, result in bge_records]
    print(
        f"hash: camera={hash_camera}/10 ipc={hash_ipc}/10 total={hash_camera + hash_ipc}/20\n"
        f"{MODEL_SPEC.model_id}: camera={bge_camera}/10 ipc={bge_ipc}/10 total={bge_total}/20\n"
        f"{MODEL_SPEC.model_id} query embed p95={_p95(query_times) * 1000:.1f}ms "
        f"retrieval p95={_p95(retrieval_times) * 1000:.1f}ms"
    )
    assert bge_camera == 10
    if MODEL_SPEC.model_id == "bge-m3":
        assert bge_ipc >= 8
        assert bge_total >= 18
    else:
        # bge-small-zh-v1.5 is the lightweight candidate; its long-query
        # IPC recall is recorded for comparison rather than used as the 18/20 gate.
        assert bge_total >= 10
    assert hash_camera + hash_ipc >= 17