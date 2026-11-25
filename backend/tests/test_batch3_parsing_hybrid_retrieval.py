import asyncio
import hashlib
from datetime import UTC, datetime

import httpx

from app.contracts.rag_workspace import LibraryDocument, RagContext
from app.rag import HashingEmbeddings, RagSettings, SqliteRagEngine
from app.rag.chunking import build_chunks
from app.rag.facts import build_scopes
from app.rag.models import Chunk, ParsedSection
from app.rag.parsing import parse_document
from app.rag.rerank import HttpReranker

from test_rag_workspace import FIXTURES, LIBRARY, stub_application, upload_body
from workspace_helpers import PREFIX, browser, login

BASES = f"{PREFIX}/library/knowledge-bases"
RETRIEVAL = f"{PREFIX}/library/retrieval-test"


def _engine(tmp_path, *, reranker=None):
    settings = RagSettings(
        index_path=tmp_path / "b3-index.sqlite3",
        embedding_provider="hash",
        chat_provider="template",
        max_retries=0,
    )
    return SqliteRagEngine(
        settings,
        embedder=HashingEmbeddings(settings.embedding_dimensions),
        chat=None,
        reranker=reranker,
    )


def test_markdown_table_chunking_repeats_headers_and_assigns_parent_id():
    header = "| 型号 | 处理器 | 接口 | 防护等级 |"
    sep = "| --- | --- | --- | --- |"
    rows = [f"| MODEL-{i:03d} | Core i7 第12代 | 双千兆网口 + 4个DB9 | IP65 |" for i in range(40)]
    table_md = "\n".join([header, sep, *rows])
    sections = (ParsedSection(page=1, section="规格参数表", text=table_md),)

    chunks = build_chunks("org-a", "doc-tbl", "v1", sections)
    assert len(chunks) >= 3
    parent_ids = {chunk.parent_id for chunk in chunks}
    assert len(parent_ids) == 1 and next(iter(parent_ids)).startswith("par_")
    for chunk in chunks:
        lines = chunk.text.splitlines()
        assert lines[0] == header
        assert lines[1] == sep
        # Every data row stays intact (starts and ends with '|').
        for row in lines[2:]:
            assert row.startswith("| MODEL-") and row.endswith("|")


def test_tokenizer_aware_chunking_eliminates_bge_small_truncation(tmp_path):
    """Simulate a 512-token model (like bge-small-zh-v1.5) and verify 0% chunk truncation."""
    long_text = "；".join(f"参数项 {i}：工业相机支持高帧率采集与稳定传输能力" for i in range(90))
    sections = (ParsedSection(page=None, section="详细规格", text=long_text),)

    # Simulate 1 character == 1 token tokenizer (conservative upper bound for CJK).
    def fake_token_counts(texts: list[str]) -> list[int]:
        return [len(t) for t in texts]

    chunks = build_chunks(
        "org-a", "doc-tok", "v1", sections, max_tokens=512, token_counter=fake_token_counts,
    )
    assert len(chunks) >= 4
    counts = fake_token_counts([chunk.text for chunk in chunks])
    assert all(count <= 512 for count in counts), f"Found truncated chunks: {counts}"


def test_multipage_pdf_assembly_decoupling_combines_same_version_and_isolates_versions():
    """PDF page 1 & page 2 for the same model and version combine into one candidate while preserving pages."""
    p1 = Chunk(
        org_id="org-a", document_id="cam-pdf", version="v1", chunk_id="chk_p1",
        model="AX-920", page=1, section="第 1 页", ordinal=0,
        text="AX-920 采用 GigE 数据接口，分辨率 12 MP。",
    )
    p2 = Chunk(
        org_id="org-a", document_id="cam-pdf", version="v1", chunk_id="chk_p2",
        model="AX-920", page=2, section="第 2 页", ordinal=1,
        text="AX-920 最高帧率 120 fps，支持 PoE 供电。",
    )
    scopes = build_scopes([p1, p2])
    assert len(scopes) == 1
    assert scopes[0].assembly == "默认装配"
    pages = {fact.page for fact in scopes[0].facts}
    assert pages == {1, 2}

    # Different document version stays strictly isolated.
    p2_v2 = p2.model_copy(update={"version": "v2", "chunk_id": "chk_p2_v2"})
    split_scopes = build_scopes([p1, p2_v2])
    assert len(split_scopes) == 2
    assert {scope.version for scope in split_scopes} == {"v1", "v2"}


def test_external_ocr_parser_hook(monkeypatch):
    def fake_post(url, *, json, headers, timeout):
        assert url == "http://parser.local/parse"
        assert headers["Authorization"] == "Bearer parser-secret"
        assert json["filename"] == "scanned.pdf"
        return httpx.Response(200, json={
            "sections": [
                {"page": 1, "section": "第 1 页", "text": "扫描件第 1 页：SP-VC-120 GigE 12 MP"},
                {"page": 2, "section": "第 2 页", "text": "扫描件第 2 页：SP-VC-120 120 fps"},
            ],
        })

    monkeypatch.setenv("SALESPILOT_PARSER_BASE_URL", "http://parser.local")
    monkeypatch.setenv("SALESPILOT_PARSER_API_KEY", "parser-secret")
    monkeypatch.setattr("app.rag.parsing.httpx.post", fake_post)

    parsed = parse_document("scanned.pdf", b"%PDF-1.4 scanned", None)
    assert len(parsed) == 2
    assert parsed[0].page == 1 and "12 MP" in parsed[0].text
    assert parsed[1].page == 2 and "120 fps" in parsed[1].text


def test_hybrid_bm25_raw_retrieval_reranker_and_cross_version_ab(tmp_path, monkeypatch):
    """Test FTS5 BM25 + RRF hybrid recall, unknown-category raw_retrieval, HTTP reranker, and version A/B."""
    battery_file = tmp_path / "bb-200-battery.md"
    battery_file.write_text(
        "# BB-200 工业储能电池组\n\n型号 BB-200 标称容量 120 Ah，内置 BMS-Q900 保护板，支持 RS-485 通信。\n",
        encoding="utf-8",
    )
    cam_file = FIXTURES / "specs/sp-vision-cameras.md"

    def fake_rerank_post(url, *, json, headers, timeout):
        assert url == "http://rerank.local/rerank"
        # Give highest rerank score to whichever document contains "BMS-Q900".
        results = [
            {"index": i, "relevance_score": 0.98 if "BMS-Q900" in doc else 0.12}
            for i, doc in enumerate(json["documents"])
        ]
        return httpx.Response(200, json={"results": results})

    monkeypatch.setattr("app.rag.rerank.httpx.post", fake_rerank_post)
    reranker = HttpReranker(base_url="http://rerank.local", model="bge-reranker-v2-m3", max_retries=0)

    async def check():
        database = tmp_path / "hybrid.sqlite3"
        engine = _engine(tmp_path, reranker=reranker)
        app = stub_application(database, engine, models_dir=tmp_path / "models")
        async with app.router.lifespan_context(app), browser(app) as client:
            await login(client)
            kb_id = (await client.post(BASES, json={
                "name": "混合检索库", "description": "", "embedding_model_id": "workspace-default",
            })).json()["knowledge_base_id"]

            d_cam = (await client.post(LIBRARY, json=upload_body(cam_file, knowledge_base_id=kb_id))).json()
            d_bat = (await client.post(LIBRARY, json=upload_body(battery_file, knowledge_base_id=kb_id))).json()
            assert (await client.post(f"{LIBRARY}/{d_cam['document_id']}/v1/publish")).status_code == 200
            assert (await client.post(f"{LIBRARY}/{d_bat['document_id']}/v1/publish")).status_code == 200

            # In raw_retrieval mode, arbitrary new category & exact code BMS-Q900 recalls cleanly via hybrid/bm25 + rerank.
            raw_hybrid = await client.post(RETRIEVAL, json={
                "text": "需要支持 BMS-Q900 保护板和 RS-485 的 120 Ah 电池组",
                "knowledge_base_id": kb_id,
                "mode": "raw_retrieval",
                "strategy": "hybrid",
            })
            assert raw_hybrid.status_code == 200, raw_hybrid.text
            body = raw_hybrid.json()
            assert body["status"] == "retrieved"
            top = body["hits"][0]
            assert top["filename"] == battery_file.name
            assert top["vector_score"] is not None
            assert top["bm25_score"] is not None and top["bm25_score"] > 0
            assert top["rerank_score"] == 0.98
            assert -1.0 <= top["score"] <= 1.0

            # Pure BM25 lexical recall works without embedding vectors.
            raw_bm25 = await client.post(RETRIEVAL, json={
                "text": "BMS-Q900",
                "knowledge_base_id": kb_id,
                "mode": "raw_retrieval",
                "strategy": "bm25",
            })
            assert raw_bm25.status_code == 200, raw_bm25.text
            assert raw_bm25.json()["status"] == "retrieved"
            assert raw_bm25.json()["hits"][0]["filename"] == battery_file.name

            # Cross-index-version A/B testing before activation.
            rebuild = await client.post(f"{BASES}/{kb_id}/rebuild", json={"background": False})
            assert rebuild.status_code == 200
            new_ver = rebuild.json()["target_index_version"]
            ab = await client.post(RETRIEVAL, json={
                "text": "BMS-Q900",
                "knowledge_base_id": kb_id,
                "mode": "raw_retrieval",
                "strategy": "hybrid",
                "index_version": new_ver,
            })
            assert ab.status_code == 200, ab.text
            assert ab.json()["hits"][0]["index_version"] == new_ver

    asyncio.run(check())
