import asyncio
import base64
import hashlib
import json as jsonlib
import math

import httpx
from cryptography.fernet import Fernet

from app.main import create_app
from app.rag.models import ModelClaim
from app.settings import Settings

from workspace_helpers import ORIGIN, PASSWORD_A, PASSWORD_B, PREFIX, browser, login

CONFIGS = f"{PREFIX}/model-configs"
LIBRARY = f"{PREFIX}/library/documents"
BASES = f"{PREFIX}/library/knowledge-bases"
EMBEDDING_MODELS = f"{PREFIX}/library/embedding-models"
RETRIEVAL_TEST = f"{PREFIX}/library/retrieval-test"
ALLOWED_CUSTOM_URL = "http://127.0.0.1:9001/v1"
FAKE_KEY = "batch4-synthetic-key-001"


def app_for(path, key=None, **values):
    return create_app(Settings(
        database_path=path,
        rag_mode="demo",
        demo_password_a=PASSWORD_A,
        demo_password_b=PASSWORD_B,
        web_origins=(ORIGIN,),
        model_config_key=key,
        model_allowed_base_urls=(ALLOWED_CUSTOM_URL,),
        **values,
    ))


def _echo_chat(url, *, json, headers, timeout):
    prompt = json["messages"][-1]["content"]
    raw = prompt[prompt.index("\n[") + 1:prompt.rindex("\n只返回")]
    claims = [{key: item[key] for key in ModelClaim.model_fields} for item in jsonlib.loads(raw)]
    return httpx.Response(200, json={"choices": [{"message": {"content": jsonlib.dumps({"claims": claims}, ensure_ascii=False)}}]})


def _deterministic_embedding(text: str, dims: int = 8) -> list[float]:
    raw = [0.0] * dims
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for idx in range(dims):
            raw[idx] += (digest[idx] - 128) / 128.0
    if not any(raw):
        raw[0] = 1.0
    norm = math.sqrt(sum(val * val for val in raw)) or 1.0
    return [val / norm for val in raw]


def _make_fake_embeddings(calls: list, dims: int = 8):
    def fake_post(url, *, json, headers, timeout):
        calls.append((url, json["model"], headers.get("Authorization"), list(json["input"])))
        data = [
            {"index": idx, "embedding": _deterministic_embedding(text, dims)}
            for idx, text in enumerate(json["input"])
        ]
        return httpx.Response(200, json={"data": data})
    return fake_post


async def _add_config(client, **values):
    body = {
        "capability": "chat",
        "name": "测试配置",
        "provider": "custom",
        "model_id": "test-model",
        "base_url": ALLOWED_CUSTOM_URL,
        **values,
    }
    response = await client.post(CONFIGS, json=body)
    assert response.status_code == 201, response.text
    return response.json()


async def _apply(client, item, action):
    return await client.post(f"{CONFIGS}/{item['config_id']}/{action}", json={"revision": item["revision"]})


def _upload_file(filename: str, payload: bytes, *, knowledge_base_id: str = "default") -> dict:
    return {
        "file": {
            "filename": filename,
            "content_base64": base64.b64encode(payload).decode("ascii"),
        },
        "knowledge_base_id": knowledge_base_id,
    }


def _upload_md(filename: str, text: str, *, knowledge_base_id: str = "default") -> dict:
    return _upload_file(filename, text.encode("utf-8"), knowledge_base_id=knowledge_base_id)


def test_four_capabilities_coexist_with_independent_active_slots(tmp_path, monkeypatch):
    emb_calls: list = []
    rerank_calls: list = []
    parser_calls: list = []
    fake_embeddings = _make_fake_embeddings(emb_calls, dims=16)

    def dispatch_post(url, *, json, headers, timeout):
        if url.endswith("/chat/completions"):
            return _echo_chat(url, json=json, headers=headers, timeout=timeout)
        if url.endswith("/embeddings"):
            return fake_embeddings(url, json=json, headers=headers, timeout=timeout)
        if url.endswith("/rerank"):
            rerank_calls.append((url, json["model"], json["query"], list(json["documents"])))
            return httpx.Response(200, json={
                "results": [{"index": idx, "relevance_score": 0.9 - idx * 0.2} for idx in range(len(json["documents"]))]
            })
        if url.endswith("/parse"):
            parser_calls.append((url, json["filename"]))
            return httpx.Response(200, json={"sections": [{"page": 1, "section": "封面", "text": "探针解析通过"}]})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("httpx.post", dispatch_post)

    app = app_for(tmp_path / "four-caps.sqlite3", Fernet.generate_key().decode())

    async def check():
        async with app.router.lifespan_context(app), browser(app) as admin:
            await login(admin)
            chat_cfg = await _add_config(admin, capability="chat", name="售前主对话", provider="deepseek", base_url=None, api_key=FAKE_KEY)
            emb_cfg_1 = await _add_config(admin, capability="embedding", name="远端向量 A", model_id="bge-m3-api")
            emb_cfg_2 = await _add_config(admin, capability="embedding", name="远端向量 B", model_id="bge-large-api")
            rerank_cfg = await _add_config(admin, capability="rerank", name="远端重排", model_id="bge-reranker-v2-m3")
            parser_cfg = await _add_config(admin, capability="parser", name="MinerU 解析", model_id="mineru-v2")

            for cfg in (chat_cfg, emb_cfg_1, emb_cfg_2, rerank_cfg, parser_cfg):
                tested = await _apply(admin, cfg, "test")
                assert tested.status_code == 200 and tested.json()["test_status"] == "passed", tested.text

            # Embedding probe must auto-discover dimensions
            emb_list = (await admin.get(CONFIGS, params={"capability": "embedding"})).json()["configs"]
            assert len(emb_list) == 2
            assert all(item["dimensions"] == 16 for item in emb_list)

            # Activate all 4 capability slots independently
            for cfg in (chat_cfg, emb_cfg_1, rerank_cfg, parser_cfg):
                activated = await _apply(admin, cfg, "activate")
                assert activated.status_code == 200 and activated.json()["active"] is True

            # Verify all 4 active simultaneously via /active?capability=...
            for cap, expected_model in (
                ("chat", "test-model"),
                ("embedding", "bge-m3-api"),
                ("rerank", "bge-reranker-v2-m3"),
                ("parser", "mineru-v2"),
            ):
                active = (await admin.get(f"{CONFIGS}/active", params={"capability": cap})).json()
                assert active["mode"] == "configured" and active["model_id"] == expected_model

            # Switching embedding active slot must NOT disturb chat / rerank / parser
            switched = await _apply(admin, emb_cfg_2, "activate")
            assert switched.status_code == 200 and switched.json()["active"] is True
            all_configs = (await admin.get(CONFIGS)).json()["configs"]
            active_by_cap = {item["capability"]: item["model_id"] for item in all_configs if item["active"]}
            assert active_by_cap == {
                "chat": "test-model",
                "embedding": "bge-large-api",
                "rerank": "bge-reranker-v2-m3",
                "parser": "mineru-v2",
            }

    asyncio.run(check())


def test_remote_embedding_config_powers_kb_publish_and_retrieval(tmp_path, monkeypatch):
    emb_calls: list = []
    monkeypatch.setattr("app.rag.embeddings.httpx.post", _make_fake_embeddings(emb_calls, dims=8))
    app = app_for(tmp_path / "remote-emb.sqlite3", Fernet.generate_key().decode())

    async def check():
        async with app.router.lifespan_context(app), browser(app) as admin:
            await login(admin)
            emb = await _add_config(
                admin, capability="embedding", name="私有 BGE-M3", provider="ollama",
                base_url=None, model_id="bge-m3", api_key=None,
            )
            tested = await _apply(admin, emb, "test")
            assert tested.status_code == 200 and tested.json()["dimensions"] == 8

            # Verify cfg-<config_id> appears in /library/embedding-models
            options = (await admin.get(EMBEDDING_MODELS)).json()
            remote_model_id = f"cfg-{emb['config_id']}"
            matched = [item for item in options if item["model_id"] == remote_model_id]
            assert len(matched) == 1 and matched[0]["selectable"] is True
            assert "8 维" in matched[0]["name"]

            # Create KB bound to remote embedding model, upload & publish document
            kb = (await admin.post(BASES, json={
                "name": "远程向量产品库",
                "description": "2C8G 远程 Embedding 测试",
                "embedding_model_id": remote_model_id,
            })).json()
            assert kb["embedding_model_id"] == remote_model_id
            kb_id = kb["knowledge_base_id"]

            doc = (await admin.post(LIBRARY, json=_upload_md(
                "sp-remote-camera.md",
                "# SP-REM-800 视觉相机\n型号 SP-REM-800 支持 240 fps 高速采集，防护等级 IP67，接口 10GigE。\n",
                knowledge_base_id=kb_id,
            ))).json()
            pub = await admin.post(f"{LIBRARY}/{doc['document_id']}/v1/publish")
            assert pub.status_code == 200, pub.text

            # Retrieval test must query remote embedding endpoint and return hits
            calls_before_query = len(emb_calls)
            res = await admin.post(RETRIEVAL_TEST, json={
                "text": "SP-REM-800 10GigE 240 fps",
                "knowledge_base_id": kb_id,
                "mode": "raw_retrieval",
                "strategy": "hybrid",
            })
            assert res.status_code == 200, res.text
            body = res.json()
            assert body["hits"] and "SP-REM-800" in body["hits"][0]["text"]
            assert len(emb_calls) > calls_before_query
            assert emb_calls[-1][0] == "http://127.0.0.1:11434/v1/embeddings"
            assert emb_calls[-1][1] == "bge-m3"

    asyncio.run(check())


def test_active_rerank_config_applies_http_reranker_dynamically(tmp_path, monkeypatch):
    rerank_calls: list = []

    def fake_rerank(url, *, json, headers, timeout):
        rerank_calls.append((url, json["model"], json["query"], list(json["documents"])))
        # Give highest score to the chunk mentioning SP-RERANK-TARGET
        results = []
        for idx, text in enumerate(json["documents"]):
            score = 0.98 if "SP-RERANK-TARGET" in text else 0.12
            results.append({"index": idx, "relevance_score": score})
        return httpx.Response(200, json={"results": results})

    monkeypatch.setattr("app.rag.rerank.httpx.post", fake_rerank)
    app = app_for(tmp_path / "remote-rerank.sqlite3", Fernet.generate_key().decode())

    async def check():
        async with app.router.lifespan_context(app), browser(app) as admin:
            await login(admin)
            for name, text in (
                ("doc-noise.md", "# 工业光源配件\n型号 SP-NOISE-01 通用电源线 24V 工业视觉配套。\n"),
                ("doc-target.md", "# 精密视觉主机\n型号 SP-RERANK-TARGET 工业视觉主控箱，24V 供电，支持多路相机同步。\n"),
            ):
                doc = (await admin.post(LIBRARY, json=_upload_md(name, text))).json()
                assert (await admin.post(f"{LIBRARY}/{doc['document_id']}/v1/publish")).status_code == 200

            rerank_cfg = await _add_config(
                admin, capability="rerank", name="云端 Reranker", model_id="bge-reranker-v2-m3",
            )
            assert (await _apply(admin, rerank_cfg, "test")).json()["test_status"] == "passed"
            assert (await _apply(admin, rerank_cfg, "activate")).json()["active"] is True

            res = await admin.post(RETRIEVAL_TEST, json={
                "text": "24V 工业视觉",
                "knowledge_base_id": "default",
                "mode": "raw_retrieval",
                "strategy": "hybrid",
            })
            assert res.status_code == 200, res.text
            hits = res.json()["hits"]
            assert hits and "SP-RERANK-TARGET" in hits[0]["text"]
            assert hits[0]["rerank_score"] is not None and hits[0]["rerank_score"] >= 0.95
            assert len(rerank_calls) >= 2  # 1 probe + 1 retrieval

    asyncio.run(check())


def test_active_parser_config_routes_pdf_to_external_ocr_parser(tmp_path, monkeypatch):
    parser_calls: list = []

    def fake_parser(url, *, json, headers, timeout):
        parser_calls.append((url, json["filename"], headers.get("Authorization")))
        return httpx.Response(200, json={
            "sections": [
                {
                    "page": 3,
                    "section": "第 3 页 扫描件规格表",
                    "text": "型号 SP-OCR-990 防爆测温仪，测温范围 -40~1200℃，防爆等级 Ex d IIC T6。",
                }
            ]
        })

    monkeypatch.setattr("app.rag.parsing.httpx.post", fake_parser)
    app = app_for(tmp_path / "remote-parser.sqlite3", Fernet.generate_key().decode())

    async def check():
        async with app.router.lifespan_context(app), browser(app) as admin:
            await login(admin)
            parser_cfg = await _add_config(
                admin, capability="parser", name="MinerU OCR 服务", model_id="mineru-ocr", api_key=FAKE_KEY,
            )
            assert (await _apply(admin, parser_cfg, "test")).json()["test_status"] == "passed"
            assert (await _apply(admin, parser_cfg, "activate")).json()["active"] is True

            # Minimal valid PDF payload (would yield empty text without OCR)
            fake_pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
            uploaded = await admin.post(LIBRARY, json=_upload_file("scanned-catalog.pdf", fake_pdf))
            assert uploaded.status_code == 201, uploaded.text
            doc_id = uploaded.json()["document_id"]

            pub = await admin.post(f"{LIBRARY}/{doc_id}/v1/publish")
            assert pub.status_code == 200, pub.text
            assert any(call[1] == "scanned-catalog.pdf" and call[2] == f"Bearer {FAKE_KEY}" for call in parser_calls)

            res = await admin.post(RETRIEVAL_TEST, json={
                "text": "SP-OCR-990 防爆等级",
                "knowledge_base_id": "default",
                "mode": "raw_retrieval",
                "strategy": "hybrid",
            })
            assert res.status_code == 200, res.text
            hits = res.json()["hits"]
            assert hits and "SP-OCR-990" in hits[0]["text"] and hits[0]["page"] == 3

    asyncio.run(check())
